"""Split each judge-stage request into hops from rj_prof / OMNI_HOP lines (never part of the PR).

    python rj_prof_analysis.py <server.log> --client-jsonl <client.jsonl> [--stage 1] [--json out.json]

One row per request sent to --stage (client_send0), in order. With
--client-jsonl, only requests from sessions the client marked warmup=false are
kept (the client warms up before *each* clip, so warmups are interleaved with
measured turns). Without it, --skip drops the first N requests; that is only
correct when all warmups come first. Hops, all in ms:

  send        client_send0 -> client_send1   orchestrator hands the request to ZMQ
  to_engine   client_send1 -> eng_recv       ZMQ + engine input thread decode
  to_add      eng_recv -> eng_add            input queue -> busy loop add_request
  to_sched    eng_add.t1 -> eng_sched.t1     waiting in the scheduler, schedule() call
  exec        eng_exec.t -> eng_exec.t1      model_executor.execute_model (runner, forward, pooler)
  update      eng_update.t0 -> eng_outq      update_from_output + queue put
  to_encode   eng_outq -> eng_encode.t1      output IO thread wake-up + encode
  to_client   eng_encode.t1 -> client_decode.t1  ZMQ + client decode
  to_orch     client_decode.t1 -> OMNI_HOP output  client queue -> orchestrator poll loop
  total       client_send0 -> OMNI_HOP output
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
from pathlib import Path

LINE = re.compile(r"(RJ_PROF|OMNI_HOP|OMNI_TIMING) (\w+) (.*)$")
KV = re.compile(r"(\w+)=(\S+)")

HOPS = [
    ("send", ("client_send0", "t"), ("client_send1", "t")),
    ("to_engine", ("client_send1", "t"), ("eng_recv", "t")),
    ("to_add", ("eng_recv", "t"), ("eng_add", "t")),
    ("to_sched", ("eng_add", "t1"), ("eng_sched", "t1")),
    ("sched_to_exec", ("eng_sched", "t1"), ("eng_exec", "t")),
    ("exec", ("eng_exec", "t"), ("eng_exec", "t1")),
    ("exec_to_update", ("eng_exec", "t1"), ("eng_update", "t0")),
    ("update", ("eng_update", "t0"), ("eng_outq", "t")),
    ("to_encode", ("eng_outq", "t"), ("eng_encode", "t1")),
    ("to_client", ("eng_encode", "t1"), ("client_decode", "t1")),
    ("to_orch", ("client_decode", "t1"), ("output", "t")),
    ("total", ("client_send0", "t"), ("output", "t")),
]


def parse(path: Path, stage: str) -> list[dict]:
    events = []
    for raw in path.read_text(errors="replace").splitlines():
        m = LINE.search(raw)
        if not m:
            continue
        kind, name, rest = m.groups()
        kv = dict(KV.findall(rest))
        if kind == "OMNI_TIMING":
            continue
        if kind == "OMNI_HOP" and not (name == "output" and kv.get("stage") == stage and kv.get("finished") == "1"):
            continue
        # client_decode runs in the API process and does not know the stage; rows() matches it by request id.
        if kind == "RJ_PROF" and name not in ("client_decode", "other_exec") and kv.get("stage") != stage:
            continue
        events.append((name, kv))
    return events


def rows(events: list[tuple[str, dict]]) -> list[dict]:
    out = []
    i = 0
    while i < len(events):
        name, kv = events[i]
        if name != "client_send0":
            i += 1
            continue
        req = kv["req"]
        row: dict[str, dict] = {"client_send0": kv}
        j = i + 1
        while j < len(events):
            n2, kv2 = events[j]
            if n2 == "client_send0":
                break
            base = kv2.get("req", "").split(":")[0]
            if n2 in ("run", "other_exec"):
                row.setdefault(n2, []).append(kv2)
            elif n2 in ("eng_exec",) or base == req or (n2 == "client_decode" and req in kv2.get("req", "")):
                row.setdefault(n2, kv2)
            if n2 == "output" and base == req:
                break
            j += 1
        out.append({"req": req, "ev": row})
        i = j if j > i else i + 1
    return out


def hop_ms(row: dict) -> dict[str, float | None]:
    ev = row["ev"]
    res = {}
    for hop, (a, ak), (b, bk) in HOPS:
        if a in ev and b in ev and ak in ev[a] and bk in ev[b]:
            res[hop] = (float(ev[b][bk]) - float(ev[a][ak])) * 1000
        else:
            res[hop] = None
    return res


def exec_breakdown(row: dict) -> dict[str, float]:
    """Runner steps (run step=...) and other stages' execute_model overlapping this request's exec window."""
    ev = row["ev"]
    if "eng_exec" not in ev:
        return {}
    lo, hi = float(ev["eng_exec"]["t"]), float(ev["eng_exec"]["t1"])
    res: dict[str, float] = {}
    for kv in ev.get("run", []):
        t0, t1 = float(kv["t0"]), float(kv["t1"])
        if lo <= t0 <= hi:
            key = "run:" + kv["step"]
            res[key] = res.get(key, 0.0) + (t1 - t0) * 1000
    for kv in ev.get("other_exec", []):
        t0, t1 = float(kv["t"]), float(kv["t1"])
        overlap = min(hi, t1) - max(lo, t0)
        if overlap > 0:
            key = f"overlap:stage{kv['stage']}"
            res[key] = res.get(key, 0.0) + overlap * 1000
    return res


def session_of(request_id: str) -> str | None:
    """duplex-s.<urlsafe b64 session id>.e.<epoch>.r.<role> -> session id."""
    parts = request_id.split(".")
    if len(parts) < 2 or parts[0] != "duplex-s":
        return None
    token = parts[1]
    try:
        return base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
    except Exception:
        return None


def measured_sessions(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["session_id"] for r in rows if r.get("warmup") is False and r.get("session_id")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--stage", default="1")
    p.add_argument("--skip", type=int, default=9)
    p.add_argument("--client-jsonl", type=Path, help="keep only requests from sessions with warmup=false")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    all_rows = rows(parse(a.log, a.stage))
    if a.client_jsonl:
        measured = measured_sessions(a.client_jsonl)
        kept = [r for r in all_rows if session_of(r["req"]) in measured]
        a.skip = len(all_rows) - len(kept)
    else:
        kept = all_rows[a.skip :]
    table = [{**hop_ms(r), **exec_breakdown(r)} for r in kept]
    extra = sorted({k for t in table for k in t if k.startswith(("run:", "overlap:"))})
    summary = {}
    for hop in [h for h, _, _ in HOPS] + extra:
        vals = sorted(v for t in table if (v := t.get(hop)) is not None)
        if vals:
            summary[hop] = {
                "n": len(vals),
                "p50": round(statistics.median(vals), 2),
                "p95": round(vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))], 2),
                "min": round(vals[0], 2),
                "max": round(vals[-1], 2),
            }
    how = "matched to measured client sessions" if a.client_jsonl else f"after skipping the first {a.skip}"
    print(f"requests: {len(all_rows)} total, {len(kept)} kept ({how})")
    print(f"{'hop':<34}{'n':>4}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}")
    for hop, s in summary.items():
        print(f"{hop:<34}{s['n']:>4}{s['p50']:>9}{s['p95']:>9}{s['min']:>9}{s['max']:>9}")
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "rows": [
            {"req": r["req"], "hops_ms": t} for r, t in zip(kept, table)]}, indent=1))


if __name__ == "__main__":
    main()
