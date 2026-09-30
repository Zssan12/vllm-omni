"""Split the time between ASR output and the main-model submit into hops.

Reads a server log written by a timing_patch.py tree. Judge on (5 stages):

  asr_out        OMNI_HOP output stage=0 finished=1
  fwd0           OMNI_TIMING forward src=0      (start of the ASR -> judge forward)
  bridge1        OMNI_HOP bridge stage=1 t0..t1 (asr2judge)
  submit1        OMNI_TIMING submit stage=1
  judge_out      OMNI_HOP output stage=1 finished=1
  fwd1           OMNI_TIMING forward src=1
  bridge2        OMNI_HOP bridge stage=2 t0..t1 (judge2aura -> asr2aura)
  submit2        OMNI_TIMING submit stage=2

Judge off (4 stages): asr_out, fwd0, bridge1 (asr2aura), submit1.

Hops are reported per request, then p50 / p95 (nearest rank) over requests
that have every point. With --client-jsonl (the onoff client's rows) each
server request is matched to its client row through the session id encoded
in the request id, warmups are dropped and results are grouped by clip; this
is the mode to quote. Without it, --skip drops the first N requests in log
order (only a rough fallback: the client warms up each clip right before its
formal rows). Rejected turns (no submit of the main stage) are counted
separately. All times are orchestrator wall clock: they include scheduling and
IPC, not pure GPU time.

    python hop_analysis.py server.log --mode on|off --client-jsonl onoff-on.jsonl [--json out.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

TIMING = re.compile(r"OMNI_TIMING (?:forward src=(\d+) dst=\d+|submit stage=(\d+)) req=(\S+) t=([0-9.]+)")
OUTPUT = re.compile(r"OMNI_HOP output stage=(\d+) req=(\S+) finished=(\d) t=([0-9.]+)")
BRIDGE = re.compile(r"OMNI_HOP bridge stage=(\d+) req=(\S+) t0=([0-9.]+) t1=([0-9.]+)")


def parse(lines):
    reqs: OrderedDict[str, dict[str, float]] = OrderedDict()

    def put(req, key, value):
        # First occurrence wins (a resumed/partial stage can log more than once).
        reqs.setdefault(req, {}).setdefault(key, value)

    for line in lines:
        if m := TIMING.search(line):
            src, stage, req, t = m.groups()
            put(req, f"fwd{src}" if src is not None else f"submit{stage}", float(t))
        elif m := OUTPUT.search(line):
            stage, req, finished, t = m.groups()
            if finished == "1":
                put(req, f"out{stage}", float(t))
        elif m := BRIDGE.search(line):
            stage, reqs_field, t0, t1 = m.groups()
            for req in reqs_field.split(","):
                put(req, f"bridge{stage}_t0", float(t0))
                put(req, f"bridge{stage}_t1", float(t1))
    return reqs


HOPS = {
    "on": [
        ("asr_out->fwd0", "out0", "fwd0"),
        ("fwd0->bridge(asr2judge) start", "fwd0", "bridge1_t0"),
        ("bridge asr2judge", "bridge1_t0", "bridge1_t1"),
        ("bridge end->judge submitted", "bridge1_t1", "submit1"),
        ("judge engine (submit->output at orchestrator)", "submit1", "out1"),
        ("judge output->fwd1", "out1", "fwd1"),
        ("fwd1->bridge(judge2aura) start", "fwd1", "bridge2_t0"),
        ("bridge judge2aura", "bridge2_t0", "bridge2_t1"),
        ("bridge end->main submitted", "bridge2_t1", "submit2"),
        ("TOTAL fwd0->main submitted", "fwd0", "submit2"),
    ],
    "off": [
        ("asr_out->fwd0", "out0", "fwd0"),
        ("fwd0->bridge(asr2aura) start", "fwd0", "bridge1_t0"),
        ("bridge asr2aura", "bridge1_t0", "bridge1_t1"),
        ("bridge end->main submitted", "bridge1_t1", "submit1"),
        ("TOTAL fwd0->main submitted", "fwd0", "submit1"),
    ],
}


def session_of(request_id: str) -> str | None:
    """duplex-s.<urlsafe b64 of the session id>.e.<epoch>.r.<stage request>"""
    parts = request_id.split(".")
    if len(parts) < 2 or parts[0] != "duplex-s":
        return None
    try:
        return base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def client_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("session_id"):
            rows[row["session_id"]] = row
    return rows


def quantile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--mode", choices=("on", "off"), required=True)
    p.add_argument("--client-jsonl", type=Path, help="onoff client rows: match requests, drop warmups")
    p.add_argument("--skip", type=int, default=0, help="fallback without --client-jsonl")
    p.add_argument("--hop-clips", nargs="+", help="clips whose passed requests enter the hop stats (e.g. a3.wav a4.wav)")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    reqs = list(parse(a.log.read_text(errors="replace").splitlines()).items())
    reqs = [(r, pts) for r, pts in reqs if "fwd0" in pts]
    report: dict = {"mode": a.mode}
    if a.client_jsonl:
        clients = client_rows(a.client_jsonl)
        matched = [(r, pts, clients.get(session_of(r) or "")) for r, pts in reqs]
        report["unmatched_server_requests"] = sum(1 for *_, row in matched if row is None)
        report["warmups_dropped"] = sum(1 for *_, row in matched if row is not None and row.get("warmup"))
        reqs = [(r, {**pts, "_clip": row.get("clip")}) for r, pts, row in matched if row is not None and not row.get("warmup")]
    else:
        reqs = reqs[a.skip :]
        report["note"] = "no --client-jsonl: log-order skip, do not quote"
    main_submit = "submit2" if a.mode == "on" else "submit1"
    passed = [(r, pts) for r, pts in reqs if main_submit in pts]
    if a.hop_clips:
        passed = [(r, pts) for r, pts in passed if pts.get("_clip") in set(a.hop_clips)]
        report["hop_clips"] = a.hop_clips
    rejected = [(r, pts) for r, pts in reqs if main_submit not in pts]
    report.update(requests=len(reqs), reached_main=len(passed), stopped_before_main=len(rejected))
    report["by_clip"] = {}
    for _, pts in reqs:
        clip = str(pts.get("_clip"))
        entry = report["by_clip"].setdefault(clip, {"requests": 0, "reached_main": 0})
        entry["requests"] += 1
        entry["reached_main"] += int(main_submit in pts)
    hops = {}
    for name, start, end in HOPS[a.mode]:
        values = [(pts[end] - pts[start]) * 1000 for _, pts in passed if start in pts and end in pts]
        if values:
            hops[name] = {"n": len(values), "p50_ms": round(quantile(values, 0.5), 3), "p95_ms": round(quantile(values, 0.95), 3)}
    report["hops_for_requests_reaching_main"] = hops
    if a.mode == "on" and rejected:
        values = [(pts["out1"] - pts["submit1"]) * 1000 for _, pts in rejected if "submit1" in pts and "out1" in pts]
        if values:
            report["rejected_judge_engine_ms"] = {"n": len(values), "p50": round(quantile(values, 0.5), 3), "p95": round(quantile(values, 0.95), 3)}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.json:
        a.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
