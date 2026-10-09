"""Cross-concurrency report for the H200 response-judge arms.

This is the layer the reproduction package does not have. `aura_probe_common.measure()`
already records `first_text_ms` / `first_audio_ms` per turn and
`aura_concurrency_real.summarize()` already counts completions and rejections, but
nothing joins judge-off against judge-on across concurrency levels. That join is
what says whether the judge costs anything a user can feel, and what it saves.

    python3 judge_bench_report.py <arm-base-dir> [<arm-base-dir> ...]

Each base dir is one `run_arms.sh conc` output: `<base>/off.jsonl`, `<base>/on.jsonl`
plus the `gpu-<mode>-{before,ready,after}.txt` snapshots.

This report reads `terminal_ms` off the per-turn rows rather than reusing
`summarize()`'s `latency_successful_questions`, because it needs the rows themselves
to split each arm by clip kind and by `listen_sources`. Both fields are measured from
the same `committed_at`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

# The judge exists to suppress a reply to a backchannel and never to suppress one
# to a real question. Both directions are counted separately.
#
# The `conc` client labels every row with `kind`; the `onoff` client does not and
# is identified by clip name instead (a3/a4 ask something, a0/a1 are backchannels
# -- see repro/README.md "Data"). The mapping is the only thing that differs.
QUESTION = "request"
BACKCHANNEL = "backchannel"
CLIP_KIND = {"a3.wav": QUESTION, "a4.wav": QUESTION, "a0.wav": BACKCHANNEL, "a1.wav": BACKCHANNEL}

# A turn can end without a reply for two unrelated reasons: the judge decided so, or
# AURA emitted `<|silent|>` on its own. Only the first is the judge doing work, and
# `listen_sources` is what distinguishes them -- judge-off suppresses backchannels too.
JUDGE_SOURCE = "response_judge"


def by_judge(row):
    """True when this turn's suppression is attributed to the judge stage."""
    return JUDGE_SOURCE in (row.get("listen_sources") or [])


def turn_kind(row):
    """Row kind, from `kind` when the client set it, else from the clip name."""
    return row.get("kind") or CLIP_KIND.get(row.get("clip"))


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def dist(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    return {
        "n": len(values),
        "p50": round(statistics.median(values), 1),
        "p95": round(percentile(values, 0.95), 1),
    }


def load(path: Path):
    """Return (turn_rows, wave_summaries, missing) for one arm file.

    `missing` is True when the file is absent, so a half-finished run reports that
    rather than reading as an arm with nothing in it.
    """
    turns, waves = [], []
    if not path.exists():
        return turns, waves, True
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wave_summary" in row:
            waves.append(row["wave_summary"])
        elif turn_kind(row):
            turns.append(row)
    return turns, waves, False


def gpu_peak(base: Path, mode: str):
    """Peak used MiB / util % across the arm's nvidia-smi snapshots.

    The snapshot is the plain `nvidia-smi` table, not `--query-gpu=csv`, so the
    numbers come out of the memory row:
    `| N/A   28C   P0  127W / 700W |  110052MiB / 143771MiB |      2%   Default |`
    """
    row_re = re.compile(r"(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%\s")
    used, util, stages = [], [], 0
    for stage in ("before", "ready", "after"):
        f = base / f"gpu-{mode}-{stage}.txt"
        if not f.exists():
            continue
        stages += 1
        for m in row_re.finditer(f.read_text(errors="replace")):
            used.append(int(m.group(1)))
            util.append(int(m.group(3)))
    if not used:
        return None
    return {
        "used_mib_max": max(used),
        "util_pct_max": max(util),
        "stages": stages,
    }


def _group_by_users(rows):
    """Rows keyed by the concurrency level they ran at (`users`, set per turn)."""
    out = {}
    for r in rows:
        out.setdefault(r.get("users"), []).append(r)
    return {n: out[n] for n in sorted(out, key=lambda x: (x is None, x))}


def arm_stats(base: Path, mode: str):
    turns, waves, missing = load(base / f"{mode}.jsonl")
    measured = [r for r in turns if not r.get("warmup")]
    qs = [r for r in measured if turn_kind(r) == QUESTION]
    bs = [r for r in measured if turn_kind(r) == BACKCHANNEL]
    q_ok = [r for r in qs if r.get("answered") and not r.get("failed")]

    # A question that went unanswered either broke or was silenced, and only the
    # second is a judge error. Separating them keeps an infrastructure failure from
    # reading as the judge blocking a real question.
    q_lost = [r for r in qs if r not in q_ok]
    q_broken = [r for r in q_lost if r.get("failed") or r.get("timed_out")]
    q_silenced = [r for r in q_lost if r not in q_broken]
    q_false_blocks = [r for r in q_silenced if by_judge(r)]

    # Same split on the other side: judge-off suppresses backchannels too, via AURA's
    # own `<|silent|>`, so an unattributed suppression is not the judge working.
    b_blocked = [r for r in bs if r.get("blocked") and not r.get("failed")]
    b_by_judge = [r for r in b_blocked if by_judge(r)]
    b_by_aura = [r for r in b_blocked if not by_judge(r)]
    b_answered = [r for r in bs if r.get("answered") and not r.get("failed")]

    # Throughput: the measured (warmup=False) wave summaries, per concurrency.
    throughput = {}
    for w in waves:
        if w.get("warmup"):
            continue
        throughput.setdefault(w.get("users"), []).append(
            {
                "questions_per_min": round(w.get("completed_questions_per_min") or 0, 2),
                "wall_s": round(w.get("wall_s") or 0, 2),
                "attempts": w.get("request_attempts"),
                "completed": w.get("requests_completed"),
                "rejected": w.get("requests_rejected"),
                "valid": w.get("valid"),
            }
        )

    return {
        "mode": mode,
        "missing": missing,
        "turns_measured": len(measured),
        "questions": len(qs),
        "questions_answered": len(q_ok),
        "backchannels": len(bs),
        "failures": sum(1 for r in measured if r.get("failed")),
        "timeouts": sum(1 for r in measured if r.get("timed_out")),
        "false_blocks": len(q_false_blocks),  # a real question the judge silenced
        "questions_silenced_other": len(q_silenced) - len(q_false_blocks),  # AURA, not the judge
        "questions_broken": len(q_broken),  # failed or timed out, not a judge decision
        "backchannels_suppressed": len(b_blocked),  # by anyone
        "backchannels_suppressed_by_judge": len(b_by_judge),  # the judge doing its job
        "backchannels_suppressed_by_aura": len(b_by_aura),  # AURA's own `<|silent|>`
        "backchannels_answered": len(b_answered),  # let a backchannel through
        "ttft_ms": dist([r.get("first_text_ms") for r in q_ok]),
        "ttfa_ms": dist([r.get("first_audio_ms") for r in q_ok]),
        "turn_ms": dist([r.get("terminal_ms") for r in q_ok]),
        "latency_by_users": {
            n: {
                "ttft_ms": dist([r.get("first_text_ms") for r in rows]),
                "ttfa_ms": dist([r.get("first_audio_ms") for r in rows]),
                "turn_ms": dist([r.get("terminal_ms") for r in rows]),
            }
            for n, rows in _group_by_users(q_ok).items()
        },
        "backchannel_latency_ms": dist([r.get("terminal_ms") for r in b_by_judge]),
        "listen_sources": sorted(
            {s for r in bs for s in (r.get("listen_sources") or []) if s}
        ),
        "throughput_by_users": throughput,
        "gpu": gpu_peak(base, mode),
    }


def fmt(d, unit=""):
    if not d:
        return "n=0"
    return f"n={d['n']}  p50 {d['p50']}{unit} / p95 {d['p95']}{unit}"


def delta(on, off, key):
    a, b = on.get(key), off.get(key)
    if not a or not b:
        return "—"
    return f"{a['p50'] - b['p50']:+.1f} p50 / {a['p95'] - b['p95']:+.1f} p95"


def print_latency_by_level(off, on):
    """Per-concurrency TTFT/TTFA, which the pooled row above can hide.

    Pooling every level into one p50 mixes populations: the judge can be faster at
    one user and slower at four, and the pooled delta lands somewhere in between
    while describing neither. These are the rows to quote. The counts per level are
    small (rounds x users answered questions each), so read them as a direction, not
    as an estimate.
    """
    levels = sorted(
        set(off["latency_by_users"]) | set(on["latency_by_users"]), key=lambda x: (x is None, x)
    )
    if len(levels) < 2:
        return
    print("\n### by concurrency level")
    print("| users | n | TTFT off | TTFT on | TTFT delta | TTFA off | TTFA on | TTFA delta |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in levels:
        o = off["latency_by_users"].get(n, {})
        w = on["latency_by_users"].get(n, {})
        cells = []
        for key in ("ttft_ms", "ttfa_ms"):
            a, b = o.get(key), w.get(key)
            cells += [
                f"{a['p50']}" if a else "—",
                f"{b['p50']}" if b else "—",
                f"{b['p50'] - a['p50']:+.1f}" if a and b else "—",
            ]
        count = (w.get("ttft_ms") or o.get("ttft_ms") or {}).get("n", "—")
        print(f"| {n} | {count} | " + " | ".join(cells) + " |")
    print("\np50 milliseconds, answered questions only.")


def main():
    bases = [Path(p) for p in sys.argv[1:]]
    if not bases:
        raise SystemExit(__doc__)

    for base in bases:
        off, on = arm_stats(base, "off"), arm_stats(base, "on")
        print(f"\n{'=' * 78}\n{base}\n{'=' * 78}")

        print("\n## Routing")
        print("| arm | questions | answered | false blocks | broken | backchannels | by judge | by AURA | let through |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in (off, on):
            print(
                f"| {s['mode']} | {s['questions']} | {s['questions_answered']} | "
                f"{s['false_blocks']} | {s['questions_broken']} | {s['backchannels']} | "
                f"{s['backchannels_suppressed_by_judge']} | {s['backchannels_suppressed_by_aura']} | "
                f"{s['backchannels_answered']} |"
            )
        print("\n`false blocks` counts only questions silenced with `response_judge` in")
        print("`listen_sources`; `broken` counts failures and timeouts, which are not judge")
        print("decisions. Suppressions are split the same way.")
        if on["listen_sources"]:
            print(f"\njudge-on listen sources: {', '.join(on['listen_sources'])}")

        print("\n## Latency (answered questions only)")
        print(f"| metric | off | on | on - off |")
        print(f"|---|---|---|---|")
        for label, key, unit in (
            ("TTFT (first text)", "ttft_ms", " ms"),
            ("TTFA (first audio)", "ttfa_ms", " ms"),
            ("turn total", "turn_ms", " ms"),
        ):
            print(f"| {label} | {fmt(off[key], unit)} | {fmt(on[key], unit)} | {delta(on, off, key)} |")
        print_latency_by_level(off, on)

        print("\n## Suppression latency (rejected backchannels, judge-on)")
        print(f"  {fmt(on['backchannel_latency_ms'], ' ms')}")

        print("\n## Useful throughput (completed questions / min)")
        levels = sorted(
            set(off["throughput_by_users"]) | set(on["throughput_by_users"]), key=lambda x: (x is None, x)
        )
        if not levels:
            print("  no wave summaries -- `onoff` arms open one session per turn and emit")
            print("  none. Run a `conc` arm for throughput.")
        else:
            print("| users | off q/min | on q/min | on/off | on wall_s | off wall_s | off valid | on valid |")
            print("|---:|---:|---:|---:|---:|---:|---|---|")
        for n in levels:
            o = (off["throughput_by_users"].get(n) or [{}])[0]
            w = (on["throughput_by_users"].get(n) or [{}])[0]
            oq, wq = o.get("questions_per_min"), w.get("questions_per_min")
            ratio = f"{wq / oq:.3f}" if oq and wq else "—"
            print(
                f"| {n} | {oq if oq is not None else '—'} | {wq if wq is not None else '—'} | {ratio} | "
                f"{w.get('wall_s', '—')} | {o.get('wall_s', '—')} | {o.get('valid')} | {w.get('valid')} |"
            )

        print("\n## GPU (peak over the arm's snapshots)")
        for s in (off, on):
            g = s["gpu"]
            print(f"  {s['mode']}: " + (f"{g['used_mib_max']} MiB used, {g['util_pct_max']}% util, {g['stages']} snapshots" if g else "no snapshots"))

        verdict = []
        for s in (off, on):
            if s["missing"]:
                verdict.append(f"WARNING: {base / (s['mode'] + '.jsonl')} is missing -- no data for this arm")
            elif not s["turns_measured"]:
                verdict.append(f"WARNING: judge-{s['mode']} has no measured turns -- arm did not complete")
        if off["false_blocks"] or on["false_blocks"]:
            verdict.append(
                f"WARNING: false block(s) attributed to the judge -- off {off['false_blocks']}, on {on['false_blocks']}"
            )
        if off["questions_broken"] or on["questions_broken"]:
            verdict.append(
                f"note: unanswered questions from failures/timeouts (not judge decisions) -- "
                f"off {off['questions_broken']}, on {on['questions_broken']}"
            )
        if off["questions_silenced_other"] or on["questions_silenced_other"]:
            verdict.append(
                f"note: questions silenced without `{JUDGE_SOURCE}` attribution -- "
                f"off {off['questions_silenced_other']}, on {on['questions_silenced_other']}"
            )
        if on["backchannels"] and not on["backchannels_suppressed_by_judge"]:
            verdict.append(f"WARNING: no suppression attributed to `{JUDGE_SOURCE}` -- judge may not be active")
        if off["backchannels_suppressed_by_aura"]:
            verdict.append(
                f"note: judge-off suppressed {off['backchannels_suppressed_by_aura']} backchannel(s) "
                "via AURA's own silence"
            )
        if any(not w.get("valid") for m in (off, on) for lvl in m["throughput_by_users"].values() for w in lvl):
            verdict.append("WARNING: at least one wave was invalid -- numbers incomplete")
        print("\n## Checks")
        print("\n".join(f"  - {v}" for v in verdict) or "  - clean")


if __name__ == "__main__":
    main()
