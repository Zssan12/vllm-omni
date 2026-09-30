"""How fast is a CPU thread right after it wakes from an idle gap? (never part of the PR)

Read-only: changes no system setting. After sleeping for each gap, the thread
immediately runs N chunks of a fixed pure-Python loop, calibrated to take about
--chunk-ms each when the core is fully ramped, and records every chunk's
duration. It also records the core it ran on and that core's scaling_cur_freq
(read-only sysfs) right after waking.

Fingerprints:
  frequency scaling  many chunks after waking are slow, recovering over ms-tens of ms; worse after longer gaps
  C-state exit       only the wake itself is late (sleep overshoot); chunk speed is normal
  cache              a loop this small stays in L1/L2, so chunk speed is unaffected

    python cpu_wake_probe.py --out <new dir> [--gaps 0 0.01 0.05 0.2 0.5 1 2] [--reps 15]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def current_cpu() -> int:
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[36])  # field 39 of /proc/<pid>/stat: CPU last executed on


def cur_freq_mhz(cpu: int) -> float | None:
    try:
        return int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text()) / 1000
    except OSError:
        return None


def work(n: int) -> int:
    acc = 0
    for i in range(n):
        acc = (acc + i * 7) & 0xFFFF
    return acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--chunks", type=int, default=120)
    p.add_argument("--chunk-ms", type=float, default=0.25)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    # Calibrate on a fully busy core: spin 1 s first, then size the loop.
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end:
        work(1000)
    n = 20000
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        work(n)
        samples.append(time.perf_counter() - t0)
    per_iter = statistics.median(samples) / n
    n = max(1, int(a.chunk_ms / 1000 / per_iter))
    base_samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        work(n)
        base_samples.append((time.perf_counter() - t0) * 1000)
    base = statistics.median(base_samples)

    rows = []
    for gap in a.gaps:
        for rep in range(a.reps):
            if gap:
                t_sleep = time.perf_counter()
                time.sleep(gap)
                overshoot_ms = (time.perf_counter() - t_sleep - gap) * 1000
            else:
                overshoot_ms = 0.0
            cpu = current_cpu()
            freq = cur_freq_mhz(cpu)
            chunks = []
            for _ in range(a.chunks):
                t0 = time.perf_counter()
                work(n)
                chunks.append((time.perf_counter() - t0) * 1000)
            rows.append({"gap_s": gap, "rep": rep, "cpu": cpu, "cur_freq_mhz_after_wake": freq,
                         "sleep_overshoot_ms": overshoot_ms, "chunk_ms": chunks})

    summary = {"base_chunk_ms": round(base, 4), "loop_iters": n, "by_gap": {}}
    for gap in a.gaps:
        mine = [r for r in rows if r["gap_s"] == gap]

        def med_ratio(lo: int, hi: int) -> float:
            return round(statistics.median(statistics.mean(r["chunk_ms"][lo:hi]) for r in mine) / base, 2)

        # time until chunks are back within 10% of the ramped speed, per rep
        recover = []
        for r in mine:
            elapsed = 0.0
            for c in r["chunk_ms"]:
                elapsed += c
                if c <= 1.1 * base:
                    break
            recover.append(elapsed)
        summary["by_gap"][str(gap)] = {
            "slowdown_first_4_chunks": med_ratio(0, 4),
            "slowdown_chunks_4_20": med_ratio(4, 20),
            "slowdown_chunks_20_60": med_ratio(20, 60),
            "slowdown_last_40": med_ratio(len(mine[0]["chunk_ms"]) - 40, len(mine[0]["chunk_ms"])),
            "ms_until_first_fast_chunk": round(statistics.median(recover), 2),
            "sleep_overshoot_ms": round(statistics.median(r["sleep_overshoot_ms"] for r in mine), 3),
            "cur_freq_mhz_after_wake": statistics.median(
                r["cur_freq_mhz_after_wake"] for r in mine if r["cur_freq_mhz_after_wake"] is not None
            ) if any(r["cur_freq_mhz_after_wake"] is not None for r in mine) else None,
        }
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
