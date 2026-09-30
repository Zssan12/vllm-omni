"""Fresh-session judge off/on latency; JSONL remains readable by analyze_onoff.py."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from aura_probe_common import (
    add_client_arguments,
    distribution,
    failure_row,
    frame_b64,
    load_clips,
    session,
    write_row,
)


async def main_async(a):
    specs = []
    for spec in a.clips:
        name, sep, repetitions = spec.rpartition("=")
        if not sep or int(repetitions) < 1:
            raise ValueError("clips must be path=positive_repetitions")
        specs.append((name, int(repetitions)))
    clips = load_clips([name for name, _ in specs])
    frame = frame_b64()
    rows = []
    with Path(a.out).open("x") as out:
        for path, repetitions in specs:
            name = Path(path).name
            for rep in range(a.warmup + repetitions):
                try:
                    async with session(a, single_turn=True) as probe:
                        pending = await probe.start(clips[name], frame)
                        row = await probe.wait(
                            pending, timeout=a.timeout, listen_wait=a.listen_wait_s
                        )
                        await asyncio.sleep(a.grace_s)
                except Exception as exc:
                    row = failure_row(exc)
                row.update(mode=a.mode, clip=name, rep=rep, warmup=rep < a.warmup)
                write_row(out, row)
                rows.append(row)
                print(
                    {
                        k: row.get(k)
                        for k in (
                            "mode",
                            "clip",
                            "rep",
                            "terminal",
                            "answered",
                            "blocked",
                            "errors",
                        )
                    },
                    flush=True,
                )
                if row.get("failed") or (
                    name in {"a3.wav", "a4.wav"} and not row.get("answered")
                ):
                    write_row(
                        out, {"invalid": "turn failed; no aggregate should be quoted"}
                    )
                    return 2
                await asyncio.sleep(a.think_s)
        summary = {}
        for path, _ in specs:
            name = Path(path).name
            measured = [r for r in rows if r["clip"] == name and not r["warmup"]]
            summary[name] = {
                "n": len(measured),
                "blocked": sum(r["blocked"] for r in measured),
                "spoke": sum(r["has_audio"] for r in measured),
            }
            for key in ("first_text_ms", "first_audio_ms", "terminal_ms"):
                summary[name][key] = distribution([r.get(key) for r in measured])
        write_row(out, {"summary": summary, "mode": a.mode})
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_client_arguments(p)
    p.add_argument("--mode", required=True, choices=("off", "on"))
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--grace-s", type=float, default=0.5)
    p.add_argument("--listen-wait-s", type=float, default=1.5)
    p.add_argument("--think-s", type=float, default=1)
    raise SystemExit(asyncio.run(main_async(p.parse_args())))
