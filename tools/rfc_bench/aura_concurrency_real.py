"""Real judge on/off throughput under a fixed, fresh-session mixed workload."""

from __future__ import annotations

import argparse
import asyncio
import time
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


def summarize(rows, wall):
    requests = [r for r in rows if r["kind"] == "request"]
    successes = [r for r in requests if r.get("answered") and not r.get("failed")]
    backchannels = [r for r in rows if r["kind"] == "backchannel"]
    return {
        "wall_s": wall,
        "request_attempts": len(requests),
        "requests_completed": len(successes),
        "completed_questions_per_min": len(successes) / wall * 60 if wall > 0 else 0,
        "requests_rejected": sum(r.get("blocked", False) for r in requests),
        "timeouts": sum(r.get("timed_out", False) for r in rows),
        "failures": sum(r.get("failed", False) for r in rows),
        "backchannels": len(backchannels),
        "backchannels_listen_only": sum(r.get("blocked", False) for r in backchannels),
        "valid": len(successes) == len(requests)
        and not any(r.get("failed") for r in rows),
        "latency_successful_questions": {
            k: distribution([r.get(k) for r in successes])
            for k in ("first_text_ms", "first_audio_ms", "turn_ms")
        },
    }


async def user(uid, a, clips, frame, stream, *, users, warmup, rows):
    for turn in range(a.rounds * 2):
        kind = "request" if turn % 2 == 0 else "backchannel"
        name = (["a3.wav", "a4.wav"] if kind == "request" else ["a0.wav", "a1.wav"])[
            (turn // 2) % 2
        ]
        identity = dict(
            mode=a.mode,
            user=uid,
            turn=turn,
            kind=kind,
            clip=name,
            users=users,
            warmup=warmup,
        )
        try:
            async with session(a, single_turn=True) as probe:
                pending = await probe.start(clips[name], frame)
                row = await probe.wait(
                    pending, timeout=a.timeout, listen_wait=a.listen_wait_s
                )
        except asyncio.CancelledError:
            row = failure_row(
                RuntimeError("cancelled because another user failed"), **identity
            )
            rows.append(row)
            write_row(stream, row)
            raise
        except Exception as exc:
            row = failure_row(exc)
        row.update(identity)
        rows.append(row)
        write_row(stream, row)
        if row.get("failed") or (kind == "request" and not row.get("answered")):
            raise RuntimeError(
                f"{users} users: {uid}/{turn} did not complete successfully; stopping this level"
            )
        if turn + 1 < a.rounds * 2:
            await asyncio.sleep(a.think_s)


async def wave(a, clips, frame, stream, n, *, warmup):
    rows = []
    start = time.monotonic()
    tasks = [
        asyncio.create_task(
            user(u, a, clips, frame, stream, users=n, warmup=warmup, rows=rows)
        )
        for u in range(n)
    ]
    error = None
    try:
        await asyncio.gather(*tasks)
    except Exception as exc:
        error = str(exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    result = summarize(rows, time.monotonic() - start)
    result.update(users=n, mode=a.mode, warmup=warmup, expected_turns=n * a.rounds * 2)
    if error or len(rows) != n * a.rounds * 2:
        result.update(valid=False, error=error or "incomplete wave")
    write_row(stream, {"wave_summary": result})
    return result


async def main_async(a):
    if any(n < 1 or n > 8 for n in a.users) or a.rounds < 1:
        raise ValueError("users must be 1..8 and rounds must be positive")
    clips = load_clips(a.clips)
    if set(clips) != {"a0.wav", "a1.wav", "a3.wav", "a4.wav"}:
        raise ValueError("exactly a0/a1/a3/a4 are required")
    frame = frame_b64()
    with Path(a.out).open("x") as stream:
        for n in a.users:
            # Two pairs warm all four clips at the concurrency being measured.
            warm_args = argparse.Namespace(**{**vars(a), "rounds": 2})
            warm = await wave(warm_args, clips, frame, stream, n, warmup=True)
            if not warm["valid"]:
                return 2
            measured = await wave(a, clips, frame, stream, n, warmup=False)
            print({f"{n}users_{a.mode}": measured}, flush=True)
            if not measured["valid"]:
                return 2
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_client_arguments(p)
    p.add_argument("--mode", required=True, choices=("off", "on"))
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--users", type=int, nargs="+", default=[4, 8])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--think-s", type=float, default=1)
    p.add_argument(
        "--listen-wait-s",
        type=float,
        default=0,
        help="AURA model-listen ends a turn; no artificial 1.5s throughput penalty",
    )
    raise SystemExit(asyncio.run(main_async(p.parse_args())))
