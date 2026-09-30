"""Play the multi-turn test dialogues against a duplex server and record what the assistant did per turn.

Reads a synthesized audio set (synth_say.py / synth_qwen3_tts.py: <audio-dir>/manifest.jsonl) and,
for each turn, streams the WAV, commits it and waits for the turn's outcome: an answer
(text + audio) or a listen. Records the judge's decision source
(response.metadata.vllm_omni.listen_source), first text / first audio latency and how much
reply audio was produced, then summarises the off/on comparison per category.

Session modes
  per-turn     (default) a fresh session per turn. The judge sees only the current transcript,
               so its decisions are the same as in a dialogue; works on #8316 as is.
  per-dialogue one session per dialogue. Needs the same-session recovery fix: without
               it a turn after a silent turn may only get response.listen.

    PYTHONPATH=<vllm-omni tree> python tools/multiturn/dialogue_client.py --model <served name> \
        --audio-dir audio-qwen3tts --split heldout --mode on --out runs/on.jsonl
    python dialogue_client.py --summarize runs/off.jsonl runs/on.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rfc_bench"))


def load_turns(audio_dir: Path, split: str | None, scenes, buckets, limit: int | None):
    dialogues: dict[str, list[dict]] = {}
    for line in (audio_dir / "manifest.jsonl").read_text().splitlines():
        m = json.loads(line)
        if split and m["split"] != split:
            continue
        if scenes and m["scene"] not in scenes:
            continue
        if buckets and m["bucket"] not in buckets:
            continue
        dialogues.setdefault(m["dialogue"], []).append(m)
    items = sorted(dialogues.items())
    return items[:limit] if limit else items


async def play(a) -> int:
    from aura_probe_common import failure_row, frame_b64, session, write_row
    from vllm_omni.clients.duplex import read_pcm16_wav

    dialogues = load_turns(a.audio_dir, a.split, a.scene, a.bucket, a.limit)
    frame = frame_b64()
    out_path = Path(a.out)
    with out_path.open("x") as out:
        warm = dialogues[0][1][0]
        for rep in range(a.warmup):  # a question turn, not scored
            async with session(a, single_turn=True) as probe:
                row = await probe.wait(await probe.start(read_pcm16_wav(a.audio_dir / warm["wav"]), frame),
                                       timeout=a.timeout, listen_wait=a.listen_wait_s)
            write_row(out, {**_slim(row), "warmup": True, "rep": rep, "mode": a.mode})
        for did, turns in dialogues:
            if a.session_mode == "per-dialogue":
                async with session(a) as probe:
                    for i, t in enumerate(turns):
                        if i:
                            await asyncio.wait_for(probe.client.clear_input(), 10)
                        row = await _turn(a, probe, t, frame)
                        _emit(out, a, t, row)
                        await asyncio.sleep(a.gap_s)
            else:
                for t in turns:
                    try:
                        async with session(a, single_turn=True) as probe:
                            row = await _turn(a, probe, t, frame)
                    except Exception as exc:
                        row = failure_row(exc)
                    _emit(out, a, t, row)
                    await asyncio.sleep(a.gap_s)
    print(json.dumps(summarize([out_path], a.output_rate), ensure_ascii=False, indent=1))
    return 0


async def _turn(a, probe, t, frame):
    from vllm_omni.clients.duplex import read_pcm16_wav

    pending = await probe.start(read_pcm16_wav(a.audio_dir / t["wav"]), frame, chunk_ms=a.chunk_ms or None)
    return await probe.wait(pending, timeout=a.timeout, listen_wait=a.listen_wait_s)


def _slim(row):
    keep = ("session_id", "terminal", "answered", "blocked", "failed", "errors", "first_text_ms", "first_audio_ms",
            "terminal_ms", "text", "audio_bytes", "transcripts", "listen_sources")
    return {k: row.get(k) for k in keep}


def _emit(out, a, t, row):
    from aura_probe_common import write_row

    rec = {"mode": a.mode, "session_mode": a.session_mode, "warmup": False,
           **{k: t[k] for k in ("dialogue", "split", "scene", "bucket", "turn", "text", "category", "reply", "voice")},
           **_slim(row)}
    write_row(out, rec)
    print({k: rec.get(k) for k in ("mode", "dialogue", "turn", "category", "reply", "answered", "blocked",
                                   "listen_sources", "first_audio_ms", "failed")}, flush=True)


def summarize(paths, output_rate=24000):
    from statistics import median

    per_mode = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("warmup") or "category" not in r:
                continue
            m = per_mode.setdefault(r["mode"], {"rows": []})
            m["rows"].append(r)
    result = {}
    for mode, m in per_mode.items():
        rows = m["rows"]
        cats = defaultdict(lambda: {"n": 0, "answered": 0, "blocked": 0, "failed": 0})
        for r in rows:
            c = cats[r["category"]]
            c["n"] += 1
            c["answered"] += bool(r.get("answered"))
            c["blocked"] += bool(r.get("blocked"))
            c["failed"] += bool(r.get("failed"))
        should, silent = [r for r in rows if r["reply"]], [r for r in rows if not r["reply"]]
        spurious = [r for r in silent if (r.get("audio_bytes") or 0) > 0]
        fa = [r["first_audio_ms"] for r in should if r.get("answered") and r.get("first_audio_ms") is not None]
        ft = [r["first_text_ms"] for r in should if r.get("answered") and r.get("first_text_ms") is not None]
        sources = defaultdict(int)
        for r in rows:
            for s in r.get("listen_sources") or []:
                sources[str(s)] += 1
        result[mode] = {
            "turns": len(rows),
            "failed": sum(bool(r.get("failed")) for r in rows),
            "should_reply_answered": f"{sum(bool(r.get('answered')) for r in should)}/{len(should)}",
            "should_reply_blocked": f"{sum(bool(r.get('blocked')) for r in should)}/{len(should)}",
            "no_reply_spoke": f"{len(spurious)}/{len(silent)}",
            "no_reply_spoken_seconds": round(sum(r["audio_bytes"] for r in spurious) / (2 * output_rate), 1),
            "first_text_ms_p50": round(median(ft), 1) if ft else None,
            "first_audio_ms_p50": round(median(fa), 1) if fa else None,
            "listen_sources": dict(sources),
            "by_category": dict(cats),
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", nargs="+", help="only summarise existing JSONL files")
    p.add_argument("--output-rate", type=int, default=24000, help="sample rate of the reply PCM16 audio")
    p.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    p.add_argument("--model")
    p.add_argument("--out")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--open-timeout", type=float, default=45)
    p.add_argument("--server-log")
    p.add_argument("--audio-dir", type=Path)
    p.add_argument("--split", choices=("tune", "heldout"))
    p.add_argument("--scene", nargs="*")
    p.add_argument("--bucket", nargs="*")
    p.add_argument("--limit", type=int, help="first N dialogues")
    p.add_argument("--mode", choices=("off", "on"), help="label only: which server this is")
    p.add_argument("--session-mode", choices=("per-turn", "per-dialogue"), default="per-turn")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gap-s", type=float, default=1.0, help="pause after a turn finishes")
    p.add_argument("--chunk-ms", type=int, default=0, help="stream audio in real time with this chunk size")
    p.add_argument("--listen-wait-s", type=float, default=1.5)
    a = p.parse_args()
    if a.summarize:
        print(json.dumps(summarize(a.summarize, a.output_rate), ensure_ascii=False, indent=1))
        return 0
    if not (a.model and a.out and a.audio_dir and a.mode):
        p.error("--model, --out, --audio-dir and --mode are required to play")
    return asyncio.run(play(a))


if __name__ == "__main__":
    raise SystemExit(main())
