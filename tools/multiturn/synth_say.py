"""Synthesize the multi-turn test dialogues with macOS `say` (16 kHz mono WAV, one file per turn).

Each dialogue uses its own voice (the dialogue's "voice" field; mainland Mandarin
voices other than Tingting are addressed by their full `say` name). Output:

  <out>/<dialogue id>/tNN.wav      one user turn
  <out>/manifest.jsonl             dialogue id, turn, text, category, reply, voice, wav, seconds, sha256

    python synth_say.py --dialogues multiturn-zh/data/dialogues-tune.jsonl \
        --dialogues multiturn-zh/data/dialogues-heldout.jsonl --out multiturn-zh/audio-say
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

SINGLE_NAME = {"Tingting"}


def say_voice(name: str) -> str:
    return name if name in SINGLE_NAME else f"{name} (Chinese (China mainland))"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dialogues", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rate", type=int, default=0, help="words per minute for `say -r` (0 = voice default)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "turn.aiff"
        for path in a.dialogues:
            for line in path.read_text().splitlines():
                d = json.loads(line)
                ddir = a.out / d["id"]
                ddir.mkdir()
                for t in d["turns"]:
                    wav = ddir / f"t{t['idx']:02d}.wav"
                    cmd = ["say", "-v", say_voice(d["voice"]), "-o", str(aiff)]
                    if a.rate:
                        cmd += ["-r", str(a.rate)]
                    subprocess.run([*cmd, t["text"]], check=True)
                    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                                   check=True)
                    with wave.open(str(wav)) as w:
                        seconds = w.getnframes() / w.getframerate()
                    rows.append({
                        "dialogue": d["id"], "split": d["split"], "scene": d["scene"], "bucket": d["bucket"],
                        "turn": t["idx"], "text": t["text"], "category": t["category"], "reply": t["reply"],
                        "voice": d["voice"], "wav": str(wav.relative_to(a.out)), "seconds": round(seconds, 3),
                        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    })
    (a.out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    total = sum(r["seconds"] for r in rows)
    print(json.dumps({"turns": len(rows), "audio_seconds": round(total, 1),
                      "min_s": min(r["seconds"] for r in rows), "max_s": max(r["seconds"] for r in rows)}))


if __name__ == "__main__":
    main()
