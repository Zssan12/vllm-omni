"""Synthesize the multi-turn test dialogues with a local Qwen3-TTS CustomVoice server (vllm-omni, /v1/audio/speech).

Same output layout as synth_say.py (<out>/<dialogue>/tNN.wav + manifest.jsonl, 16 kHz mono PCM16),
so every downstream tool reads either set. tune and held-out dialogues use disjoint speakers.
Backchannels, fillers and side talk get a short delivery instruction, because a flat read of
"嗯嗯" is not how people say it.

    vllm-omni serve <Qwen3-TTS-12Hz-1.7B-CustomVoice> --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
        --host 127.0.0.1 --port 8091 --trust-remote-code --omni
    python synth_qwen3_tts.py --dialogues dialogues-tune.jsonl --dialogues dialogues-heldout.jsonl --out audio-qwen3tts
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import wave
from pathlib import Path

SPEAKERS = {"tune": ["vivian", "uncle_fu", "eric"], "heldout": ["serena", "dylan"]}
STYLE = {
    "backchannel": "随口轻声应一声，语气自然，像在边听边附和",
    "filler": "犹豫、拖长音，像在想事情",
    "side_talk": "转头对身边的家里人说话，语气随意",
}


def to_pcm16_16k(data: bytes) -> bytes:
    import numpy as np

    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected PCM16 from the server, got sample width {width}")
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        n = int(round(len(x) * 16000 / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())
    return out.getvalue()


def speak(conn, model, text, speaker, instructions):
    payload = {"model": model, "input": text, "voice": speaker, "language": "Chinese",
               "task_type": "CustomVoice", "response_format": "wav"}
    if instructions:
        payload["instructions"] = instructions
    conn.request("POST", "/v1/audio/speech", body=json.dumps(payload, ensure_ascii=False).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body[:300]!r}")
    return body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dialogues", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--model", required=True, help="served model name or path")
    p.add_argument("--no-style", action="store_true", help="plain reading for every turn")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=120)
    rows, count = [], {"tune": 0, "heldout": 0}
    for path in a.dialogues:
        for line in path.read_text().splitlines():
            d = json.loads(line)
            pool = SPEAKERS[d["split"]]
            speaker = pool[count[d["split"]] % len(pool)]
            count[d["split"]] += 1
            ddir = a.out / d["id"]
            ddir.mkdir()
            for t in d["turns"]:
                style = None if a.no_style else STYLE.get(t["category"])
                wav = ddir / f"t{t['idx']:02d}.wav"
                wav.write_bytes(to_pcm16_16k(speak(conn, a.model, t["text"], speaker, style)))
                with wave.open(str(wav)) as w:
                    seconds = w.getnframes() / w.getframerate()
                rows.append({
                    "dialogue": d["id"], "split": d["split"], "scene": d["scene"], "bucket": d["bucket"],
                    "turn": t["idx"], "text": t["text"], "category": t["category"], "reply": t["reply"],
                    "voice": f"qwen3-tts:{speaker}", "style": style, "wav": str(wav.relative_to(a.out)),
                    "seconds": round(seconds, 3), "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                })
    (a.out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    total = sum(r["seconds"] for r in rows)
    print(json.dumps({"turns": len(rows), "audio_seconds": round(total, 1),
                      "min_s": min(r["seconds"] for r in rows), "max_s": max(r["seconds"] for r in rows)}))


if __name__ == "__main__":
    main()
