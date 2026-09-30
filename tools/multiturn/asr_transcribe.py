"""Transcribe the multi-turn test WAVs with a local (loopback) ASR server, e.g. the pipeline's Qwen3-ASR:

    vllm serve <Qwen3-ASR-1.7B> --served-model-name qwen3-asr --host 127.0.0.1 --port 8095 ...
    python asr_transcribe.py --audio-dir multiturn-zh/audio-say --out asr.jsonl

One row per turn: dialogue, turn, reference text, transcript. Feed it to
`judge_text_eval.py convert --asr` to judge the ASR output instead of the reference text.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import uuid
from pathlib import Path


def post_wav(conn, model, path):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", model), ("language", "zh"), ("temperature", "0")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    conn.request("POST", "/v1/audio/transcriptions", body=b"".join(parts),
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"{path}: HTTP {resp.status} {body[:200]!r}")
    return json.loads(body)["text"].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", type=Path, required=True, help="directory with manifest.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--model", default="qwen3-asr")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=60)
    rows = []
    for line in (a.audio_dir / "manifest.jsonl").read_text().splitlines():
        m = json.loads(line)
        t0 = time.perf_counter()
        text = post_wav(conn, a.model, a.audio_dir / m["wav"])
        rows.append({"dialogue": m["dialogue"], "turn": m["turn"], "reference": m["text"], "transcript": text,
                     "category": m["category"], "asr_ms": round((time.perf_counter() - t0) * 1000, 1)})
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    same = sum(r["transcript"].rstrip("。？！，.?!") == r["reference"].rstrip("。？！，.?!") for r in rows)
    print(json.dumps({"turns": len(rows), "exact_match_ignoring_final_punct": same}))


if __name__ == "__main__":
    main()
