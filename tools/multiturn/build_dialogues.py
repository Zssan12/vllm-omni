"""Build the multi-turn test dialogues from sentence pools (deterministic).

Each dialogue is one user talking to the assistant over several turns. The first
turn always needs a reply, so the assistant has something to answer before any
backchannel. The remaining turns mix reply turns (question / request) with
no-reply turns, at one of three densities:

  low   about 20% of turns need no reply
  mid   about 40%
  high  about 60%

Scenes:
  S1  "listening and acknowledging": no-reply turns are backchannels
  S2  "someone else in the room": no-reply turns are side talk, fillers and some backchannels

tune and heldout come from separate sentence pools and separate voices, so the
held-out numbers also cover unseen sentences and speakers.

    python build_dialogues.py --pools multiturn-zh/data/pools.json --out multiturn-zh/data
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

VOICES = {
    "tune": ["Tingting", "Eddy", "Flo", "Reed", "Sandy", "Shelley"],
    "heldout": ["Grandma", "Grandpa", "Rocko"],
}
COUNTS = {"tune": 10, "heldout": 5}  # dialogues per (bucket), split across scenes
SHARES = {"low": 0.2, "mid": 0.4, "high": 0.6}
REPLY_CATEGORIES = ("question", "request")


def no_reply_category(scene: str, rng: random.Random) -> str:
    if scene == "S1":
        return "backchannel"
    return rng.choices(["side_talk", "filler", "backchannel"], weights=[0.5, 0.25, 0.25])[0]


def positions(n_turns: int, k: int, rng: random.Random) -> set[int]:
    """k no-reply positions among 1..n-1, at most two in a row."""
    for _ in range(1000):
        pos = set(rng.sample(range(1, n_turns), k))
        runs = sorted(pos)
        if not any(runs[i] + 1 == runs[i + 1] and runs[i] + 2 in pos for i in range(len(runs) - 1)):
            return pos
    raise RuntimeError("could not place no-reply turns")


def build_split(split: str, pools: dict, rng: random.Random) -> list[dict]:
    dialogues = []
    voices = VOICES[split]
    for bucket, share in SHARES.items():
        for i in range(COUNTS[split]):
            scene = "S1" if i % 2 == 0 else "S2"
            n_turns = rng.randint(6, 8)
            k = max(1, round(share * n_turns))
            no_reply_at = positions(n_turns, k, rng)
            reply_pool = [(c, t) for c in REPLY_CATEGORIES for t in pools[split][c]]
            rng.shuffle(reply_pool)
            used: dict[str, set[str]] = {}
            turns = []
            for idx in range(n_turns):
                if idx in no_reply_at:
                    category = no_reply_category(scene, rng)
                    choices = [t for t in pools[split][category] if t not in used.get(category, set())]
                    text = rng.choice(choices or pools[split][category])
                    used.setdefault(category, set()).add(text)
                else:
                    category, text = reply_pool.pop()
                turns.append({"idx": idx, "text": text, "category": category,
                              "reply": category in REPLY_CATEGORIES})
            dialogues.append({
                "id": f"{split}-{scene}-{bucket}-{i:02d}",
                "split": split,
                "scene": scene,
                "bucket": bucket,
                "voice": voices[len(dialogues) % len(voices)],
                "turns": turns,
            })
    return dialogues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260930)
    a = p.parse_args()
    pools = json.loads(a.pools.read_text())
    rng = random.Random(a.seed)
    summary = {}
    for split in ("tune", "heldout"):
        dialogues = build_split(split, pools, rng)
        path = a.out / f"dialogues-{split}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists; choose a new --out")
        path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n")
        turns = [t for d in dialogues for t in d["turns"]]
        summary[split] = {
            "dialogues": len(dialogues),
            "turns": len(turns),
            "reply_turns": sum(t["reply"] for t in turns),
            "no_reply_turns": sum(not t["reply"] for t in turns),
            "by_category": dict(Counter(t["category"] for t in turns)),
            "by_bucket_no_reply_share": {
                b: round(sum(not t["reply"] for d in dialogues if d["bucket"] == b for t in d["turns"])
                         / sum(len(d["turns"]) for d in dialogues if d["bucket"] == b), 2)
                for b in SHARES
            },
            "voices": sorted({d["voice"] for d in dialogues}),
        }
    (a.out / "summary.json").write_text(json.dumps({"seed": a.seed, **summary}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
