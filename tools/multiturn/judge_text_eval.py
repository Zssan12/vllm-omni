"""Text-level judge check on the multi-turn test dialogues, reusing the existing tuners.

The stage judges only the current transcript (no history), so every dialogue turn
becomes one labelled case in the tuners' input format (judge-quality-zh.json):
id = <dialogue>-tNN, group = category, expected.response_needed = yes/no.

    # 1) convert (local or GPU host, no deps)
    python judge_text_eval.py convert --dialogues dialogues-tune.jsonl --out tune.json
    # optional: use ASR transcripts instead of the reference text (asr.jsonl from asr_transcribe.py)
    python judge_text_eval.py convert --dialogues ... --asr asr.jsonl --out tune-asr.json

    # 2) run an existing tuner unchanged except for its 4 demo-clip ids (absent here)
    python judge_text_eval.py run qwen3_prompt_tuning.py -- --model ... --tune tune.json --heldout heldout.json --out <new dir>

    # 3) per-category breakdown of a tuner's raw.jsonl with its chosen thresholds
    python judge_text_eval.py breakdown --run <tuner out dir> --cases tune.json heldout.json
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path


def convert(a):
    asr = {}
    if a.asr:
        for line in Path(a.asr).read_text().splitlines():
            r = json.loads(line)
            asr[(r["dialogue"], r["turn"])] = r["transcript"]
    cases = []
    for path in a.dialogues:
        for line in Path(path).read_text().splitlines():
            d = json.loads(line)
            for t in d["turns"]:
                key = (d["id"], t["idx"])
                if a.asr and key not in asr:
                    raise SystemExit(f"no ASR transcript for {key}")
                cases.append({
                    "id": f"{d['id']}-t{t['idx']:02d}",
                    "group": t["category"],
                    "expected": {"response_needed": "yes" if t["reply"] else "no"},
                    "state": {"history": [], "transcript": asr[key] if a.asr else t["text"],
                              "assistant_speaking": False},
                    "reference_text": t["text"],
                    "scene": d["scene"], "bucket": d["bucket"],
                })
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(json.dumps({"cases": len(cases), "reply": sum(c["expected"]["response_needed"] == "yes" for c in cases)}))


def run(a):
    # The tuners check four fixed demo clips by id; they are not in this dataset.
    sys.argv = [a.script, *a.args]
    main = runpy.run_path(a.script, run_name="__tuner__")["main"]
    main.__globals__["DEMO"] = ()  # run_path returns a copy; patch the live namespace
    if a.variants:  # replace the tuner's VARIANTS (same shape as the tuner's own)
        main.__globals__["VARIANTS"] = json.loads(Path(a.variants).read_text())
    main()


def decide(row, thresholds):
    if "answer" in row:  # generative YES/NO: the stage rejects only a clear NO
        return row["answer"].strip().upper() != "NO"
    return row["p_reply"] >= thresholds[row["variant"]]


def breakdown(a):
    run_dir = Path(a.run)
    summary = json.loads((run_dir / "summary.json").read_text())
    thresholds = {name: s.get("threshold_from_tune") for name, s in summary.items()}
    group = {}
    for path in a.cases:
        for c in json.loads(Path(path).read_text()):
            group[c["id"]] = c["group"]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for line in (run_dir / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] not in group:
            continue
        cell = table[(r["variant"], r["split"])][group[r["id"]]]
        cell[0] += decide(r, thresholds) == r["reply"]
        cell[1] += 1
    result = {f"{v} | {s}": {g: f"{ok}/{n}" for g, (ok, n) in sorted(cats.items())}
              for (v, s), cats in sorted(table.items())}
    print(json.dumps({"thresholds": thresholds, "correct_by_category": result}, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--dialogues", action="append", required=True)
    c.add_argument("--asr")
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variants", help="JSON file replacing the tuner's VARIANTS dict")
    r.add_argument("script")
    r.add_argument("args", nargs=argparse.REMAINDER)
    b = sub.add_parser("breakdown")
    b.add_argument("--run", required=True)
    b.add_argument("--cases", nargs="+", required=True)
    a = p.parse_args()
    if a.cmd == "run" and a.args[:1] == ["--"]:
        a.args = a.args[1:]
    {"convert": convert, "run": run, "breakdown": breakdown}[a.cmd](a)


if __name__ == "__main__":
    main()
