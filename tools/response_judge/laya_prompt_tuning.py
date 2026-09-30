"""Prompt / threshold tuning for the LAYA response judge (laya package, same maths as the stage).

The stage judges only the current ASR transcript, so every variant here sees
exactly that (no history, no assistant state). For each variant:

  * P(reply) is computed for every labelled case of the tune and held-out sets;
  * the threshold is chosen on the tune set only (max balanced accuracy, ties ->
    fewer blocked requests, then closest to 0.5);
  * the held-out set is scored with that frozen threshold and with 0.5.

Labels are the author's own (judge-quality-zh.json / judge-heldout-zh.json,
17 reply + 17 no-reply each); this is a small directional check, not a benchmark.

    python laya_prompt_tuning.py --model-dir <laya snapshot> --laya-pkg <dir> \
        --tune judge-quality-zh.json --heldout judge-heldout-zh.json --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ZH_ACTION = {
    "yes": "开口回答用户：用户提了问题、提出请求或指令，或者在回答助手刚才的问题",
    "no": "不出声，继续听：用户只是附和（嗯、好的、哦）、在笑、在咳嗽，或者在和别人说话",
}
EN_ACTION = {
    "respond": "Speak and answer: the user asked a question, made a request or gave an instruction, or answered the assistant",
    "stay_silent": "Keep listening without speaking: the user only gave a backchannel (uh-huh, okay, oh), laughed, coughed, "
    "the text is noise, or the user is talking to someone else",
}
FRAMED_ZH = "用户刚刚说：「{t}」"
FRAMED_EN = "The user just said to the voice assistant: \"{t}\""

VARIANTS = {
    "zh-choice-plain": {"type": "choice", "ins": "语音助手接下来应该怎么做？", "crit": ZH_ACTION, "reply": "yes", "state": "{t}"},
    "zh-choice-framed": {"type": "choice", "ins": "语音助手接下来应该怎么做？", "crit": ZH_ACTION, "reply": "yes", "state": FRAMED_ZH},
    "en-choice-plain": {"type": "choice", "ins": "What should the voice assistant do next?", "crit": EN_ACTION, "reply": "respond", "state": "{t}"},
    "en-choice-framed": {"type": "choice", "ins": "What should the voice assistant do next?", "crit": EN_ACTION, "reply": "respond", "state": FRAMED_EN},
    "en-noul-framed": {
        "type": "noul",
        "ins": "The voice assistant should reply to what the user just said.",
        "crit": {
            "true": "a reply is needed: a question, request, instruction, or an answer to the assistant",
            "false": "no reply: a backchannel, laughter, a cough, noise, or speech addressed to someone else",
        },
        "state": FRAMED_EN,
    },
    "en-noul-plain": {
        "type": "noul",
        "ins": "This utterance is addressed to the voice assistant and needs a spoken reply.",
        "crit": {"true": "needs a reply", "false": "backchannel, filler, noise or side talk; no reply"},
        "state": "{t}",
    },
    "zh-noul-framed": {
        "type": "noul",
        "ins": "语音助手需要回应用户刚说的这句话。",
        "crit": {"true": "需要回应：提问、请求、指令，或者在回答助手", "false": "不需要回应：附和、笑、咳嗽、噪声，或在和别人说话"},
        "state": FRAMED_ZH,
    },
    "en-choice-3way": {
        "type": "choice",
        "ins": "What kind of input is this for a voice assistant?",
        "crit": {
            "request": "a question, request, instruction, or an answer to the assistant",
            "backchannel": "a backchannel or filler such as uh-huh, okay, oh, or laughter",
            "not_for_assistant": "noise, a cough, or speech addressed to someone else",
        },
        "reply": "request",
        "state": FRAMED_EN,
    },
    "en-choice-short": {
        "type": "choice",
        "ins": "Does the assistant need to answer?",
        "crit": {"yes": "yes, answer the user", "no": "no, just keep listening"},
        "reply": "yes",
        "state": FRAMED_EN,
    },
    "zh-choice-short": {
        "type": "choice",
        "ins": "助手需要回答吗？",
        "crit": {"yes": "需要回答用户", "no": "不需要，继续听"},
        "reply": "yes",
        "state": FRAMED_ZH,
    },
}


def labelled(path: Path):
    for case in json.loads(path.read_text()):
        label = case["expected"].get("response_needed")
        if label in ("yes", "no"):
            yield case["id"], case["group"], case["state"]["transcript"], label == "yes"


def p_reply(agent, variant, transcript) -> float:
    state = variant["state"].format(t=transcript)
    question = {"instructions": variant["ins"], "type": variant["type"], "criteria": variant["crit"]}
    result = agent.predict(state, {"q": question})
    answer = result.get("answers", result)["q"]
    if variant["type"] == "noul":
        return float(answer["noul"])
    return float(answer["probabilities"][variant["reply"]])


def score(rows, threshold):
    reply = [p >= threshold for p, _ in rows]
    pos = [r for r, (_, y) in zip(reply, rows) if y]
    neg = [not r for r, (_, y) in zip(reply, rows) if not y]
    tpr, tnr = sum(pos) / len(pos), sum(neg) / len(neg)
    return {
        "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "blocked_requests": f"{len(pos) - sum(pos)}/{len(pos)}",
        "filtered_non_requests": f"{sum(neg)}/{len(neg)}",
    }


def choose_threshold(rows):
    candidates = sorted({round(p, 6) for p, _ in rows} | {0.5})
    best = None
    for t in candidates:
        s = score(rows, t)
        blocked = int(s["blocked_requests"].split("/")[0])
        key = (s["balanced_accuracy"], -blocked, -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--laya-pkg", required=True)
    p.add_argument("--tune", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new directory")
    a.out.mkdir(parents=True)
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, a.laya_pkg)
    import laya

    agent = laya.load(a.model_dir, device="cuda")
    summary = {}
    with (a.out / "raw.jsonl").open("w") as raw:
        for name, variant in VARIANTS.items():
            sets = {}
            for split, path in (("tune", a.tune), ("heldout", a.heldout)):
                rows = []
                for cid, group, transcript, is_reply in labelled(path):
                    prob = p_reply(agent, variant, transcript)
                    rows.append((prob, is_reply))
                    raw.write(json.dumps({"variant": name, "split": split, "id": cid, "group": group,
                                          "transcript": transcript, "reply": is_reply, "p_reply": prob},
                                         ensure_ascii=False) + "\n")
                sets[split] = rows
            t = choose_threshold(sets["tune"])
            summary[name] = {
                "threshold_from_tune": t,
                "tune@threshold": score(sets["tune"], t),
                "heldout@threshold": score(sets["heldout"], t),
                "heldout@0.5": score(sets["heldout"], 0.5),
            }
            print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    (a.out / "variants.json").write_text(json.dumps(VARIANTS, ensure_ascii=False, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
