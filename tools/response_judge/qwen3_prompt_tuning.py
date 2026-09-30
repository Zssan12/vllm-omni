"""Qwen3 chat judge prompt check on a GPU host, with the stage's exact input format.

Each variant is a system prompt for the response_judge ``chat_yes_no`` format;
the user message is the stage default (current transcript only, no history)
and the chat template is rendered with enable_thinking=False, exactly like
response_judge._chat_prompt. One token, temperature 0. The variant is chosen
on the tune set; report the held-out numbers. Also records in-process judge
latency (vllm.LLM, prefix cache on): the compute lower bound for the stage.

    PYTHONPATH=<vllm-omni tree> python qwen3_prompt_tuning.py --model <Qwen3-1.7B dir> --tune judge-quality-zh.json \
        --heldout judge-heldout-zh.json --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RULES = (
    "你是语音助手前面的门控判断器，只判断语音助手要不要回应用户刚说的话。"
    "只输出 YES 或 NO，不要输出其他内容。\n"
    "YES：用户在提问、提出请求或指令，或者在回答助手刚才的问题。\n"
    "NO：用户只是在附和（嗯、好的、哦、对对对）、在笑、在咳嗽，这段文字是噪声或无意义的字幕，"
    "或者用户在和身边的其他人说话。"
)
EXAMPLES = (
    "\n\n示例（与评测题不同）：\n"
    "用户刚刚说：「嗯。」 → NO\n"
    "用户刚刚说：「明天是星期几？」 → YES\n"
    "用户刚刚说：「音乐吧」 → YES\n"
    "用户刚刚说：「爸，你手机响了。」 → NO\n"
    "用户刚刚说：「(咳嗽)」 → NO\n"
    "用户刚刚说：「好嘞」 → NO\n"
    "用户刚刚说：「帮我查一下快递到哪了」 → YES"
)
MINIMAL = (
    "判断语音助手是否需要回应用户刚说的话。附和、笑、咳嗽、噪声、对别人说的话输出 NO；"
    "提问、请求、回答助手的问题输出 YES。只输出 YES 或 NO。"
)
VARIANTS = {"rules (stage default)": RULES, "fewshot": RULES + EXAMPLES, "minimal": MINIMAL}
USER = "对话记录：\n（还没有对话）\n用户刚刚说：「{transcript}」"  # response_judge.DEFAULT_CHAT_USER_TEMPLATE
DEMO = ("A-request-0", "A-request-1", "B-backchannel-0", "B-backchannel-1")


def labelled(path):
    for case in json.loads(Path(path).read_text()):
        label = case["expected"].get("response_needed")
        if label in ("yes", "no"):
            yield case["id"], case["state"]["transcript"], label == "yes"


def score(rows):
    pos = [r for r, y in rows if y]
    neg = [not r for r, y in rows if not y]
    return {"balanced_accuracy": round((sum(pos) / len(pos) + sum(neg) / len(neg)) / 2, 4),
            "blocked_requests": f"{len(pos) - sum(pos)}/{len(pos)}", "filtered_non_requests": f"{sum(neg)}/{len(neg)}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tune", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new directory")
    a.out.mkdir(parents=True)
    from vllm import LLM, SamplingParams
    from vllm.plugins import load_general_plugins

    # Load the stage's own class (ResponseJudgeQwen3ForCausalLM) through vllm-omni's
    # plugin; run with PYTHONPATH pointing at the vllm-omni tree under test.
    load_general_plugins()
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=2048,
              enable_prefix_caching=True, seed=0,
              hf_overrides={"architectures": ["ResponseJudgeQwen3ForCausalLM"]})
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def prompt(system, transcript):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": USER.format(transcript=transcript)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    for system in VARIANTS.values():  # warm the prefix cache and kernels
        llm.generate([prompt(system, "你好")], sp, use_tqdm=False)
    summary, raw = {}, []
    for name, system in VARIANTS.items():
        sets, latency = {}, []
        for split, path in (("tune", a.tune), ("heldout", a.heldout)):
            rows = []
            for cid, transcript, y in labelled(path):
                t0 = time.perf_counter()
                out = llm.generate([prompt(system, transcript)], sp, use_tqdm=False)[0].outputs[0]
                latency.append((time.perf_counter() - t0) * 1000)
                answer = out.text.strip().upper()
                reply = answer != "NO"  # the stage rejects only a clear NO
                rows.append((reply, y))
                raw.append({"variant": name, "split": split, "id": cid, "transcript": transcript, "reply": y,
                            "answer": out.text, "token_ids": list(out.token_ids)})
            sets[split] = rows
        latency.sort()
        demo = {r["id"]: r["answer"].strip().upper() != "NO" for r in raw
                if r["variant"] == name and r["split"] == "tune" and r["id"] in DEMO}
        summary[name] = {"tune": score(sets["tune"]), "heldout": score(sets["heldout"]),
                         "demo_clips_correct": all(demo[c] == c.startswith("A") for c in DEMO),
                         "inproc_latency_ms_p50": round(latency[len(latency) // 2], 2),
                         "inproc_latency_ms_p95": round(latency[min(len(latency) - 1, int(0.95 * len(latency)))], 2),
                         "system_prompt": VARIANTS[name]}
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != "system_prompt"}, ensure_ascii=False),
              flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (a.out / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n")


if __name__ == "__main__":
    main()
