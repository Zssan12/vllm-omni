"""CLM judge on a GPU host: tune the prompt, prepare the model directory, check parity.

    # 1) one Qwen3-8B pooling engine: embed every state/option text once, score all
    #    prompt variants with the CLM heads, choose threshold (tune set only) and a
    #    variant, write <out>/ (vLLM model dir) + tuning summary + reference logits
    python clm_tune_and_prepare.py tune --qwen3-8b <dir> --head CLM_v0.1-8B.pt \
        --tune judge-quality-zh.json --heldout judge-heldout-zh.json --out <new dir> [--variant NAME]

    # 2) separate process: serve <out>/model as ClmDecisionModel (the stage's model
    #    class) and compare its logits with the reference
    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=<vllm-omni tree> \
        python clm_tune_and_prepare.py parity --out <same dir>

Reference maths is clm.engine / clm.heads (Apache-2.0): score = scale * cos(
state_head(embed(state)), action_head(embed(option))), softmax over options.
Labels are the author's own small sets (17 reply + 17 no-reply each).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CONV_EN = 'Conversation so far:\n(no conversation yet)\nThe assistant is not speaking right now.\nThe user just said: "{transcript}"'
SAID_EN = 'The user just said to the voice assistant: "{transcript}"'
MULTI5 = {
    "answer": "Answer the question the user asked",
    "act": "Carry out the action the user requested",
    "ack": 'Stay silent: the user is only acknowledging, like "uh-huh", "okay", "oh"',
    "other": "Stay silent: the user is talking to someone else nearby, not to the assistant",
    "noise": "Stay silent: this is only a cough, laughter, throat clearing, noise or meaningless caption text",
}
VARIANTS = {
    "en-multi5-conv": {"instructions": "What should the voice assistant do next?", "options": MULTI5,
                       "reply": ["answer", "act"], "state": CONV_EN},
    "en-multi5-said": {"instructions": "What should the voice assistant do next?", "options": MULTI5,
                       "reply": ["answer", "act"], "state": SAID_EN},
    "en-action-said": {
        "instructions": "What should the voice assistant do next?",
        "options": {
            "respond": "Speak and answer: the user asked a question, made a request or gave an instruction, "
            "or answered the assistant",
            "stay_silent": "Keep listening without speaking: the user only gave a backchannel (uh-huh, okay, oh), "
            "laughed, coughed, the text is noise, or the user is talking to someone else",
        },
        "reply": ["respond"], "state": SAID_EN},
    "en-short-said": {"instructions": "Does the assistant need to answer?",
                      "options": {"yes": "yes, answer the user", "no": "no, just keep listening"},
                      "reply": ["yes"], "state": SAID_EN},
    "zh-action-said": {
        "instructions": "语音助手接下来应该怎么做？",
        "options": {"yes": "开口回答用户：用户提了问题、提出请求或指令，或者在回答助手刚才的问题",
                    "no": "不出声，继续听：用户只是附和（嗯、好的、哦）、在笑、在咳嗽，或者在和别人说话"},
        "reply": ["yes"], "state": "用户刚刚说：「{transcript}」"},
}
DEMO = ("A-request-0", "A-request-1", "B-backchannel-0", "B-backchannel-1")  # the four GPU demo clips


def state_text(variant, transcript):  # clm.schema.state_text / response_judge._clm_prompt
    s, i = variant["state"].format(transcript=transcript).strip(), variant["instructions"].strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def labelled(path):
    for case in json.loads(Path(path).read_text()):
        label = case["expected"].get("response_needed")
        if label in ("yes", "no"):
            yield case["id"], case["state"]["transcript"], label == "yes"


def score(rows, t):
    pos = [p >= t for p, y in rows if y]
    neg = [p < t for p, y in rows if not y]
    return {"balanced_accuracy": round((sum(pos) / len(pos) + sum(neg) / len(neg)) / 2, 4),
            "blocked_requests": f"{len(pos) - sum(pos)}/{len(pos)}", "filtered_non_requests": f"{sum(neg)}/{len(neg)}"}


def choose_threshold(rows):
    best = None
    for t in sorted({round(p, 6) for p, _ in rows} | {0.5}):
        s = score(rows, t)
        key = (s["balanced_accuracy"], -int(s["blocked_requests"].split("/")[0]), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def tune(a):
    import torch
    import torch.nn.functional as F
    from vllm import LLM

    sys.path.insert(0, str(a.omni)) if a.omni else None
    from vllm_omni.model_executor.models.response_judge.clm import ClmHead

    ck = torch.load(a.head, map_location="cpu")
    cfg = dict(ck["cfg"])
    head_cfg = {"hidden": cfg.get("hidden_size", 4096), "width": cfg["width"], "depth": cfg["depth"],
                "proj": ck.get("projection_dim", cfg.get("projection_dim", 512)),
                "activation": cfg.get("activation", "gelu"), "layernorm": cfg.get("layernorm", False),
                "residual": cfg.get("residual", False)}
    state_head, action_head = ClmHead(**head_cfg).eval(), ClmHead(**head_cfg).eval()
    state_head.load_state_dict(ck["state_head"])
    action_head.load_state_dict(ck["action_head"])
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

    cases = {"tune": list(labelled(a.tune)), "heldout": list(labelled(a.heldout))}
    texts = set()
    for v in VARIANTS.values():
        texts |= {d or k for k, d in v["options"].items()}
        texts |= {state_text(v, t) for rows in cases.values() for _, t, _ in rows}
    texts = sorted(texts)
    llm = LLM(model=str(a.qwen3_8b), runner="pooling", max_model_len=2048, enforce_eager=True,
              gpu_memory_utilization=a.gpu_memory_utilization)
    emb = torch.tensor([o.outputs.embedding for o in llm.embed(texts, use_tqdm=False)], dtype=torch.float32)
    emb = dict(zip(texts, F.normalize(emb, dim=-1)))
    del llm

    def logits(v, transcript):
        with torch.no_grad():
            zs = F.normalize(state_head(emb[state_text(v, transcript)]), dim=-1)
            zo = F.normalize(action_head(torch.stack([emb[d or k] for k, d in v["options"].items()])), dim=-1)
            return scale * (zo @ zs)

    def p_reply(v, lg):
        keys = list(v["options"])
        probs = torch.softmax(lg, -1)
        return float(sum(probs[keys.index(k)] for k in v["reply"]))

    a.out.mkdir(parents=True)
    summary, raw = {}, []
    for name, v in VARIANTS.items():
        sets = {}
        for split, rows in cases.items():
            sets[split] = []
            for cid, transcript, y in rows:
                lg = logits(v, transcript)
                p = p_reply(v, lg)
                sets[split].append((p, y))
                raw.append({"variant": name, "split": split, "id": cid, "transcript": transcript, "reply": y,
                            "p_reply": p, "logits": lg.tolist(), "state_text": state_text(v, transcript)})
        t = choose_threshold(sets["tune"])
        demo = {r["id"]: r["p_reply"] for r in raw if r["variant"] == name and r["split"] == "tune" and r["id"] in DEMO}
        summary[name] = {"threshold_from_tune": t, "tune@threshold": score(sets["tune"], t),
                         "heldout@threshold": score(sets["heldout"], t), "heldout@0.5": score(sets["heldout"], 0.5),
                         "demo_clips_correct": all((demo[c] >= t) == c.startswith("A") for c in DEMO), "demo_p": demo}
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (a.out / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n")
    (a.out / "variants.json").write_text(json.dumps(VARIANTS, ensure_ascii=False, indent=1))

    chosen = a.variant or max(
        summary, key=lambda n: (summary[n]["demo_clips_correct"], summary[n]["tune@threshold"]["balanced_accuracy"],
                                -int(summary[n]["tune@threshold"]["blocked_requests"].split("/")[0])))
    v, t = VARIANTS[chosen], summary[chosen]["threshold_from_tune"]
    write_model_dir(a, v, t, head_cfg, state_head, action_head, scale, emb)
    ref = [r for r in raw if r["variant"] == chosen]
    (a.out / "reference.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ref) + "\n")
    print(json.dumps({"chosen": chosen, "threshold": t}), flush=True)


def write_model_dir(a, v, threshold, head_cfg, state_head, action_head, scale, emb):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    model = a.out / "model"
    model.mkdir()
    config = json.loads((a.qwen3_8b / "config.json").read_text())
    config["architectures"] = ["ClmDecisionModel"]
    config["clm_head"] = head_cfg
    config["clm_num_options"] = len(v["options"])
    config["response_judge"] = {"format": "clm", "instructions": v["instructions"], "option_keys": list(v["options"]),
                                "reply_option": v["reply"], "threshold": threshold, "state_template": v["state"]}
    (model / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    with torch.no_grad():
        option_proj = F.normalize(action_head(torch.stack([emb[d or k] for k, d in v["options"].items()])), dim=-1)
    tensors = {f"clm.state_head.{k}": t.contiguous() for k, t in state_head.state_dict().items()}
    tensors["clm.option_proj"] = option_proj.contiguous()
    tensors["clm.scale"] = torch.tensor(scale)
    save_file(tensors, str(model / "clm_head.safetensors"))
    index_path = a.qwen3_8b / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in tensors:
        index["weight_map"][name] = "clm_head.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    for src in a.qwen3_8b.iterdir():
        if src.name.endswith(".safetensors") or src.name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
            (model / src.name).symlink_to(src.resolve())


def parity(a):
    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    ref = [json.loads(line) for line in (a.out / "reference.jsonl").read_text().splitlines() if line]
    llm = LLM(model=str(a.out / "model"), runner="pooling", max_model_len=2048, enforce_eager=True,
              gpu_memory_utilization=a.gpu_memory_utilization)
    params = PoolingParams(task="classify")
    for r in ref[:3]:  # warmup
        llm.encode([r["state_text"]], pooling_params=params, pooling_task="classify", use_tqdm=False)
    outs, latency = [], []
    for r in ref:  # one request at a time: the judge's per-turn cost
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs += llm.encode([r["state_text"]], pooling_params=params, pooling_task="classify", use_tqdm=False)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - t0) * 1000)
    latency.sort()
    config = json.loads((a.out / "model" / "config.json").read_text())["response_judge"]
    keys, reply, threshold = config["option_keys"], config["reply_option"], config["threshold"]

    def p_reply(logits):
        probs = torch.softmax(logits, -1)
        return float(sum(probs[keys.index(k)] for k in reply))

    diffs, agree, decisions_agree, rows = [], 0, 0, []
    for r, o in zip(ref, outs):
        got = o.outputs.data.float().cpu()
        want = torch.tensor(r["logits"])
        diffs.append(float((got - want).abs().max()))
        agree += int(got.argmax()) == int(want.argmax())
        p_got, p_want = p_reply(got), p_reply(want)
        same = (p_got >= threshold) == (p_want >= threshold)
        decisions_agree += same
        rows.append({"id": r["id"], "split": r["split"], "transcript": r["transcript"], "p_reply_stage": p_got,
                     "p_reply_reference": p_want, "decision_agree": same, "argmax_agree": int(got.argmax()) == int(want.argmax()),
                     "stage_logits": got.tolist(), "reference_logits": r["logits"]})
    (a.out / "parity-rows.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    report = {"rows": len(ref), "argmax_agree": agree, "reply_decision_agree": decisions_agree,
              "max_abs_logit_diff": max(diffs),
              "max_abs_p_reply_diff": max(abs(x["p_reply_stage"] - x["p_reply_reference"]) for x in rows),
              "latency_ms_p50": round(latency[len(latency) // 2], 2),
              "latency_ms_p95": round(latency[min(len(latency) - 1, int(0.95 * len(latency)))], 2),
              "note": "stage model (ClmDecisionModel, pooling runner, one request at a time) vs reference CLM maths "
                      "on vLLM embeddings computed in one batch"}
    (a.out / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("tune", "parity"))
    p.add_argument("--qwen3-8b", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--tune", type=Path)
    p.add_argument("--heldout", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=sorted(VARIANTS))
    p.add_argument("--omni", type=Path, help="vllm-omni tree (if not on PYTHONPATH)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = p.parse_args()
    if a.phase == "tune":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; choose a new directory")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        tune(a)
    else:
        parity(a)


if __name__ == "__main__":
    main()
