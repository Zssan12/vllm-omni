"""Check that response_judge's chat_yes_no prompt equals the old AURA judge prompt byte for byte.

The old prompt (vllm-omni-judge-stage aura_omni/judge.py build_judge_prompt) is
the one used for all earlier GPU evidence. The new bridge renders the Qwen3
chat template with enable_thinking=False; this renders that template with
jinja2 (as transformers does) from a tokenizer_config.json.

    python qwen3_prompt_equality.py <tokenizer_config.json> <old aura_omni dir> <new stage_input_processors dir>
"""

import importlib.util
import json
import sys
import types

import jinja2


def main() -> None:
    tmpl = json.load(open(sys.argv[1]))["chat_template"]
    sys.modules["vllm_omni.engine.duplex.contracts"] = types.SimpleNamespace(
        duplex_resource_request_belongs_to_session=lambda *a: False
    )
    spec = importlib.util.spec_from_file_location("oldjudge", sys.argv[2] + "/judge.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["oldjudge"] = old
    spec.loader.exec_module(old)
    src = open(sys.argv[3] + "/response_judge.py").read()
    ns: dict = {}
    exec(src.split("@dataclass")[0].replace("from vllm.logger import init_logger", "init_logger=lambda n: None"), ns)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    template = env.from_string(tmpl)
    cases = ["嗯嗯", "好的", "今天北京天气怎么样？", "帮我定一个明天早上七点的闹钟。", " 对对对 "]
    results = []
    for c in cases:
        msgs = [
            {"role": "system", "content": ns["DEFAULT_CHAT_SYSTEM_PROMPT"]},
            {"role": "user", "content": ns["DEFAULT_CHAT_USER_TEMPLATE"].format(transcript=c.strip())},
        ]
        new = template.render(messages=msgs, add_generation_prompt=True, enable_thinking=False, tools=None)
        results.append({"transcript": c, "identical": new == old.build_judge_prompt(c, old.DEFAULT_AURA_JUDGE_PROMPT)})
    print(json.dumps({"identical": sum(r["identical"] for r in results), "cases": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
