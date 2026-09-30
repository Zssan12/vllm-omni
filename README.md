# Response judge stage: reproduction materials

Scripts, configs and data behind the numbers reported for
[vllm-project/vllm-omni#8316](https://github.com/vllm-project/vllm-omni/pull/8316)
and discussed in [#8211](https://github.com/vllm-project/vllm-omni/issues/8211).
None of this is part of the PR. The measurement hooks patch a *copy* of the
source tree and are never meant to be merged.

## Source under test

| Numbers | vllm-omni source |
| --- | --- |
| Profiling before/after tables below (P1–P9, repeats) | `4c1da86187d366149fae028f9f4fa20be8adb16c` (the first #8316 head) |
| End-to-end check of the PR update, client `listen_source` (P10/P11) | `4c1da86` plus an earlier revision of the update (`listen_source` forwarding, LAYA capture sizes). That patch is not included, so P10/P11 are **reference results only** |
| Final #8316 update | `ecf9cb08e87eb6e363c491ca94e265d241968b83`: covered by regression tests, not re-run end to end |
| Prototype numbers (84 GB, Qwen3 judge) | an earlier AURA-specific prototype on base `235c4303`; not reproducible with this package |

Every end-to-end run applies two measurement patches to a copy of that source,
in this order: `tools/response_judge/timing_patch.py`, then
`tools/response_judge/prof_patch.py`.

## Prerequisites

- vLLM 0.30.0 and torch 2.13.0 (CUDA 13.0 build) in one virtualenv, with the
  vllm-omni source above importable (`PYTHONPATH=<patched tree>`) and the
  `vllm-omni` CLI installed in the same virtualenv (`run_arm.py` starts
  `<venv>/bin/vllm-omni` unless `--serve-bin` is given).
- `gcc` and `ninja` on `PATH` (the first run compiles kernels; `ninja` lives in
  the virtualenv's `bin/`).
- Python packages used by the tools beyond vLLM/torch: `PyYAML`, `jinja2`
  (`qwen3_prompt_equality.py`), `nvidia-ml-py`/`pynvml` (the standalone
  timing scripts) and, for LAYA parity only, the `laya` package 0.3.20
  (`--laya-pkg <dir>`). The runner and Realtime clients also need `psutil`
  (`run_arm.py`), `Pillow` (image frames in the client) and `websockets`
  (the duplex client); these are in vllm-omni's `dev` extras, not in the
  base install.
- Offline model directories. Replace every `/path/to/models/...` in the YAML
  files with your paths; `run_arm.py --models` only locates the host model and
  the TTS tokenizer and does not rewrite the YAML. Use absolute paths for the
  source tree, configs, audio and output directories, because the runner
  starts the server from the source tree.

## Models and revisions

| Model | Revision | How it is prepared |
| --- | --- | --- |
| `convaiinnovations/laya-multilingual` | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | `tools/response_judge/prepare_laya_dir.py <snapshot> <out>` writes a vLLM-loadable directory (`config.json` with `architectures: [LayaDecisionModel]`, the tokenizer, a symlink to the weights) |
| `Contrastive-LM/CLM-v0.1-8B` | `e939398d4556fcd9400c76fa8c5a513202f42b0a` | `tools/response_judge/clm_tune_and_prepare.py tune ... --variant en-multi5-conv` (see its docstring) writes the model directory for `ClmDecisionModel`; rerunning it chose the same variant and threshold (0.300773) |
| `Qwen/Qwen3-8B` (CLM encoder) | `b968826d9c46dd6066d109eabc6255188de91218` | used as is |
| CLM reference code | `Contrastive-LM/CLM` @ `bb42c6c` | only for the parity check of the scoring heads |
| `Qwen/Qwen3-1.7B` (chat judge) | not recorded | see `model-hashes/` |
| AURA, `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | not recorded | see `model-hashes/` |

The last four checkpoints were pre-installed on the rented machines and their
revisions were not recorded. `model-hashes/*.sha256` lists the SHA-256 of every
file in those directories (and in the LAYA snapshot), so you can check that
yours match. The directory name `AURA-88ee5506` on that machine is not proof of
a revision.

## Data

- `tools/rfc_bench/judge-quality-zh.json` (tune) and
  `tools/rfc_bench/judge-heldout-zh.json` (held-out): Chinese text cases written
  by the author, 39 each. Five "unfinished utterance" cases per set have no
  `response_needed` label and are not scored, so each set scores 34 cases:
  17 that need a reply and 17 that do not. Prompts and thresholds were chosen on
  the tune set only. Three transcripts appear in both sets ("嗯", "哦",
  "(咳嗽)"). The cases carry conversation history and an
  `assistant_speaking` flag, but the judge stage currently reads only the
  current transcript.
- `audio/`: the four end-to-end clips (synthetic Mandarin from macOS
  `say -v Tingting`, 16 kHz mono) and their SHA-256 sums: `a0` "嗯嗯", `a1`
  "好的", `a3` "今天北京天气怎么样？", `a4` "帮我定一个明天早上七点的闹钟。".

## Configs

Use these frozen files to reproduce the P1–P9 rows of the latency table below.
`configs/compact-32gb/reference-p10-laya-pr-overlay.yaml` is the configuration
of the reference P10 run (judge stage as in the updated LAYA overlay,
`max_num_seqs: 8`).

- `configs/compact-32gb/`: the profile used on one 32 GB GPU (RTX 4080 SUPER);
  all stages in eager mode with small KV caches unless noted.
  - `judge-off.yaml`: `aura_omni`, no judge.
  - `judge-laya-eager.yaml`: `aura_omni_judged` with the LAYA judge in eager
    mode and `max_num_seqs: 1` (the setting of the first report).
  - `judge-laya-graph-capture-sizes.yaml`: the same with CUDA graphs on the
    judge stage and `compilation_config.cudagraph_capture_sizes`.
  - `judge-qwen3.yaml`: the Qwen3-1.7B judge on the same profile. AURA + Qwen3
    did not fit in 32 GB, so this was not run end to end.
- `configs/prototype-84gb/`: the official AURA duplex smoke profile on one
  RTX 6000D (84 GB), used with the earlier prototype (`smoke-*` judge off/on;
  `conc-*` with `max_sessions` and every stage's `max_num_seqs` set to 8).
- `tools/response_judge/make_configs.py` derives deploy files from a base
  profile, but with its defaults it does not produce the frozen compact files
  (different context lengths, memory budgets and no capture sizes). It is
  included for reference only.

## Tools

Standalone judge checks (no main model loaded; all need a GPU):

| Script | What it does |
| --- | --- |
| `laya_vllm_parity.py` | LAYA through the stage's `LayaDecisionModel` vs the `laya` package: logits, argmax, latency |
| `laya_head_parity.py` (+ `laya_vllm_model.py`) | the LAYA scoring head vs the `laya` package on CUDA |
| `laya_cpu_checks.py` | CPU checks of the LAYA prompt layout |
| `laya_prompt_tuning.py` | LAYA prompt variants on the tune / held-out sets |
| `qwen3_prompt_tuning.py`, `qwen3_prompt_equality.py` | Qwen3 prompt variants; the stage's prompt against the earlier measured one |
| `clm_tune_and_prepare.py` | CLM prompt tuning, model directory, parity with the reference maths |
| `laya_standalone_prof.py`, `qwen3_capture_check.py`, `clm_capture_check.py` | latency back to back vs after idle gaps, with and without capture sizes |
| `cpu_wake_probe.py` | how fast a CPU thread runs right after an idle gap (read-only; changes no system setting) |

End to end (AURA duplex, judge off vs on):

| Script | What it does |
| --- | --- |
| `tools/response_judge/timing_patch.py` | copies a source tree and adds `OMNI_TIMING` / `OMNI_HOP` log lines |
| `tools/response_judge/prof_patch.py`, `rj_prof.py` | adds per-request timestamps inside each stage's EngineCore and the orchestrator (`RJ_PROF` lines) |
| `tools/response_judge/run_arm.py` | starts the server with one config, runs a client, stops the server, collects logs |
| `tools/rfc_bench/aura_judge_onoff.py`, `aura_concurrency_real.py`, `aura_probe_common.py` | Realtime duplex clients (fresh session per turn; concurrency). `listen_sources` in the output JSONL comes from `response.listen` |
| `tools/rfc_bench/analyze_onoff.py` | added latency, judge on vs off |
| `tools/response_judge/rj_prof_analysis.py`, `hop_analysis.py` | per-hop breakdown of the judge stage |
| `tools/rfc_bench/aura_final_gpu_suite.py` | only its process helpers are used, by `run_arm.py`; the historical prototype suite it contains cannot run from this package |
| `tools/rfc_bench/apply_silent_turn_fix_prototype.py` | imported by the file above; a prototype fix for the same-session silent-turn issue, not part of #8316 |

Typical end-to-end pair (absolute paths; `ORCH=1` for the optimized pair,
`ORCH=0` for the eager baseline pair, the same in both arms):

```bash
S=/abs/source-tree; W=/abs/work; C=/abs/response-judge-repro; ORCH=1
python $C/tools/response_judge/timing_patch.py $S $W/code-timing
python $C/tools/response_judge/prof_patch.py $W/code-timing $W/code-prof
export VLLM_OMNI_EVENT_DRIVEN_ORCH=$ORCH
RJ_PROF_STAGES=1 python $C/tools/response_judge/run_arm.py --code $W/code-prof \
    --models /path/to/models --audio-dir $C/audio --client onoff \
    --config $C/configs/compact-32gb/judge-off.yaml --name off --mode off --out $W/off
RJ_PROF_STAGES=1 python $C/tools/response_judge/run_arm.py --code $W/code-prof \
    --models /path/to/models --audio-dir $C/audio --client onoff \
    --config $C/configs/compact-32gb/judge-laya-graph-capture-sizes.yaml --name on-laya --mode on --out $W/on
cp $W/off/onoff-off.jsonl $W/off/server-off.log $W/on/
python $C/tools/rfc_bench/analyze_onoff.py --directory $W/on
python $C/tools/response_judge/rj_prof_analysis.py $W/on/on-laya.server.log
```

For the eager baseline, set `ORCH=0` and use `judge-laya-eager.yaml`. Each
arm opens one fresh duplex session per turn, runs 2 warmups per clip, then 8
runs of each question clip and 3 of each backchannel clip.

## Numbers these produced

One RTX 4080 SUPER 32 GB, compact profile, LAYA judge, one session at a time.

- **Routing** uses all 22 measured turns per arm: 16/16 questions answered and
  6/6 backchannels ended at the judge in every run.
- **Added latency** = p50(on) − p50(off) (and the same for p95) of "ASR output
  forwarded → AURA submitted", over the **16 question turns per arm**; the 6
  rejected turns have no AURA submit. It is a difference of group quantiles,
  not time to first audio.
- **Judge stage** = submit → output back at the orchestrator, over all
  **22 measured judge requests**.

| Setting | Added latency p50 / p95 | Judge stage p50 |
| --- | --- | --- |
| LAYA eager, 1 ms orchestrator poll (two runs) | 43.0 / 45.9 ms; 44.5 / 60.3 ms | 41.2 ms; 40.6 ms |
| LAYA capture sizes + `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` in both arms (two runs) | 26.1 / 35.9 ms; 28.5 / 29.5 ms | 24.7 ms; 26.5 ms |
| Reference: the PR update's LAYA overlay (`max_num_seqs: 8`), earlier revision of the update (P10/P11) | 27.2 / 30.8 ms | 20.9 ms |

In P10 the client read `response.metadata.vllm_omni.listen_source ==
"response_judge"` on all 6 rejected turns.

Standalone (in-process), the same host, 20 held-out cases:

| Judge | Back to back | After a 2 s idle gap |
| --- | --- | --- |
| LAYA, CUDA graphs, `execute_model` | 2.8 ms | 11.3 ms |
| Qwen3-1.7B, default capture sizes / up to 256 | 6.4 / 6.3 ms | 18.2 / 18.3 ms |
| CLM, default capture sizes / up to 256 | 28.2 / 27.6 ms | 39.3 / 31.3 ms |

Larger capture sizes gave no gain for Qwen3 and did not change any Qwen3 or
CLM decision. The host used the `powersave` CPU governor; the idle-gap
slowdown is consistent with CPU frequency scaling, but no other governor was
tested.
