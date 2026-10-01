# Multi-turn check

Data, scripts and raw results for the multi-turn check of the response judge stage
([vllm-project/vllm-omni#8316](https://github.com/vllm-project/vllm-omni/pull/8316)).
None of this is part of the PR.

## Data

I wrote Chinese multi-turn dialogues that mix turns that need a reply (questions,
requests) with turns that do not (backchannels, fillers, side talk to someone
nearby). `tools/multiturn/build_dialogues.py` builds them deterministically from
`data/pools.json`.

| Split | Dialogues | Turns | Need a reply | Need no reply |
| --- | --- | --- | --- | --- |
| tune | 30 | 210 | 125 (question 74, request 51) | 85 (backchannel 51, side talk 19, filler 15) |
| held-out | 15 | 103 | 63 (question 44, request 19) | 40 (backchannel 28, side talk 9, filler 3) |

- The two splits use separate sentence pools; the only sentence in both is the
  filler "呃……".
- Audio (`audio-qwen3tts/`, 16 kHz mono) was synthesized with Qwen3-TTS
  CustomVoice (`tools/multiturn/synth_qwen3_tts.py`). tune uses the speakers
  vivian / uncle_fu / eric, held-out uses serena / dylan. Backchannels, fillers
  and side talk get a short delivery instruction. Two clips that came out broken
  (one turned "对。" into 23.7 s of babble) were regenerated once; the manifest
  marks them with `regenerated`.
- The `voice` field in `data/dialogues-*.jsonl` names macOS `say` voices, used
  by `synth_say.py` for a first pass. That audio is not included; the ASR often
  misheard its short backchannels, so the numbers below use the Qwen3-TTS audio.
- ASR: the pipeline's Qwen3-ASR-1.7B (`tools/multiturn/asr_transcribe.py`); 266
  of 313 transcripts match the text apart from punctuation.

## How prompts and thresholds were chosen

Prompt variants and thresholds were chosen on the tune split only
(`tools/multiturn/judge_text_eval.py` runs the existing tuners in
`tools/response_judge/`). No held-out sentence appears in any prompt. All
numbers below are on the held-out split.

## Results

### Judges on their own (held-out, Qwen3-TTS audio → Qwen3-ASR → judge)

"Blocked" means a turn that needs a reply was rejected; "filtered" means a turn
that needs no reply was rejected.

| Judge | Prompt | Blocked | Filtered |
| --- | --- | --- | --- |
| Qwen3-1.7B | stage default | 0/63 | 28/40 |
| Qwen3-1.7B | `sidetalk-fewshot` | 0/63 | 32/40 |
| LAYA | previous example config in #8316 (threshold 0.633, before `932fd24`) | 35/63 | 34/40 |
| LAYA | `en-choice-4way-heard`, threshold 0.0173 (the example config since `932fd24`) | 0/63 | 37/40 |
| CLM | example prompt in #8316, threshold chosen on tune (0.437) | 6/63 | 34/40 |

- The previous LAYA example config was chosen on the earlier 34-case set
  (`tools/rfc_bench/judge-*-zh.json`) and did not carry over to this set, so
  #8316 now uses the config chosen here.
- These dialogues contain no short answers to the assistant ("七点") and no
  interruptions ("等一下"). On the earlier 34-case held-out set, which has
  them, the new config blocks 6/17 reply-needed cases (the previous one
  13/17), mostly those two kinds. Describing them in the prompt made both
  sets worse; they need the assistant's last turn, not a better prompt.
- Results on the reference text and on `say` audio are in
  `results/judge-text/` (`t1`–`t3`); `t4` is the table above.
- The LAYA config through the stage's own code path (`_laya_prompt` →
  `LayaDecisionModel` → `_option_scores_reject`) gives the same decision as the
  `laya` package on 313/313 transcripts (`results/parity/`).

### End to end (held-out, AURA, judge off vs on)

32 GB compact profile (RTX 4080 SUPER), `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` in both
arms, one fresh session per turn, one run per arm. The "on" arm uses the LAYA
config above; otherwise the deploy files match the P10 / P11 profiles of the
main package (`results/e2e/*.deploy.yaml`).

| | judge off | judge on (LAYA) |
| --- | --- | --- |
| **Turns that need a reply** (questions, requests; 63): replied | 63 | 63 |
| **Turns that need no reply** (40): still got a reply | 40 | 3 |
| 　backchannels (28) | 28 | 0 |
| 　fillers (3) | 3 | 0 |
| 　side talk (9) | 9 | 3 |
| Audio in those unneeded replies (24 kHz output) | ~109 s | ~9 s |

- No turn failed in either arm. Every rejected turn ended with
  `response.listen` whose `listen_source` is `response_judge` (37/37).
- The 3 misses are the same sentence, side talk phrased as a question
  ("你作业写完了没有？").
- Latency: the two arms are separate server starts with different replies, so
  this run cannot resolve tens of milliseconds. Median first text was 640 ms
  (off) and 616 ms (on). The added latency of the judge is the dedicated
  measurement in the main package (~27 ms, p50); re-measured with this
  config it is 27.7 ms against 27.4 ms for the previous prompt in the same
  run. First audio (~2.3–2.7 s) reflects the compact eager profile, not a
  production setup.
- A dialogue in one session needs the same-session recovery fix, so this run
  uses one session per turn. The judge reads only the current transcript, so
  its decisions do not depend on that.

### Stability of the judge on its own

`tools/multiturn/laya_stability.py`, LAYA config above, the judge stage's engine
settings (pooling, fp32, `max_num_seqs` 8, capture sizes 16–256), prompts and
decisions from the stage's own functions.

- Batching (`results/stability/flip.json`): the 313 transcripts scored one at a
  time and in batches of 2, 4 and 8. No decision flipped; P(reply) differed by
  at most 0.0014. A second one-at-a-time pass was identical.
- Repeated calls (`results/stability/soak.json`): 12 minutes, bursts of 20 calls
  with random 0.2–1.5 s pauses, 14,060 calls. Every score matched the first pass
  exactly. Per-minute latency stayed flat (median of an `llm.encode` call: about
  8.2 ms back to back, about 17.7 ms for the first call after a pause; p99 at
  most 26 ms). CUDA allocated/reserved memory (1265.2 / 1366.0 MB) and peak RSS
  did not change.

## Reproduce

Paths are relative to this package. Replace `/path/to/models` in the deploy
files with your model directories; the prerequisites are those of the main
README.

```bash
# text-level judge check (GPU, judge model only)
python tools/multiturn/judge_text_eval.py convert --dialogues multiturn-zh/data/dialogues-heldout.jsonl \
    --asr multiturn-zh/results/asr/asr-qwen3tts-v2.jsonl --out heldout.json
python tools/multiturn/judge_text_eval.py run --variants tools/multiturn/variants/laya-best.json \
    tools/response_judge/laya_prompt_tuning.py -- --model-dir <laya snapshot> --laya-pkg <dir> \
    --tune tune.json --heldout heldout.json --out <new dir>

# end to end, one arm (starts the server, runs the client, stops the server)
VLLM_OMNI_EVENT_DRIVEN_ORCH=1 python tools/response_judge/run_arm.py --code <vllm-omni tree> \
    --models /path/to/models --audio-dir multiturn-zh/audio-qwen3tts --client dialogue \
    --dialogue-args "--split heldout --warmup 2 --gap-s 1.0" --client-timeout 3600 \
    --config multiturn-zh/results/e2e/on.deploy.yaml --name on --mode on --out <new dir>
python tools/multiturn/dialogue_client.py --summarize <off dir>/off.jsonl <on dir>/on.jsonl
```

Source under test: the tree of the #8316 head `ecf9cb08e87eb6e363c491ca94e265d241968b83`,
without measurement patches, with the LAYA config passed in the deploy file.
The checks in `results/config-update/` ran on that tree plus the changes of
`932fd24`, with the measurement patches of the main package.

## Files

- `data/`: sentence pools, dialogues (tune / held-out), counts.
- `audio-qwen3tts/`: the synthesized turns and `manifest.jsonl` (text, label,
  speaker, duration, SHA-256).
- `results/cases/`: judge inputs (reference text, `say` + ASR, Qwen3-TTS + ASR).
- `results/asr/`: ASR transcripts.
- `results/judge-text/`: per-run `summary.json` and `raw.jsonl` of the tuners.
- `results/parity/`, `results/stability/`: see above.
- `results/e2e/`: per-turn rows of both arms, the summary and the deploy files.
- `results/config-update/`: checks for updating the LAYA example config in
  #8316. Added latency with the previous and the new prompt, from one run that
  shares its off arm (`analyze_onoff.py` output and the judge-stage breakdown,
  same method as P10). The new prompt's quality on the earlier 34-case sets.
  Prompt variants that describe short answers and interruptions.
