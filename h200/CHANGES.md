# Changes from the H200 benchmark work

These are the edits the H200 benchmark needed in order to run. Each is kept as
small as it can be, and each preserves the original behaviour by default.

## `tools/response_judge/run_arm.py` — the `conc` concurrency ladder

The `conc` branch hard-coded the client's arguments:

```python
extra = ["--mode", a.mode, "--users", "4", "8", "--rounds", "3", ...]
```

Two problems. The levels were not settable, so an arm could not run 1/2/4
sessions; and `--rounds` was pinned at 3 with no way to change it, so the
measured wave per user was fixed at 3 questions and 3 backchannels. Replaced
with:

```python
p.add_argument("--conc-users", nargs="+", default=["4", "8"],
               help="conc arms only: space-separated concurrency levels")
p.add_argument("--rounds", type=int, default=3,
               help="conc arms only: measured rounds per user (turns per user = 2x this)")
...
extra = ["--mode", a.mode, "--users", *a.conc_users, "--rounds", str(a.rounds), ...]
```

Both defaults reproduce the previous behaviour exactly. `--conc-users` takes
`nargs="+"` rather than a space-joined string so `--conc-users 1 2 4` parses the
way it reads.

## `tools/response_judge/timing_patch.py` — one anchor, both upstream forms

`4e860735` (the current #8316 head) reformatted the `response_judge_rejected`
argument in `vllm_omni/engine/duplex_orchestrator.py` from one line to three, so
the patch's original one-line anchor no longer matched. Narrowing the anchor to
the new form alone would break the other direction: `4c1da861` is what the
repro README's profiling walkthrough uses.

```python
# 4c1da861 (what the patch was written against)
response_judge_rejected=finished and self._is_response_judge_stage(stage_id) and judge_rejects(output),

# 4e860735
response_judge_rejected=(
    finished and self._is_response_judge_stage(stage_id) and judge_rejects(output, req_state.streaming)
),
```

The patch now accepts **both** forms. The anchor for that edit is a tuple of
candidates tried in order, and the edit is marked `APPEND`, so the injected
`OMNI_HOP judge_rejected` line is added after whichever form matched instead of
the patch having to restate it:

```python
(
    (THREE_LINE_FORM, ONE_LINE_FORM),   # tried in order
    (APPEND, '        if context.response_judge_rejected:\n...'),
)
```

Appending is what makes one patch serve several anchor forms; a replacing patch
may not carry alternatives, because the replacement would depend on which form
matched, and `main()` rejects that combination rather than guessing. An anchor
that appears more than once is never used, so the exactly-once guarantee the
patch relied on is kept per candidate.

Semantics are unchanged. Both trees verify: `h200/check_anchors.py` reports 0
mismatches against `4c1da861` and `4e860735` for both patches, the patched
output compiles on each, and against `4e860735` all four patched files are
byte-identical to the tree the published arms ran on.

## Not a code change: the startup timeout

`run_arm.py --startup-timeout` defaults to 420 s. On the benchmark host that was
too tight: engine init took 373.8 s and the duplex readiness warmup needs
another ~47 s, so the arm was aborted mid-warmup while the server was coming up
fine. `h200/run_arms.sh` passes `--startup-timeout 900` rather than changing the
default, since 420 s is a reasonable default on a host where nothing else is
compiling.

## `h200/` — the harness itself

Not part of the package's own tooling; a layer on top of it.

- `run_arms.sh` runs the off/on pair and records the environment per arm.
- `judge_bench_report.py` joins the two arms across concurrency levels. The
  package has no such layer: `aura_probe_common.measure()` records TTFT/TTFA per
  turn and `aura_concurrency_real.summarize()` counts completions, but nothing
  compares the arms.
- `check_anchors.py` reports every patch anchor against a tree without applying.
  It resolves alternative anchor forms through `timing_patch.resolve_anchor`
  itself, so the check cannot disagree with what applying would do.
- `fetch_models.sh` downloads the seven checkpoints at pinned revisions. It
  never deletes an existing destination: a directory without `.complete` is
  skipped with a message rather than removed, since it may still be usable or
  resumable, and tens of GB over a mirror is expensive to redo. Each download
  goes to a sibling `<dest>.tmp-$$` on the same filesystem and is moved into
  place only on success, so an interrupted or failed run leaves whatever was
  already there untouched. `FORCE=1` re-downloads, and even then the existing
  directory is replaced only after the new one has finished.

### Attribution in `judge_bench_report.py`

A turn can end without a reply for reasons that are not the judge's doing, so
the report separates them by `"response_judge" in listen_sources`:

- `false blocks` counts only questions the judge silenced. Questions that failed
  or timed out are counted as `broken`, and a question silenced without judge
  attribution is reported separately — previously all three were one number.
- Backchannel suppressions are split into `by judge` and `by AURA`, the latter
  being AURA's own `<|silent|>`, which judge-off emits too. Suppression latency
  is measured over judge-attributed turns only.
- A missing arm file is flagged in the checks instead of reading as an arm with
  no turns, which used to print `clean`.

On the published `conc` run these splits change no number: 35/35 answered in
both arms, 0 false blocks, 0 broken, and all 21 suppressions attributed to
`response_judge` with 0 to AURA.

It also reads the two client output shapes: the `conc` client
(`aura_concurrency_real.py`) labels every row with `kind`, while the `onoff`
client (`aura_judge_onoff.py`) does not and is identified by clip name. Both are
handled, so no arm needs a post-processing step.

## Known quirks worked around in `h200/`, not patched here

- `hop_analysis.py` without `--client-jsonl` falls back to dropping the first N
  requests in log order and labels its output "do not quote". The `conc` client
  emits no `--client-jsonl`, so `conc` hop numbers are not reported.
- The configs carry a `/path/to/models` placeholder. `run_arms.sh` stages a copy
  with it substituted rather than shipping a duplicate set of configs that would
  drift.

## Not modified

`aura_probe_common.py`, `aura_concurrency_real.py`, `aura_judge_onoff.py`,
`analyze_onoff.py`, `hop_analysis.py`, `aura_final_gpu_suite.py`, the configs,
the audio, and the model-hash files are untouched.
