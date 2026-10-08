# H200 end-to-end arms

Our half of the split agreed in
[vllm-project/vllm-omni#8211](https://github.com/vllm-project/vllm-omni/issues/8211):
the judge stage itself is PR
[#8316](https://github.com/vllm-project/vllm-omni/pull/8316), and this directory
is the end-to-end benchmark harness that runs judge-off against judge-on and
reports what the judge costs and what it saves.

Results from this harness are in
[#8211 comment](https://github.com/vllm-project/vllm-omni/issues/8211#issuecomment-6057006709).

Everything here sits on top of the package's own tools. The clients, configs,
patches and clips are the package's; this adds the arm driver and the
cross-concurrency report, and makes `run_arm.py`'s concurrency ladder settable
(see `CHANGES.md` in this directory).

## Why a separate report

ReviewBot asked #8316 for the value of the judge, which a latency table cannot
answer: a latency delta is measured on the turns the judge *lets through*, and
says nothing about the turns it *suppresses*. Those are the turns where the
main model and the TTS stage do no work at all. So the question needs three
numbers, not one:

- **cost** — what an answered question pays, as TTFT and TTFA, judge-on minus
  judge-off;
- **saving** — how much main-model and TTS work never happened because the turn
  ended at the judge;
- **safety** — how often the judge suppressed a turn that wanted an answer.

`aura_probe_common.measure()` already records TTFT/TTFA per turn and
`aura_concurrency_real.summarize()` already counts completions and rejections.
What the package has no layer for is the join: judge-off against judge-on,
across concurrency levels. That join is `judge_bench_report.py`.

Two things to know when quoting the package's own output:

1. Its headline "added latency" is a difference of group quantiles, not time to
   first audio. The README says so; it should not be presented as a
   user-visible latency.
2. `aura_concurrency_real.summarize()` reads `turn_ms`, but `measure()` emits
   `terminal_ms`, so the package's per-concurrency latency summary is always
   empty. `judge_bench_report.py` reads `terminal_ms` off the per-turn rows.

## Files

| File | What it does |
| --- | --- |
| `run_arms.sh` | runs the judge-off and judge-on arms back to back, one `smoke` or `conc` pair, recording the environment per arm |
| `judge_bench_report.py` | joins the two arms: routing, latency pooled and per concurrency level, useful throughput, GPU peak, and checks |
| `check_anchors.py` | reports every `timing_patch` / `prof_patch` anchor against a source tree without applying, so a moved anchor is caught before an arm runs |
| `fetch_models.sh` | downloads all seven checkpoints at the pinned revisions |

## Running

```bash
MODELS=/abs/models ./h200/fetch_models.sh

# patch the PR head, then confirm every anchor still matches it
python3 h200/check_anchors.py /abs/source-tree
python3 tools/response_judge/timing_patch.py /abs/source-tree /abs/work/code-timing
python3 tools/response_judge/prof_patch.py /abs/work/code-timing /abs/work/code-prof

CODE=/abs/work/code-prof MODELS=/abs/models ./h200/run_arms.sh smoke   # liveness
CODE=/abs/work/code-prof MODELS=/abs/models ./h200/run_arms.sh conc    # the numbers
python3 h200/judge_bench_report.py h200/out/conc-<stamp>
```

`run_arms.sh` refuses to start unless the target GPU is idle and the port is
free, and it never kills anything it did not start. Each arm's directory records
the config it ran, the environment, both arms' JSONL, and three `nvidia-smi`
snapshots.

Defaults worth knowing: `GPU=4`, `PORT=8099`, `CONC_USERS="1 2 4"`,
`CONC_ROUNDS=5`, `STARTUP_TIMEOUT=900`. The concurrency ladder and rounds are
pinned in the script rather than left to `run_arm.py`'s defaults, so every
`conc` arm has the same shape and the numbers stay comparable. Five rounds is
5 real questions and 5 backchannels per user, the minimum that gives a p50
worth quoting at one user.

Both arms use the package's frozen `../configs/prototype-84gb` files. Those
carry the `/path/to/models` placeholder, so `run_arms.sh` stages a copy with it
substituted for the container's mount point and leaves the checked-in files
untouched. `run_arm.py` copies the staged file into the arm directory, so each
arm still records exactly what it ran.

## Reading the report

- **Routing** — questions answered, false blocks, backchannels suppressed
  against let through, and the `listen_sources` behind each suppression.
- **Latency** — TTFT and TTFA of answered questions, pooled and per concurrency
  level. Quote the per-level rows: pooling mixes populations, and the judge can
  be faster at one user and slower at four while the pooled delta describes
  neither.
- **Useful throughput** — completed questions per minute per level.
- **GPU** — peak memory and utilization over the arm's snapshots.
- **Checks** — flags a false block, a judge that suppressed nothing (so is not
  wired in), an invalid wave, and judge-off turns AURA itself silenced.

Judge-off suppressing backchannels is expected rather than a bug, because AURA
emits `<|silent|>` on its own. That is why `listen_source` matters: only
`response_judge` is the judge doing the work.

## Environment used

| | |
| --- | --- |
| Host | 8× NVIDIA H200 143 GB, driver 580.65.06 |
| GPU | one, serial pipeline (every stage in the `prototype-84gb` configs pins `devices: '0'`) |
| Interpreter | Python 3.12.13, torch 2.13.0+cu130, vLLM 0.30.0, CUDA 13.0 |
| Source under test | #8316 head `4e860735dcf31b0f90c29ebff05dc696a3f07630` |

Two notes for anyone reproducing this:

**Python ≥ 3.11 is required, and the prerequisites do not say so.** On 3.10 the
PR head does not import: `DuplexCommand` is declared `@dataclass(slots=True)`,
and 3.10 repeats inherited fields in `__slots__`, so every concrete subclass
dies with `TypeError: multiple bases have instance lay-out conflict`. Upstream
fixed exactly this in `aa41a3ac` (#7985), which is **not** an ancestor of
`4e860735`, so the PR branch does not carry it. A 3.10 base image hits this
before the server starts.

**The CPU governor could not be pinned on this host.**
`/sys/devices/system/cpu/cpu0/cpufreq/` does not exist and we are unprivileged,
so it is recorded as `UNAVAILABLE` in each arm's `environment.txt` rather than
set. Since it cannot be controlled, every arm keeps the same shape, with warmup
immediately before the measured wave, so the comparison stays like for like.

## Not covered here

- LAYA and CLM judges. Only the Qwen3-1.7B judge was run.
- Rejection-then-another-turn, cancellation, and cross-session isolation.
- Per-block `listen_source` accounting on `prototype-84gb`. The package reports
  it for the `compact-32gb` P10 run, so it needs checking on this profile
  before being quoted.
- Hop analysis on `conc` arms. The `conc` client emits no `--client-jsonl`, so
  `hop_analysis.py` falls back to dropping the first N requests in log order and
  labels its own output "do not quote". Those numbers are left out rather than
  reported.
