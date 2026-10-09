#!/usr/bin/env bash
# H200 end-to-end arms for the response-judge stage.
#
#   CODE=<patched tree> MODELS=<model dir> ./h200/run_arms.sh smoke
#   CODE=<patched tree> MODELS=<model dir> ./h200/run_arms.sh conc [extra...]
#
# smoke = judge off/on, 4 clips, one session: liveness and clip-by-clip routing.
# conc  = judge off/on at 1/2/4 sessions: the routing, latency and throughput numbers.
#
# Everything a run produces lands in h200/out/<arm>-<stamp>/; nothing is overwritten.
# Paths are resolved from this script's own location, so the package runs from
# wherever it is checked out.
#
#   CODE       tree of the PR head with timing_patch.py and prof_patch.py applied (required)
#   MODELS     directory holding the four checkpoints (required; see h200/fetch_models.sh)
#   IMG        image carrying the Python 3.12 venv        (default jb-vllm-omni:py312)
#   PY         interpreter inside IMG                     (default /opt/issue8211-py312/bin/python)
#   SERVE_BIN  vllm-omni launcher inside IMG              (default /opt/issue8211-py312/bin/vllm-omni)
#   GPU        device index                               (default 4)
#   PORT       server port                                (default 8099)
set -euo pipefail

H200="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$H200")"

ARM="${1:-smoke}"; shift || true
EXTRA="${*:-}"

# The image used here carries a Python 3.12 venv because neither interpreter in the
# upstream image can run the PR head as shipped: the image's own vllm_omni is 0.28.0
# and cannot import against vLLM 0.30.0 (vllm.entrypoints.openai.cli_args is gone),
# and the PR head does not import under Python 3.10 at all (DuplexCommand is declared
# @dataclass(slots=True); 3.10 repeats inherited fields in __slots__, so every concrete
# subclass dies with "multiple bases have instance lay-out conflict"). Upstream fixed
# that in aa41a3ac (#7985), which is not an ancestor of 4e860735.
CODE=${CODE:?set CODE to the timing-patched tree of the PR head}
MODELS=${MODELS:?set MODELS to the directory holding the four checkpoints}
IMG=${IMG:-jb-vllm-omni:py312}
PY=${PY:-/opt/issue8211-py312/bin/python}
SERVE_BIN=${SERVE_BIN:-/opt/issue8211-py312/bin/vllm-omni}
CFGS=$PKG/configs
OUT=$H200/out
PORT=${PORT:-8099}
GPU=${GPU:-4}

# Fixed and recorded; both move judge latency. The CPU governor is not exposed on
# the benchmark host at all -- /sys/devices/system/cpu/cpu0/cpufreq/ does not exist
# and we are not root -- so it is recorded as uncontrolled rather than pinned.
EVENT_DRIVEN_ORCH=1

# Engine init measured at 373.8 s on the H200 host under load (loadavg ~8.7, other
# tenants compiling on neighbouring GPUs), and the duplex readiness warmup needs
# another ~47 s on top of that. run_arm.py's default of 420 s expires mid-warmup,
# which aborts the arm even though the server came up fine. 900 s clears it with room.
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}

mkdir -p "$OUT"
for m in AURA-88ee5506 Qwen3-1.7B Qwen3-ASR-1.7B Qwen3-TTS-12Hz-1.7B-CustomVoice; do
  [[ -d $MODELS/$m ]] || { echo "missing model: $m -- run h200/fetch_models.sh first"; exit 1; }
done

# Refuse to start unless GPU $GPU is idle and the port is free. Nothing is killed.
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$GPU | tr -d ' ')
if (( busy > 200 )); then echo "GPU $GPU busy (${busy} MiB in use); set GPU=<idle index>"; exit 1; fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then echo "port $PORT already listening"; exit 1; fi

case "$ARM" in
  smoke) cfg_off=smoke-off.yaml
         cfg_on=smoke-on.yaml
         client=onoff ;;

  # The concurrency ladder and rounds are pinned here, not left to run_arm.py's
  # defaults (4 and 8 users, 3 rounds), so every conc arm is the same shape and the
  # numbers stay comparable. 5 rounds = 5 real questions + 5 backchannels per user,
  # which is the minimum that gives a p50 worth quoting at one user.
  conc)  cfg_off=conc-off.yaml
         cfg_on=conc-on.yaml
         client=conc
         EXTRA="--conc-users ${CONC_USERS:-1 2 4} --rounds ${CONC_ROUNDS:-5} $EXTRA" ;;
  *) echo "usage: $0 {smoke|conc} [extra run_arm.py args]"; exit 2 ;;
esac

stamp=$(date +%Y%m%d-%H%M%S)
base=$OUT/$ARM-$stamp
mkdir -p "$base"

# The package's configs carry the placeholder /path/to/models. The host model
# directory is bind-mounted at /models inside the container, so that is what the
# placeholder becomes. Staging it here keeps the checked-in configs unmodified,
# and every arm still records the exact config it ran (run_arm.py copies it into
# the arm directory).
stage=/tmp/jb-cfg-$stamp
mkdir -p "$stage"
for cfg in "$cfg_off" "$cfg_on"; do
  sed "s#/path/to/models#/models#g" "$CFGS/prototype-84gb/$cfg" > "$stage/$cfg"
done
CFGS=$stage

{
  echo "arm=$ARM  started=$(date -Is)"
  echo "code=$CODE"
  echo "image=$IMG"
  echo "gpu=$GPU  $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader --id=$GPU)"
  echo "VLLM_OMNI_EVENT_DRIVEN_ORCH=$EVENT_DRIVEN_ORCH"
  gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  if [[ -r $gov ]]; then echo "cpu_governor=$(cat $gov)"; else echo "cpu_governor=UNAVAILABLE (no cpufreq sysfs; unprivileged) -- uncontrolled"; fi
  echo "nproc=$(nproc)  loadavg=$(cat /proc/loadavg)"
  echo "extra_args=$EXTRA"
} > "$base/environment.txt"

run_one() {  # run_one <mode> <config basename> <extra>
  local mode="$1" cfg="$2" extra="$3"
  echo "--- $ARM / $mode"
  docker run --rm --gpus "device=$GPU" --name "jb-$ARM-$mode" \
    --shm-size 32g --ipc host \
    -v "$CODE:/code:ro" -v "$MODELS:/models:ro" -v "$PKG:/repro:ro" -v "$CFGS:/configs:ro" -v "$base:/out" \
    -e PYTHONPATH=/code \
    -e VLLM_OMNI_EVENT_DRIVEN_ORCH="$EVENT_DRIVEN_ORCH" \
    -e RJ_PROF_STAGES=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -w /code --entrypoint "$PY" "$IMG" \
      /repro/tools/response_judge/run_arm.py \
        --code /code --config "/configs/$cfg" \
        --name "$mode" --mode "$mode" --client "$client" \
        --models /models --audio-dir /repro/audio --out /out --port "$PORT" \
        --serve-bin "$SERVE_BIN" --startup-timeout "$STARTUP_TIMEOUT" $extra 2>&1 | tee "$base/$mode.run.log"
}

run_one off "$cfg_off" "$EXTRA"
run_one on  "$cfg_on"  "$EXTRA"

echo
echo "arm base: $base"
echo "report:   python3 $H200/judge_bench_report.py $base"
