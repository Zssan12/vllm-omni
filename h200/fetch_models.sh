#!/usr/bin/env bash
# Download every checkpoint the response-judge repro needs, pinned to revision.
# Runs on the host inside the py3.12 image (it ships huggingface_hub).
#
#   MODELS=/models ./h200/fetch_models.sh
#   MODELS=/models FORCE=1 ./h200/fetch_models.sh   # re-download even if present
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
# The mirror does not proxy HF's Xet CAS backend (cas-server.xethub.hf.co), which
# answers 401 there. Force the plain HTTP/LFS path instead.
export HF_HUB_DISABLE_XET=1

MODELS=${MODELS:-/models}
FORCE=${FORCE:-}
mkdir -p "$MODELS"

# Never delete an existing destination: these are tens of GB over a mirror, and a
# directory without `.complete` may still be a usable or resumable download. Fetch
# into a sibling temp dir (same filesystem, so the move is atomic) and only put it
# in place once the download has succeeded.
fetch() {  # fetch <repo> <sha> <dest-name>
  local repo="$1" sha="$2" dest="$MODELS/$3"
  if [ -f "$dest/.complete" ]; then echo "== skip $dest (already complete)"; return 0; fi
  if [ -e "$dest" ] && [ -z "$FORCE" ]; then
    echo "== skip $dest (exists without .complete; verify it, or re-run with FORCE=1)"
    return 0
  fi
  local tmp="$dest.tmp-$$"
  rm -rf "$tmp"
  echo "== $repo@${sha:0:8} -> $dest"
  if hf download "$repo" --revision "$sha" --local-dir "$tmp"; then
    touch "$tmp/.complete"
    # Only now is the existing destination replaced, and only under FORCE.
    if [ -e "$dest" ]; then rm -rf "$dest"; fi
    mv "$tmp" "$dest"
  else
    echo "!! FAILED $repo -- leaving $dest untouched, continuing with the rest"
    rm -rf "$tmp"
    return 1
  fi
}

# Pipeline checkpoints, exact directory names the repro configs expect.
fetch aurateam/AURA                            88ee550629fcb4d84428cdbfb346d07f01ea6e03 AURA-88ee5506
fetch Qwen/Qwen3-ASR-1.7B                      7278e1e70fe206f11671096ffdd38061171dd6e5 Qwen3-ASR-1.7B
fetch Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice     0c0e3051f131929182e2c023b9537f8b1c68adfe Qwen3-TTS-12Hz-1.7B-CustomVoice
fetch Qwen/Qwen3-1.7B                          70d244cc86ccca08cf5af4e1e306ecf908b1ad5e Qwen3-1.7B

# Judge candidates: LAYA snapshot (turned into a vLLM-loadable dir later) and the CLM pair.
# The LAYA revision is the one the repro README pins, NOT the repo's live head: the
# head's tokenizer/tokenizer_config.json does not match
# model-hashes/rfc-laya-multilingual-e4e9ddf.sha256.
fetch convaiinnovations/laya-multilingual      e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 laya-snapshot
fetch Contrastive-LM/CLM-v0.1-8B               e939398d4556fcd9400c76fa8c5a513202f42b0a CLM-v0.1-8B
fetch Qwen/Qwen3-8B                            b968826d9c46dd6066d109eabc6255188de91218 Qwen3-8B

echo
echo "== done. sizes:"
du -sh "$MODELS"/* 2>/dev/null
