#!/usr/bin/env bash
# Runs ON the GPU droplet, from a copy of this repository at /root/unalloc.
#
#   bash remote.sh setup          # uv, vLLM, the repo, and the model weights
#   bash remote.sh serve          # start vLLM + nvidia-smi logging, wait until healthy
#   bash remote.sh bench [args]   # run bench.py, results in $OUT
#   bash remote.sh stop           # stop vLLM and the GPU logger
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VENV=/opt/unalloc-venv
REPO=/root/unalloc
OUT="${OUT:-/root/results}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"

step="${1:-}"
shift || true

case "$step" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --allow-existing -p 3.12 "$VENV"
    # Weights download in the background while the (large) vLLM install runs.
    ( uvx --from huggingface_hub hf download "$MODEL" > "$OUT/download.log" 2>&1 \
        && echo "download complete" >> "$OUT/download.log" ) &
    uv pip install -p "$VENV/bin/python" vllm
    uv pip install -p "$VENV/bin/python" -e "$REPO"
    wait
    tail -1 "$OUT/download.log"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    ;;
  serve)
    nohup nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits -l 1 > "$OUT/nvidia_smi.csv" 2>&1 &
    echo $! > /root/nvsmi.pid
    # The default FlashInfer sampler JIT-compiles a kernel during warm-up; on
    # the DigitalOcean AI/ML image (CUDA 13.1 toolkit, cu130 wheels) that build
    # failed. The PyTorch sampler is equivalent for greedy decoding.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    nohup "$VENV/bin/vllm" serve "$MODEL" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --enable-prefix-caching \
      --enable-prompt-tokens-details \
      --port 8000 > "$OUT/vllm.log" 2>&1 &
    echo $! > /root/vllm.pid
    for _ in $(seq 1 240); do
      if curl -sf localhost:8000/health >/dev/null; then echo "vLLM healthy"; exit 0; fi
      if ! kill -0 "$(cat /root/vllm.pid)" 2>/dev/null; then tail -40 "$OUT/vllm.log"; exit 1; fi
      sleep 2
    done
    echo "vLLM did not become healthy"; tail -40 "$OUT/vllm.log"; exit 1
    ;;
  bench)
    cd "$REPO"
    "$VENV/bin/python" -m case_studies.gpu_validation.bench \
      --base http://localhost:8000 --model "$MODEL" --out "$OUT" "$@"
    ;;
  stop)
    kill "$(cat /root/vllm.pid)" "$(cat /root/nvsmi.pid)" 2>/dev/null || true
    ;;
  *)
    echo "usage: remote.sh setup|serve|bench|stop" >&2
    exit 2
    ;;
esac
