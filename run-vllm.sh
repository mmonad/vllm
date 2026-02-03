#!/bin/bash
# vLLM ROCm launch script for AMD Radeon AI PRO R9700 (gfx1201/RDNA4)
# Optimized based on: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html

# Configuration - edit these as needed
MODEL="${MODEL:-QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ}"
SERVED_NAME="${SERVED_NAME:-My_Model}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"        # Increased from 8 for better throughput
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"       # Increased from 0.9 to maximize KV-cache
SWAP_SPACE="${SWAP_SPACE:-4}"
TP_SIZE="${TP_SIZE:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# =============================================================================
# ROCm Environment Variables (RDNA4/gfx1201 optimized)
# =============================================================================

# GPU Selection
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# Disable ROCm profiling/tracing - causes assertion failure on RDNA4
# "hsa_amd_profiling_async_copy_enable failed" crash fix
export HSA_TOOLS_LIB=""
export ROCR_TRACER_ENABLE=0

# Kernel launch optimization (documented to improve performance)
export HIP_FORCE_DEV_KERNARG=1

# Triton-based kernels for RDNA4 (AITER not supported on gfx12)
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export VLLM_USE_TRITON_AWQ="1"

# IMPORTANT: AITER is for MI300X/MI355X (gfx9) only - explicitly disable for RDNA4
export VLLM_ROCM_USE_AITER="0"

# Prefer hipBLASLt for GEMM operations (if available)
export TORCH_BLAS_PREFER_HIPBLASLT=1

# =============================================================================
# Script setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting vLLM server (RDNA4 optimized)..."
echo "  Model: $MODEL"
echo "  GPU: $HIP_VISIBLE_DEVICES"
echo "  Context: $MAX_MODEL_LEN tokens"
echo "  Max sequences: $MAX_NUM_SEQS"
echo "  GPU memory: ${GPU_MEM_UTIL}%"
echo "  Port: $PORT"
echo ""

# =============================================================================
# Launch vLLM
# =============================================================================

exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --swap-space "$SWAP_SPACE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --quantization awq \
    --trust-remote-code \
    --disable-log-requests \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
