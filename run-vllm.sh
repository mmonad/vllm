#!/bin/bash
# vLLM ROCm launch script for AMD Radeon AI PRO R9700 (gfx1201/RDNA4)
#
# Model configuration now lives in YAML config files under configs/.
# This script only sets ROCm/RDNA4 environment variables and delegates to vllm serve.
#
# Usage:
#   ./run-vllm.sh configs/qwen3-14b-fp8.yaml              # tool-calling (FP8)
#   ./run-vllm.sh configs/qwen3-coder-30b-a3b-awq.yaml   # coding model
#   ./run-vllm.sh configs/qwen3-think-30b-a3b-awq.yaml   # reasoning model
#   ./run-vllm.sh configs/qwen3-vl-30b-a3b-awq.yaml      # vision model
#   ./run-vllm.sh configs/glm-ocr-0.9b.yaml              # OCR model
#   ./run-vllm.sh configs/voxtral-3b-batch.yaml           # batch transcription
#   ./run-vllm.sh configs/voxtral-4b-realtime.yaml        # streaming STT

CONFIG="${1:?Usage: $0 <config.yaml> [extra vllm args...]}"
shift

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

# Triton JIT kernels aren't serializable by torch.compile's cache on ROCm
export VLLM_DISABLE_COMPILE_CACHE=1

# =============================================================================
# Launch vLLM
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

exec vllm serve --config "$CONFIG" "$@"
