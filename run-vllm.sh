#!/bin/bash
# vLLM launch script — auto-selects CUDA or ROCm environment from config path.
#
# Usage:
#   ./run-vllm.sh configs/cuda/qwen3.5-27b-fp8.yaml       # NVIDIA RTX Pro 6000
#   ./run-vllm.sh configs/rocm/qwen3.5-27b-fp8.yaml       # AMD RDNA4 (gfx1201)
#   ./run-vllm.sh configs/cuda/voxtral-4b-realtime.yaml    # streaming STT

CONFIG="${1:?Usage: $0 <config.yaml> [extra vllm args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Detect backend from config path
# =============================================================================
case "$CONFIG" in
    configs/cuda/*)
        BACKEND="cuda"
        ;;
    configs/rocm/*)
        BACKEND="rocm"
        ;;
    *)
        echo "error: cannot determine backend from config path." >&2
        echo "Place configs under configs/cuda/ or configs/rocm/." >&2
        exit 1
        ;;
esac

# =============================================================================
# GPU auto-selection (ROCm)
# =============================================================================
# Reads tensor-parallel-size from the config, queries rocm-smi for GPUs that
# have no active compute PIDs, and picks the first N free ones.
select_rocm_gpus() {
    local tp_size="$1"

    # If the user already pinned GPUs, respect that
    if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
        echo "HIP_VISIBLE_DEVICES already set: $HIP_VISIBLE_DEVICES"
        return
    fi

    # Three numbering schemes coexist on this system:
    #   - rocm-smi / DRM / KFD index (--showuse, --showpidgpus, --showbus): PCI scan order
    #   - HIP runtime index (HIP_VISIBLE_DEVICES, hipGetDeviceCount): HSA topology order
    #   - PCI bus ID (B:D.F): the only cross-runtime stable identifier
    # The two index schemes can disagree (e.g., rocm-smi GPU[2] == HIP[1] on a
    # 4× R9700 box where HSA walks NUMA-affinity order, not PCI order).  We
    # pick free GPUs in rocm-smi's view, then translate to HIP indices via PCI
    # bus IDs before exporting.  Doing it in Python because libamdhip64 is the
    # authoritative source for HIP's index → PCI map.
    local selection
    selection=$("$SCRIPT_DIR/.venv/bin/python" - "$tp_size" <<'PY'
import ctypes
import re
import subprocess
import sys

tp_size = int(sys.argv[1])

def run(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        # Fail closed: if rocm-smi can't tell us what's busy, treating every
        # GPU as free will mispin onto an actively-running workload.
        print(
            f"error: {' '.join(args)} failed (rc={r.returncode}): "
            f"{r.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return r.stdout

# rocm-smi index → PCI bus (lowercased BDF without the 0000: domain prefix)
def normalize_bdf(s):
    s = s.strip().lower()
    if s.startswith("0000:"):
        s = s[5:]
    return s

bus_lines = run("rocm-smi", "--showbus")
smi_to_bdf = {}
for m in re.finditer(r"GPU\[(\d+)\][^\n]*PCI Bus:\s*([0-9a-fA-F:.]+)", bus_lines):
    smi_to_bdf[int(m.group(1))] = normalize_bdf(m.group(2))

if not smi_to_bdf:
    print("error: rocm-smi --showbus returned no GPUs", file=sys.stderr)
    sys.exit(1)

# Busy DRM indices.  rocm-smi --showpidgpus formats vary; handle both modern
# "PID N is using K DRM device(s):\n<ids>" and legacy "GPU[N]" inline tokens.
pid_text = run("rocm-smi", "--showpidgpus")
busy_drm = set()
capture = False
for line in pid_text.splitlines():
    if re.match(r"^PID \d+ is using [1-9]\d* DRM device\(s\):", line):
        capture = True
        continue
    if capture and re.match(r"^\s*\d+(\s+\d+)*\s*$", line):
        busy_drm.update(int(x) for x in line.split())
        capture = False
        continue
    if line.startswith("PID ") or line.startswith("="):
        capture = False
if not busy_drm:
    busy_drm = {int(m) for m in re.findall(r"GPU\[(\d+)\]", pid_text)}

# DRM index aligns with rocm-smi GPU[N] index in practice (both walk
# /dev/dri/renderD128+N).  Treat them as the same here.
free_smi = sorted(set(smi_to_bdf) - busy_drm)
if len(free_smi) < tp_size:
    print(
        f"error: need {tp_size} free GPUs but only {len(free_smi)} available "
        f"(free smi={free_smi}, busy smi={sorted(busy_drm)})",
        file=sys.stderr,
    )
    sys.exit(1)
chosen_smi = free_smi[:tp_size]
chosen_bdfs = [smi_to_bdf[i] for i in chosen_smi]

# HIP index → PCI bus, via libamdhip64.so
hip = ctypes.CDLL("libamdhip64.so")
n = ctypes.c_int()
if hip.hipGetDeviceCount(ctypes.byref(n)) != 0:
    print("error: hipGetDeviceCount failed", file=sys.stderr)
    sys.exit(1)
hip_to_bdf = {}
for i in range(n.value):
    buf = ctypes.create_string_buffer(64)
    if hip.hipDeviceGetPCIBusId(buf, 64, ctypes.c_int(i)) != 0:
        print(f"error: hipDeviceGetPCIBusId failed for HIP[{i}]", file=sys.stderr)
        sys.exit(1)
    hip_to_bdf[i] = normalize_bdf(buf.value.decode())
bdf_to_hip = {v: k for k, v in hip_to_bdf.items()}

try:
    chosen_hip = [bdf_to_hip[bdf] for bdf in chosen_bdfs]
except KeyError as e:
    print(
        f"error: PCI bus {e.args[0]} not found in HIP enumeration "
        f"(hip_to_bdf={hip_to_bdf})",
        file=sys.stderr,
    )
    sys.exit(1)

# Output: <hip_csv>|<smi_csv>|<bdf_csv>|<num_free>
print(
    f"{','.join(str(i) for i in chosen_hip)}|"
    f"{','.join(str(i) for i in chosen_smi)}|"
    f"{','.join(chosen_bdfs)}|"
    f"{len(free_smi)}"
)
PY
)
    if [ -z "$selection" ]; then
        echo "error: GPU selection helper failed" >&2
        exit 1
    fi

    local hip_csv smi_csv bdf_csv num_free
    IFS='|' read -r hip_csv smi_csv bdf_csv num_free <<< "$selection"
    export HIP_VISIBLE_DEVICES="$hip_csv"
    echo "Auto-selected GPUs: HIP=[$hip_csv] (rocm-smi=[$smi_csv], PCI=[$bdf_csv], tp=$tp_size, $num_free free)"
}

# =============================================================================
# ROCm Environment Variables (RDNA4/gfx1201 optimized)
# =============================================================================
setup_rocm() {

    # Disable ROCm profiling/tracing — causes assertion failure on RDNA4
    export HSA_TOOLS_LIB=""
    export ROCR_TRACER_ENABLE=0

    # Kernel launch optimization
    export HIP_FORCE_DEV_KERNARG=1

    # Limit HW command queues to 1 — reduces per-queue bookkeeping overhead.
    # Recommended for single-stream inference on RDNA4 where async overlap
    # across multiple queues gives little benefit and costs submit latency.
    export GPU_MAX_HW_QUEUES=1

    # Optional: PyTorch TunableOp — autotunes GEMM kernel shapes for each
    # model at runtime.  First run with TUNING=1 is slow (builds a cache
    # under ~/.cache/torch/tunableop); subsequent runs with TUNING=0 reuse
    # it for ~10-20% GEMM speedup on RDNA4.  Opt-in per shell session:
    #   export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1

    # Triton-based kernels for RDNA4 (AITER not supported on gfx12)
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
    export VLLM_USE_TRITON_AWQ="1"

    # AITER is for MI300X/MI355X (gfx9) only
    export VLLM_ROCM_USE_AITER="0"

    # Prefer hipBLASLt for GEMM operations
    export TORCH_BLAS_PREFER_HIPBLASLT=1

    # FP8 weight padding is a CDNA cache-alignment trick that allocates a
    # transient 2× weight copy during load (F.pad + slice).  HIP's allocator
    # has no expandable_segments, so the spike OOMs large FP8 models on load;
    # RDNA4 also gains ~nothing from the alignment, so disable it outright.
    # No-op for non-FP8 models and ignored on CUDA.
    export VLLM_ROCM_FP8_PADDING=0

    # Accelerate safetensors model loading via GPU
    export SAFETENSORS_FAST_GPU=1

    # Triton JIT kernels aren't serializable by torch.compile's cache on ROCm
    export VLLM_DISABLE_COMPILE_CACHE=1

    # Tuned MoE kernel configs
    export VLLM_TUNED_CONFIG_FOLDER="$SCRIPT_DIR/tuned-moe-configs"

    # RCCL LL/LL128 protocols hang on gfx1201 (RDNA4) — wave32 fine-grained
    # sync paths are broken.  Force the Simple protocol which uses coarse
    # barrier sync and works.  Negligible perf hit for inference workloads
    # since large AllReduces already use Simple by default.
    export NCCL_PROTO=Simple
}

# =============================================================================
# CUDA Environment Variables (RTX Pro 6000 / Blackwell)
# =============================================================================
setup_cuda() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
}

# =============================================================================
# Launch
# =============================================================================

# Read parallelism sizes from the YAML config (default: 1 each).
# Total GPUs needed = TP * PP (one worker per (tp_rank, pp_rank) pair).
TP_SIZE=$(grep -oP '^\s*tensor-parallel-size:\s*\K[0-9]+' "$CONFIG" 2>/dev/null || echo 1)
PP_SIZE=$(grep -oP '^\s*pipeline-parallel-size:\s*\K[0-9]+' "$CONFIG" 2>/dev/null || echo 1)
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

case "$BACKEND" in
    rocm)
        # setup_rocm first: the RDNA4 safety env vars (HSA_TOOLS_LIB="",
        # ROCR_TRACER_ENABLE=0) must be in the environment before the GPU
        # selection helper dlopens libamdhip64 and triggers HIP init.
        setup_rocm
        select_rocm_gpus "$WORLD_SIZE"
        ;;
    cuda) setup_cuda ;;
esac

source "$SCRIPT_DIR/.venv/bin/activate"

# Default bind is 0.0.0.0:8080, but only when the YAML doesn't pin host/port
# itself — vLLM's CLI > config precedence (see argparse_utils.py
# _pull_args_from_config) means flags emitted here would otherwise shadow
# YAML values.  Configs that need loopback-only or a non-default port (e.g.
# realtime audio bound to 127.0.0.1:8088) can declare `host:` / `port:` in
# the YAML and have them honored.
DEFAULT_FLAGS=()
# Anchor at column 0: vLLM only pulls top-level YAML keys into argparse, so
# a nested "  host:" under another mapping must not suppress the default.
if ! grep -qE '^host\s*:' "$CONFIG"; then
    DEFAULT_FLAGS+=(--host 0.0.0.0)
fi
if ! grep -qE '^port\s*:' "$CONFIG"; then
    DEFAULT_FLAGS+=(--port 8080)
fi

exec vllm serve "${DEFAULT_FLAGS[@]}" --config "$CONFIG" "$@"
