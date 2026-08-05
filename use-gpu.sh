#!/usr/bin/env bash
# use-gpu: Switch vllm between NVIDIA CUDA and AMD ROCm virtual environments.
#
# Each target gets its own venv (.venv-cuda / .venv-rocm) and .venv is a
# symlink to the active one.
#
# Why the CUDA path is different from ROCm:
#   pyproject.toml is configured for ROCm (AMD wheel URLs in [tool.uv.sources],
#   ROCm-only deps like amdsmi/conch-triton-kernels, and vllm is listed in
#   no-build-isolation-package). For CUDA we bypass all of that by:
#     1. Installing build deps from requirements/build/cuda.txt
#     2. Installing runtime deps from requirements/cuda.txt (+ common.txt)
#     3. Building vllm with --no-deps --no-build-isolation to skip the
#        ROCm dependency list entirely.
set -euo pipefail

VLLM_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="3.13"

# ---- usage ----
usage() {
    cat <<'EOF'
Usage: use-gpu.sh <cuda|rocm> [-u|--update] [-r|--rebuild] [-h|--help]
       use-gpu.sh                          # show current target

Targets:
  cuda, nvidia    Switch to NVIDIA CUDA venv (create on first run)
  rocm, amd       Switch to AMD ROCm venv (create on first run)

Options:
  -u, --update    Update deps + incremental rebuild (changed C++ only)
  -r, --rebuild   Nuke venv and recreate from scratch
  -h, --help      Show this help
EOF
}

# ---- helpers ----
show_current() {
    if [ -L "$VLLM_DIR/.venv" ]; then
        local target
        target="$(readlink "$VLLM_DIR/.venv")"
        case "$target" in
            *.venv-cuda*) echo "current: cuda  ($target)" ;;
            *.venv-rocm*) echo "current: rocm  ($target)" ;;
            *)            echo "current: $target" ;;
        esac
    elif [ -d "$VLLM_DIR/.venv" ]; then
        echo "current: .venv (not managed — rename it to .venv-rocm or .venv-cuda first)"
    else
        echo "current: none"
    fi
}

preflight_cuda() {
    if [ -z "${CUDA_HOME:-}" ]; then
        echo "error: CUDA_HOME is not set. Source your shell profile or set it manually." >&2
        exit 1
    fi
    if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
        exit 1
    fi
}

create_cuda_venv() {
    local venv="$VLLM_DIR/.venv-cuda"
    echo "==> Creating CUDA venv..."
    # --seed for parity with the rocm path; harmless on cuda.
    uv venv --python "$PYTHON_VERSION" --seed "$venv"

    echo "==> Installing build deps (requirements/build/cuda.txt)..."
    VIRTUAL_ENV="$venv" uv pip install \
        -r "$VLLM_DIR/requirements/build/cuda.txt" \
        --torch-backend cu129

    echo "==> Installing runtime deps (requirements/cuda.txt)..."
    VIRTUAL_ENV="$venv" uv pip install \
        -r "$VLLM_DIR/requirements/cuda.txt" \
        --torch-backend cu129

    echo "==> Installing vllm (editable, --no-deps)..."
    VIRTUAL_ENV="$venv" VLLM_TARGET_DEVICE=cuda \
        uv pip install -e "$VLLM_DIR" --no-deps --no-build-isolation
}

install_rocm_build_deps() {
    # vllm is in no-build-isolation-package, so build deps (including torch)
    # must be in the venv before the editable install.  We install
    # everything from requirements/build/cuda.txt except torch, then install
    # the ROCm torch wheel separately (URL from [tool.uv.sources]).  Using
    # the cuda variant is intentional — build/rocm.txt pulls in runtime
    # extras (torchvision, triton, amdsmi, timm, -r ../common.txt) that we
    # don't want during the pre-build step.
    local venv="$1"
    local python_bin="$venv/bin/python"
    local tmp
    tmp="$(mktemp)"
    grep -v -i '^torch\b' "$VLLM_DIR/requirements/build/cuda.txt" > "$tmp"
    echo "==> Installing build deps (requirements/build/cuda.txt, minus torch)..."
    VIRTUAL_ENV="$venv" uv pip install -r "$tmp"
    rm -f "$tmp"

    # flash-attn uses legacy setup.py (no pyproject.toml).  Setuptools >=80
    # removed the 'install_lib' command that legacy installs need.  Cap it.
    # pip is also required: setup.py:223 shells out to `python -m pip install
    # third_party/aiter`, which fails on a uv-only venv ("No module named pip")
    # — uv venvs don't seed pip by default.
    VIRTUAL_ENV="$venv" uv pip install 'setuptools>=77.0.3,<80' pip

    # Install only the pinned ROCm stack here. A full `uv sync` also resolves
    # the project's Git dependencies and can block while updating them.
    echo "==> Installing ROCm torch stack from AMD index..."
    uv pip install \
        --python "$python_bin" \
        --index "https://repo.amd.com/rocm/whl-multi-arch/" \
        --index-strategy unsafe-best-match \
        "torch[device-gfx1201]==2.12.0+rocm7.14.0" \
        "torchvision[device-gfx1201]==0.27.0+rocm7.14.0" \
        "torchaudio==2.11.0+rocm7.14.0"

    # flash-attn setup imports torch, so install the official release only
    # after the ROCm stack is available.
    echo "==> Installing official flash-attn release..."
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        uv pip install \
            --python "$python_bin" \
            --no-build-isolation \
            "flash-attn==2.8.3.post1"
}

create_rocm_venv() {
    local venv="$VLLM_DIR/.venv-rocm"
    echo "==> Creating ROCm venv..."
    # --seed installs pip; flash-attention's setup.py shells out to
    # `python -m pip install third_party/aiter` (the Triton kernels for FA
    # live in the aiter package) and a uv-only venv has no pip.
    uv venv --python "$PYTHON_VERSION" --seed "$venv"

    install_rocm_build_deps "$venv"

    # FLASH_ATTENTION_TRITON_AMD_ENABLE selects the Triton backend in
    # flash-attention's setup.py and skips compiling the CK (Composable
    # Kernel) C++ extension — required on RDNA4/gfx1201 where the CK
    # submodule's headers drift from what mha_varlen_bwd.hip expects
    # ("type 'float' cannot be narrowed to ck_tile::index_t").  run-vllm.sh
    # exports the same var at runtime; the build needs it independently.
    echo "==> Installing vllm (editable, uses pyproject.toml ROCm sources)..."
    VIRTUAL_ENV="$venv" \
    VLLM_TARGET_DEVICE=rocm \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        uv pip install -e "$VLLM_DIR"
}

update_cuda_venv() {
    local venv="$VLLM_DIR/.venv-cuda"
    echo "==> Updating deps (requirements/cuda.txt)..."
    VIRTUAL_ENV="$venv" uv pip install \
        -r "$VLLM_DIR/requirements/build/cuda.txt" \
        -r "$VLLM_DIR/requirements/cuda.txt" \
        --torch-backend cu129

    echo "==> Rebuilding vllm (incremental, only changed C++ extensions)..."
    VIRTUAL_ENV="$venv" VLLM_TARGET_DEVICE=cuda \
        uv pip install -e "$VLLM_DIR" --no-deps --no-build-isolation
}

update_rocm_venv() {
    local venv="$VLLM_DIR/.venv-rocm"

    install_rocm_build_deps "$venv"

    # See create_rocm_venv for why FLASH_ATTENTION_TRITON_AMD_ENABLE is set
    # here — picks the Triton backend so the CK C++ extension is skipped.
    echo "==> Updating vllm (editable, uses pyproject.toml ROCm sources)..."
    VIRTUAL_ENV="$venv" \
    VLLM_TARGET_DEVICE=rocm \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        uv pip install -e "$VLLM_DIR"
}

switch_to() {
    local target_name="$1"
    local action="${2:-}"
    local venv_dir=".venv-${target_name}"

    # Refuse if .venv is a real directory (not a symlink)
    if [ -d "$VLLM_DIR/.venv" ] && [ ! -L "$VLLM_DIR/.venv" ]; then
        echo "error: .venv is a real directory, not a symlink." >&2
        echo "Rename it first: mv .venv .venv-rocm  (or .venv-cuda)" >&2
        exit 1
    fi

    # Handle --rebuild (nuke and recreate)
    if [ "$action" = "--rebuild" ] || [ "$action" = "-r" ]; then
        if [ -d "$VLLM_DIR/$venv_dir" ]; then
            echo "Removing $venv_dir..."
            rm -rf "${VLLM_DIR:?}/${venv_dir:?}"
        fi
    fi

    # Handle --update (incremental update of existing venv)
    if [ "$action" = "--update" ] || [ "$action" = "-u" ]; then
        if [ ! -d "$VLLM_DIR/$venv_dir" ]; then
            echo "error: $venv_dir does not exist. Run without -u first." >&2
            exit 1
        fi
        case "$target_name" in
            cuda)
                preflight_cuda
                update_cuda_venv
                ;;
            rocm)
                update_rocm_venv
                ;;
        esac
    fi

    # Create venv if it doesn't exist
    if [ ! -d "$VLLM_DIR/$venv_dir" ]; then
        case "$target_name" in
            cuda)
                preflight_cuda
                create_cuda_venv
                ;;
            rocm)
                create_rocm_venv
                ;;
        esac
    fi

    # Swap symlink
    ln -sfn "$venv_dir" "$VLLM_DIR/.venv"

    echo ""
    echo "Switched .venv -> $venv_dir"
    echo "Run:  source .venv/bin/activate"
}

# ---- parse args ----
TARGET=""
ACTION=""

for arg in "$@"; do
    case "$arg" in
        cuda|nvidia)    TARGET="cuda" ;;
        rocm|amd)       TARGET="rocm" ;;
        -u|--update)    ACTION="--update" ;;
        -r|--rebuild)   ACTION="--rebuild" ;;
        -h|--help)      usage; exit 0 ;;
        *)
            echo "error: unknown argument '$arg'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    show_current
else
    switch_to "$TARGET" "$ACTION"
fi
