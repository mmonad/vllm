#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RDNA4 tuning for FP8 dense and MoE kernels across multiple model deployments.

This combines the dense block-wise FP8 tuner and the fused-MoE FP8 tuner into
one script while keeping the generated JSON names compatible with vLLM:

  * Dense block-wise W8A8 projection configs:
    vllm/model_executor/layers/quantization/utils/configs/

  * Fused-MoE FP8 W8A8 block configs:
    vllm/model_executor/layers/fused_moe/configs/

Both tuning paths run eager kernels directly on one GPU. CUDA graphs and Ray are
intentionally avoided because this workflow targets single-GPU RDNA4.

Models are selected via --model. The registered specs encode each deployment's
exact (N, K) GEMM shapes (after the configured tensor-parallel split) and any
fused-MoE parameters; see MODELS below. Add a new entry there to onboard a
new model rather than hard-coding shapes at the call site.

Usage:
  .venv/bin/python tune-rdna4-fp8.py --model gemma-4-31b-fp8
  .venv/bin/python tune-rdna4-fp8.py --model qwen3.6-27b-fp8 --target dense
  .venv/bin/python tune-rdna4-fp8.py --model qwen3.6-35b-a3b-fp8 --target moe
  .venv/bin/python tune-rdna4-fp8.py --model qwen3.6-27b-fp8 --shapes 5120,3072
  .venv/bin/python tune-rdna4-fp8.py --model gemma-4-31b-fp8 --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any


def _preparse_gpu(argv: list[str]) -> str | None:
    """Extract --gpu early so HIP_VISIBLE_DEVICES is set before torch imports."""
    for i, tok in enumerate(argv):
        if tok == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--gpu="):
            return tok.split("=", 1)[1]
    return None


_GPU_ARG = _preparse_gpu(sys.argv)
if _GPU_ARG is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = _GPU_ARG
elif "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

# HIP_VISIBLE_DEVICES is the least surprising knob here; keeping both set can
# make HIP and rocm-smi disagree about which card is selected.
os.environ.pop("ROCR_VISIBLE_DEVICES", None)

# AITER kernels are not the target of this RDNA4 tuner. Respect an explicit user
# setting, but default to the Triton paths used by these benchmarks.
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = "./tuned-rdna4-configs"
IN_TREE_DENSE_CONFIGS = os.path.join(
    SCRIPT_DIR,
    "vllm",
    "model_executor",
    "layers",
    "quantization",
    "utils",
    "configs",
)
IN_TREE_MOE_CONFIGS = os.path.join(
    SCRIPT_DIR,
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
    "configs",
)


# Geometric spacing through the prefill regime. max-num-batched-tokens=8192 in
# the deployment YAMLs means a chunked-prefill step can carry up to ~8k tokens;
# typical single-prompt prefills land at 128/512/2048 (in the benchmark set).
# Decode M stays small (single-digit per active sequence), so the low end keeps
# tight steps. vLLM's nearest-M snap picks whichever key is closest at runtime.
DENSE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DENSE_BLOCK_N = 128
DENSE_BLOCK_K = 128


@dataclass(frozen=True)
class MoESpec:
    """Fused-MoE FP8 W8A8 block-wise tuning parameters for a model."""

    experts: int
    topk: int
    intermediate_size: int  # per-expert intermediate (pre-shard)
    hidden_size: int
    tp_size: int
    block_n: int = 128
    block_k: int = 128
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)

    @property
    def shard_intermediate(self) -> int:
        # Matches benchmark_moe.py: 2 * (intermediate / tp) so gate+up are
        # combined along N before the column-parallel split.
        return 2 * self.intermediate_size // self.tp_size

    @property
    def block_shape(self) -> list[int]:
        return [self.block_n, self.block_k]


@dataclass(frozen=True)
class ModelSpec:
    """One tunable deployment: dense GEMM shapes plus optional MoE config."""

    name: str  # human-readable label printed in logs
    notes: str  # one-line context (architecture / TP topology)
    dense_shapes: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    moe: MoESpec | None = None


# Registry of supported deployments. Shapes are post-TP per-rank dimensions —
# match what each rank's QKVParallelLinear / RowParallelLinear / MergedColumn
# layer actually instantiates after sharding.
MODELS: dict[str, ModelSpec] = {
    "qwen3.6-27b-fp8": ModelSpec(
        name="Qwen3.6-27B-FP8",
        notes="dense FP8 block-wise; tensor-parallel-size=2",
        dense_shapes=(
            (5120, 3072),  # o_proj
            (17408, 5120),  # gate_up_proj (fused)
            (5120, 8704),  # down_proj
            (7168, 5120),  # qkv_proj
        ),
    ),
    "qwen3.6-35b-a3b-fp8": ModelSpec(
        name="Qwen3.6-35B-A3B-FP8",
        notes="fused-MoE FP8 block-wise; pipeline-parallel-size=3, no TP",
        moe=MoESpec(
            experts=256,
            topk=8,
            intermediate_size=512,
            hidden_size=2048,
            tp_size=1,
        ),
    ),
    "gemma-4-31b-fp8": ModelSpec(
        name="Gemma-4-31B-IT-FP8-block",
        notes=(
            "hybrid sliding/full attention dense FP8 block-wise; tensor-parallel-size=2"
        ),
        dense_shapes=(
            # Sliding-attention layers (50 of 60, head_dim=256, KV heads=16):
            (8192, 5376),  # qkv_proj sliding (16Q + 8K + 8V)*256 = 8192
            (5376, 4096),  # o_proj sliding (16*256 = 4096 in)
            # Full-attention layers (10 of 60, global_head_dim=512, KV heads=4
            # under attention_k_eq_v):
            (10240, 5376),  # qkv_proj full (16Q + 2K + 2V)*512 = 10240
            (5376, 8192),  # o_proj full (16*512 = 8192 in)
            # Shared MLP (intermediate_size=21504):
            (21504, 5376),  # gate_up_proj fused (2 * 21504/2)
            (5376, 10752),  # down_proj (21504/2 in)
        ),
    ),
}


CONFIG_KEY_ORDER = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
    "SPLIT_K",
)


def sort_kernel_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in CONFIG_KEY_ORDER if key in config}


def parse_dense_shapes(
    raw: str | None,
    default: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    if raw is None:
        return list(default)

    shapes = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        parts = [part.strip() for part in pair.split(",")]
        if len(parts) != 2:
            raise ValueError(f"expected N,K pair, got {pair!r}")
        shapes.append((int(parts[0]), int(parts[1])))

    if not shapes:
        raise ValueError("--shapes did not contain any N,K pairs")
    return shapes


def sanitize_device_name(device_name: str) -> str:
    return device_name.replace(" ", "_")


def dense_config_name(N: int, K: int, device_name: str) -> str:
    return (
        f"N={N},K={K},device_name={sanitize_device_name(device_name)},"
        f"dtype=fp8_w8a8,block_shape=[{DENSE_BLOCK_N},{DENSE_BLOCK_K}].json"
    )


def copy_into_tree(src_path: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copyfile(src_path, dst_path)
    return dst_path


def json_has_batch_sizes(path: str, batch_sizes: list[int]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(data, dict):
        return False

    expected = {str(m) for m in batch_sizes}
    return expected.issubset(data.keys())


def build_dense_rdna4_search_space() -> list[dict[str, int]]:
    """
    RDNA4 WMMA-friendly search space for block-wise FP8 GEMM across decode
    (M<=16) and prefill (M up to a few thousand) regimes.

    Shape constraints for the kernel:
      - BLOCK_SIZE_M/N must be multiples of 16 (WMMA m16n16k16/m16n16k32).
      - BLOCK_SIZE_K must divide BLOCK_K=128 so block-scale indexing is exact.
      - BLOCK_SIZE_M extends up to 128 because large prefill-M amortizes a
        larger M-tile across more rows, avoiding repeated B-tile reloads.
    Tile-pressure filter caps per-thread output work at 64 elements. Triton
    still rejects register-hungry configs via OutOfResources at compile time.
    """
    param_ranges = {
        "BLOCK_SIZE_M": [16, 32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [64, 128],
        "GROUP_SIZE_M": [1, 4, 8, 16, 32],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
    }
    keys, values = zip(*param_ranges.items())
    configs = []
    for vals in product(*values):
        cfg = dict(zip(keys, vals))
        if DENSE_BLOCK_K % cfg["BLOCK_SIZE_K"] != 0:
            continue
        tile_elems = cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_N"]
        if tile_elems // (cfg["num_warps"] * 32) > 64:
            continue
        configs.append(cfg)
    return configs


def make_dense_inputs(M: int, N: int, K: int):
    """Allocate FP8 inputs and FP32 scales matching the block-wise layout."""
    import torch

    fp8 = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * (
        2 * fp8_max
    )
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(fp8)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * (
        2 * fp8_max
    )
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(fp8)

    k_tiles = (K + DENSE_BLOCK_K - 1) // DENSE_BLOCK_K
    n_tiles = (N + DENSE_BLOCK_N - 1) // DENSE_BLOCK_N
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2

    return A, B, As, Bs


def benchmark_dense_config(A, B, As, Bs, M, N, K, config, num_iters=20):
    """Time one dense FP8 config; returns mean latency in microseconds."""
    import torch

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _w8a8_triton_block_scaled_mm,
    )
    from vllm.triton_utils import triton

    C = A.new_empty((M, N), dtype=torch.bfloat16)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    def run():
        _w8a8_triton_block_scaled_mm[grid](
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            DENSE_BLOCK_N,
            DENSE_BLOCK_K,
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            **config,
        )

    run()
    torch.accelerator.synchronize()
    for _ in range(3):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for _ in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    return (sum(latencies) / num_iters) * 1000.0


def write_dense_json(out_path: str, best_per_bs: dict[int, dict[str, Any]]) -> None:
    serialized = {
        str(m): dict(sorted(cfg.items())) for m, cfg in sorted(best_per_bs.items())
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=4)
        f.write("\n")


def tune_dense_shape(
    N: int,
    K: int,
    search_space: list[dict[str, int]],
    save_dir: str,
    device_name: str,
    install: bool,
) -> str:
    """Tune all dense batch sizes for one (N, K), flushing after each M."""
    import torch

    from vllm.triton_utils import triton

    print(f"\n{'=' * 60}")
    print(f"Dense FP8 tuning N={N}, K={K}")
    print(f"{'=' * 60}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, dense_config_name(N, K, device_name))

    best_per_bs: dict[int, dict[str, Any]] = {}
    for M in DENSE_BATCH_SIZES:
        bs_start = time.time()
        print(f"\n--- dense M={M} (N={N}, K={K}) ---")
        A, B, As, Bs = make_dense_inputs(M, N, K)

        best_config = None
        best_time = float("inf")
        skipped = 0
        total = len(search_space)
        for idx, cfg in enumerate(search_space):
            try:
                t_us = benchmark_dense_config(
                    A,
                    B,
                    As,
                    Bs,
                    M,
                    N,
                    K,
                    cfg,
                    num_iters=20,
                )
            except triton.runtime.autotuner.OutOfResources:
                skipped += 1
                continue
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    [skip] cfg {idx}: {type(e).__name__}: {e}")
                continue

            if t_us < best_time:
                best_time = t_us
                best_config = cfg

            if (idx + 1) % 25 == 0 or idx == total - 1:
                print(
                    f"    [{idx + 1}/{total}] best so far: "
                    f"{best_time:.1f} us (skipped {skipped})"
                )

            if idx > 0 and idx % 50 == 0:
                gc.collect()
                torch.accelerator.empty_cache()

        elapsed_bs = time.time() - bs_start
        assert best_config is not None, f"No valid config for M={M}, N={N}, K={K}"
        best_per_bs[M] = dict(best_config)
        print(f"  Best: {best_time:.1f} us  config={best_config}  ({elapsed_bs:.0f}s)")

        write_dense_json(out_path, best_per_bs)
        print(
            f"    -> flushed {out_path} ({len(best_per_bs)}/{len(DENSE_BATCH_SIZES)})"
        )
        if install:
            dst_path = copy_into_tree(out_path, IN_TREE_DENSE_CONFIGS)
            print(f"    -> installed {dst_path}")

        del A, B, As, Bs
        gc.collect()
        torch.accelerator.empty_cache()

    print(f"\n  -> wrote {out_path}")
    return out_path


def run_dense_tuning(
    *,
    model: ModelSpec,
    shapes: list[tuple[int, int]],
    save_dir: str,
    device_name: str,
    dry_run: bool,
    skip_existing: bool,
    install: bool,
) -> list[str]:
    search_space = build_dense_rdna4_search_space()

    print(f"\nDense block-wise FP8 tuning: {model.name}")
    print(f"  Notes:           {model.notes}")
    print(f"  Block quant:     [{DENSE_BLOCK_N}, {DENSE_BLOCK_K}]")
    print(f"  Batch sizes:     {DENSE_BATCH_SIZES}")
    print(f"  Shapes (N, K):   {shapes}")
    print(f"  Search space:    {len(search_space)} configs")
    print(
        f"  Estimated evals: {len(search_space) * len(DENSE_BATCH_SIZES) * len(shapes)}"
    )
    print(f"  Save dir:        {save_dir}")
    print(f"  Auto-install:    {install}")

    if dry_run:
        print(f"  Example config:  {search_space[0]}")
        return []

    written = []
    for N, K in shapes:
        out_path = os.path.join(save_dir, dense_config_name(N, K, device_name))
        if skip_existing and json_has_batch_sizes(out_path, DENSE_BATCH_SIZES):
            print(f"\n[skip] dense N={N}, K={K} - {out_path} is complete")
            if install:
                dst_path = copy_into_tree(out_path, IN_TREE_DENSE_CONFIGS)
                print(f"    -> installed {dst_path}")
            continue
        written.append(
            tune_dense_shape(N, K, search_space, save_dir, device_name, install)
        )
    return written


def build_moe_rdna4_search_space(moe: MoESpec) -> list[dict[str, int]]:
    """
    RDNA4 search space for FP8 block-wise fused MoE.

    The checkpoint's weight_block_size dictates BLOCK_SIZE_N/BLOCK_SIZE_K — the
    fused-MoE kernel requires them to match the quantization tile exactly.
    """
    param_ranges = {
        "BLOCK_SIZE_M": [16, 32, 64],
        "BLOCK_SIZE_N": [moe.block_n],
        "BLOCK_SIZE_K": [moe.block_k],
        "GROUP_SIZE_M": [1, 4, 8, 16, 32],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [0],
    }
    keys, values = zip(*param_ranges.items())
    configs = []
    for vals in product(*values):
        config = dict(zip(keys, vals))
        config["SPLIT_K"] = 1
        configs.append(config)
    return configs


def benchmark_moe_config_eager(
    config,
    num_tokens,
    num_experts,
    shard_intermediate_size,
    hidden_size,
    topk,
    dtype,
    block_quant_shape,
    num_iters=20,
):
    """Benchmark a single MoE config without CUDA graphs."""
    import torch

    from vllm.model_executor.layers.fused_moe import fused_topk, override_config
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        disable_inplace,
        fused_experts,
    )
    from vllm.platforms import current_platform

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    init_dtype = torch.float16
    intermediate_size = shard_intermediate_size // 2
    w1 = torch.randn(
        num_experts,
        shard_intermediate_size,
        hidden_size,
        dtype=init_dtype,
    ).to(current_platform.fp8_dtype())
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=init_dtype,
    ).to(current_platform.fp8_dtype())

    block_n, block_k = block_quant_shape
    factor_for_scale = 1e-2
    n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
    n_tiles_w2 = (hidden_size + block_n - 1) // block_n
    k_tiles_w1 = (hidden_size + block_k - 1) // block_k
    k_tiles_w2 = (intermediate_size + block_k - 1) // block_k
    w1_scale = (
        torch.rand(
            (num_experts, n_tiles_w1, k_tiles_w1),
            dtype=torch.float32,
        )
        * factor_for_scale
    )
    w2_scale = (
        torch.rand(
            (num_experts, n_tiles_w2, k_tiles_w2),
            dtype=torch.float32,
        )
        * factor_for_scale
    )
    a1_scale = torch.randn(1, dtype=torch.float32)
    a2_scale = torch.randn(1, dtype=torch.float32)

    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=current_platform.fp8_dtype(),
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_quant_shape,
    )

    inplace = not disable_inplace()

    def run(i):
        input_gating.copy_(gating_output[i])
        with override_config(config):
            topk_weights, topk_ids, _ = fused_topk(
                x, input_gating, topk, renormalize=True
            )
            return fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=inplace,
                quant_config=quant_config,
            )

    run(0)
    torch.accelerator.synchronize()
    for i in range(min(3, num_iters)):
        run(i)
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run(i)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    return sum(latencies) / num_iters * 1000.0


def tune_moe_batch_size(num_tokens, search_space, dtype, moe: MoESpec):
    """Find the best MoE config for a given batch size."""
    import torch

    from benchmarks.kernels.benchmark_moe import prune_rocm_search_space
    from vllm.triton_utils import triton

    pruned = prune_rocm_search_space(
        num_tokens,
        moe.shard_intermediate,
        moe.hidden_size,
        search_space.copy(),
        is_fp16=False,
        topk=moe.topk,
    )

    best_config = None
    best_time = float("inf")
    total = len(pruned)

    for idx, config in enumerate(pruned):
        try:
            kernel_time = benchmark_moe_config_eager(
                config,
                num_tokens,
                moe.experts,
                moe.shard_intermediate,
                moe.hidden_size,
                moe.topk,
                dtype,
                moe.block_shape,
                num_iters=20,
            )
        except triton.runtime.autotuner.OutOfResources:
            continue
        except Exception as e:
            print(f"    [skip] config {idx}: {type(e).__name__}: {e}")
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"    [{idx + 1}/{total}] best so far: {best_time:.1f} us")

        if idx > 0 and idx % 50 == 0:
            gc.collect()
            torch.accelerator.empty_cache()

    assert best_config is not None, f"No valid MoE config for M={num_tokens}"
    return best_config, best_time


def run_moe_tuning(
    *,
    model: ModelSpec,
    save_dir: str,
    dry_run: bool,
    skip_existing: bool,
    install: bool,
) -> str | None:
    import torch

    from benchmarks.kernels.benchmark_moe import (
        prune_rocm_search_space,
        save_configs,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name

    moe = model.moe
    assert moe is not None, "run_moe_tuning called for a model without MoE spec"
    batch_sizes = list(moe.batch_sizes)

    dtype = torch.float16
    search_space = build_moe_rdna4_search_space(moe)
    total_before_prune = len(search_space)

    print(f"\nMoE FP8 W8A8 block tuning: {model.name}")
    print(f"  Notes:           {model.notes}")
    print(
        f"  MoE shape:       E={moe.experts}, topk={moe.topk}, "
        f"intermediate={moe.intermediate_size}, hidden={moe.hidden_size}, "
        f"tp={moe.tp_size}"
    )
    print(f"  GEMM1:           (M, {moe.shard_intermediate}, {moe.hidden_size})")
    print(f"  GEMM2:           (M, {moe.hidden_size}, {moe.shard_intermediate // 2})")
    print(f"  Quant:           fp8_w8a8, block_shape={moe.block_shape}")
    print(f"  Batch sizes:     {batch_sizes}")
    print(f"  Search space:    {total_before_prune} configs before pruning")
    print(f"  Save dir:        {save_dir}")
    print(f"  Auto-install:    {install}")

    example_config = None
    for m in batch_sizes:
        pruned = prune_rocm_search_space(
            m,
            moe.shard_intermediate,
            moe.hidden_size,
            search_space.copy(),
            is_fp16=False,
            topk=moe.topk,
        )
        if example_config is None and pruned:
            example_config = sort_kernel_config(pruned[0])
        print(f"  M={m:>4d}:        {len(pruned):>4d} configs after pruning")

    total_evals = sum(
        len(
            prune_rocm_search_space(
                m,
                moe.shard_intermediate,
                moe.hidden_size,
                search_space.copy(),
                is_fp16=False,
                topk=moe.topk,
            )
        )
        for m in batch_sizes
    )
    print(f"  Estimated evals: {total_evals}")

    tuned_filename = get_config_file_name(
        moe.experts,
        moe.shard_intermediate // 2,
        "fp8_w8a8",
        moe.block_shape,
    )
    src_path = os.path.join(save_dir, tuned_filename)

    if dry_run:
        if example_config is not None:
            print(f"  Example config:  {example_config}")
        print(f"  Output file:     {src_path}")
        return None

    if skip_existing and json_has_batch_sizes(src_path, batch_sizes):
        print(f"\n[skip] MoE - {src_path} is complete")
        if install:
            dst_path = copy_into_tree(src_path, IN_TREE_MOE_CONFIGS)
            print(f"    -> installed {dst_path}")
        return None

    if install:
        os.makedirs(IN_TREE_MOE_CONFIGS, exist_ok=True)

    best_configs = {}
    for batch_size in batch_sizes:
        bs_start = time.time()
        print(f"\n--- MoE M={batch_size} ---")
        config, kernel_time = tune_moe_batch_size(
            batch_size,
            search_space,
            dtype,
            moe,
        )
        elapsed_bs = time.time() - bs_start
        best_configs[batch_size] = sort_kernel_config(config)
        print(f"  Best: {kernel_time:.1f} us  config={config}  ({elapsed_bs:.0f}s)")

        save_configs(
            best_configs,
            moe.experts,
            moe.shard_intermediate,
            moe.hidden_size,
            moe.topk,
            dtype,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            block_quant_shape=moe.block_shape,
            save_dir=save_dir,
        )

        if install:
            dst_path = copy_into_tree(src_path, IN_TREE_MOE_CONFIGS)
            print(
                f"    -> installed {dst_path} ({len(best_configs)}/{len(batch_sizes)})"
            )

    print(f"\n  -> wrote {src_path}")
    return src_path


def resolve_targets(target: str, model: ModelSpec) -> tuple[bool, bool]:
    """Compute (run_dense, run_moe) from --target, gated by what the model has."""
    want_dense = target in ("all", "dense")
    want_moe = target in ("all", "moe")
    has_dense = bool(model.dense_shapes)
    has_moe = model.moe is not None
    if target == "dense" and not has_dense:
        raise SystemExit(
            f"--target dense requested but {model.name!r} has no dense shapes"
        )
    if target == "moe" and not has_moe:
        raise SystemExit(f"--target moe requested but {model.name!r} has no MoE spec")
    return want_dense and has_dense, want_moe and has_moe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODELS.keys()),
        required=True,
        help="Model deployment to tune for. See MODELS registry for entries.",
    )
    parser.add_argument(
        "--target",
        choices=("all", "dense", "moe"),
        default="all",
        help=(
            "Which tuner to run. 'all' tunes whatever the chosen model has "
            "(dense shapes and/or a MoE spec). Default: all."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Physical GPU index. Pre-parsed into HIP_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory for all tuned RDNA4 JSON configs.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help=(
            "Override dense N,K pairs to tune, separated by semicolons. "
            "Example: '5120,3072;17408,5120'. Falls back to the model's "
            "registered dense_shapes when omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print search-space and output details, then exit.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip dense shapes or MoE configs that already have all batch sizes.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-tune even when complete JSON configs already exist.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not copy tuned JSON files into vLLM's in-tree config folders.",
    )
    args = parser.parse_args()

    model = MODELS[args.model]
    run_dense, run_moe = resolve_targets(args.target, model)

    if run_dense:
        try:
            dense_shapes = parse_dense_shapes(args.shapes, model.dense_shapes)
        except ValueError as e:
            parser.error(str(e))
    else:
        dense_shapes = []
        if args.shapes is not None:
            print("Warning: --shapes is ignored when dense tuning is not selected.")

    install = not args.no_install

    import torch

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_random_seed

    torch.set_default_device("cuda")

    raw_device_name = current_platform.get_device_name()
    device_name = sanitize_device_name(raw_device_name)
    props = torch.cuda.get_device_properties(0)
    free_memory, total_memory = torch.accelerator.get_memory_info(0)
    free_gb = free_memory / 1e9
    total_gb = total_memory / 1e9

    print("=" * 60)
    print(f"RDNA4 FP8 tuning: {model.name}")
    print("=" * 60)
    print(f"Model key:     {args.model}")
    print(f"Notes:         {model.notes}")
    print(f"Device:        {raw_device_name} (PCI {props.pci_bus_id})")
    print(f"Visible GPU:   HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"Memory:        {free_gb:.1f}/{total_gb:.1f} GB free")
    print(f"AITER:         VLLM_ROCM_USE_AITER={os.environ.get('VLLM_ROCM_USE_AITER')}")
    print("Mode:          eager (no CUDA graphs, no Ray)")

    start = time.time()
    dense_written: list[str] = []
    moe_written: str | None = None

    if run_dense:
        set_random_seed(args.seed)
        dense_written = run_dense_tuning(
            model=model,
            shapes=dense_shapes,
            save_dir=args.save_dir,
            device_name=device_name,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            install=install,
        )

    if run_moe:
        set_random_seed(args.seed)
        moe_written = run_moe_tuning(
            model=model,
            save_dir=args.save_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            install=install,
        )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete.")
        return

    print(f"Tuning complete in {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    if dense_written:
        print("\nDense configs:")
        for path in dense_written:
            print(f"  {path}")
    if moe_written:
        print("\nMoE config:")
        print(f"  {moe_written}")
    if install:
        print("\nInstalled config directories:")
        if run_dense:
            print(f"  {IN_TREE_DENSE_CONFIGS}")
        if run_moe:
            print(f"  {IN_TREE_MOE_CONFIGS}")
    else:
        print("\nInstall skipped (--no-install).")


if __name__ == "__main__":
    main()
