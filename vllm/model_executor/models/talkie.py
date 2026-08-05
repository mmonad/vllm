# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Talkie 1930 13B model.

Reference HF implementation:
https://huggingface.co/lewtun/talkie-1930-13b-it-hf

Architectural notes:
- Decoder-only transformer with full multi-head attention (40 heads, no GQA),
  bf16 weights, head_dim 128, hidden 5120, intermediate 13696, ctx 2048.
- RoPE rotates by ``-θ`` (sign of sin flipped vs the standard NeoX
  convention), so we apply it inline rather than reuse vLLM's kernels.
- Parameter-free RMSNorm everywhere (input pre-norm, QK-norm, final norm).
- Per-head learnable gain on Q after QK-norm.
- Per-layer learnable scalars scale the attention output, the MLP output,
  and an embedding-skip path that adds a scaled copy of the initial
  post-norm token embedding to every block's residual stream.
- ``lm_head`` is stored as a bare ``nn.Parameter`` (no ``.weight`` suffix)
  with a global learnable scalar gain.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free RMSNorm, matching the HF reference exactly.

    The reference calls ``F.rms_norm(x, (x.shape[-1],))`` with no explicit
    ``eps`` — PyTorch then uses the dtype-dependent default. Pinning ``eps``
    to a fixed value would shift QK-norm output for low-magnitude bf16
    activations, breaking numerical fidelity to the published checkpoint.
    """
    return F.rms_norm(x, (x.shape[-1],))


class _TalkieRotaryEmbedding(nn.Module):
    """RoPE with Talkie's negative-θ rotation direction.

    Reference applies ``y1 = x1*cos + x2*sin`` and ``y2 = x2*cos - x1*sin``
    over a half-then-half split of the head dimension, which is the standard
    NeoX layout but rotated by ``-θ`` instead of ``+θ``. The cos/sin tables
    are sized at construction to the actual deployed context length passed
    in by ``TalkieModel`` (``max(config.max_position_embeddings,
    vllm_config.model_config.max_model_len)``) so a runtime branch isn't
    needed inside the compiled forward path.
    """

    def __init__(self, head_dim: int, max_position: int, base: float):
        super().__init__()
        self.head_dim = head_dim

        channels = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channels / head_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q/k: (num_tokens, num_heads, head_dim)
        cos = self.cos.index_select(0, positions).to(q.dtype)
        sin = self.sin.index_select(0, positions).to(q.dtype)
        # Broadcast across heads: (num_tokens, 1, head_dim/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        d = q.shape[-1] // 2

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., :d]
            x2 = x[..., d:]
            y1 = x1 * cos + x2 * sin
            y2 = x2 * cos - x1 * sin
            return torch.cat([y1, y2], dim=-1)

        return _rotate(q), _rotate(k)


class TalkieAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        max_position: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        # Talkie is full MHA — KV heads track attention heads.
        self.total_num_kv_heads = self.total_num_heads
        self.num_kv_heads = self.num_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = _TalkieRotaryEmbedding(
            head_dim=self.head_dim,
            max_position=max_position or config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Per-head Q gain — sharded along heads to match TP slicing of Q.
        self.head_gain = nn.Parameter(torch.ones(self.num_heads))

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)

        # Reference order (modeling_talkie.py): RoPE first, then RMSNorm on
        # Q/K, then per-head gain on Q. Reversing this changes the logits.
        q, k = self.rotary_emb(positions, q, k)
        q = _rms_norm(q)
        k = _rms_norm(k)
        q = q * self.head_gain.to(q.dtype).view(1, -1, 1)

        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class TalkieMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        out, _ = self.down_proj(x)
        return out


class TalkieDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        max_position: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = TalkieAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            max_position=max_position,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = TalkieMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Per-layer gains (scalars).
        self.attn_gain = nn.Parameter(torch.ones(1))
        self.mlp_gain = nn.Parameter(torch.ones(1))
        self.embed_skip_gain = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        e_x: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = (
            residual + self.attn_gain.to(hidden_states.dtype) * hidden_states
        )

        residual = hidden_states
        hidden_states = _rms_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_gain.to(hidden_states.dtype) * hidden_states

        hidden_states = (
            hidden_states + self.embed_skip_gain.to(hidden_states.dtype) * e_x
        )
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class TalkieModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False)
            and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Size the RoPE cache to whichever is larger: the config's native
        # window or the deployed vLLM context. Avoids OOB ``index_select``
        # when a user passes ``--max-model-len`` above the trained ctx.
        max_position = max(
            getattr(config, "max_position_embeddings", 0),
            getattr(vllm_config.model_config, "max_model_len", 0) or 0,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: TalkieDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                max_position=max_position,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "e_x"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = _rms_norm(hidden_states)
            e_x = hidden_states
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            e_x = intermediate_tensors["e_x"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states, e_x)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "e_x": e_x})

        hidden_states = _rms_norm(hidden_states)
        return hidden_states


class TalkieForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["attn_query", "attn_key", "attn_value"],
        "gate_up_proj": ["mlp_gate", "mlp_linear"],
    }

    # Remap the upstream HF parameter names onto vLLM's standard naming so the
    # qkv / gate-up packing logic and ParallelLMHead can find their tensors.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.blocks.": "model.layers.",
            ".attn.attn_query.": ".self_attn.attn_query.",
            ".attn.attn_key.": ".self_attn.attn_key.",
            ".attn.attn_value.": ".self_attn.attn_value.",
            ".attn.attn_resid.": ".self_attn.o_proj.",
            ".attn.head_gain.head_g": ".self_attn.head_gain",
            ".mlp.mlp_resid.": ".mlp.down_proj.",
            ".mlp.mlp_gate.": ".mlp.mlp_gate.",
            ".mlp.mlp_linear.": ".mlp.mlp_linear.",
            ".attn_gain.a_g": ".attn_gain",
            ".mlp_gain.a_g": ".mlp_gain",
            ".embed_skip.a_g": ".embed_skip_gain",
            "lm_head_gain.w_g": "lm_head_gain",
        },
        orig_to_new_prefix={
            "model.embed.": "model.embed_tokens.",
        },
        orig_to_new_regex={},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config

        self.model = TalkieModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if getattr(config, "tie_word_embeddings", False):
                # Tied case: lm_head reuses the input embedding. The bare
                # ``lm_head`` weight in the checkpoint, if present, is then
                # ignored via ``skip_prefixes`` semantics in load_weights.
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.lm_head_gain = nn.Parameter(torch.ones(1))
        else:
            self.lm_head = PPMissingLayer()
            self.lm_head_gain = None

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Apply the global lm_head gain as a pre-scale on the hidden states,
        # which is equivalent to multiplying the lm_head weight by w_g.
        gain = self.lm_head_gain.to(hidden_states.dtype)
        return self.logits_processor(self.lm_head, hidden_states * gain)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The bare ``lm_head`` parameter (no ``.weight``) needs to land on
        # ``lm_head.weight`` of ParallelLMHead. When weights are tied, drop
        # the checkpoint's lm_head entirely so it doesn't overwrite the
        # embedding tensor that ``self.lm_head`` now aliases.
        tie = getattr(self.config, "tie_word_embeddings", False)

        def _rename_lm_head(stream):
            for name, w in stream:
                if name == "lm_head":
                    if tie:
                        continue
                    yield "lm_head.weight", w
                else:
                    yield name, w

        # head_gain is per-head (40,) and must be sharded along TP.
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        def _shard_head_gain(stream):
            for name, w in stream:
                if name.endswith(".self_attn.head_gain") and tp_size > 1:
                    n_per_rank = w.shape[0] // tp_size
                    w = w[tp_rank * n_per_rank : (tp_rank + 1) * n_per_rank]
                yield name, w

        weights = _shard_head_gain(
            self.hf_to_vllm_mapper.apply(_rename_lm_head(weights))
        )

        # Standard QKV / gate-up packing. Subsequent loads are 1:1 by name.
        stacked_params_mapping = [
            ("qkv_proj", "attn_query", "q"),
            ("qkv_proj", "attn_key", "k"),
            ("qkv_proj", "attn_value", "v"),
            ("gate_up_proj", "mlp_gate", 0),
            ("gate_up_proj", "mlp_linear", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, w in weights:
            matched = False
            for param_name, src_name, shard_id in stacked_params_mapping:
                if src_name not in name:
                    continue
                mapped = name.replace(src_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = param.weight_loader
                weight_loader(param, w, shard_id)
                loaded.add(mapped)
                matched = True
                break
            if matched:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)
        return loaded
