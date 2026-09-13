"""A tiny decoder-only transformer with an explicit KV cache, written from scratch.

The point of this model is not quality, it is *shape*. Serving cost on real
hardware is set by how many tokens are prefilled, how many decode steps run,
and how many key/value bytes are held while they run. All three depend only on
the architecture and the request shape, never on the weight values: a dense
matmul over random numbers takes the same time and memory as one over trained
numbers. So the weights here are random with a fixed seed, and the model is
still a faithful stand-in for "what does serving this request cost".

Architecture: byte-level vocab (256), pre-norm RMSNorm blocks, rotary position
embeddings, multi-head attention through `scaled_dot_product_attention`, and a
GELU MLP. The KV cache is preallocated per layer and filled through a write
index, which is how production servers (and OpenCost's KV-cache-corrected
inference pricing) think about memory: a reservation plus an occupancy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 1280
    rope_base: float = 10_000.0
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def bytes_per_element(self) -> int:
        return torch.empty((), dtype=self.torch_dtype).element_size()

    @property
    def kv_bytes_per_token(self) -> int:
        """2 (K and V) x layers x heads x head_dim x bytes, for one token of context."""
        return 2 * self.n_layers * self.n_heads * self.head_dim * self.bytes_per_element

    def kv_bytes(self, seq_len: int) -> int:
        """Analytic KV cache size for `seq_len` tokens of context."""
        return self.kv_bytes_per_token * seq_len

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = asdict(self)
        out["head_dim"] = self.head_dim
        out["kv_bytes_per_token"] = self.kv_bytes_per_token
        return out


class KVCache:
    """Preallocated per-layer key/value tensors plus a write index.

    `capacity` is the reservation (what a static allocator holds for the whole
    request); `length` is the occupancy (what a paged allocator such as vLLM's
    would actually have mapped). Cost attribution can use either, so both are
    exposed.
    """

    def __init__(self, cfg: ModelConfig, capacity: int, batch: int = 1) -> None:
        if capacity > cfg.max_seq_len:
            raise ValueError(f"capacity {capacity} exceeds max_seq_len {cfg.max_seq_len}")
        shape = (batch, cfg.n_heads, capacity, cfg.head_dim)
        self.cfg = cfg
        self.capacity = capacity
        self.keys = [torch.zeros(shape, dtype=cfg.torch_dtype) for _ in range(cfg.n_layers)]
        self.values = [torch.zeros(shape, dtype=cfg.torch_dtype) for _ in range(cfg.n_layers)]
        self.length = 0

    @property
    def nbytes(self) -> int:
        """Measured bytes of the reserved tensors."""
        return sum(t.nbytes for t in (*self.keys, *self.values))

    @property
    def used_bytes(self) -> int:
        """Bytes occupied by tokens actually written so far."""
        return self.cfg.kv_bytes(self.length)

    def reset(self, length: int = 0) -> None:
        """Rewind the write index. Stale entries past it are never attended."""
        self.length = length

    def load_prefix(self, prefix: KVCache) -> None:
        """Copy a precomputed shared prefix (e.g. a system prompt) into this cache."""
        n = prefix.length
        if n > self.capacity:
            raise ValueError("prefix longer than cache capacity")
        for layer in range(self.cfg.n_layers):
            self.keys[layer][:, :, :n].copy_(prefix.keys[layer][:, :, :n])
            self.values[layer][:, :, :n].copy_(prefix.values[layer][:, :, :n])
        self.length = n


def _rope_tables(cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
    angles = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float64), inv_freq)
    return angles.cos().to(cfg.torch_dtype), angles.sin().to(cfg.torch_dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE. x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: KVCache | None,
        start: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(cfg.d_model, dim=-1)
        q = q.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if cache is not None:
            cache.keys[layer][:, :, start : start + t] = k
            cache.values[layer][:, :, start : start + t] = v
            k = cache.keys[layer][:, :, : start + t]
            v = cache.values[layer][:, :, : start + t]

        if start == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:
            # A single new query may attend to every cached position.
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Suffix prefill after a cached prefix. SDPA's is_causal is top-left
            # aligned for non-square masks, which would be wrong here, so the
            # bottom-right causal mask is built explicitly.
            rows = torch.arange(t).unsqueeze(1) + start
            cols = torch.arange(start + t).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=cols <= rows)
        return self.proj(y.transpose(1, 2).reshape(b, t, cfg.d_model))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: KVCache | None,
        start: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, layer, cache, start)
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    """Decoder-only transformer. Weight values are random and irrelevant to cost."""

    def __init__(self, cfg: ModelConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = _rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights(seed)
        self.to(cfg.torch_dtype)
        self.eval()

    @torch.no_grad()
    def _init_weights(self, seed: int) -> None:
        # A private generator keeps the global RNG untouched, so building the
        # model never perturbs the trace generator or anything else seeded.
        gen = torch.Generator().manual_seed(seed)
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                param.normal_(0.0, 0.02, generator=gen)
                if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                    param.mul_(1.0 / math.sqrt(2 * self.cfg.n_layers))
            else:
                param.fill_(1.0)

    def n_params(self, *, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel() + self.head.weight.numel()
        return total

    def forward(
        self,
        tokens: torch.Tensor,
        cache: KVCache | None = None,
        *,
        last_only: bool = True,
    ) -> torch.Tensor:
        """Run `tokens` (batch, seq). With a cache, positions continue from its write index.

        Returns logits for the last position (batch, vocab) when `last_only`,
        otherwise for every position (batch, seq, vocab).
        """
        t = tokens.shape[1]
        start = cache.length if cache is not None else 0
        if start + t > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if cache is not None and start + t > cache.capacity:
            raise ValueError("sequence exceeds cache capacity")
        cos = self.rope_cos[start : start + t]
        sin = self.rope_sin[start : start + t]
        x = self.embed(tokens)
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, layer, cache, start)
        if cache is not None:
            cache.length = start + t
        if last_only:
            x = x[:, -1]
        return self.head(self.norm(x))


# --- greedy decoding ------------------------------------------------------------
# Greedy with a fixed number of new tokens and no EOS stop: completion length is
# a property of the request, not of whatever a random model happens to emit.


@torch.inference_mode()
def decode_cached(
    model: TinyDecoder,
    prompt: torch.Tensor,
    n_new: int,
    *,
    prefix: KVCache | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Prefill once, then one single-token forward per new token.

    With `prefix`, the first `prefix.length` tokens of `prompt` are taken from
    that precomputed cache and only the suffix is prefilled.
    """
    cache = KVCache(model.cfg, capacity=prompt.shape[1] + n_new)
    start = 0
    if prefix is not None:
        cache.load_prefix(prefix)
        start = prefix.length
    logits = model(prompt[:, start:], cache)
    out: list[int] = []
    steps: list[torch.Tensor] = []
    for i in range(n_new):
        steps.append(logits[0].clone())
        token = int(logits[0].argmax())
        out.append(token)
        if i < n_new - 1:
            logits = model(torch.tensor([[token]]), cache)
    return out, torch.stack(steps)


@torch.inference_mode()
def decode_uncached(
    model: TinyDecoder, prompt: torch.Tensor, n_new: int
) -> tuple[list[int], torch.Tensor]:
    """Naive decoding: recompute attention over the whole prefix at every step."""
    seq = prompt.clone()
    out: list[int] = []
    steps: list[torch.Tensor] = []
    for _ in range(n_new):
        logits = model(seq, None)
        steps.append(logits[0].clone())
        token = int(logits[0].argmax())
        out.append(token)
        seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
    return out, torch.stack(steps)


def equivalence_check(
    model: TinyDecoder,
    *,
    prompt_len: int = 48,
    prefix_len: int = 16,
    n_new: int = 24,
    seed: int = 0,
    tol: float = 1e-4,
) -> dict[str, object]:
    """Cached, prefix-reused and uncached decoding must agree.

    If they did not, every cost number downstream would be measuring a
    different computation than the one it claims to measure.
    """
    gen = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), generator=gen)
    ref_tokens, ref_logits = decode_uncached(model, prompt, n_new)
    tokens, logits = decode_cached(model, prompt, n_new)

    prefix = KVCache(model.cfg, capacity=prefix_len)
    with torch.inference_mode():
        model(prompt[:, :prefix_len], prefix)
    pre_tokens, pre_logits = decode_cached(model, prompt, n_new, prefix=prefix)

    cached_diff = float((logits - ref_logits).abs().max())
    prefix_diff = float((pre_logits - ref_logits).abs().max())
    result = {
        "prompt_len": prompt_len,
        "prefix_len": prefix_len,
        "n_new": n_new,
        "tolerance": tol,
        "cached_tokens_equal": tokens == ref_tokens,
        "cached_max_abs_logit_diff": cached_diff,
        "prefix_reuse_tokens_equal": pre_tokens == ref_tokens,
        "prefix_reuse_max_abs_logit_diff": prefix_diff,
    }
    result["passed"] = bool(
        result["cached_tokens_equal"]
        and result["prefix_reuse_tokens_equal"]
        and cached_diff <= tol
        and prefix_diff <= tol
    )
    return result
