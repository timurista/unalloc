"""Reference decoder-only transformer with a KV cache.

Written as plain tensor code rather than nn.Module layers for two reasons: the
tensor-parallel and pipeline variants have to slice *exactly* these weights, and
generating the weights from an explicit torch.Generator keeps every process in a
spawn group bit-identical without shipping state dicts between them.

A `Stage` holds a contiguous range of layers, optionally with the embedding
(first stage) and the LM head (last stage). The single-process reference is just
a stage holding everything, which is what makes the pipeline split a pure
re-partition of the reference rather than a second implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    vocab: int = 256
    max_seq: int = 512
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return self.d_model * self.mlp_ratio

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "head_dim": self.head_dim, "d_ff": self.d_ff}


TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, vocab=64, max_seq=96)
"""Config for tests: same code paths, a fraction of the compute."""


def init_weights(cfg: ModelConfig, seed: int) -> dict[str, Tensor]:
    """Deterministic float32 weights.

    Linear weights use std 1/sqrt(fan_in) so the final logits have O(1) spread;
    with the usual 0.02 init the top-2 logit margin of an untrained model is so
    small that float noise from sharding could flip a greedy token and make the
    correctness check meaningless.
    """
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    gen = torch.Generator().manual_seed(seed)

    def normal(*shape: int, std: float) -> Tensor:
        return torch.randn(*shape, generator=gen) * std

    d, ff = cfg.d_model, cfg.d_ff
    weights: dict[str, Tensor] = {
        "tok_emb": normal(cfg.vocab, d, std=1.0),
        "pos_emb": normal(cfg.max_seq, d, std=0.1),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        weights[p + "ln1.weight"] = 1.0 + normal(d, std=0.05)
        weights[p + "ln1.bias"] = normal(d, std=0.05)
        weights[p + "attn.qkv"] = normal(3 * d, d, std=d**-0.5)
        weights[p + "attn.out"] = normal(d, d, std=d**-0.5)
        weights[p + "ln2.weight"] = 1.0 + normal(d, std=0.05)
        weights[p + "ln2.bias"] = normal(d, std=0.05)
        weights[p + "mlp.up"] = normal(ff, d, std=d**-0.5)
        weights[p + "mlp.down"] = normal(d, ff, std=ff**-0.5)
    weights["ln_f.weight"] = 1.0 + normal(d, std=0.05)
    weights["ln_f.bias"] = normal(d, std=0.05)
    weights["lm_head"] = normal(cfg.vocab, d, std=d**-0.5)
    return weights


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-5)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, start: int) -> Tensor:
    """Attention of T new queries over `start + T` cached keys.

    q: [B, H, T, hd]; k, v: [B, H, start + T, hd]. Written out rather than via
    scaled_dot_product_attention so the reference and the shards provably run
    the same kernel sequence per head.
    """
    t_new, t_all = q.shape[2], k.shape[2]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if t_new > 1:
        query_pos = torch.arange(start, start + t_new).unsqueeze(1)
        key_pos = torch.arange(t_all).unsqueeze(0)
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class KVCache:
    """Preallocated per-layer K and V buffers: [B, heads_local, max_len, head_dim]."""

    def __init__(self, n_layers: int, batch: int, heads: int, max_len: int, head_dim: int):
        shape = (batch, heads, max_len, head_dim)
        self.k = [torch.zeros(shape) for _ in range(n_layers)]
        self.v = [torch.zeros(shape) for _ in range(n_layers)]

    def update(self, layer: int, start: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        end = start + k.shape[2]
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.k, *self.v))


class Decoder(Protocol):
    """What greedy decoding needs from a model, sharded or not."""

    def reset_cache(self, batch: int, max_len: int) -> None: ...

    def forward(self, tokens: Tensor, start: int) -> Tensor: ...


class Stage:
    """Layers [lo, hi) of the reference model, plus embedding/head if at an end."""

    def __init__(
        self,
        weights: dict[str, Tensor],
        cfg: ModelConfig,
        lo: int = 0,
        hi: int | None = None,
    ):
        self.cfg = cfg
        self.lo, self.hi = lo, cfg.n_layers if hi is None else hi
        self.first = self.lo == 0
        self.last = self.hi == cfg.n_layers
        keep = [f"layers.{i}." for i in range(self.lo, self.hi)]
        self.w: dict[str, Tensor] = {}
        for name, tensor in weights.items():
            if name.startswith("layers."):
                if any(name.startswith(p) for p in keep):
                    self.w[name] = tensor
            elif name in ("tok_emb", "pos_emb"):
                if self.first:
                    self.w[name] = tensor
            elif self.last:
                self.w[name] = tensor
        self.cache: KVCache | None = None

    @property
    def param_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.w.values())

    def reset_cache(self, batch: int, max_len: int) -> None:
        cfg = self.cfg
        self.cache = KVCache(self.hi - self.lo, batch, cfg.n_heads, max_len, cfg.head_dim)

    def forward(self, x: Tensor, start: int) -> Tensor:
        """Token ids in (first stage) or hidden states; logits or hidden states out."""
        assert self.cache is not None, "reset_cache() first"
        w, cfg = self.w, self.cfg
        if self.first:
            x = w["tok_emb"][x] + w["pos_emb"][start : start + x.shape[1]]
        bsz, t_new, d = x.shape
        hd = cfg.head_dim
        for slot, i in enumerate(range(self.lo, self.hi)):
            p = f"layers.{i}."
            h = layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            q, k, v = F.linear(h, w[p + "attn.qkv"]).split(d, dim=-1)
            q, k, v = (t.view(bsz, t_new, cfg.n_heads, hd).transpose(1, 2) for t in (q, k, v))
            k_all, v_all = self.cache.update(slot, start, k, v)
            a = causal_attention(q, k_all, v_all, start)
            x = x + F.linear(a.transpose(1, 2).reshape(bsz, t_new, d), w[p + "attn.out"])
            h = layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            x = x + F.linear(F.gelu(F.linear(h, w[p + "mlp.up"])), w[p + "mlp.down"])
        if self.last:
            x = F.linear(layer_norm(x, w["ln_f.weight"], w["ln_f.bias"]), w["lm_head"])
        return x


def reference_model(weights: dict[str, Tensor], cfg: ModelConfig) -> Stage:
    return Stage(weights, cfg)


def prompt_tokens(cfg: ModelConfig, batch: int, length: int, seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed + 7919)
    return torch.randint(0, cfg.vocab, (batch, length), generator=gen)


@torch.inference_mode()
def greedy_generate(
    model: Decoder, prompt: Tensor, n_new: int, clock: list[float] | None = None
) -> tuple[Tensor, Tensor]:
    """Prefill once, then decode one token per step against the KV cache.

    Returns (token ids [B, n_new], the logits each token was picked from
    [B, n_new, vocab]) so a sharded run can be compared logit-for-logit. If
    `clock` is given, the time to the first token is appended to it.
    """
    t0 = time.perf_counter()
    bsz, t_prompt = prompt.shape
    model.reset_cache(bsz, t_prompt + n_new)
    logits = model.forward(prompt, 0)[:, -1]
    tokens, picked = [], []
    for step in range(n_new):
        nxt = logits.argmax(dim=-1)
        if step == 0 and clock is not None:
            clock.append(time.perf_counter() - t0)
        tokens.append(nxt)
        picked.append(logits)
        if step == n_new - 1:
            break
        logits = model.forward(nxt.unsqueeze(1), t_prompt + step)[:, -1]
    return torch.stack(tokens, dim=1), torch.stack(picked, dim=1)


def top2_margin(logits: Tensor) -> float:
    """Smallest gap between the best and second-best logit over all steps.

    Reported next to the max abs logit diff: a token match only means something
    if the margin is much larger than the numerical noise.
    """
    top = logits.topk(2, dim=-1).values
    return float((top[..., 0] - top[..., 1]).min())
