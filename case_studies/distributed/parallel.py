"""Tensor and pipeline parallel serving on gloo, with every transfer timed.

Timing model: gloo on CPU executes collectives and point-to-point ops
synchronously, so wall time spent inside `all_reduce` / `send` / `recv` is time
the rank could not compute. We call that "communication" and everything else in
the generate loop "compute". Note what that includes: a rank blocked in `recv`
while its peer is still computing is counted as communication. For TP the ranks
do symmetric work so that wait is small; for PP it *is* the pipeline bubble,
which is exactly the GPU time a showback cannot attach to a token.

Process hygiene: every worker destroys its process group in `finally`, init has
a timeout so a crashed peer cannot hang the others forever, and `mp.spawn` with
join=True terminates surviving ranks if any rank raises.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from case_studies.distributed.model import (
    KVCache,
    ModelConfig,
    Stage,
    Tensor,
    causal_attention,
    greedy_generate,
    init_weights,
    layer_norm,
    prompt_tokens,
    reference_model,
    top2_margin,
)

THREAD_BUDGET = 4
"""Total intra-op threads shared by all ranks of one run.

The dev laptop has 4 performance cores; giving each rank `THREAD_BUDGET //
world_size` threads keeps TP4 from oversubscribing and keeps configs comparable
on a fixed hardware budget, the way W GPUs are a fixed budget.
"""


def threads_per_rank(world_size: int) -> int:
    budget = min(THREAD_BUDGET, os.cpu_count() or 1)
    return max(1, budget // world_size)


# --- timed communication ------------------------------------------------------


@dataclass
class OpStats:
    calls: int = 0
    seconds: float = 0.0
    bytes: int = 0


class TimedComm:
    """Thin wrapper over torch.distributed that accounts every call."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.ops: dict[str, OpStats] = {}

    def _record(self, op: str, tensor: Tensor, t0: float) -> None:
        stats = self.ops.setdefault(op, OpStats())
        stats.calls += 1
        stats.seconds += time.perf_counter() - t0
        stats.bytes += tensor.numel() * tensor.element_size()

    def all_reduce(self, tensor: Tensor) -> Tensor:
        if self.world_size == 1:
            return tensor
        t0 = time.perf_counter()
        dist.all_reduce(tensor)
        self._record("all_reduce", tensor, t0)
        return tensor

    def send(self, tensor: Tensor, dst: int) -> None:
        t0 = time.perf_counter()
        dist.send(tensor, dst)
        self._record("send", tensor, t0)

    def recv(self, tensor: Tensor, src: int) -> Tensor:
        t0 = time.perf_counter()
        dist.recv(tensor, src)
        self._record("recv", tensor, t0)
        return tensor

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.ops.values())

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {op: asdict(s) for op, s in sorted(self.ops.items())}

    def reset(self) -> None:
        self.ops = {}


# --- tensor parallelism ------------------------------------------------------------


def shard_rows(t: Tensor, rank: int, world: int) -> Tensor:
    """Column-parallel: split the output dimension (rows of a Linear weight)."""
    return t.chunk(world, dim=0)[rank].contiguous()


def shard_cols(t: Tensor, rank: int, world: int) -> Tensor:
    """Row-parallel: split the input dimension (columns of a Linear weight)."""
    return t.chunk(world, dim=1)[rank].contiguous()


class TPModel:
    """Megatron-style TP built by slicing the reference weights.

    Attention: the fused QKV projection is split per head block (Q, K and V each
    column-parallel), so rank r owns heads [r*H/W, (r+1)*H/W) and keeps only
    their KV cache. The output projection is row-parallel and followed by an
    all_reduce. MLP: up-projection column-parallel, down-projection row-parallel,
    all_reduce. Embeddings, LayerNorms and the LM head stay replicated, so each
    rank sees full logits and can pick the same greedy token without a gather.
    """

    def __init__(
        self,
        weights: dict[str, Tensor],
        cfg: ModelConfig,
        rank: int,
        world: int,
        comm: TimedComm,
    ):
        if cfg.n_heads % world or cfg.d_ff % world:
            raise ValueError(f"n_heads={cfg.n_heads} and d_ff={cfg.d_ff} must divide by {world}")
        self.cfg, self.rank, self.world, self.comm = cfg, rank, world, comm
        self.heads_local = cfg.n_heads // world
        self.w: dict[str, Tensor] = {}
        for name, t in weights.items():
            if name.endswith("attn.qkv"):
                q, k, v = t.chunk(3, dim=0)
                self.w[name] = torch.cat([shard_rows(x, rank, world) for x in (q, k, v)])
            elif name.endswith("mlp.up"):
                self.w[name] = shard_rows(t, rank, world)
            elif name.endswith(("attn.out", "mlp.down")):
                self.w[name] = shard_cols(t, rank, world)
            else:
                self.w[name] = t
        self.cache: KVCache | None = None

    @property
    def param_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.w.values())

    def reset_cache(self, batch: int, max_len: int) -> None:
        cfg = self.cfg
        self.cache = KVCache(cfg.n_layers, batch, self.heads_local, max_len, cfg.head_dim)

    def forward(self, tokens: Tensor, start: int) -> Tensor:
        assert self.cache is not None, "reset_cache() first"
        w, cfg, comm = self.w, self.cfg, self.comm
        x = w["tok_emb"][tokens] + w["pos_emb"][start : start + tokens.shape[1]]
        bsz, t_new, _ = x.shape
        hd, h_loc = cfg.head_dim, self.heads_local
        d_loc = h_loc * hd
        for i in range(cfg.n_layers):
            p = f"layers.{i}."
            h = layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            q, k, v = F.linear(h, w[p + "attn.qkv"]).split(d_loc, dim=-1)
            q, k, v = (t.view(bsz, t_new, h_loc, hd).transpose(1, 2) for t in (q, k, v))
            k_all, v_all = self.cache.update(i, start, k, v)
            a = causal_attention(q, k_all, v_all, start)
            out = F.linear(a.transpose(1, 2).reshape(bsz, t_new, d_loc), w[p + "attn.out"])
            x = x + comm.all_reduce(out)
            h = layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            out = F.linear(F.gelu(F.linear(h, w[p + "mlp.up"])), w[p + "mlp.down"])
            x = x + comm.all_reduce(out)
        return F.linear(layer_norm(x, w["ln_f.weight"], w["ln_f.bias"]), w["lm_head"])


# --- pipeline parallelism ----------------------------------------------------------


@torch.inference_mode()
def pp_first_stage(
    stage: Stage, comm: TimedComm, prompt: Tensor, n_new: int
) -> tuple[Tensor, float]:
    """Stage 0: embed + first half; ships activations, receives the next token.

    No micro-batching: one request batch flows through the pipe per step, so the
    idle stage waits a full stage-time every step (the naive-PP bubble).
    Returns (tokens, seconds until the first token was back on this rank).
    """
    t0 = time.perf_counter()
    bsz, t_prompt = prompt.shape
    stage.reset_cache(bsz, t_prompt + n_new)
    cur, pos, tokens = prompt, 0, []
    first_token_s = 0.0
    for step in range(n_new):
        hidden = stage.forward(cur, pos)
        comm.send(hidden.contiguous(), 1)
        nxt = comm.recv(torch.empty(bsz, dtype=torch.long), 1)
        if step == 0:
            first_token_s = time.perf_counter() - t0
        tokens.append(nxt)
        pos += cur.shape[1]
        cur = nxt.unsqueeze(1)
    return torch.stack(tokens, dim=1), first_token_s


@torch.inference_mode()
def pp_last_stage(
    stage: Stage, comm: TimedComm, batch: int, t_prompt: int, n_new: int
) -> tuple[Tensor, Tensor]:
    """Stage 1: second half + head; greedy-picks and returns the token upstream."""
    d = stage.cfg.d_model
    stage.reset_cache(batch, t_prompt + n_new)
    pos, tokens, picked = 0, [], []
    for step in range(n_new):
        t_new = t_prompt if step == 0 else 1
        hidden = comm.recv(torch.empty(batch, t_new, d), 0)
        logits = stage.forward(hidden, pos)[:, -1]
        nxt = logits.argmax(dim=-1)
        comm.send(nxt, 0)
        tokens.append(nxt)
        picked.append(logits)
        pos += t_new
    return torch.stack(tokens, dim=1), torch.stack(picked, dim=1)


# --- per-rank worker ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Case:
    batch: int
    prompt_len: int
    new_tokens: int

    @property
    def key(self) -> str:
        return f"b{self.batch}_p{self.prompt_len}_n{self.new_tokens}"


@dataclass(frozen=True, slots=True)
class RunPlan:
    mode: str  # "tp" or "pp"
    world_size: int
    cfg: ModelConfig
    seed: int
    correctness: Case
    cases: tuple[Case, ...]
    warmup: int
    repeats: int
    threads: int
    tmpdir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _generate_on_rank(
    plan: RunPlan, rank: int, model: Any, comm: TimedComm, prompt: Tensor, n_new: int
) -> tuple[Tensor | None, Tensor | None, float]:
    """One generate call in this rank's role. Returns (tokens, logits, first-token s)."""
    if plan.mode == "tp":
        clock: list[float] = []
        tokens, logits = greedy_generate(model, prompt, n_new, clock)
        return tokens, logits, clock[0]
    if rank == 0:
        tokens, first = pp_first_stage(model, comm, prompt, n_new)
        return tokens, None, first
    tokens, logits = pp_last_stage(model, comm, prompt.shape[0], prompt.shape[1], n_new)
    return tokens, logits, float("nan")


def _worker(rank: int, plan: RunPlan, port: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(plan.threads)
    torch.set_num_threads(plan.threads)
    with contextlib.suppress(RuntimeError):  # only settable before parallel work starts
        torch.set_num_interop_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=plan.world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        _run_rank(rank, plan)
    finally:
        dist.destroy_process_group()


def _run_rank(rank: int, plan: RunPlan) -> None:
    cfg, world = plan.cfg, plan.world_size
    weights = init_weights(cfg, plan.seed)
    comm = TimedComm(world)
    model: Any
    if plan.mode == "tp":
        model = TPModel(weights, cfg, rank, world, comm)
    else:
        half = cfg.n_layers // 2
        model = Stage(weights, cfg, 0, half) if rank == 0 else Stage(weights, cfg, half)
    del weights
    result: dict[str, Any] = {"rank": rank, "param_bytes": model.param_bytes}

    # correctness against the parent's single-process reference
    ref = torch.load(Path(plan.tmpdir) / "reference.pt")
    case = plan.correctness
    tokens, logits, _ = _generate_on_rank(plan, rank, model, comm, ref["prompt"], case.new_tokens)
    result["correctness"] = {
        "tokens_match": bool(torch.equal(tokens, ref["tokens"])) if tokens is not None else None,
        "max_abs_logit_diff": (
            float((logits - ref["logits"]).abs().max()) if logits is not None else None
        ),
        "has_logits": logits is not None,
    }
    comm.reset()

    # timing
    timings: dict[str, Any] = {}
    for bench in plan.cases:
        prompt = prompt_tokens(cfg, bench.batch, bench.prompt_len, plan.seed)
        reps = []
        for i in range(plan.warmup + plan.repeats):
            dist.barrier()  # untimed: every rank starts the repeat together
            comm.reset()
            t0 = time.perf_counter()
            _, _, first = _generate_on_rank(plan, rank, model, comm, prompt, bench.new_tokens)
            total = time.perf_counter() - t0
            if i >= plan.warmup:
                reps.append(
                    {
                        "total_s": total,
                        "comm_s": comm.seconds,
                        "first_token_s": first,
                        "ops": comm.snapshot(),
                    }
                )
        timings[bench.key] = {
            "repeats": reps,
            "kv_cache_bytes": model.cache.nbytes if model.cache is not None else 0,
        }
        comm.reset()
    result["timings"] = timings
    (Path(plan.tmpdir) / f"rank{rank}.json").write_text(json.dumps(result))


# --- launcher ----------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch(plan: RunPlan) -> list[dict[str, Any]]:
    """Run `plan` on `world_size` spawned ranks and return each rank's result.

    The parent computes the single-process reference first (with the full thread
    budget) and hands it to the ranks as a file, so correctness is judged
    against one canonical output rather than a per-rank recomputation.
    """
    cfg, case = plan.cfg, plan.correctness
    with tempfile.TemporaryDirectory(prefix="unalloc-dist-") as tmp:
        prompt = prompt_tokens(cfg, case.batch, case.prompt_len, plan.seed)
        ref_model = reference_model(init_weights(cfg, plan.seed), cfg)
        tokens, logits = _with_threads(
            THREAD_BUDGET, greedy_generate, ref_model, prompt, case.new_tokens
        )
        torch.save(
            {"prompt": prompt, "tokens": tokens, "logits": logits}, Path(tmp) / "reference.pt"
        )
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        mp.spawn(
            _worker,
            args=(replace(plan, tmpdir=tmp), free_port()),
            nprocs=plan.world_size,
            join=True,
        )
        results = [
            json.loads((Path(tmp) / f"rank{r}.json").read_text()) for r in range(plan.world_size)
        ]
    margin = top2_margin(logits)
    for r in results:
        r["reference_top2_margin"] = margin
    return results


def _with_threads(n: int, fn: Callable[..., Any], *args: Any) -> Any:
    before = torch.get_num_threads()
    torch.set_num_threads(min(n, os.cpu_count() or 1))
    try:
        return fn(*args)
    finally:
        torch.set_num_threads(before)


def median(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return statistics.median(clean) if clean else float("nan")
