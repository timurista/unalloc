"""A discrete-event model of a vLLM-style serving engine, instrumented for cost.

What is modelled, because it moves cost between tenants:

* **Paged KV cache.** Fixed-size blocks drawn from a finite GPU pool.
* **Automatic prefix caching.** Full blocks are content-addressed; a request
  whose prompt starts with already-computed blocks reuses them, and freed blocks
  stay cached (evictable, LRU) until the pool needs them.
* **Continuous batching with chunked prefill.** One token budget per engine step
  shared by decodes first, then prefills.
* **Recompute preemption.** When decode growth exhausts the pool, the most
  recently admitted request is evicted and later recomputed.

What is not modelled, because it only scales absolute numbers: kernels, real
latency variance, speculative decoding, multi-LoRA. Step latency is an analytic
function of the batch (see `StepCost`), with order-of-magnitude constants for
an 8B model on one H100-class GPU. The study compares *shares*, which are
insensitive to those constants.

Every engine step's duration is split across the sequences in it
(`Request.compute_s`), and every block-second of KV memory is split across the
requests holding that block (`Request.kv_block_s`). Idle GPU time and unheld
blocks are what no per-request meter can see; they are tracked separately.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

# --- configuration ------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str = "llama-3.1-8b"
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_element: int = 2  # fp16 / bf16

    @property
    def kv_bytes_per_token(self) -> int:
        # K and V, per layer, per KV head (GQA), per head dim.
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element


@dataclass(frozen=True)
class StepCost:
    """Step latency = base + prefill tokens + decode sequences + attended context."""

    base_s: float = 0.004
    per_prefill_token_s: float = 0.00006
    per_decode_seq_s: float = 0.0003
    per_context_token_s: float = 0.00000012


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 12_000
    max_num_seqs: int = 96
    max_batched_tokens: int = 2048
    prefix_caching: bool = True
    watermark: float = 0.01
    """Fraction of blocks kept free at admission so decodes can grow."""


@dataclass(frozen=True)
class TenantSpec:
    name: str
    team: str | None
    """Owner label as it would appear on the showback row. None = unlabeled."""
    weight: float
    """Share of request (or session) starts."""
    prompt: tuple[int, int]
    output: tuple[int, int]
    system_prefix: int = 0
    """Tokens of shared system prompt at the start of every prompt."""
    prefix_groups: int = 1
    """Distinct system prompts in use, e.g. one per product surface."""
    turns: tuple[int, int] | None = None
    """Multi-turn sessions: each turn re-sends the whole conversation so far."""
    think_s: tuple[float, float] = (1.0, 5.0)


TENANTS: tuple[TenantSpec, ...] = (
    # RAG: long retrieved context, short answers, a few shared instruction prompts.
    TenantSpec("search", "search", 0.40, prompt=(600, 1400), output=(16, 64),
               system_prefix=512, prefix_groups=3),
    # Agent loops: short new messages, long outputs, the conversation re-sent every turn.
    TenantSpec("agents", "agents", 0.20, prompt=(80, 240), output=(150, 400),
               system_prefix=1024, turns=(3, 8)),
    TenantSpec("platform", "platform", 0.25, prompt=(200, 600), output=(64, 256),
               system_prefix=256, prefix_groups=2),
    # A shared playground key nobody owns.
    TenantSpec("sandbox", None, 0.15, prompt=(50, 2000), output=(10, 500)),
)


# --- requests ---------------------------------------------------------------------------


@dataclass
class Session:
    sid: int
    tenant: TenantSpec
    group: int
    turns_left: int
    history: int = 0


@dataclass
class Request:
    rid: int
    tenant: TenantSpec
    arrival: float
    prompt_len: int
    output_len: int
    group: int
    session: Session | None = None

    computed: int = 0
    generated: int = 0
    blocks: list[int] = field(default_factory=list)
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    cached_tokens: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    compute_s: float = 0.0
    kv_block_s: float = 0.0

    @property
    def shareable_len(self) -> int:
        """Tokens whose content another request could have computed identically."""
        if self.session is not None:
            return self.prompt_len + self.output_len
        return min(self.prompt_len, self.tenant.system_prefix)

    def block_key(self, index: int, block_size: int) -> tuple[Any, ...] | None:
        end = (index + 1) * block_size
        if end <= self.tenant.system_prefix:
            return ("sys", self.tenant.name, self.group, index)
        if self.session is not None and end <= self.shareable_len:
            return ("conv", self.session.sid, index)
        return None

    @property
    def prefill_target(self) -> int:
        """KV that must exist before the next decode step can run."""
        if self.generated == 0:
            return self.prompt_len
        return self.prompt_len + self.generated - 1

    @property
    def in_decode(self) -> bool:
        return self.computed >= self.prefill_target and self.generated > 0


# --- block manager ----------------------------------------------------------------------


class BlockManager:
    """Ref-counted, content-addressed KV blocks with time-integrated ownership."""

    def __init__(self, num_blocks: int, caching: bool) -> None:
        self.num_blocks = num_blocks
        self.caching = caching
        self.ref = [0] * num_blocks
        self.key: list[tuple[Any, ...] | None] = [None] * num_blocks
        self.holders: list[list[Request]] = [[] for _ in range(num_blocks)]
        self.since = [0.0] * num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.cached: dict[tuple[Any, ...], int] = {}
        self.evictions = 0
        self.cache_reserve_block_s = 0.0

    def available(self) -> int:
        return len(self.free) + len(self.evictable)

    def _settle(self, block: int, now: float) -> None:
        elapsed = now - self.since[block]
        if elapsed > 0:
            holders = self.holders[block]
            if holders:
                share = elapsed / len(holders)
                for req in holders:
                    req.kv_block_s += share
            elif self.key[block] is not None:
                self.cache_reserve_block_s += elapsed
        self.since[block] = now

    def match_prefix(self, req: Request, tokens: int, block_size: int) -> list[int]:
        if not self.caching:
            return []
        matched: list[int] = []
        for index in range(tokens // block_size):
            key = req.block_key(index, block_size)
            block = self.cached.get(key) if key is not None else None
            if block is None:
                break
            matched.append(block)
        # At least one token must be computed to produce logits.
        while matched and len(matched) * block_size >= tokens:
            matched.pop()
        return matched

    def acquire(self, block: int, req: Request, now: float) -> None:
        self._settle(block, now)
        if self.ref[block] == 0:
            self.evictable.pop(block, None)
        self.ref[block] += 1
        self.holders[block].append(req)

    def take(self, req: Request, now: float) -> int:
        if self.free:
            block = self.free.popleft()
        else:
            block, _ = self.evictable.popitem(last=False)
            self._settle(block, now)
            del self.cached[self.key[block]]  # type: ignore[arg-type]
            self.key[block] = None
            self.evictions += 1
        self._settle(block, now)
        self.ref[block] = 1
        self.holders[block].append(req)
        return block

    def release(self, block: int, req: Request, now: float) -> None:
        self._settle(block, now)
        self.ref[block] -= 1
        self.holders[block].remove(req)
        if self.ref[block] == 0:
            if self.key[block] is not None:
                self.evictable[block] = None
            else:
                self.free.append(block)

    def register(self, block: int, key: tuple[Any, ...] | None) -> None:
        if not self.caching or key is None or self.key[block] is not None:
            return
        if key not in self.cached:
            self.cached[key] = block
            self.key[block] = key

    def close(self, now: float) -> None:
        for block in range(self.num_blocks):
            self._settle(block, now)


# --- engine ---------------------------------------------------------------------------------


@dataclass
class RunResult:
    config: dict[str, Any]
    requests: list[Request]
    wall_s: float
    idle_s: float
    steps: int
    evictions: int
    cache_reserve_block_s: float
    dropped: int
    tokens_generated: int


def _uniform(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


class Engine:
    def __init__(
        self,
        engine: EngineConfig,
        cost: StepCost = StepCost(),
        model: ModelSpec = ModelSpec(),
    ) -> None:
        self.cfg = engine
        self.cost = cost
        self.model = model
        self.blocks = BlockManager(engine.num_blocks, engine.prefix_caching)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.done: list[Request] = []

    # -- block helpers --

    def _blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.cfg.block_size)

    def _register_full_blocks(self, req: Request, before: int) -> None:
        size = self.cfg.block_size
        for index in range(before // size, req.computed // size):
            self.blocks.register(req.blocks[index], req.block_key(index, size))

    def _free(self, req: Request, now: float) -> None:
        # Reverse order: tail blocks are the least likely to be shared, so they
        # should be the first evicted.
        for block in reversed(req.blocks):
            self.blocks.release(block, req, now)
        req.blocks = []

    def _preempt(self, victim: Request, now: float) -> None:
        self.running.remove(victim)
        victim.preemptions += 1
        victim.recomputed_tokens += victim.computed
        self._free(victim, now)
        victim.computed = 0
        self.waiting.appendleft(victim)

    def _try_admit(self, req: Request, now: float) -> bool:
        target = req.prefill_target
        needed_total = self._blocks_for(target)
        if needed_total > self.cfg.num_blocks * (1 - self.cfg.watermark):
            raise ValueError("request larger than the KV pool")
        matched = self.blocks.match_prefix(req, target, self.cfg.block_size)
        reclaimed = sum(1 for b in matched if self.blocks.ref[b] == 0)
        new = needed_total - len(matched)
        headroom = self.blocks.available() - reclaimed - new
        if headroom < self.cfg.num_blocks * self.cfg.watermark:
            return False
        for block in matched:
            self.blocks.acquire(block, req, now)
        req.blocks = list(matched)
        for _ in range(new):
            req.blocks.append(self.blocks.take(req, now))
        req.computed = len(matched) * self.cfg.block_size
        req.cached_tokens += req.computed
        req.recomputed_tokens = max(0, req.recomputed_tokens - req.computed)
        if req.admitted_at is None:
            req.admitted_at = now
        return True

    # -- one engine step --

    def step(self, now: float) -> float:
        budget = self.cfg.max_batched_tokens
        work: list[tuple[Request, int]] = []

        for req in list(self.running):
            if budget <= 0:
                break
            if req not in self.running or not req.in_decode:
                continue
            if req.computed + 1 > len(req.blocks) * self.cfg.block_size:
                while self.blocks.available() == 0:
                    victim = self.running[-1]
                    self._preempt(victim, now)
                    if victim is req:
                        break
                if req not in self.running:
                    continue
                req.blocks.append(self.blocks.take(req, now))
            work.append((req, 1))
            budget -= 1

        for req in self.running:
            if budget <= 0:
                break
            if not req.in_decode:
                chunk = min(req.prefill_target - req.computed, budget)
                if chunk > 0:
                    work.append((req, chunk))
                    budget -= chunk

        while (
            self.waiting
            and budget > 0
            and len(self.running) < self.cfg.max_num_seqs
            and self.waiting[0].arrival <= now
        ):
            req = self.waiting[0]
            if not self._try_admit(req, now):
                break
            self.waiting.popleft()
            self.running.append(req)
            chunk = min(req.prefill_target - req.computed, budget)
            if chunk > 0:
                work.append((req, chunk))
                budget -= chunk

        if not work:
            return 0.0

        weights: list[float] = []
        prefill_tokens = 0
        decodes = 0
        context = 0
        for req, tokens in work:
            if req.in_decode:
                decodes += 1
                context += req.computed
                weights.append(
                    self.cost.per_decode_seq_s + self.cost.per_context_token_s * req.computed
                )
            else:
                prefill_tokens += tokens
                weights.append(self.cost.per_prefill_token_s * tokens)
        duration = (
            self.cost.base_s
            + self.cost.per_prefill_token_s * prefill_tokens
            + self.cost.per_decode_seq_s * decodes
            + self.cost.per_context_token_s * context
        )
        total_weight = sum(weights)
        end = now + duration

        for (req, tokens), weight in zip(work, weights, strict=True):
            req.compute_s += duration * weight / total_weight
            decoding = req.in_decode
            before = req.computed
            req.computed += tokens
            self._register_full_blocks(req, before)
            if decoding:
                req.generated += 1
            elif req.computed >= req.prefill_target and req.generated == 0:
                req.generated = 1
                req.first_token_at = end

        for req, _ in work:
            if req in self.running and req.generated >= req.output_len:
                req.finished_at = end
                self.running.remove(req)
                self._free(req, end)
                self.done.append(req)
        return duration


def simulate(
    *,
    rate: float,
    duration_s: float,
    engine: EngineConfig = EngineConfig(),
    tenants: tuple[TenantSpec, ...] = TENANTS,
    cost: StepCost = StepCost(),
    model: ModelSpec = ModelSpec(),
    seed: int = 7,
) -> RunResult:
    """Serve a Poisson workload for `duration_s`, then drain in-flight work."""
    rng = random.Random(seed)
    sim = Engine(engine, cost, model)
    counter = itertools.count()
    arrivals: list[tuple[float, int, Request]] = []

    def new_request(tenant: TenantSpec, at: float, session: Session | None) -> Request:
        message = _uniform(rng, tenant.prompt)
        if session is not None:
            prompt = (session.history or tenant.system_prefix) + message
            group = session.group
        else:
            prompt = tenant.system_prefix + message
            group = rng.randrange(tenant.prefix_groups)
        return Request(
            rid=next(counter),
            tenant=tenant,
            arrival=at,
            prompt_len=prompt,
            output_len=_uniform(rng, tenant.output),
            group=group,
            session=session,
        )

    weights = [t.weight for t in tenants]
    t = 0.0
    sessions = itertools.count()
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            break
        tenant = rng.choices(tenants, weights)[0]
        session = None
        if tenant.turns is not None:
            session = Session(
                sid=next(sessions),
                tenant=tenant,
                group=0,
                turns_left=_uniform(rng, tenant.turns) - 1,
            )
        req = new_request(tenant, t, session)
        heapq.heappush(arrivals, (req.arrival, req.rid, req))

    now = 0.0
    idle = 0.0
    steps = 0
    dropped = 0
    while arrivals or sim.waiting or sim.running:
        while arrivals and arrivals[0][0] <= now:
            _, _, req = heapq.heappop(arrivals)
            if engine.num_blocks * (1 - engine.watermark) < math.ceil(
                req.prompt_len / engine.block_size
            ):
                dropped += 1
                continue
            sim.waiting.append(req)

        if not sim.running and not sim.waiting:
            if not arrivals:
                break
            idle += arrivals[0][0] - now
            now = arrivals[0][0]
            continue

        before = len(sim.done)
        elapsed = sim.step(now)
        if elapsed == 0.0:
            # Nothing runnable: wait for the next arrival.
            if not arrivals:
                raise RuntimeError("engine stalled with work queued")
            idle += arrivals[0][0] - now
            now = arrivals[0][0]
            continue
        now += elapsed
        steps += 1

        for req in sim.done[before:]:
            session = req.session
            if session is None or session.turns_left <= 0:
                continue
            follow_up_at = now + rng.uniform(*req.tenant.think_s)
            if follow_up_at >= duration_s:
                continue
            session.history = req.prompt_len + req.output_len
            session.turns_left -= 1
            nxt = new_request(req.tenant, follow_up_at, session)
            heapq.heappush(arrivals, (nxt.arrival, nxt.rid, nxt))

    sim.blocks.close(now)
    return RunResult(
        config={
            "rate_rps": rate,
            "duration_s": duration_s,
            "seed": seed,
            "engine": engine.__dict__,
            "step_cost": cost.__dict__,
            "model": {**model.__dict__, "kv_bytes_per_token": model.kv_bytes_per_token},
        },
        requests=sim.done,
        wall_s=now,
        idle_s=idle,
        steps=steps,
        evictions=sim.blocks.evictions,
        cache_reserve_block_s=sim.blocks.cache_reserve_block_s,
        dropped=dropped,
        tokens_generated=sum(r.generated for r in sim.done),
    )
