"""Token-level request traces with the same tenants as the serving simulator.

Prompts are token-id lists rather than text so lengths are exact and shared
prefixes are byte-identical, which is what prefix caching hashes. Token values
are arbitrary: serving cost does not depend on what the tokens mean.

One deliberate difference from the simulator: a multi-turn session's next
prompt appends *synthetic* assistant tokens, not the model's actual output
(the server returns text, not ids). Prefix reuse across turns therefore covers
the previous prompt but not the previous answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from case_studies.kv_cache.sim import TENANTS, TenantSpec

VOCAB_LOW, VOCAB_HIGH = 1_000, 100_000


@dataclass
class Session:
    sid: int
    turns_left: int
    history: list[int] = field(default_factory=list)


@dataclass
class TraceRequest:
    rid: int
    tenant: str
    team: str | None
    at: float
    prompt: list[int]
    output_len: int
    session: Session | None = None


class TraceBuilder:
    def __init__(self, seed: int, tenants: tuple[TenantSpec, ...] = TENANTS) -> None:
        self.rng = random.Random(seed)
        self.tenants = tenants
        self._rid = 0
        self._sid = 0
        self._prefixes: dict[tuple[str, int], list[int]] = {}

    def _ids(self, n: int, rng: random.Random | None = None) -> list[int]:
        source = rng or self.rng
        return [source.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n)]

    def _system(self, tenant: TenantSpec, group: int) -> list[int]:
        key = (tenant.name, group)
        if key not in self._prefixes:
            # Seeded by name so every run and every rate shares the same prefixes.
            self._prefixes[key] = self._ids(tenant.system_prefix, random.Random(f"{key}"))
        return self._prefixes[key]

    def request(self, tenant: TenantSpec, at: float, session: Session | None) -> TraceRequest:
        message = self._ids(self.rng.randint(*tenant.prompt))
        if session is not None:
            prompt = (session.history or self._system(tenant, 0)) + message
        else:
            prompt = self._system(tenant, self.rng.randrange(tenant.prefix_groups)) + message
        self._rid += 1
        return TraceRequest(
            rid=self._rid,
            tenant=tenant.name,
            team=tenant.team,
            at=at,
            prompt=prompt,
            output_len=self.rng.randint(*tenant.output),
            session=session,
        )

    def arrivals(self, rate: float, duration: float) -> list[TraceRequest]:
        weights = [t.weight for t in self.tenants]
        out: list[TraceRequest] = []
        t = 0.0
        while True:
            t += self.rng.expovariate(rate)
            if t >= duration:
                return out
            tenant = self.rng.choices(self.tenants, weights)[0]
            session = None
            if tenant.turns is not None:
                self._sid += 1
                session = Session(self._sid, self.rng.randint(*tenant.turns) - 1)
            out.append(self.request(tenant, t, session))

    def follow_up(
        self, req: TraceRequest, finished_at: float, duration: float, max_len: int
    ) -> TraceRequest | None:
        session = req.session
        if session is None or session.turns_left <= 0:
            return None
        tenant = next(t for t in self.tenants if t.name == req.tenant)
        at = finished_at + self.rng.uniform(*tenant.think_s)
        history = req.prompt + self._ids(req.output_len)
        # Leave room for the next message and answer inside the context window.
        if at >= duration or len(history) + tenant.prompt[1] + tenant.output[1] > max_len:
            return None
        session.history = history
        session.turns_left -= 1
        return self.request(tenant, at, session)
