"""A deterministic multi-tenant request trace.

Four callers share one deployment, chosen so that tokens and serving work
disagree as much as they do in production:

- `search` is RAG: long retrieved-context prompts, short answers. Many tokens,
  but nearly all of them are prefilled in one parallel forward pass.
- `agents` is tool-using loops: short prompts, long generations. Few tokens,
  but every output token is its own sequential decode step holding the cache.
- `platform` is a medium/medium internal assistant.
- `sandbox` is an unlabeled experimentation key with no owner, mixed shapes.

`search` and `agents` share a common system prompt prefix whose KV cache the
server computes once and reuses, the way prefix caching works in vLLM/SGLang.
Gateways still bill those prefix tokens on every request.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SYSTEM_PREFIX_TOKENS = 64
SYSTEM_PROMPT = (
    b"You are a careful assistant for Acme Corp. Cite sources, keep answers short, "
    b"and never reveal internal tooling."
)


@dataclass(frozen=True, slots=True)
class CallerProfile:
    caller: str
    team: str | None
    """Owner label as the deployment would see it. None means unlabeled."""
    weight: float
    prompt_range: tuple[int, int]
    """Inclusive total prompt length, shared prefix included."""
    completion_range: tuple[int, int]
    shared_prefix: bool
    shape: str


PROFILES: tuple[CallerProfile, ...] = (
    CallerProfile("search", "search", 0.33, (400, 700), (16, 32), True, "RAG: long in, short out"),
    CallerProfile("agents", "agents", 0.21, (80, 160), (150, 250), True, "short in, long out"),
    CallerProfile("platform", "platform", 0.25, (150, 300), (60, 120), False, "medium/medium"),
    CallerProfile("sandbox", None, 0.21, (16, 600), (16, 200), False, "unlabeled key, mixed"),
)

CALLERS: tuple[str, ...] = tuple(p.caller for p in PROFILES)


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    caller: str
    team: str | None
    prompt_tokens: int
    completion_tokens: int
    shared_prefix: bool
    prompt_seed: int
    """Seed for the non-prefix prompt bytes. Content does not affect cost."""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def system_prefix_ids() -> list[int]:
    """The shared system prompt as byte-level token ids, exactly SYSTEM_PREFIX_TOKENS long."""
    raw = (SYSTEM_PROMPT * (SYSTEM_PREFIX_TOKENS // len(SYSTEM_PROMPT) + 1))[:SYSTEM_PREFIX_TOKENS]
    return list(raw)


def _counts(n_requests: int) -> dict[str, int]:
    """Largest-remainder apportionment so counts sum to exactly n_requests."""
    total = sum(p.weight for p in PROFILES)
    raw = {p.caller: p.weight / total * n_requests for p in PROFILES}
    counts = {k: int(v) for k, v in raw.items()}
    for caller in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[
        : n_requests - sum(counts.values())
    ]:
        counts[caller] += 1
    return counts


def build_trace(n_requests: int, seed: int) -> list[Request]:
    """Interleaved requests with per-caller shapes; identical for identical inputs."""
    rng = random.Random(seed)
    counts = _counts(n_requests)
    order = [p for p in PROFILES for _ in range(counts[p.caller])]
    rng.shuffle(order)
    trace: list[Request] = []
    for i, profile in enumerate(order):
        trace.append(
            Request(
                request_id=f"req-{i:04d}",
                caller=profile.caller,
                team=profile.team,
                prompt_tokens=rng.randint(*profile.prompt_range),
                completion_tokens=rng.randint(*profile.completion_range),
                shared_prefix=profile.shared_prefix,
                prompt_seed=rng.randrange(2**31),
            )
        )
    return trace
