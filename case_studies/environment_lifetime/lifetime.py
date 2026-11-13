"""Exact allocation of environment-lifetime memory GB-seconds to requests.

Model (synthetic, offline):

* Each environment has an observed lifetime ``[lifetime_start_s, lifetime_end_s]``
  and a constant synthetic memory capacity ``memory_gb``. Lifetime GB-seconds are
  ``memory_gb * (lifetime_end_s - lifetime_start_s)``.
* Each request belongs to exactly one environment and has a span
  ``[start_s, end_s]`` that must lie inside that environment's lifetime.
* Per environment, the lifetime is cut into non-overlapping slices at every
  lifetime and request boundary. A slice covered by ``n >= 1`` requests is busy;
  its GB-seconds are split equally, ``1/n`` to each covering request. A slice
  covered by no request is uncovered and its GB-seconds go to the residual.
* Environments are integrated independently: requests on different
  environments never share a slice, even when their wall-clock spans overlap.
* Zero-length requests (``start_s == end_s``) are allowed. They cover no slice,
  receive exactly 0 GB-seconds and are flagged ``zero_duration``.
* The residual is labelled ``unobserved`` unless the fixture declares itself
  both ``synthetic`` and ``telemetry_complete``; only then is it labelled idle.

All arithmetic uses ``fractions.Fraction``. Inputs are ints, ``Decimal``,
``Fraction`` or decimal strings; binary floats, booleans, NaN and infinities are
rejected. Results are serialised as exact strings: a plain decimal when the
value terminates in base 10, otherwise ``"numerator/denominator"``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "unalloc.case_studies.environment_lifetime/v1"
RESULT_SCHEMA = "unalloc.case_studies.environment_lifetime.result/v1"

RESIDUAL_IDLE = "idle_by_complete_synthetic_fixture"
RESIDUAL_UNOBSERVED = "unobserved"

SCOPE_NOTE = (
    "Offline synthetic arithmetic. Not validated AWS ingestion, not an AWS bill, "
    "not CPU billing. No prices, no dollar amounts."
)

_TOP_REQUIRED = {"schema", "synthetic", "telemetry_complete", "environments", "requests"}
_TOP_OPTIONAL = {"description", "notes"}
_ENV_REQUIRED = {"id", "memory_gb", "lifetime_start_s", "lifetime_end_s"}
_REQ_REQUIRED = {"id", "environment", "start_s", "end_s"}
_ITEM_OPTIONAL = {"note"}


class FixtureError(ValueError):
    """The fixture is malformed; the message names the offending field."""


@dataclass(frozen=True)
class Environment:
    id: str
    memory_gb: Fraction
    start: Fraction
    end: Fraction


@dataclass(frozen=True)
class Request:
    id: str
    environment: str
    start: Fraction
    end: Fraction


@dataclass(frozen=True)
class Fixture:
    synthetic: bool
    telemetry_complete: bool
    environments: tuple[Environment, ...]
    requests: tuple[Request, ...]


# --- exact number handling ------------------------------------------------------


def exact(value: Any, field: str) -> Fraction:
    """Convert ``value`` to an exact, finite Fraction or raise FixtureError."""
    if isinstance(value, bool):
        raise FixtureError(f"{field}: boolean {value!r} is not a number")
    if isinstance(value, float):
        raise FixtureError(
            f"{field}: binary float {value!r} rejected; use int, str, Decimal or Fraction"
        )
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FixtureError(f"{field}: non-finite value {value!r}")
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text != value or "_" in text:
            raise FixtureError(f"{field}: malformed number string {value!r}")
        try:
            parsed = Decimal(text)
        except InvalidOperation:
            raise FixtureError(f"{field}: malformed number string {value!r}") from None
        if not parsed.is_finite():
            raise FixtureError(f"{field}: non-finite value {value!r}")
        return Fraction(parsed)
    raise FixtureError(f"{field}: unsupported type {type(value).__name__}")


def non_negative(value: Any, field: str) -> Fraction:
    q = exact(value, field)
    if q < 0:
        raise FixtureError(f"{field}: negative value {format_exact(q)}")
    return q


def positive(value: Any, field: str) -> Fraction:
    q = exact(value, field)
    if q <= 0:
        raise FixtureError(f"{field}: must be > 0, got {format_exact(q)}")
    return q


def format_exact(q: Fraction) -> str:
    """Exact string: terminating decimal when possible, else ``"n/d"``."""
    d = q.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f"{q.numerator}/{q.denominator}"
    places = max(twos, fives)
    scaled = q.numerator * (10**places) // q.denominator  # exact: denominator | 10**places
    with localcontext() as ctx:
        ctx.prec = len(str(abs(scaled))) + 2
        text = format(Decimal(scaled).scaleb(-places), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


# --- parsing ----------------------------------------------------------------------


def _identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FixtureError(
            f"{field}: identity must be a non-empty string without "
            f"surrounding whitespace, got {value!r}"
        )
    return value


def _check_keys(item: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(item, Mapping):
        raise FixtureError(f"{where}: expected an object, got {type(item).__name__}")
    missing = required - item.keys()
    if missing:
        raise FixtureError(f"{where}: missing field(s) {sorted(missing)}")
    unknown = item.keys() - required - optional
    if unknown:
        raise FixtureError(f"{where}: unknown field(s) {sorted(unknown)}")


def _flag(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise FixtureError(f"{field}: expected true/false, got {value!r}")
    return value


def parse_fixture(data: Any) -> Fixture:
    """Validate a decoded fixture object. Raises FixtureError on any defect."""
    _check_keys(data, _TOP_REQUIRED, _TOP_OPTIONAL, "fixture")
    if data["schema"] != SCHEMA:
        raise FixtureError(f"fixture.schema: expected {SCHEMA!r}, got {data['schema']!r}")
    synthetic = _flag(data["synthetic"], "fixture.synthetic")
    complete = _flag(data["telemetry_complete"], "fixture.telemetry_complete")
    raw_envs, raw_reqs = data["environments"], data["requests"]
    if not isinstance(raw_envs, list):
        raise FixtureError("fixture.environments: expected a list")
    if not isinstance(raw_reqs, list):
        raise FixtureError("fixture.requests: expected a list")

    envs: dict[str, Environment] = {}
    for i, item in enumerate(raw_envs):
        where = f"environments[{i}]"
        _check_keys(item, _ENV_REQUIRED, _ITEM_OPTIONAL, where)
        env_id = _identity(item["id"], f"{where}.id")
        if env_id in envs:
            raise FixtureError(f"{where}.id: duplicate environment id {env_id!r}")
        start = non_negative(item["lifetime_start_s"], f"{where}.lifetime_start_s")
        end = non_negative(item["lifetime_end_s"], f"{where}.lifetime_end_s")
        if end <= start:
            raise FixtureError(
                f"{where}: lifetime_end_s ({format_exact(end)}) must be > "
                f"lifetime_start_s ({format_exact(start)})"
            )
        memory = positive(item["memory_gb"], f"{where}.memory_gb")
        envs[env_id] = Environment(env_id, memory, start, end)

    reqs: dict[str, Request] = {}
    for i, item in enumerate(raw_reqs):
        where = f"requests[{i}]"
        _check_keys(item, _REQ_REQUIRED, _ITEM_OPTIONAL, where)
        req_id = _identity(item["id"], f"{where}.id")
        if req_id in reqs:
            raise FixtureError(f"{where}.id: duplicate request id {req_id!r}")
        env_id = _identity(item["environment"], f"{where}.environment")
        env = envs.get(env_id)
        if env is None:
            raise FixtureError(f"{where}.environment: unknown environment {env_id!r}")
        start = non_negative(item["start_s"], f"{where}.start_s")
        end = non_negative(item["end_s"], f"{where}.end_s")
        if end < start:
            raise FixtureError(
                f"{where}: end_s ({format_exact(end)}) is before start_s ({format_exact(start)})"
            )
        if start < env.start or end > env.end:
            raise FixtureError(
                f"{where}: span [{format_exact(start)}, {format_exact(end)}] is outside "
                f"environment {env_id!r} lifetime "
                f"[{format_exact(env.start)}, {format_exact(env.end)}]"
            )
        reqs[req_id] = Request(req_id, env_id, start, end)

    return Fixture(synthetic, complete, tuple(envs.values()), tuple(reqs.values()))


def _reject_constant(name: str) -> Any:
    raise FixtureError(f"non-finite JSON literal {name!r} rejected")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise FixtureError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def loads_fixture(text: str) -> Fixture:
    """Decode JSON (floats as Decimal, no NaN/Infinity, no duplicate keys) and validate."""
    try:
        data = json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise FixtureError(f"invalid JSON: {exc}") from None
    return parse_fixture(data)


def load_fixture(path: Path) -> Fixture:
    return loads_fixture(Path(path).read_text(encoding="utf-8"))


# --- integration -------------------------------------------------------------------


def _integrate(env: Environment, requests: Iterable[Request]) -> dict[str, Any]:
    reqs = sorted(requests, key=lambda r: r.id)
    points = {env.start, env.end}
    for r in reqs:
        points.update((r.start, r.end))
    bounds = sorted(points)

    allocated = {r.id: Fraction(0) for r in reqs}
    busy = uncovered = Fraction(0)
    max_concurrency = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        width = b - a  # > 0: bounds are distinct and sorted
        active = [r for r in reqs if r.start <= a and r.end >= b]
        if not active:
            uncovered += width
            continue
        busy += width
        max_concurrency = max(max_concurrency, len(active))
        share = env.memory_gb * width / len(active)
        for r in active:
            allocated[r.id] += share

    lifetime_s = env.end - env.start
    lifetime_gbs = env.memory_gb * lifetime_s
    raw_s = sum((r.end - r.start for r in reqs), Fraction(0))
    allocated_total = sum(allocated.values(), Fraction(0))
    residual = env.memory_gb * uncovered

    # Exact invariants. A failure here is a bug in this module, not in the input.
    if busy + uncovered != lifetime_s:
        raise AssertionError(f"{env.id}: slice seconds do not sum to the lifetime")
    if allocated_total != env.memory_gb * busy:
        raise AssertionError(f"{env.id}: allocated GB-s != memory * busy seconds")
    if allocated_total + residual != lifetime_gbs:
        raise AssertionError(f"{env.id}: allocated + residual != lifetime GB-s")

    return {
        "env": env,
        "requests": reqs,
        "allocated": allocated,
        "raw_s": raw_s,
        "busy_s": busy,
        "uncovered_s": uncovered,
        "lifetime_s": lifetime_s,
        "lifetime_gbs": lifetime_gbs,
        "allocated_gbs": allocated_total,
        "residual_gbs": residual,
        "max_concurrency": max_concurrency,
    }


def residual_label(fixture: Fixture) -> str:
    if fixture.synthetic and fixture.telemetry_complete:
        return RESIDUAL_IDLE
    return RESIDUAL_UNOBSERVED


def compute(fixture: Fixture) -> dict[str, Any]:
    """Allocate every environment's lifetime GB-seconds. Output is order-independent."""
    by_env: dict[str, list[Request]] = {e.id: [] for e in fixture.environments}
    for r in fixture.requests:
        by_env[r.environment].append(r)

    label = residual_label(fixture)
    f = format_exact
    environments = []
    request_rows = []
    totals = dict.fromkeys(
        (
            "raw_s",
            "busy_s",
            "uncovered_s",
            "lifetime_s",
            "lifetime_gbs",
            "allocated_gbs",
            "residual_gbs",
        ),
        Fraction(0),
    )
    for env in sorted(fixture.environments, key=lambda e: e.id):
        res = _integrate(env, by_env[env.id])
        for key in totals:
            totals[key] += res[key]
        environments.append(
            {
                "id": env.id,
                "memory_gb": f(env.memory_gb),
                "lifetime_start_s": f(env.start),
                "lifetime_end_s": f(env.end),
                "request_count": len(res["requests"]),
                "max_concurrency": res["max_concurrency"],
                "raw_request_seconds": f(res["raw_s"]),
                "union_busy_seconds": f(res["busy_s"]),
                "overlap_seconds": f(res["raw_s"] - res["busy_s"]),
                "uncovered_seconds": f(res["uncovered_s"]),
                "lifetime_seconds": f(res["lifetime_s"]),
                "lifetime_gb_seconds": f(res["lifetime_gbs"]),
                "allocated_gb_seconds": f(res["allocated_gbs"]),
                "residual_gb_seconds": f(res["residual_gbs"]),
                "residual_label": label,
                "reconciliation": _reconciliation(
                    res["allocated_gbs"], res["residual_gbs"], res["lifetime_gbs"]
                ),
            }
        )
        for r in res["requests"]:
            request_rows.append(
                {
                    "id": r.id,
                    "environment": env.id,
                    "start_s": f(r.start),
                    "end_s": f(r.end),
                    "duration_seconds": f(r.end - r.start),
                    "zero_duration": r.end == r.start,
                    "allocated_gb_seconds": f(res["allocated"][r.id]),
                }
            )

    return {
        "schema": RESULT_SCHEMA,
        "scope": SCOPE_NOTE,
        "synthetic": fixture.synthetic,
        "telemetry_complete": fixture.telemetry_complete,
        "residual_label": label,
        "units": {
            "time": "seconds",
            "memory": "GB (synthetic constant capacity)",
            "allocation": "GB-seconds",
        },
        "totals": {
            "environment_count": len(environments),
            "request_count": len(request_rows),
            "raw_request_seconds": f(totals["raw_s"]),
            "union_busy_seconds": f(totals["busy_s"]),
            "uncovered_seconds": f(totals["uncovered_s"]),
            "lifetime_seconds": f(totals["lifetime_s"]),
            "lifetime_gb_seconds": f(totals["lifetime_gbs"]),
            "allocated_gb_seconds": f(totals["allocated_gbs"]),
            "residual_gb_seconds": f(totals["residual_gbs"]),
            "reconciliation": _reconciliation(
                totals["allocated_gbs"], totals["residual_gbs"], totals["lifetime_gbs"]
            ),
        },
        "environments": environments,
        "requests": request_rows,
    }


def _reconciliation(allocated: Fraction, residual: Fraction, lifetime: Fraction) -> dict:
    diff = lifetime - (allocated + residual)
    return {
        "identity": "allocated_gb_seconds + residual_gb_seconds == lifetime_gb_seconds",
        "allocated_plus_residual_gb_seconds": format_exact(allocated + residual),
        "lifetime_gb_seconds": format_exact(lifetime),
        "difference_gb_seconds": format_exact(diff),
        "exact": diff == 0,
    }


def without_requests(fixture: Fixture, drop: Iterable[str]) -> Fixture:
    """Simulate missing request telemetry. The result never claims completeness."""
    drop = set(drop)
    known = {r.id for r in fixture.requests}
    unknown = drop - known
    if unknown:
        raise FixtureError(f"cannot drop unknown request id(s) {sorted(unknown)}")
    kept = tuple(r for r in fixture.requests if r.id not in drop)
    return Fixture(fixture.synthetic, False, fixture.environments, kept)


def dumps_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, indent=2) + "\n"
