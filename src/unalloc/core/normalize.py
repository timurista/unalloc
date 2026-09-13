"""Label canonicalization.

The whole premise of unalloc is joining ledgers that never agreed on a key.
OpenCost gives you `label_cost_center` or `labels.costCenter`. LiteLLM gives you
`team_id` or a request tag. OpenAI gives you a project name. None of them will
match on a naive dict lookup, so every adapter runs its labels through here
before emitting a CostRow.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Prefixes providers bolt onto label keys that carry no meaning for us.
_STRIP_PREFIXES = (
    "label_",
    "labels_",
    "annotation_",
    "kubernetes_io_",
    "app_kubernetes_io_",
    "tag_",
    "request_tag_",
)

# Default alias table: many spellings of the same business dimension.
# Extend this per-org via `aliases` in config rather than editing code.
DEFAULT_ALIASES: dict[str, str] = {
    "costcenter": "cost_center",
    "cost_centre": "cost_center",
    "cc": "cost_center",
    "team_id": "team",
    "teamid": "team",
    "owning_team": "team",
    "owner": "team",
    "squad": "team",
    "project_id": "project",
    "projectid": "project",
    "proj": "project",
    "workspace_id": "workspace",
    "svc": "service",
    "app": "service",
    "application": "service",
    "env": "environment",
    "stage": "environment",
}


def canonical_key(key: str) -> str:
    """Normalize one label key to snake_case with provider noise stripped.

    >>> canonical_key("label_costCenter")
    'cost_center'
    >>> canonical_key("app.kubernetes.io/name")
    'name'
    """
    key = key.strip()
    # camelCase -> camel_case before lowercasing, or we lose word boundaries.
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    key = key.lower()
    # Path-style keys: app.kubernetes.io/name -> name
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    key = _NON_ALNUM.sub("_", key).strip("_")
    # Strip until nothing matches: OpenCost's Prometheus-style keys stack
    # prefixes (label_app_kubernetes_io_name), and stopping after one made the
    # same Kubernetes label canonicalize differently depending on the endpoint.
    stripped = True
    while stripped:
        stripped = False
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix) and len(key) > len(prefix):
                key = key[len(prefix) :]
                stripped = True
                break
    return key


def _precedence(raw_key: str, canon: str, aliased: bool) -> tuple[int, str]:
    """Rank competing raw keys that land on the same canonical key.

    Lower wins: a key already written canonically, then an alias, then a key
    that needed rewriting (prefix, camelCase), then a path-derived key. Ties
    break on the raw key itself, so the result never depends on the order a
    provider happened to serialize its labels in.
    """
    written = raw_key.strip().lower()
    if written == canon and not aliased:
        rank = 0
    elif aliased:
        rank = 1
    elif "/" not in written:
        rank = 2
    else:
        rank = 3
    return rank, written


def normalize_labels(
    raw: Mapping[str, object] | None,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Canonicalize keys, apply the alias table, drop empty values.

    When several raw keys collapse onto one canonical key, `_precedence`
    decides deterministically: an explicit `team` label wins over a `team_id`
    that aliases onto it, and `app.kubernetes.io/name` vs
    `leaderworkerset.sigs.k8s.io/name` resolves the same way whatever order
    the provider sent them in.
    """
    if not raw:
        return {}
    table = {**DEFAULT_ALIASES, **(aliases or {})}
    best: dict[str, tuple[tuple[int, str], str]] = {}
    for key, value in raw.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        shaped = canonical_key(str(key))
        if not shaped:
            continue
        canon = table.get(shaped, shaped)
        rank = _precedence(str(key), canon, aliased=canon != shaped)
        if canon not in best or rank < best[canon][0]:
            best[canon] = (rank, text)
    return {canon: text for canon, (_, text) in best.items()}


def dimensions(rows: Iterable) -> dict[str, int]:
    """Count how many rows carry each label key.

    Useful for `unalloc labels`: it tells you which dimensions are actually
    viable to attribute on before you pick one.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for key in row.labels:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
