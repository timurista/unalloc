"""A LeaderWorkerSet month in OpenCost shape, and what unalloc makes of it.

Multi-host inference on Kubernetes is usually a LeaderWorkerSet (LWS): each
replica is one leader pod plus W-1 worker pods, one GPU each, all running shards
of one model for one tenant. LWS renders leader and worker pods from two
templates (`leaderTemplate`, `workerTemplate`) and stamps its own bookkeeping
labels on both. The failure this study models is mundane and common: the owner
label (`team`) was added to the leader template only. Every worker GPU-hour then
reaches the ledger with no owner, and because the workers are where most of the
collective-communication time is spent, so does most of the communication cost.

GPU time per pod is split into busy compute, communication and in-pod idle using
the measured per-rank communication fractions from the TP/PP runs, scaled by an
assumed utilization. OpenCost bills a pod for its whole allocated GPU either way;
the split only matters for the showback question at the end.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from case_studies.common import (
    CPU_CORE_HOUR_USD,
    GPU_HOUR_USD,
    RAM_GB_HOUR_USD,
    WINDOW_END,
    WINDOW_START,
    load_rows,
    opencost_allocation,
    opencost_payload,
    summarize,
    write_payloads,
)
from unalloc.core.models import UNALLOCATED, CostRow
from unalloc.core.normalize import canonical_key, normalize_labels

HOURS = (WINDOW_END - WINDOW_START).total_seconds() / 3600
NAMESPACE = "inference"
CHART_NAME = "vllm"
"""app.kubernetes.io/name from a shared serving chart, identical for every tenant."""
LWS = "leaderworkerset.sigs.k8s.io/"
DIMENSION = "team"

IDLE_GPU_HOURS = HOURS / 2
"""Unscheduled capacity: half a GPU for the month, billed to OpenCost's __idle__."""
IDLE_CPU_CORES = 16.0
IDLE_RAM_GB = 64.0

COMM_FRACTION_SENSITIVITY = (0.20, 0.10, 0.05)
"""GPU-plausible mean communication fractions (NVLink/InfiniBand), for contrast with
the CPU/gloo-measured fractions, which are an upper bound. Applied by rescaling the
measured per-rank fractions, see `build_pods`."""


@dataclass(frozen=True, slots=True)
class Tenant:
    team: str
    lws_name: str
    mode: str
    size: int
    """Pods per replica (LWS `size`), one GPU each: TP or PP world size."""
    replicas: int
    utilization: float
    """Busy fraction of allocated GPU time (compute + communication)."""
    model: str
    cpu_cores: float = 8.0
    ram_gb: float = 64.0


TENANTS: tuple[Tenant, ...] = (
    Tenant("search", "search-llama-70b", "tp", 4, 2, 0.60, "llama-3.1-70b"),
    Tenant("agents", "agents-qwen-32b", "pp", 2, 3, 0.45, "qwen2.5-32b"),
)


@dataclass(frozen=True, slots=True)
class Pod:
    tenant: Tenant
    group: int
    worker_index: int
    comm_fraction: float

    @property
    def role(self) -> str:
        return "leader" if self.worker_index == 0 else "worker"

    @property
    def name(self) -> str:
        # LWS naming: leader `<lws>-<group>`, workers `<lws>-<group>-<index>`.
        base = f"{self.tenant.lws_name}-{self.group}"
        return base if self.worker_index == 0 else f"{base}-{self.worker_index}"

    @property
    def allocation_name(self) -> str:
        return f"{NAMESPACE}/{self.name}"

    @property
    def controller(self) -> str:
        # Leader StatefulSet is `<lws>`; each group's worker StatefulSet `<lws>-<group>`.
        lws = self.tenant.lws_name
        return lws if self.worker_index == 0 else f"{lws}-{self.group}"

    @property
    def busy_hours(self) -> float:
        return HOURS * self.tenant.utilization

    @property
    def comm_hours(self) -> float:
        return self.busy_hours * self.comm_fraction

    @property
    def compute_hours(self) -> float:
        return self.busy_hours - self.comm_hours

    @property
    def gpu_cost(self) -> float:
        return HOURS * GPU_HOUR_USD

    @property
    def cpu_cost(self) -> float:
        return self.tenant.cpu_cores * HOURS * CPU_CORE_HOUR_USD

    @property
    def ram_cost(self) -> float:
        return self.tenant.ram_gb * HOURS * RAM_GB_HOUR_USD


def build_pods(
    fractions: Mapping[str, Sequence[float]],
    tenants: Sequence[Tenant] = TENANTS,
    target_mean: float | None = None,
) -> list[Pod]:
    """One Pod per LWS pod. Worker index i takes the measured fraction of rank i.

    If the measured world size differs from the tenant's LWS size, every pod gets
    the mean measured fraction rather than an invented per-rank profile.

    `target_mean` rescales all measured fractions by one factor so their
    pod-weighted mean equals it. Scaling (rather than overriding with a constant)
    keeps the measured *structure* — TP ranks spending more time in collectives
    than PP stages — which is what moves money between tenants in a showback.
    """
    pods = []
    for tenant in tenants:
        measured = list(fractions[tenant.mode])
        for group in range(tenant.replicas):
            for idx in range(tenant.size):
                frac = measured[idx] if len(measured) == tenant.size else statistics.fmean(measured)
                pods.append(Pod(tenant, group, idx, frac))
    if target_mean is not None and pods:
        scale = target_mean / statistics.fmean(p.comm_fraction for p in pods)
        pods = [Pod(p.tenant, p.group, p.worker_index, p.comm_fraction * scale) for p in pods]
    return pods


# --- labels and payloads -------------------------------------------------------------


def _digest(text: str, length: int) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:length]


def raw_labels(pod: Pod, *, owner_on_workers: bool, key_style: str = "slash") -> dict[str, str]:
    """Pod labels as the API server holds them.

    Keys are emitted sorted because that is how Go's encoding/json serialises a
    map, which matters: unalloc's normalize keeps the *first* value per canonical
    key, so the order decides which of two colliding labels survives.

    key_style "prom" applies Prometheus label-name sanitisation (`.`/`/`/`-` ->
    `_`), which is the form OpenCost's allocation API actually returns because
    it reads labels from kube-state-metrics.
    """
    t = pod.tenant
    labels = {
        "app.kubernetes.io/component": pod.role,
        "app.kubernetes.io/managed-by": "Helm",
        "app.kubernetes.io/name": CHART_NAME,
        LWS + "group-index": str(pod.group),
        LWS + "group-key": _digest(f"{NAMESPACE}/{t.lws_name}/{pod.group}", 40),
        LWS + "name": t.lws_name,
        LWS + "template-revision-hash": _digest(t.lws_name, 10),
        LWS + "worker-index": str(pod.worker_index),
    }
    if pod.role == "leader" or owner_on_workers:
        labels["team"] = t.team
    if key_style == "prom":
        labels = {re.sub(r"[^A-Za-z0-9_]", "_", k): v for k, v in labels.items()}
    return dict(sorted(labels.items()))


def build_payload(pods: Sequence[Pod], *, owner_on_workers: bool, key_style: str) -> dict:
    allocations = [
        opencost_allocation(
            pod.allocation_name,
            namespace=NAMESPACE,
            controller=pod.controller,
            controller_kind="statefulset",
            labels=raw_labels(pod, owner_on_workers=owner_on_workers, key_style=key_style),
            cpu_cost=pod.cpu_cost,
            ram_cost=pod.ram_cost,
            gpu_cost=pod.gpu_cost,
            gpu_hours=HOURS,
            model=pod.tenant.model,
            efficiency=pod.tenant.utilization,
        )
        for pod in pods
    ]
    allocations.append(
        opencost_allocation(
            "__idle__",
            namespace="",
            controller="",
            controller_kind="",
            cpu_cost=IDLE_CPU_CORES * HOURS * CPU_CORE_HOUR_USD,
            ram_cost=IDLE_RAM_GB * HOURS * RAM_GB_HOUR_USD,
            gpu_cost=IDLE_GPU_HOURS * GPU_HOUR_USD,
            gpu_hours=IDLE_GPU_HOURS,
        )
    )
    return opencost_payload(allocations)


# --- scenarios -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    title: str
    owner_on_workers: bool
    key_style: str
    fallback: tuple[str, ...]
    variants: tuple[tuple[str, ...], ...] = ()
    """Other fallback lists evaluated on the same payload."""


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("s1", "team only; owner label on the LWS leader template only", False, "slash", ()),
    Scenario(
        "s2",
        "fallback to `name`, which every LWS pod carries (leaderworkerset.sigs.k8s.io/name)",
        False,
        "slash",
        ("name",),
        (("controller",), ("namespace",), ("group_key",)),
    ),
    Scenario(
        "s2_prom",
        "S2 with Prometheus-sanitised label keys, as OpenCost's /allocation serialises them",
        False,
        "prom",
        ("name",),
        (("leaderworkerset_sigs_k8s_io_name",),),
    ),
    Scenario("s3", "owner label fixed on the LWS worker template", True, "slash", ()),
)


def _assigned(row: CostRow, dimension: str, fallback: Sequence[str]) -> str:
    """The bucket unalloc's attribute() puts a row in (same lookup order)."""
    key = row.label(dimension)
    if key is None:
        for alt in fallback:
            key = row.label(alt)
            if key is not None:
                break
    return key or UNALLOCATED


def evaluate(
    rows: Sequence[CostRow], pods: Sequence[Pod], fallback: Sequence[str]
) -> dict[str, Any]:
    """unalloc's summary plus a check against ground truth we know and it doesn't.

    A fallback that drives the unallocated share to zero by landing spend in a
    bucket that is not a team is not a fix; `misattributed_usd` makes that visible.
    """
    by_alloc = {pod.allocation_name: pod for pod in pods}
    zero = Decimal("0")
    acc = {
        "correct_usd": zero,
        "misattributed_usd": zero,
        "unallocated_owned_usd": zero,
        "unallocated_idle_usd": zero,
    }
    comm_unallocated = comm_misattributed = 0.0
    non_owner: dict[str, Decimal] = {}
    for row in rows:
        got = _assigned(row, DIMENSION, fallback)
        pod = by_alloc.get(row.name)
        want = pod.tenant.team if pod else None
        comm_usd = pod.comm_hours * GPU_HOUR_USD if pod else 0.0
        if got == UNALLOCATED:
            acc["unallocated_owned_usd" if pod else "unallocated_idle_usd"] += row.amount_usd
            comm_unallocated += comm_usd
        elif got == want:
            acc["correct_usd"] += row.amount_usd
        else:
            acc["misattributed_usd"] += row.amount_usd
            non_owner[got] = non_owner.get(got, zero) + row.amount_usd
            comm_misattributed += comm_usd
    summary = summarize(rows, DIMENSION, fallback=fallback)
    total = summary["total_usd"]
    summary["accuracy"] = {
        **acc,
        "correct_pct": float(acc["correct_usd"] / total * 100) if total else 0.0,
        "misattributed_pct": float(acc["misattributed_usd"] / total * 100) if total else 0.0,
        "non_owner_buckets": dict(sorted(non_owner.items())),
        "communication_usd_unallocated": round(comm_unallocated, 2),
        "communication_usd_misattributed": round(comm_misattributed, 2),
    }
    return summary


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "fallback": summary["fallback"],
        "unallocated_pct": summary["unallocated_pct"],
        "unallocated_usd": summary["unallocated_usd"],
        "buckets": [{"key": b["key"], "amount_usd": b["amount_usd"]} for b in summary["buckets"]],
        "accuracy": summary["accuracy"],
    }


# --- communication economics -----------------------------------------------------------


def communication_economics(pods: Sequence[Pod]) -> dict[str, Any]:
    """Dollars of GPU time spent in collectives, and where a per-token showback puts them.

    A per-token showback charges every tenant one blended $/token. Tokens are
    produced by compute time, not communication time, so we take each tenant's
    token volume as proportional to its busy-compute GPU-hours. The blended rate
    then spreads the communication dollars by compute share, regardless of which
    tenant's parallelism strategy actually incurred them.
    """
    tenants: dict[str, dict[str, float]] = {}
    for pod in pods:
        t = tenants.setdefault(
            pod.tenant.team,
            {"compute_hours": 0.0, "comm_hours": 0.0, "leader_comm_usd": 0.0,
             "worker_comm_usd": 0.0},
        )
        t["compute_hours"] += pod.compute_hours
        t["comm_hours"] += pod.comm_hours
        t[f"{pod.role}_comm_usd"] += pod.comm_hours * GPU_HOUR_USD
    total_comm = sum(t["comm_hours"] for t in tenants.values()) * GPU_HOUR_USD
    total_compute = sum(t["compute_hours"] for t in tenants.values())
    shifted = 0.0
    for t in tenants.values():
        t["comm_usd"] = t["comm_hours"] * GPU_HOUR_USD
        t["token_share"] = t["compute_hours"] / total_compute if total_compute else 0.0
        t["per_token_showback_comm_usd"] = total_comm * t["token_share"]
        t["subsidy_usd"] = t["per_token_showback_comm_usd"] - t["comm_usd"]
        shifted += max(t["subsidy_usd"], 0.0)
    worker_comm = sum(t["worker_comm_usd"] for t in tenants.values())
    return {
        "communication_usd_total": round(total_comm, 2),
        "communication_usd_on_workers": round(worker_comm, 2),
        "cross_tenant_shift_usd": round(shifted, 2),
        "per_tenant": {
            team: {k: round(v, 4) for k, v in t.items()} for team, t in sorted(tenants.items())
        },
    }


# --- canonicalization ------------------------------------------------------------------


def canonicalization_findings(pods: Sequence[Pod]) -> dict[str, Any]:
    """How unalloc's canonical_key treats LWS label keys, computed not asserted."""

    def key_map(style: str) -> tuple[dict[str, str], dict[str, list[str]]]:
        keys = sorted(
            {k for pod in pods for k in raw_labels(pod, owner_on_workers=True, key_style=style)}
        )
        mapping = {k: canonical_key(k) for k in keys}
        groups: dict[str, list[str]] = {}
        for raw, canon in mapping.items():
            groups.setdefault(canon, []).append(raw)
        return mapping, {c: ks for c, ks in groups.items() if len(ks) > 1}

    slash_map, slash_collisions = key_map("slash")
    prom_map, prom_collisions = key_map("prom")
    worker = next(p for p in pods if p.role == "worker")
    sorted_labels = raw_labels(worker, owner_on_workers=False)
    reversed_labels = dict(reversed(list(sorted_labels.items())))
    prefixed = ("app_kubernetes_io_name", "label_app_kubernetes_io_name",
                "label_leaderworkerset_sigs_k8s_io_name", "labels.app.kubernetes.io/name")
    return {
        "slash_keys": slash_map,
        "slash_collisions": slash_collisions,
        "prometheus_keys": prom_map,
        "prometheus_collisions": prom_collisions,
        "order_dependence": {
            "pod": worker.allocation_name,
            "name_when_keys_sorted_as_go_emits": normalize_labels(sorted_labels).get("name"),
            "name_when_key_order_reversed": normalize_labels(reversed_labels).get("name"),
        },
        "prefix_stripping": {k: canonical_key(k) for k in prefixed},
    }


# --- driver ------------------------------------------------------------------------------


def assumptions() -> dict[str, Any]:
    return {
        "window": {"start": WINDOW_START, "end": WINDOW_END, "hours": HOURS},
        "gpu_hour_usd": GPU_HOUR_USD,
        "cpu_core_hour_usd": CPU_CORE_HOUR_USD,
        "ram_gb_hour_usd": RAM_GB_HOUR_USD,
        "namespace": NAMESPACE,
        "chart_app_name": CHART_NAME,
        "tenants": [
            {
                "team": t.team,
                "lws_name": t.lws_name,
                "parallelism": t.mode,
                "pods_per_replica": t.size,
                "replicas": t.replicas,
                "gpus": t.size * t.replicas,
                "utilization": t.utilization,
                "model": t.model,
                "cpu_cores_per_pod": t.cpu_cores,
                "ram_gb_per_pod": t.ram_gb,
            }
            for t in TENANTS
        ],
        "cluster_idle": {
            "gpu_hours": IDLE_GPU_HOURS,
            "cpu_cores": IDLE_CPU_CORES,
            "ram_gb": IDLE_RAM_GB,
        },
        "gpu_time_split": (
            "per pod: busy = allocated GPU-hours x utilization; communication = busy x "
            "measured comm fraction of the matching rank (worker-index i <- rank i); "
            "compute = busy - communication; in-pod idle = allocated - busy. OpenCost bills "
            "the whole allocated GPU to the pod regardless."
        ),
        "tokens_proxy": "tenant token volume proportional to busy-compute GPU-hours",
        "labels": (
            "owner label `team` on the leader template only (S1/S2), on both templates (S3); "
            "LWS labels and shared-chart app.kubernetes.io/* labels on every pod"
        ),
    }


def run_attribution(outdir: Path, fractions: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Write one OpenCost payload per scenario, run unalloc on each, collect results."""
    pods = build_pods(fractions)
    scenarios: dict[str, Any] = {}
    for sc in SCENARIOS:
        target = outdir / sc.id
        payload = build_payload(pods, owner_on_workers=sc.owner_on_workers, key_style=sc.key_style)
        write_payloads(target, opencost=payload)
        rows = load_rows(target, ["opencost"])
        scenarios[sc.id] = {
            "title": sc.title,
            "payload": str(Path(sc.id) / "opencost_allocation.json"),
            "owner_on_workers": sc.owner_on_workers,
            "label_key_style": sc.key_style,
            "summary": evaluate(rows, pods, sc.fallback),
            "fallback_variants": [_compact(evaluate(rows, pods, fb)) for fb in sc.variants],
        }
    sensitivity: list[dict[str, Any]] = [
        {
            "mean_comm_fraction": round(statistics.fmean(p.comm_fraction for p in pods), 4),
            "source": "measured",
            **_sens(communication_economics(pods)),
        }
    ]
    for frac in COMM_FRACTION_SENSITIVITY:
        econ = communication_economics(build_pods(fractions, target_mean=frac))
        sensitivity.append(
            {"mean_comm_fraction": frac, "source": "measured scaled", **_sens(econ)}
        )
    return {
        "assumptions": assumptions(),
        "measured_comm_fractions": {k: list(v) for k, v in fractions.items()},
        "pods": [
            {
                "allocation": p.allocation_name,
                "team": p.tenant.team,
                "role": p.role,
                "group": p.group,
                "worker_index": p.worker_index,
                "comm_fraction": round(p.comm_fraction, 4),
                "gpu_hours": HOURS,
                "compute_hours": round(p.compute_hours, 2),
                "comm_hours": round(p.comm_hours, 2),
                "in_pod_idle_hours": round(HOURS - p.busy_hours, 2),
                "total_cost_usd": round(p.gpu_cost + p.cpu_cost + p.ram_cost, 2),
            }
            for p in pods
        ],
        "scenarios": scenarios,
        "communication": {**communication_economics(pods), "sensitivity": sensitivity},
        "canonicalization": canonicalization_findings(pods),
    }


def _sens(econ: dict[str, Any]) -> dict[str, float]:
    return {
        "communication_usd_total": econ["communication_usd_total"],
        "communication_usd_on_workers": econ["communication_usd_on_workers"],
        "cross_tenant_shift_usd": econ["cross_tenant_shift_usd"],
    }
