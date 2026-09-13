from __future__ import annotations

from unalloc.core.normalize import canonical_key, normalize_labels


def test_canonical_key_strips_provider_noise():
    assert canonical_key("label_costCenter") == "cost_center"
    assert canonical_key("app.kubernetes.io/name") == "name"
    # canonical_key normalizes shape only; the alias table lives in normalize_labels
    assert canonical_key("  Team_ID ") == "team_id"


def test_aliases_collapse_spellings_onto_one_dimension():
    labels = normalize_labels({"team_id": "search", "environment": "prod"})
    assert labels == {"team": "search", "environment": "prod"}


def test_explicit_key_wins_over_alias():
    labels = normalize_labels({"team": "platform", "owner": "search"})
    assert labels["team"] == "platform"


def test_stacked_prefixes_are_all_stripped():
    # OpenCost's Prometheus-style key for app.kubernetes.io/name.
    assert canonical_key("label_app_kubernetes_io_name") == "name"
    assert canonical_key("label_app_kubernetes_io_name") == canonical_key("app.kubernetes.io/name")


def test_collisions_resolve_the_same_regardless_of_key_order():
    keys = [
        ("app.kubernetes.io/name", "vllm"),
        ("leaderworkerset.sigs.k8s.io/name", "search-llama-70b"),
    ]
    forward = normalize_labels(dict(keys))
    backward = normalize_labels(dict(reversed(keys)))
    assert forward == backward


def test_canonically_written_key_beats_rewritten_and_path_keys():
    labels = normalize_labels({"app.kubernetes.io/name": "chart", "labelName": "x", "name": "api"})
    assert labels["name"] == "api"


def test_empty_values_are_dropped_not_stored_as_blank():
    assert normalize_labels({"team": "  ", "project": None, "env": "prod"}) == {
        "environment": "prod"
    }
