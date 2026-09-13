"""The ledger explorer: how a month of AI spend resolves to owners.

    python -m case_studies.ui                      # serve on http://127.0.0.1:8765
    python -m case_studies.ui --export demo.html   # self-contained page, no server

The page embeds the normalized ledger rows - produced by unalloc's real
adapters - and re-runs the attribution grouping in the browser so dimension
and fallback changes are instant. Each dataset also carries the Python CLI's
own `team` result, and the page shows whether the two agree.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.hybrid_e2e import workload
from unalloc.cli import FIXTURE_DIR
from unalloc.core.attribute import attribute
from unalloc.core.models import CostRow
from unalloc.sources import REGISTRY

APP = Path(__file__).with_name("app.html")

# Curated, ordered datasets with a sentence of context each. Anything else
# found under case_studies/results is appended after these.
CURATED: dict[str, tuple[str, str]] = {
    "hybrid_org": (
        "Hybrid org · gateway + cluster",
        "One month for a mid-size org: a LiteLLM gateway in front of OpenAI and Anthropic, "
        "plus a vLLM fleet, a vector DB and batch jobs on Kubernetes. Try falling back to "
        "namespace, then feature.",
    ),
    "fixtures": (
        "README fixtures · all four sources",
        "The payloads bundled with unalloc. The $19K vLLM pod has a costCenter but no team.",
    ),
    "distributed/s1": (
        "Distributed serving · owner label on leader pods only",
        "LeaderWorkerSet replicas where team is set on the leader template only: every worker "
        "pod's GPU bill is unowned. Fall back to name and see it land in a chart name, not a team.",
    ),
    "distributed/s3": (
        "Distributed serving · owner label fixed on workers",
        "The same fleet after adding team to the worker template. Only cluster idle remains.",
    ),
    "kv_cache/compute": (
        "Shared vLLM pod · split by measured compute",
        "One 8-GPU fleet split across four callers by simulated engine-step time, with idle "
        "time left as an unlabeled overhead row.",
    ),
    "kv_cache/memory": (
        "Shared vLLM pod · split by KV-cache memory",
        "The same traffic split by KV block-seconds. The empty KV pool becomes a large "
        "unowned overhead row.",
    ),
    "torch_kv/tokens": (
        "Real inference · split by tokens",
        "A tiny transformer actually served the trace; its pod bill split by token counts.",
    ),
    "torch_kv/compute_seconds": (
        "Real inference · split by measured compute",
        "The same trace split by measured prefill + decode seconds. Compare search's share.",
    ),
}


def _rows_for(key: str) -> list[CostRow]:
    if key == "hybrid_org":
        data = workload.build()
        return REGISTRY["opencost"]().parse(data["opencost"]) + REGISTRY["litellm"]().parse(
            data["litellm"]
        )
    folder = FIXTURE_DIR if key == "fixtures" else common.RESULTS / key
    rows: list[CostRow] = []
    for name, filename in common.FIXTURE_FILES.items():
        if (folder / filename).exists():
            rows.extend(REGISTRY[name]().from_fixture(folder / filename))
    return rows


def discover() -> list[str]:
    keys = [k for k in CURATED if k in ("hybrid_org", "fixtures") or (common.RESULTS / k).is_dir()]
    if common.RESULTS.exists():
        for path in sorted(common.RESULTS.rglob("opencost_allocation.json")):
            key = str(path.parent.relative_to(common.RESULTS))
            if key not in keys:
                keys.append(key)
    return keys


def _collapse(rows: list[CostRow]) -> list[dict[str, Any]]:
    """Merge rows that differ only in time window, e.g. a gateway's daily logs.

    Attribution only reads source, labels and amount, so summing rows with the
    same identity changes no result - it just makes the ledger readable.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        labels = {k: v for k, v in r.labels.items() if k != "aggregation"}
        key = (r.source, r.name, tuple(sorted(labels.items())))
        entry = merged.setdefault(key, {"s": r.source, "n": r.name, "a": Decimal("0"), "l": labels})
        entry["a"] += r.amount_usd
    return [{**e, "a": str(e["a"])} for e in merged.values()]


def payload(keys: list[str] | None = None) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for key in keys or discover():
        rows = _rows_for(key)
        if not rows:
            continue
        label, note = CURATED.get(key, (key.replace("/", " · "), ""))
        datasets[key] = {
            "label": label,
            "note": note,
            "expected_team_pct": str(attribute(rows, "team").unallocated_pct),
            "rows": _collapse(rows),
        }
    return {
        "default": next(iter(datasets)),
        "generated": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "datasets": datasets,
    }


def fragment(keys: list[str] | None = None) -> str:
    """The page body: styles, markup, script and data, with no document shell."""
    data = json.dumps(payload(keys), separators=(",", ":")).replace("</", "<\\/")
    script = f'<script id="unalloc-data" type="application/json">{data}</script>'
    return APP.read_text().replace("<!--DATA-->", script)


def document(keys: list[str] | None = None) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>\n{fragment(keys)}\n</body>\n</html>\n"
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.split("?")[0] in ("/", "/index.html"):
            body, status, kind = document().encode(), 200, "text/html; charset=utf-8"
        else:
            body, status, kind = b"not found", 404, "text/plain"
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path, help="write a self-contained page and exit")
    parser.add_argument("--fragment", action="store_true",
                        help="with --export, omit the <html> shell (for embedding)")
    args = parser.parse_args(argv)

    if args.export:
        keys = [k for k in CURATED if k in discover()]
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(fragment(keys) if args.fragment else document(keys))
        print(f"wrote {args.export}")
        return 0

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"unalloc ledger explorer on http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
