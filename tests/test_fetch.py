"""The fetch path, against a real local HTTP server rather than mocked httpx.

Fixtures only prove `parse()`. These prove that URLs, auth headers and
pagination are right, which is where adapters actually break in production.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from unalloc.sources import AnthropicSource, OpenAISource, OpenCostSource

START = datetime(2026, 8, 1, tzinfo=UTC)
END = datetime(2026, 9, 1, tzinfo=UTC)

Handler = Callable[[str, dict[str, list[str]], dict[str, str]], Any]


@pytest.fixture
def server() -> Iterator[Callable[[Handler], str]]:
    started: list[HTTPServer] = []

    def start(handler: Handler) -> str:
        class _Request(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlparse(self.path)
                body = json.dumps(
                    handler(url.path, parse_qs(url.query), dict(self.headers))
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        httpd = HTTPServer(("127.0.0.1", 0), _Request)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start
    for httpd in started:
        httpd.shutdown()


def _bucket(amount: str, project: str) -> dict[str, Any]:
    return {
        "start_time": int(START.timestamp()),
        "end_time": int(END.timestamp()),
        "results": [{"amount": {"value": amount, "currency": "usd"}, "project_id": project}],
    }


def test_provider_default_base_url_survives_an_empty_env_var():
    # The CLI passes base_url=None when UNALLOC_OPENAI_URL is unset. That used
    # to override the public default and make every live fetch fail.
    assert OpenAISource(base_url=None).base_url == "https://api.openai.com/v1"
    assert AnthropicSource(base_url="").base_url == "https://api.anthropic.com/v1"
    assert OpenAISource(base_url="http://proxy/v1/").base_url == "http://proxy/v1"


def test_openai_follows_pagination_cursors(server):
    pages = {
        None: {"data": [_bucket("10.50", "proj_a")], "has_more": True, "next_page": "p2"},
        "p2": {"data": [_bucket("4.25", "proj_b")], "has_more": False, "next_page": None},
    }
    seen: list[dict[str, str]] = []

    def handler(path, query, headers):
        assert path == "/v1/organization/costs"
        seen.append(headers)
        return pages[query.get("page", [None])[0]]

    source = OpenAISource(base_url=server(handler) + "/v1", token="sk-admin")
    rows = source.fetch(START, END)
    assert sum((r.amount_usd for r in rows), Decimal("0")) == Decimal("14.75")
    assert {r.labels["project"] for r in rows} == {"proj_a", "proj_b"}
    assert seen[0]["Authorization"] == "Bearer sk-admin"


def test_anthropic_uses_x_api_key_not_bearer(server):
    captured: dict[str, str] = {}

    def handler(path, query, headers):
        assert path == "/v1/organizations/cost_report"
        captured.update(headers)
        return {
            "data": [
                {
                    "starting_at": "2026-08-01T00:00:00Z",
                    "ending_at": "2026-09-01T00:00:00Z",
                    "results": [{"amount": "12.00", "workspace_id": "wrk_1"}],
                }
            ],
            "has_more": False,
        }

    rows = AnthropicSource(base_url=server(handler) + "/v1", token="sk-ant-admin").fetch(
        START, END
    )
    assert rows[0].amount_usd == Decimal("12.00")
    assert captured["x-api-key"] == "sk-ant-admin"
    assert "Authorization" not in captured
    assert captured["anthropic-version"]


def test_opencost_component_sum_stays_exact_without_total(server):
    window = {
        "a": {
            "name": "a",
            "properties": {"namespace": "ns"},
            "cpuCost": 0.1,
            "ramCost": 0.2,
        }
    }
    rows = OpenCostSource(base_url=server(lambda *_: {"data": [window]})).fetch(START, END)
    # float(0.1) + float(0.2) == 0.30000000000000004
    assert rows[0].amount_usd == Decimal("0.3")


def test_missing_base_url_names_the_env_var():
    with pytest.raises(ValueError, match="UNALLOC_OPENCOST_URL"):
        OpenCostSource().fetch(START, END)
