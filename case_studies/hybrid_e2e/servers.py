"""Mock provider APIs that behave like the real ones where it matters to unalloc.

Each mock enforces the provider's real auth scheme and pages its responses, so
a wrong header or an ignored cursor fails loudly instead of passing quietly:

* OpenCost   GET /allocation                     no auth
* LiteLLM    GET /spend/logs                     Bearer master key; Postgres-backed
                                                 when UNALLOC_CASE_PG_DSN is set
* OpenAI     GET /v1/organization/costs          Bearer admin key; cursor pages
* Anthropic  GET /v1/organizations/cost_report   x-api-key; cursor pages

Page sizes are deliberately small so pagination is always exercised.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

PAGE_SIZE = 7

Reply = tuple[int, Any]
Route = Callable[[dict[str, list[str]], dict[str, str]], Reply]


@dataclass
class MockServer:
    name: str
    routes: dict[str, Route]
    hits: list[str] = field(default_factory=list)
    httpd: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.httpd is not None
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def start(self) -> MockServer:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                server.hits.append(parsed.path)
                route = server.routes.get(parsed.path)
                if route is None:
                    status, body = 404, {"error": f"no route {parsed.path}"}
                else:
                    status, body = route(parse_qs(parsed.query), dict(self.headers))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def _paged(items: list[Any], query: dict[str, list[str]]) -> dict[str, Any]:
    start = int(query.get("page", ["0"])[0])
    limit = min(int(query.get("limit", [str(PAGE_SIZE)])[0]), PAGE_SIZE)
    chunk = items[start : start + limit]
    more = start + limit < len(items)
    return {"data": chunk, "has_more": more, "next_page": str(start + limit) if more else None}


def _unauthorised(message: str) -> Reply:
    return 401, {"error": {"type": "authentication_error", "message": message}}


# --- LiteLLM spend log store --------------------------------------------------------------


class SpendStore:
    """LiteLLM keeps spend logs in Postgres; mirror that when a DSN is available."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.backend = "memory"
        dsn = os.environ.get("UNALLOC_CASE_PG_DSN")
        if not dsn:
            return
        try:
            import psycopg
        except ImportError:
            return
        self._dsn = dsn
        try:
            self._load(psycopg)
        except psycopg.OperationalError:
            return
        self.backend = "postgres"

    def _load(self, psycopg: Any) -> None:
        with psycopg.connect(self._dsn, connect_timeout=3) as conn:
            conn.execute('DROP TABLE IF EXISTS "LiteLLM_SpendLogs"')
            # Column names follow LiteLLM's schema, including the quoted camelCase.
            conn.execute(
                """
                CREATE TABLE "LiteLLM_SpendLogs" (
                    request_id text PRIMARY KEY,
                    model text,
                    spend double precision,
                    prompt_tokens bigint,
                    completion_tokens bigint,
                    total_tokens bigint,
                    "startTime" timestamptz,
                    "endTime" timestamptz,
                    team_id text,
                    api_key_alias text,
                    request_tags jsonb,
                    metadata jsonb,
                    cache_hit text
                )
                """
            )
            with conn.cursor() as cur:
                cur.executemany(
                    'INSERT INTO "LiteLLM_SpendLogs" VALUES '
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            r["request_id"], r["model"], float(r["spend"]), r["prompt_tokens"],
                            r["completion_tokens"], r["total_tokens"], r["startTime"],
                            r["endTime"], r.get("team_id"), r.get("api_key_alias"),
                            json.dumps(r["request_tags"]), json.dumps(r["metadata"]),
                            str(r["cache_hit"]).lower(),
                        )
                        for r in self.records
                    ],
                )

    def rows(self) -> list[dict[str, Any]]:
        if self.backend == "memory":
            return self.records
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            fetched = conn.execute(
                'SELECT * FROM "LiteLLM_SpendLogs" ORDER BY "startTime", request_id'
            ).fetchall()
        return [
            {
                **row,
                # Postgres hands back double precision; LiteLLM's API does too.
                "spend": row["spend"],
                "startTime": row["startTime"].isoformat(),
                "endTime": row["endTime"].isoformat(),
                "cache_hit": row["cache_hit"] == "true",
            }
            for row in fetched
        ]


# --- the stack ----------------------------------------------------------------------------


@dataclass
class Stack:
    opencost: MockServer
    litellm: MockServer
    openai: MockServer
    anthropic: MockServer
    litellm_backend: str

    MASTER_KEY = "sk-litellm-master"
    OPENAI_KEY = "sk-admin-openai"
    ANTHROPIC_KEY = "sk-ant-admin-anthropic"

    def env(self) -> dict[str, str]:
        return {
            "UNALLOC_OPENCOST_URL": self.opencost.url,
            "UNALLOC_LITELLM_URL": self.litellm.url,
            "UNALLOC_LITELLM_KEY": self.MASTER_KEY,
            "UNALLOC_OPENAI_URL": self.openai.url + "/v1",
            "OPENAI_ADMIN_KEY": self.OPENAI_KEY,
            "UNALLOC_ANTHROPIC_URL": self.anthropic.url + "/v1",
            "ANTHROPIC_ADMIN_KEY": self.ANTHROPIC_KEY,
        }


@contextmanager
def serve(workload: dict[str, Any]) -> Iterator[Stack]:
    store = SpendStore(workload["litellm"]["data"])

    def opencost(query: dict[str, list[str]], headers: dict[str, str]) -> Reply:
        return 200, workload["opencost"]

    def litellm(query: dict[str, list[str]], headers: dict[str, str]) -> Reply:
        if headers.get("Authorization") != f"Bearer {Stack.MASTER_KEY}":
            return _unauthorised("invalid litellm key")
        return 200, {"data": store.rows()}

    def openai(query: dict[str, list[str]], headers: dict[str, str]) -> Reply:
        if headers.get("Authorization") != f"Bearer {Stack.OPENAI_KEY}":
            return _unauthorised("invalid admin key")
        return 200, {"object": "page", **_paged(workload["openai_buckets"], query)}

    def anthropic(query: dict[str, list[str]], headers: dict[str, str]) -> Reply:
        if "Authorization" in headers:
            return _unauthorised("admin API takes x-api-key, not Authorization")
        if headers.get("x-api-key") != Stack.ANTHROPIC_KEY:
            return _unauthorised("invalid x-api-key")
        return 200, _paged(workload["anthropic_buckets"], query)

    servers = [
        MockServer("opencost", {"/allocation": opencost}).start(),
        MockServer("litellm", {"/spend/logs": litellm}).start(),
        MockServer("openai", {"/v1/organization/costs": openai}).start(),
        MockServer("anthropic", {"/v1/organizations/cost_report": anthropic}).start(),
    ]
    try:
        yield Stack(*servers, litellm_backend=store.backend)
    finally:
        for server in servers:
            server.stop()
