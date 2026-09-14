"""A stand-in for vLLM's OpenAI server, for dry-running bench.py without a GPU.

    python -m case_studies.gpu_validation.fake_server --port 8001

Implements only what bench.py touches: /health, /version, /metrics and
streaming /v1/completions with a final usage chunk. Latency is a crude function
of token counts; the point is to exercise the client, not to model a GPU.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

BLOCK = 16
_lock = threading.Lock()
_seen_blocks: set[tuple[int, ...]] = set()
_state = {"running": 0, "tokens_held": 0, "hits": 0.0, "queries": 0.0, "prompt": 0.0, "gen": 0.0}
CAPACITY_TOKENS = 200_000


def _cached_prefix(prompt: list[int]) -> int:
    cached = 0
    with _lock:
        for end in range(BLOCK, len(prompt) + 1, BLOCK):
            block = tuple(prompt[:end])
            if block in _seen_blocks:
                cached = end
            else:
                break
        for end in range(BLOCK, len(prompt) + 1, BLOCK):
            _seen_blocks.add(tuple(prompt[:end]))
    return min(cached, len(prompt) - 1) // BLOCK * BLOCK


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({})
        elif self.path == "/version":
            self._json({"version": "fake"})
        elif self.path == "/metrics":
            with _lock:
                s = dict(_state)
            usage = s["tokens_held"] / CAPACITY_TOKENS
            text = "\n".join(
                [
                    "# HELP vllm:kv_cache_usage_perc fake",
                    f'vllm:kv_cache_usage_perc{{model_name="fake"}} {usage}',
                    f'vllm:num_requests_running{{model_name="fake"}} {s["running"]}',
                    'vllm:num_requests_waiting{model_name="fake"} 0',
                    f'vllm:prefix_cache_hits_total{{model_name="fake"}} {s["hits"]}',
                    f'vllm:prefix_cache_queries_total{{model_name="fake"}} {s["queries"]}',
                    f'vllm:prompt_tokens_total{{model_name="fake"}} {s["prompt"]}',
                    f'vllm:generation_tokens_total{{model_name="fake"}} {s["gen"]}',
                ]
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt: list[int] = body["prompt"]
        n_out: int = body["max_tokens"]
        cached = _cached_prefix(prompt)
        with _lock:
            _state["running"] += 1
            _state["tokens_held"] += len(prompt)
            _state["hits"] += cached
            _state["queries"] += len(prompt)
            _state["prompt"] += len(prompt)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep((len(prompt) - cached) * 2e-6)
        for i in range(n_out):
            chunk = {"choices": [{"index": 0, "text": " x", "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if i % 32 == 0:
                self.wfile.flush()
            time.sleep(0.0005)
        with _lock:
            _state["running"] -= 1
            _state["tokens_held"] -= len(prompt)
            _state["gen"] += n_out
        usage = {"prompt_tokens": len(prompt), "completion_tokens": n_out,
                 "total_tokens": len(prompt) + n_out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args: Any) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
