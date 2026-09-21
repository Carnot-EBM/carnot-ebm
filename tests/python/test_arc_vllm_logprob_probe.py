"""REQ-INFRA-7090 tests the bounded scored-server log-probability probe."""

from __future__ import annotations

import ast
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic.arc_vllm_logprob_probe import run_logprob_probe


ROOT = Path(__file__).resolve().parents[2]
KERNEL_MAIN = ROOT / "scripts" / "kaggle" / "submission_kernel" / "main.py"
PROBE_MODULE = ROOT / "python" / "carnot" / "agentic" / "arc_vllm_logprob_probe.py"


def _openapi() -> dict[str, Any]:
    return {
        "paths": {
            "/v1/completions": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CompletionRequest"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "CompletionRequest": {
                    "type": "object",
                    "properties": {
                        "logprobs": {"type": "integer"},
                        "prompt_logprobs": {"type": "integer"},
                        "allowed_token_ids": {"type": "array", "items": {"type": "integer"}},
                        "logprob_token_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                }
            }
        },
    }


@contextmanager
def _fake_vllm(mode: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            return

        def _send_json(self, status: int, body: Any) -> None:
            encoded = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/openapi.json":
                self._send_json(200, _openapi())
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            requests.append({"path": self.path, "payload": payload})
            if mode == "hang":
                time.sleep(2)
                self._send_json(200, {})
                return
            if self.path == "/tokenize":
                prompt = str(payload.get("prompt", ""))
                if len(prompt) == 2 and prompt.startswith(" "):
                    token_id = 100 + ord(prompt[1]) - ord("A")
                    self._send_json(200, {"count": 1, "tokens": [token_id]})
                else:
                    self._send_json(200, {"count": 12, "tokens": list(range(12))})
                return
            if self.path == "/v1/completions" and mode == "reject_logprobs":
                self._send_json(400, {"error": "logprobs are disabled"})
                return
            if self.path == "/v1/completions":
                scores = {f" {letter}": -float(index) for index, letter in enumerate("ABCDEF")}
                self._send_json(
                    200,
                    {
                        "choices": [
                            {
                                "text": " A",
                                "logprobs": {"top_logprobs": [scores]},
                            }
                        ]
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("the execution sandbox denies loopback sockets")
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_good_server_returns_full_option_distributions(tmp_path: Path) -> None:
    """REQ-INFRA-7090 records complete distributions from the live server."""
    out_path = tmp_path / "probe.json"
    launch_argv = ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8919"]

    with _fake_vllm("good") as (base_url, requests):
        result = run_logprob_probe(
            base_url,
            "m",
            out_path,
            budget_s=8,
            server_launch_argv=launch_argv,
        )

    assert result["honest_verdict"] == "complete_exact_declared_option_score_probe"
    assert result["server_launch_argv"] == launch_argv
    assert result["phases"]["logprob_request_variants"]["fixture_validation"]["count"] == 45
    variants = result["phases"]["logprob_request_variants"]["variants"]
    assert variants["top_n_logprobs"]["complete_option_distribution_count"] == 90
    assert variants["logprob_token_ids"]["complete_option_distribution_count"] == 90
    assert json.loads(out_path.read_text()) == result
    completion_payloads = [row["payload"] for row in requests if row["path"] == "/v1/completions"]
    assert completion_payloads
    assert all("max_logprobs" not in payload for payload in completion_payloads)


def test_logprob_http_400_returns_blocked_verdict(tmp_path: Path) -> None:
    """REQ-INFRA-7090 reports a log-probability rejection without raising."""
    out_path = tmp_path / "probe.json"

    with _fake_vllm("reject_logprobs") as (base_url, _requests):
        result = run_logprob_probe(base_url, "m", out_path, budget_s=3)

    assert result["honest_verdict"] == "blocked_logprobs_http_400"
    assert out_path.exists()
    assert json.loads(out_path.read_text())["honest_verdict"] == "blocked_logprobs_http_400"


def test_hung_server_hits_timeout_and_returns(tmp_path: Path) -> None:
    """REQ-INFRA-7090 keeps a hung HTTP call inside the total wall budget."""
    out_path = tmp_path / "probe.json"
    budget_s = 0.6

    with _fake_vllm("hang") as (base_url, _requests):
        started = time.monotonic()
        result = run_logprob_probe(base_url, "m", out_path, budget_s=budget_s)
        elapsed = time.monotonic() - started

    assert elapsed < budget_s
    assert result["honest_verdict"].startswith("blocked_")
    assert out_path.exists()


def test_submission_probe_is_off_by_default() -> None:
    """REQ-INFRA-7090 keeps the scored submission path unchanged by default."""
    tree = ast.parse(KERNEL_MAIN.read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RUN_LOGPROB_PROBE"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value is False


def test_probe_has_no_server_lifecycle_commands() -> None:
    """REQ-INFRA-7090 leaves the scored server command and process untouched."""
    source = PROBE_MODULE.read_text()

    assert "--max-logprobs" not in source
    assert "subprocess" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
