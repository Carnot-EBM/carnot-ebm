"""Minimal HTTP/REST surface for certified abstention scoring.

This module intentionally uses the Python standard library HTTP server.  It is
the third product integration surface for the Exp 3771 certified abstention
operating point and delegates scoring to the existing `score_candidates` path.

Spec: REQ-SPOE-3801, SCENARIO-SPOE-3801.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, cast

from carnot.pipeline import second_pair_detector as spd


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
HTTP_SURFACE = "http_rest_score_candidates"
POST_PATH = "/v1/score-candidates"
MAX_BODY_BYTES = 1_000_000


class AbstentionHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying Carnot scoring configuration for request handlers."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        root: Path | str = REPO_ROOT,
        examples: Sequence[spd.LabeledDetectorExample] | None = None,
    ) -> None:
        self.carnot_root = Path(root)
        self.carnot_examples = examples
        super().__init__(server_address, AbstentionRequestHandler)


class AbstentionRequestHandler(BaseHTTPRequestHandler):
    """Handle JSON POSTs for the abstention-capable score surface."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != POST_PATH:
            self._send_json(404, _error_response("NOT_FOUND", f"POST {POST_PATH} only"))
            return
        try:
            payload = self._read_json_payload()
            server = cast(AbstentionHTTPServer, self.server)
            response = score_candidates_http_payload(
                payload,
                root=server.carnot_root,
                examples=server.carnot_examples,
            )
        except json.JSONDecodeError as exc:
            self._send_json(400, _error_response("INVALID_JSON", str(exc)))
            return
        except ValueError as exc:
            self._send_json(400, _error_response("INVALID_REQUEST", str(exc)))
            return
        self._send_json(200, response)

    def _read_json_payload(self) -> object:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length <= 0:
            raise ValueError("request body must not be empty")
        if length > MAX_BODY_BYTES:  # pragma: no cover
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:  # pragma: no cover
        return


def make_server(
    server_address: tuple[str, int],
    *,
    root: Path | str = REPO_ROOT,
    examples: Sequence[spd.LabeledDetectorExample] | None = None,
) -> AbstentionHTTPServer:
    """Create an in-process HTTP server for tests or local integration."""

    return AbstentionHTTPServer(server_address, root=root, examples=examples)


def score_candidates_http_payload(
    payload: object,
    *,
    root: Path | str = REPO_ROOT,
    examples: Sequence[spd.LabeledDetectorExample] | None = None,
    domain: str | None = None,
    abstention_mode: bool | None = None,
    abstention_threshold: float | None = None,
) -> JsonDict:
    """Score a JSON-compatible HTTP payload through the product verifier path."""

    candidates = _payload_candidates(payload)
    request_domain = _payload_field(payload, "domain", domain)
    request_abstention_mode = bool(_payload_field(payload, "abstention_mode", abstention_mode))
    threshold_value = _payload_field(payload, "abstention_threshold", abstention_threshold)
    result = spd.score_candidates(
        candidates,
        root=root,
        examples=examples,
        default_domain=str(request_domain) if request_domain is not None else None,
        abstention_mode=request_abstention_mode,
        abstention_threshold=float(threshold_value) if threshold_value is not None else None,
    )
    abstention_summary = result.get("abstention_mode", {})
    return {
        "surface": HTTP_SURFACE,
        "abstention_mode_enabled": request_abstention_mode,
        "batch": {"n_candidates": len(candidates)},
        "scores": [
            _http_row(row, abstention_summary, abstention_enabled=request_abstention_mode)
            for row in result["scores"]
        ],
        **({"abstention_mode": abstention_summary} if request_abstention_mode else {}),
    }


def _payload_candidates(payload: object) -> list[JsonDict]:
    if isinstance(payload, list):
        raw_candidates = payload
    elif isinstance(payload, Mapping):
        if "candidates" in payload:
            raw_candidates = payload["candidates"]
        elif "candidate" in payload:
            raw_candidates = [payload["candidate"]]
        else:
            raise ValueError("provide candidate or candidates")
    else:
        raise ValueError("request body must be a JSON object or list")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates must be a JSON list")
    if not raw_candidates:
        raise ValueError("candidate batch is empty")

    candidates: list[JsonDict] = []
    for idx, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate at index {idx} must be a JSON object")
        candidates.append(dict(candidate))
    return candidates


def _payload_field(payload: object, field: str, override: Any) -> Any:
    if override is not None:
        return override
    if isinstance(payload, Mapping):
        return payload.get(field)
    return None


def _http_row(
    row: Mapping[str, Any],
    abstention_summary: Mapping[str, Any],
    *,
    abstention_enabled: bool,
) -> JsonDict:
    if not abstention_enabled:
        return dict(row)

    cert = row.get("certified_abstention")
    metadata = cert if isinstance(cert, Mapping) else abstention_summary
    route_to_review = bool(row.get("route_to_review", row.get("abstained", False)))
    if row.get("calibrated_error_score") is None:
        route_to_review = True
    threshold = metadata.get("threshold", metadata.get("certified_threshold"))
    risk = metadata.get("certified_risk_bound")
    return {
        "candidate_id": row.get("candidate_id"),
        "domain": row.get("domain"),
        "verdict": "abstain" if route_to_review else "confident",
        "score": row.get("abstention_score"),
        "coverage": metadata.get("coverage"),
        "risk": risk,
        "delta": metadata.get("delta"),
        "threshold": threshold,
        "threshold_source": metadata.get("threshold_source"),
    }


def _error_response(code: str, detail: str) -> JsonDict:
    return {"error": True, "error_code": code, "detail": detail}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    server = make_server((args.host, args.port))
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
