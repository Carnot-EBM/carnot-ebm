"""Tests for the Exp 3801 abstention HTTP/REST surface.

Spec: REQ-SPOE-3801, SCENARIO-SPOE-3801.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import threading

import pytest

from carnot.pipeline import abstention_http_rest as rest
from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


ROOT = Path(__file__).resolve().parents[2]


def _domain_examples(domain: str = "math", *, n: int = 80) -> list[spd.LabeledDetectorExample]:
    examples: list[spd.LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        ensemble = 0.95 - 0.004 * idx if label else 0.05 + 0.001 * (idx - n // 2)
        confidence_error = 0.82 - 0.003 * idx if label else 0.18 + 0.001 * (idx - n // 2)
        examples.append(
            spd.LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence_error,
                example_id=f"{domain}-3801-{idx}",
            )
        )
    return examples


def _batch_candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "confident-error",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "uncertain-midpoint",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.5,
        },
    ]


@contextmanager
def _running_server() -> Iterator[rest.AbstentionHTTPServer]:
    server = rest.make_server(("127.0.0.1", 0), root=ROOT, examples=_domain_examples())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def _post_json(
    server: rest.AbstentionHTTPServer,
    payload: object,
    *,
    path: str = rest.POST_PATH,
) -> tuple[int, dict[str, object]]:
    host, port = server.server_address
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=5.0)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, json.loads(data)
    finally:
        conn.close()


def _post_raw(
    server: rest.AbstentionHTTPServer,
    body: bytes,
    *,
    content_length: str | None = None,
) -> tuple[int, dict[str, object]]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5.0)
    try:
        conn.putrequest("POST", rest.POST_PATH)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)) if content_length is None else content_length)
        conn.endheaders(body)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, json.loads(data)
    finally:
        conn.close()


def test_scenario_spoe_3801_http_batch_default_off_and_abstention_on() -> None:
    """SCENARIO-SPOE-3801: HTTP POST preserves default-off and opt-in verdicts."""

    config = abstention.load_certified_abstention_config()
    with _running_server() as server:
        default_status, default_payload = _post_json(
            server,
            {"domain": "math", "candidates": _batch_candidates()},
        )
        enabled_status, enabled_payload = _post_json(
            server,
            {
                "domain": "math",
                "abstention_mode": True,
                "candidates": _batch_candidates(),
            },
        )

    assert default_status == 200
    assert default_payload["surface"] == rest.HTTP_SURFACE
    assert default_payload["abstention_mode_enabled"] is False
    assert default_payload["batch"] == {"n_candidates": 2}
    assert all("verdict" not in row for row in default_payload["scores"])  # type: ignore[index]

    assert enabled_status == 200
    assert enabled_payload["abstention_mode_enabled"] is True
    assert enabled_payload["batch"] == {"n_candidates": 2}
    rows = {row["candidate_id"]: row for row in enabled_payload["scores"]}  # type: ignore[index]
    confident = rows["confident-error"]
    uncertain = rows["uncertain-midpoint"]

    assert confident["verdict"] == "confident"
    assert confident["score"] >= config.threshold
    assert confident["coverage"] == pytest.approx(config.coverage)
    assert confident["risk"] == pytest.approx(config.certified_risk_bound)
    assert confident["delta"] == pytest.approx(config.delta)
    assert confident["threshold"] == pytest.approx(config.threshold)
    assert confident["threshold_source"] == config.threshold_source

    assert uncertain["verdict"] == "abstain"
    assert uncertain["score"] < config.threshold
    assert uncertain["coverage"] == pytest.approx(config.coverage)
    assert uncertain["risk"] == pytest.approx(config.certified_risk_bound)
    assert uncertain["delta"] == pytest.approx(config.delta)
    assert uncertain["threshold"] == pytest.approx(config.threshold)


def test_req_spoe_3801_single_candidate_payload_and_http_errors() -> None:
    """REQ-SPOE-3801: single-candidate POSTs work and malformed input fails closed."""

    with _running_server() as server:
        status, payload = _post_json(
            server,
            {
                "domain": "math",
                "abstention_mode": True,
                "candidate": _batch_candidates()[0],
            },
        )
        missing_status, missing_payload = _post_json(
            server,
            {"domain": "math"},
        )
        not_found_status, not_found_payload = _post_json(
            server,
            {"candidates": _batch_candidates()},
            path="/missing",
        )
        invalid_json_status, invalid_json_payload = _post_raw(server, b"{")
        empty_status, empty_payload = _post_raw(server, b"")
        bad_length_status, bad_length_payload = _post_raw(
            server,
            b"{}",
            content_length="not-an-int",
        )

    assert status == 200
    assert payload["batch"] == {"n_candidates": 1}
    assert payload["scores"][0]["verdict"] == "confident"  # type: ignore[index]

    assert missing_status == 400
    assert missing_payload["error"] is True
    assert missing_payload["error_code"] == "INVALID_REQUEST"
    assert "candidate" in str(missing_payload["detail"])

    assert not_found_status == 404
    assert not_found_payload["error"] is True
    assert not_found_payload["error_code"] == "NOT_FOUND"

    assert invalid_json_status == 400
    assert invalid_json_payload["error_code"] == "INVALID_JSON"

    assert empty_status == 400
    assert empty_payload["error_code"] == "INVALID_REQUEST"
    assert "empty" in str(empty_payload["detail"])

    assert bad_length_status == 400
    assert bad_length_payload["error_code"] == "INVALID_REQUEST"
    assert "Content-Length" in str(bad_length_payload["detail"])


def test_req_spoe_3801_direct_payload_accepts_json_list_and_threshold_override() -> None:
    """REQ-SPOE-3801: direct JSON-list batches use real scoring and explicit overrides."""

    payload = rest.score_candidates_http_payload(
        _batch_candidates(),
        root=ROOT,
        examples=_domain_examples(),
        domain="math",
        abstention_mode=True,
        abstention_threshold=0.5,
    )

    assert payload["batch"] == {"n_candidates": 2}
    assert payload["abstention_mode"]["operator_threshold_override"] is True
    assert all(row["verdict"] == "confident" for row in payload["scores"])


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (None, "JSON object or list"),
        ({"candidates": {}}, "JSON list"),
        ([], "candidate batch is empty"),
        ([1], "candidate at index 0"),
    ],
)
def test_req_spoe_3801_payload_validation_edges(payload: object, match: str) -> None:
    """REQ-SPOE-3801: malformed candidate payloads fail before scoring."""

    with pytest.raises(ValueError, match=match):
        rest._payload_candidates(payload)


def test_req_spoe_3801_code_domain_abstain_row_still_has_certified_metadata() -> None:
    """REQ-SPOE-3801: unscoreable rows return abstain with request metadata."""

    row = rest._http_row(
        {
            "candidate_id": "code-row",
            "domain": "code",
            "calibrated_error_score": None,
            "abstained": True,
        },
        {
            "coverage": 0.998218,
            "certified_risk_bound": 0.037646,
            "delta": 0.05,
            "certified_threshold": 0.733216,
            "threshold_source": "exp3771",
        },
        abstention_enabled=True,
    )

    assert row == {
        "candidate_id": "code-row",
        "domain": "code",
        "verdict": "abstain",
        "score": None,
        "coverage": 0.998218,
        "risk": 0.037646,
        "delta": 0.05,
        "threshold": 0.733216,
        "threshold_source": "exp3771",
    }
    assert rest._payload_field([], "domain", None) is None
