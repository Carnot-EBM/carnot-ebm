"""Repair tests for the Exp 3810 abstention HTTP/REST surface.

Spec: REQ-SPOE-3810, SCENARIO-SPOE-3810.
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


def _repair_candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "exp3810_above_threshold",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "exp3810_below_threshold",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.5,
        },
    ]


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
                example_id=f"{domain}-3810-{idx}",
            )
        )
    return examples


@contextmanager
def _running_server(
    *,
    threshold_path: Path | None = None,
) -> Iterator[rest.AbstentionHTTPServer]:
    server = rest.make_server(
        ("127.0.0.1", 0),
        root=ROOT,
        examples=_domain_examples(),
        certified_threshold_path=threshold_path,
    )
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
) -> tuple[int, dict[str, object]]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=10.0)
    try:
        conn.request(
            "POST",
            rest.POST_PATH,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, json.loads(data)
    finally:
        conn.close()


def _write_threshold_artifact(path: Path, *, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected_threshold": threshold,
                "coverage_at_operating_point": 0.998218,
                "certified_risk_bound": 0.037646,
                "certification_method": (
                    "split-conformal (Hoeffding upper bound, assumes exchangeability, delta=0.05)"
                ),
                "n_calibration": 2619,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_req_spoe_3810_http_uses_configured_threshold_path_not_default(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3810: endpoint loads the threshold config selected by wiring."""

    threshold_path = tmp_path / "results/experiment_3771_custom.json"
    _write_threshold_artifact(threshold_path, threshold=1.0)

    with _running_server(threshold_path=threshold_path) as server:
        status, payload = _post_json(
            server,
            {
                "domain": "math",
                "abstention_mode": True,
                "candidate": _repair_candidates()[0],
            },
        )

    assert status == 200
    row = payload["scores"][0]  # type: ignore[index]
    assert row["verdict"] == "abstain"
    assert row["threshold"] == pytest.approx(1.0)
    assert row["threshold_source"] == str(threshold_path.resolve())


def test_scenario_spoe_3810_real_cached_http_batch_confident_and_abstain() -> None:
    """SCENARIO-SPOE-3810: cached HTTP E2E exercises both abstention branches."""

    config = abstention.load_certified_abstention_config()
    with _running_server() as server:
        default_status, default_payload = _post_json(
            server,
            {"domain": "math", "candidates": _repair_candidates()},
        )
        enabled_status, enabled_payload = _post_json(
            server,
            {
                "domain": "math",
                "abstention_mode": True,
                "candidates": _repair_candidates(),
            },
        )

    assert default_status == 200
    assert all("verdict" not in row for row in default_payload["scores"])  # type: ignore[index]

    assert enabled_status == 200
    assert enabled_payload["batch"] == {"n_candidates": 2}
    rows = {row["candidate_id"]: row for row in enabled_payload["scores"]}  # type: ignore[index]
    confident = rows["exp3810_above_threshold"]
    abstained = rows["exp3810_below_threshold"]
    assert confident["verdict"] == "confident"
    assert confident["score"] >= config.threshold
    assert abstained["verdict"] == "abstain"
    assert abstained["score"] < config.threshold
    assert abstained["coverage"] == pytest.approx(config.coverage)
    assert abstained["risk"] == pytest.approx(config.certified_risk_bound)
    assert abstained["delta"] == pytest.approx(config.delta)
