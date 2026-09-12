"""Focused tests for the V638 semantic audit.

Spec refs: REQ-VERIFY-7252, SCENARIO-VERIFY-7252-BLOCK, and
SCENARIO-VERIFY-7252-REPLAY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7252_v638_semantic_audit as exp


REPO = Path(__file__).resolve().parents[2]


def test_req_verify_7252_contract_is_frozen() -> None:
    """REQ-VERIFY-7252 fixes the no-model audit and its paired sample plan."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7252" in spec
    assert "SCENARIO-VERIFY-7252-BLOCK" in spec
    assert exp.MODEL_SPECS == []
    assert exp.BOOTSTRAP_DRAWS == 10_000
    assert exp.RESULT_PATH == Path("results/experiment_7252_v638_semantic_audit.json")
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)
    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE
    with pytest.raises(Exception, match="run date must be 20260912"):
        exp._date_argument("20260911")


def test_scenario_verify_7252_block_names_missing_capture(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7252-BLOCK records the absent Exp7251 field."""

    missing = tmp_path / "absent-exp7251.json"
    artifact = exp.run_experiment(
        REPO,
        exp.RUN_DATE,
        output_root=tmp_path,
        path_overrides={"capture": missing},
    )

    assert artifact["experiment_id"] == "exp7252-semantic-audit"
    assert artifact["milestone"] == "2026.09.638"
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_exp7252_missing_upstream_artifact"
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "upstream_capture_ready",
        "upstream": "experiment_7251_v638_mention_heldout",
        "artifact_field": "mention_capture_complete_score",
        "expected_value": 1,
        "observed_value": "missing_artifact",
    }
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["current_invocation_counts"] == {
        "model_loads": 0,
        "generations": 0,
        "model_invocations": 0,
    }
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["semantic_audit_complete_score"] == 0
    assert artifact["semantic_value_score"] == 0
    assert artifact["rows"] == []
    assert artifact["paired_interval_rows"] == []
    assert artifact["mutation_rows"] == []
    assert artifact["sample_size_budget"]["censored_independent_units"] == 64
    assert all(row["evaluated"] is False for row in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact, REPO, path_overrides={"capture": missing}) == []

    result = tmp_path / exp.RESULT_PATH
    checkpoint = tmp_path / exp.CHECKPOINT_PATH
    assert result.is_file() and checkpoint.is_file()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert (
        exp.replay_terminal_artifact(
            REPO,
            output_root=tmp_path,
            path_overrides={"capture": missing},
        )
        == artifact
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("semantic_audit_complete_score", 1, "blocked_terminal_state"),
        ("semantic_value_score", 1, "blocked_terminal_state"),
        ("honest_verdict", "complete_false_claim", "blocked_terminal_state"),
        ("current_invocation_counts", {"model_loads": 1}, "model_contract"),
        ("gate_check_summary", {"passed": True}, "gate_check_summary"),
    ],
)
def test_scenario_verify_7252_replay_rejects_terminal_drift(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    """SCENARIO-VERIFY-7252-REPLAY rejects success-shaped blocked evidence."""

    missing = tmp_path / "missing.json"
    artifact = exp.run_experiment(
        REPO,
        output_root=tmp_path,
        path_overrides={"capture": missing},
    )
    changed = deepcopy(artifact)
    changed[field] = value
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    errors = exp.validate_artifact(changed, REPO, path_overrides={"capture": missing})
    assert error in errors


def test_req_verify_7252_source_hash_and_checksum_drift_fail(tmp_path: Path) -> None:
    """REQ-VERIFY-7252 binds the missing path and every stable artifact field."""

    missing = tmp_path / "missing.json"
    artifact = exp.run_experiment(
        REPO,
        output_root=tmp_path,
        path_overrides={"capture": missing},
    )
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["capture"]["sha256"] = "sha256:bad"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "source_artifact_hashes" in exp.validate_artifact(
        changed, REPO, path_overrides={"capture": missing}
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in exp.validate_artifact(
        changed, REPO, path_overrides={"capture": missing}
    )
    assert exp.validate_artifact([], REPO) == ["artifact_mapping"]
    missing_field = deepcopy(artifact)
    del missing_field["mutation_rows"]
    assert exp.validate_artifact(missing_field, REPO)[0] == "missing_required_field:mutation_rows"


def test_req_verify_7252_cold_validator_rejects_all_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7252 checks identity, provenance, gates, and execution fields."""

    missing = tmp_path / "missing.json"
    artifact = exp.run_experiment(
        REPO,
        output_root=tmp_path,
        path_overrides={"capture": missing},
    )
    changed = deepcopy(artifact)
    changed.update(
        {
            "schema": {},
            "experiment_id": "wrong",
            "milestone": "wrong",
            "field_principles": {},
            "run_date": "wrong",
            "duration_s": True,
            "execution_venue": "wrong",
            "execution_host": "",
            "random_seed": -1,
            "verifier_is_oracle": False,
            "acceptance_gate_results": {},
        }
    )
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert set(exp.validate_artifact(changed, REPO, path_overrides={"capture": missing})) >= {
        "schema",
        "ordinary_identity",
        "field_principles",
        "run_date",
        "duration_s",
        "execution_identity",
        "random_seed",
        "verifier_is_oracle",
        "acceptance_gate_results",
    }


@pytest.mark.parametrize(
    ("body", "expected", "capture_loaded"),
    [
        (b"not-json", "invalid_json", False),
        (b"[]", "artifact_not_mapping", False),
        (b'{"mention_capture_complete_score": 0}', 0, True),
        (
            b'{"mention_capture_complete_score": {"principle": "why", "value": 1}, '
            b'"quarantined": true}',
            1,
            True,
        ),
    ],
)
def test_req_verify_7252_observes_present_capture_states(
    tmp_path: Path, body: bytes, expected: object, capture_loaded: bool
) -> None:
    """REQ-VERIFY-7252 distinguishes malformed, gated, and quarantined inputs."""

    capture = tmp_path / "capture.json"
    capture.write_bytes(body)
    checks, loaded = exp.collect_preconditions(
        REPO,
        exp.RUN_DATE,
        tmp_path,
        path_overrides={"capture": capture},
    )
    ready = next(row for row in checks if row["check"] == "upstream_capture_ready")
    quarantine = next(row for row in checks if row["check"] == "structured_quarantine")
    assert ready["observed_value"] == expected
    assert (loaded is not None) is capture_loaded
    if expected == 1:
        assert quarantine["observed_value"] is True


def test_req_verify_7252_present_capture_and_changed_state_stop(tmp_path: Path) -> None:
    """REQ-VERIFY-7252 never converts a present producer file into invented rows."""

    capture = tmp_path / "capture.json"
    capture.write_text('{"mention_capture_complete_score": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="complete replay requires"):
        exp.run_experiment(
            REPO,
            output_root=tmp_path / "gated",
            path_overrides={"capture": capture},
        )

    missing = tmp_path / "later.json"
    artifact = exp.run_experiment(
        REPO,
        output_root=tmp_path / "blocked",
        path_overrides={"capture": missing},
    )
    missing.write_text("{}", encoding="utf-8")
    errors = exp.validate_artifact(artifact, REPO, path_overrides={"capture": missing})
    assert "upstream_state_changed" in errors


def test_req_verify_7252_internal_and_replay_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7252 stops writes and replay when cold validation fails."""

    missing = tmp_path / "missing.json"
    real_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp7252 artifact"):
        exp.run_experiment(
            REPO,
            output_root=tmp_path / "invalid",
            path_overrides={"capture": missing},
        )
    monkeypatch.setattr(exp, "validate_artifact", real_validate)

    valid = exp.run_experiment(
        REPO,
        output_root=tmp_path / "replay",
        path_overrides={"capture": missing},
    )
    valid["status"] = "running"
    result = tmp_path / "replay" / exp.RESULT_PATH
    result.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal_artifact_validation"):
        exp.replay_terminal_artifact(
            REPO,
            output_root=tmp_path / "replay",
            path_overrides={"capture": missing},
        )


def test_req_verify_7252_validation_receipts_are_observed(tmp_path: Path) -> None:
    """REQ-VERIFY-7252 attaches only exact receipts to a cold-valid result."""

    missing = tmp_path / "missing.json"
    artifact = exp.run_experiment(
        REPO,
        output_root=tmp_path,
        path_overrides={"capture": missing},
    )
    receipts = [
        {
            "command": "pytest focused",
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": "sha256:" + "0" * 64,
        }
    ]
    attached = exp.attach_validation_receipts(
        artifact,
        receipts,
        REPO,
        path_overrides={"capture": missing},
    )
    assert attached["validation_receipts"] == receipts
    assert exp.validate_artifact(attached, REPO, path_overrides={"capture": missing}) == []
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(
            artifact,
            [{"command": "missing fields"}],
            REPO,
            path_overrides={"capture": missing},
        )
    with pytest.raises(ValueError, match="validation_receipt_source_artifact"):
        exp.attach_validation_receipts(
            {**artifact, "status": "running"},
            receipts,
            REPO,
            path_overrides={"capture": missing},
        )


def test_req_verify_7252_thin_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7252 keeps the executable wrapper limited to module dispatch."""

    called: list[object] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert raised.value.code == 0
    assert called == [None]


def test_req_verify_7252_main_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7252 exposes cold validation through its CLI return code."""

    artifact = exp.base_artifact(exp.RUN_DATE)
    artifact["honest_verdict"] = "blocked_test"
    monkeypatch.setattr(exp, "run_experiment", lambda root, date: artifact)
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: [])
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: ["forced"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
