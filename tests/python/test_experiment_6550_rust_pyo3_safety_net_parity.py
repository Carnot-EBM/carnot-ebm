"""Tests for Exp6550 Safety-Net Rust/PyO3 parity artifact.

Spec refs: REQ-RUSTPY-6550, REQ-RUSTPY-6550-SCHEMA,
REQ-RUSTPY-6550-NUMERIC, REQ-RUSTPY-6550-SERIALIZATION,
REQ-RUSTPY-6550-ERRORS, REQ-RUSTPY-6550-PARITY,
REQ-RUSTPY-6550-FALLBACK, REQ-RUSTPY-6550-ROLLBACK,
REQ-RUSTPY-6550-NO-AUTHORITY, SCENARIO-RUSTPY-6550-BOUNDARY-PARITY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6550_rust_pyo3_safety_net_parity as mod


pytest.importorskip("carnot._rust")

TESTS_RUN = [{"command": "focused-exp6550", "exit_code": 0}]


def test_req_rustpy_6550_artifact_schema_and_positive_reducer(tmp_path: Path) -> None:
    """REQ-RUSTPY-6550: the terminal artifact recomputes parity readiness."""

    artifact = mod.build_artifact(
        result_path=tmp_path / "experiment_6550.json",
        write=False,
        duration_s=0.0,
        tests_run=TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_rust_pyo3_safety_net_parity_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["cross_language_router_parity_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["upstream_gate_receipt"]["gate_passed"] is True
    assert artifact["abi_schema_and_version_contract"]["schema_version"] == (
        "carnot.safety_net.router_abi.v1"
    )
    assert artifact["build_and_binding_receipts"]["binding_importable"] is True
    assert artifact["serialization_equality_receipt"]["all_decision_bytes_equal"] is True
    assert artifact["exact_downstream_equality_receipt"]["all_exact_downstream_equal"] is True
    assert artifact["fallback_and_python_rollback_receipt"]["python_rollback_exact"] is True
    assert artifact["abi_attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_rustpy_6550_rows_cover_supported_and_unsupported_conditions() -> None:
    """SCENARIO-RUSTPY-6550-BOUNDARY-PARITY: rows expose every condition."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    rows = artifact["parity_rows"]
    conditions = {row["condition"] for row in rows}

    assert rows
    assert artifact["per_unit_rows"] == rows
    assert {
        "held_compact",
        "boundary_abstention",
        "exception_lookup",
        "malformed_duplicate",
        "null_candidate_ids",
        "extreme_numeric",
        "version_skew",
        "unknown_feature",
        "nan_invalid_json",
    } <= conditions
    assert all(row["decision_equal"] for row in rows)
    assert all(row["output_bytes_equal"] for row in rows)
    assert any(row["supported"] for row in rows)
    assert any(not row["supported"] for row in rows)
    assert all(
        row["python_output"]["route"] == "native_exact_fallback"
        for row in rows
        if not row["supported"]
    )
    assert {row["error_type_equal"] for row in artifact["error_semantics_rows"]} == {True}


def test_scenario_rustpy_6550_validation_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-RUSTPY-6550-ERRORS: artifact tampering cannot keep a positive claim."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    mutations = (
        (
            "required field set mismatch",
            lambda data: data.pop("status"),
        ),
        (
            "inference_substrate mismatch",
            lambda data: data.__setitem__("inference_substrate", "wrong"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda data: data.__setitem__("verifier_is_oracle", True),
        ),
        (
            "honest_verdict terminal prefix mismatch",
            lambda data: data.__setitem__("honest_verdict", "not-terminal"),
        ),
        (
            "ready score mismatch",
            lambda data: (
                data.__setitem__("cross_language_router_parity_ready_score", 1.0),
                data["aggregate_row_recomputation"].__setitem__("ready_score_from_rows", 0.0),
            ),
        ),
        (
            "supported parity failed",
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "supported_rows_byte_equal", False
            ),
        ),
        (
            "unsupported fail-closed failed",
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "unsupported_rows_fail_closed", False
            ),
        ),
        (
            "exact downstream equality failed",
            lambda data: data["exact_downstream_equality_receipt"].__setitem__(
                "all_exact_downstream_equal", False
            ),
        ),
        (
            "rollback failed",
            lambda data: data["fallback_and_python_rollback_receipt"].__setitem__(
                "python_rollback_exact", False
            ),
        ),
        (
            "protected files changed",
            lambda data: data["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
    )
    for expected, mutate in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        candidate["reproducibility_checksum"] = mod.reproducibility_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    result_path = tmp_path / "cli-exp6550.json"
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["status"] == "complete_rust_pyo3_safety_net_parity_positive"
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0


def test_req_rustpy_6550_blocked_and_disqualified_reducers() -> None:
    """REQ-RUSTPY-6550-FALLBACK: reducers expose blocked and disqualified states."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)

    blocked = deepcopy(artifact)
    blocked["upstream_gate_receipt"]["gate_passed"] = False
    blocked_aggregate = mod.aggregate_row_recomputation(blocked)
    assert blocked_aggregate["verdict_class_from_rows"] == "blocked"
    assert mod._status_and_verdict(blocked_aggregate)[2] == "blocked"  # noqa: SLF001

    partial = deepcopy(artifact)
    partial["parity_rows"][0]["output_bytes_equal"] = False
    partial["serialization_equality_receipt"] = mod.serialization_equality_receipt(
        partial["parity_rows"]
    )
    partial_aggregate = mod.aggregate_row_recomputation(partial)
    assert partial_aggregate["verdict_class_from_rows"] == "partial"
    assert mod._status_and_verdict(partial_aggregate)[2] == "partial"  # noqa: SLF001

    disqualified = deepcopy(artifact)
    disqualified["exact_downstream_equality_receipt"]["changed_exact_output_count"] = 1
    disqualified["exact_downstream_equality_receipt"]["all_exact_downstream_equal"] = False
    disqualified_aggregate = mod.aggregate_row_recomputation(disqualified)
    assert disqualified_aggregate["verdict_class_from_rows"] == "disqualified"
    assert mod._status_and_verdict(disqualified_aggregate)[2] == "disqualified"  # noqa: SLF001


def test_req_rustpy_6550_defensive_helpers_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-RUSTPY-6550-ERRORS: reducer helper failures remain explicit."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001

    monkeypatch.setattr(
        mod.importlib, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("forced"))
    )
    assert mod._load_rust_module() is None  # noqa: SLF001
    assert mod.parity_rows(None) == []

    validation_mutations = (
        (
            "verdict_class outside Exp6550 enum",
            lambda data: data.__setitem__("verdict_class", "surprise"),
        ),
        (
            "field_provenance must cover required fields",
            lambda data: data.__setitem__("field_provenance", {}),
        ),
        (
            "cross_language_router_parity_ready_score must be 0.0 or 1.0",
            lambda data: data.__setitem__("cross_language_router_parity_ready_score", 0.5),
        ),
        (
            "positive verdict requires ready score 1.0",
            lambda data: (
                data.__setitem__("verdict_class", "positive"),
                data.__setitem__("cross_language_router_parity_ready_score", 0.0),
                data["aggregate_row_recomputation"].__setitem__("ready_score_from_rows", 0.0),
            ),
        ),
        (
            "serialization equality failed",
            lambda data: data["serialization_equality_receipt"].__setitem__(
                "all_decision_bytes_equal", False
            ),
        ),
        (
            "fallback unreachable",
            lambda data: data["fallback_and_python_rollback_receipt"].__setitem__(
                "fallback_reachable", False
            ),
        ),
        (
            "ABI attack false accept",
            lambda data: data["abi_attack_matrix"].__setitem__("all_attacks_fail_closed", False),
        ),
        (
            "native binding did not run",
            lambda data: data["build_and_binding_receipts"].__setitem__("native_code_ran", False),
        ),
    )
    for expected, mutate in validation_mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        candidate["reproducibility_checksum"] = mod.reproducibility_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)

    bad_validate = tmp_path / "bad.json"
    bad_validate.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_validate)]) == 1

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": "artifact"})
    assert mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad-build.json")]) == 1
