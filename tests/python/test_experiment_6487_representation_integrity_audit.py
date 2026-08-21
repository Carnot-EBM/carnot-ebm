"""Tests for Exp6487 raw representation integrity audit.

Spec refs: REQ-VERIFY-6487, SCENARIO-VERIFY-6487-RAW-REPLAY,
SCENARIO-VERIFY-6487-SHORTCUTS, SCENARIO-VERIFY-6487-ATTACKS,
SCENARIO-VERIFY-6487-CELLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6487_representation_integrity_audit as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6487_representation_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6487_representation_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6487_representation_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6487_representation_integrity_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6487_representation_integrity_audit.py"
)
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6487_representation_integrity_audit "
    "--date 20260821"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6487_representation_integrity_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6487_representation_integrity_audit.json"
)
E2E_COMMAND = ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _raw_payload(
    *,
    unit_id: str,
    candidate_id: str,
    family: str,
    model_hf_id: str,
    prompt_hash: str,
    candidate_hash: str,
    vector: list[float],
    exact_label: bool | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": mod.RAW_ROW_SCHEMA,
        "unit_id": unit_id,
        "candidate_id": candidate_id,
        "family": family,
        "model_hf_id": model_hf_id,
        "model_hash": mod.sha256_json({"model": model_hf_id}),
        "prompt_hash": prompt_hash,
        "candidate_hash": candidate_hash,
        "prompt_length": 40,
        "candidate_length": len(candidate_id),
        "token_length": 64,
        "vector": vector,
    }
    if exact_label is not None:
        row["exact_label"] = exact_label
    return row


def _fixture(
    tmp_path: Path,
    *,
    exp6486_style_ids: bool,
    gate_score: float = 1.0,
    mutate_first_raw_after_manifest: bool = False,
) -> dict[str, Any]:
    raw_root = tmp_path / "raw"
    families = {
        "family_a": ("model/a", 2),
        "family_b": ("model/b", 3),
    }
    unit_task_pairs = [
        ("exp6482-alpha-task-00", "development"),
        ("exp6482-beta-task-01", "held"),
    ]
    entries: list[dict[str, Any]] = []
    storage_by_split: dict[str, list[str]] = {"development": [], "held": []}
    for family_index, (family, (model_hf_id, width)) in enumerate(families.items()):
        for unit_index, (unit_id, split) in enumerate(unit_task_pairs):
            prompt_hash = mod.sha256_json({"prompt": unit_id})
            if exp6486_style_ids:
                candidates = [
                    ("exact_correct", True, [1.0 + family_index, 0.0, 0.0][:width]),
                    ("controlled_wrong_alternate", False, [0.0, 1.0 + unit_index, 0.0][:width]),
                    ("controlled_wrong_protected", False, [0.0, 0.0, 1.0 + unit_index][:width]),
                ]
            else:
                candidates = [
                    ("candidate_a", unit_index == 0, [1.0, 1.0, 1.0][:width]),
                    ("candidate_b", unit_index == 1, [1.0, 1.0, 1.0][:width]),
                ]
            for candidate_kind, label, vector in candidates:
                candidate_id = (
                    f"{unit_id}:{candidate_kind}" if exp6486_style_ids else candidate_kind
                )
                payload = _raw_payload(
                    unit_id=unit_id,
                    candidate_id=candidate_id,
                    family=family,
                    model_hf_id=model_hf_id,
                    prompt_hash=prompt_hash,
                    candidate_hash=mod.sha256_json(
                        {"candidate": candidate_id}
                        if not exp6486_style_ids
                        else {"candidate": candidate_id, "unit": unit_id}
                    ),
                    vector=vector,
                    exact_label=None if exp6486_style_ids else label,
                )
                path = raw_root / split / family / f"{unit_id}__{candidate_kind}__{family}__1.json"
                _write_json(path, payload)
                file_hash = mod.sha256_file(path)
                entries.append(
                    {
                        "cell_id": f"{unit_id}|{candidate_id}|{model_hf_id}|1|{split}",
                        "path": str(path),
                        "sha256": file_hash,
                        "write_count": 1,
                        "split": split,
                        "family": family,
                        "native_dimension": width,
                    }
                )
                storage_by_split[split].append(str(path))
    if mutate_first_raw_after_manifest:
        first = Path(entries[0]["path"])
        payload = json.loads(first.read_text(encoding="utf-8"))
        payload["vector"][0] = 99.0
        _write_json(first, payload)
    artifact = {
        "status": "complete",
        "prospective_representation_stream_ready_score": gate_score,
        "raw_vector_manifest": {
            "schema_version": "fixture.raw_vector_manifest",
            "vectors": entries,
            "vector_count": len(entries),
            "all_write_once": True,
            "paths_unique": True,
            "hash_root": mod.sha256_json([entry["sha256"] for entry in entries]),
            "storage_by_split": {
                split: sorted(paths) for split, paths in storage_by_split.items()
            },
        },
        "rows": [{"aggregate_row_that_must_not_be_trusted": True}],
    }
    artifact_path = tmp_path / "exp6486.json"
    _write_json(artifact_path, artifact)
    return {"artifact_path": artifact_path, "artifact": artifact, "entries": entries}


def _run_fixture(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    fixture = _fixture(tmp_path, **kwargs)
    return mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp6486_artifact_path=fixture["artifact_path"],
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_verify_6487_spec_declares_raw_audit_contract() -> None:
    """REQ-VERIFY-6487: OpenSpec names fields, principles, and scenarios."""

    spec = VERIFY_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-6487") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6487",
        "SCENARIO-VERIFY-6487-RAW-REPLAY",
        "SCENARIO-VERIFY-6487-SHORTCUTS",
        "SCENARIO-VERIFY-6487-ATTACKS",
        "SCENARIO-VERIFY-6487-CELLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`representation_integrity_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6487_raw_replay_and_shortcuts_disqualify(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6487-RAW-REPLAY/SHORTCUTS: raw rows drive verdicts."""

    artifact = _run_fixture(tmp_path, exp6486_style_ids=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "disqualified"
    assert artifact["honest_verdict"].startswith("disqualified:")
    assert artifact["representation_integrity_ready_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["upstream_hash_receipts"]["aggregate_rows_trusted"] is False
    assert artifact["upstream_hash_receipts"]["raw_file_count"] == 12
    assert artifact["reconstructed_stream_counts"]["raw_record_count"] == 12
    assert artifact["reconstructed_stream_counts"]["pair_count"] == 8
    assert artifact["aggregate_row_recomputation"]["row_type_counts"]["raw_record"] == 12
    assert all(row["wrong_support"] > 0 for row in artifact["within_family_cell_rows"])
    assert all(row["correct_support"] > 0 for row in artifact["within_family_cell_rows"])
    survived = {
        row["control_name"]
        for row in artifact["shortcut_control_rows"]
        if row["survived_shortcut"]
    }
    assert {"candidate_identity", "candidate_identifier_length", "row_order_modulo_pair"}.issubset(
        survived
    )


def test_scenario_verify_6487_attacks_fail_closed_and_clean_fixture_can_pass(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6487-ATTACKS/CELLS: attacks are caught per family."""

    artifact = _run_fixture(tmp_path, exp6486_style_ids=False)

    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["representation_integrity_ready_score"] == 1.0
    assert artifact["missing_verifier_gaps"] == []
    assert all(row["attack_detected"] for row in artifact["permutation_attack_rows"])
    assert all(row["attack_detected"] for row in artifact["provenance_attack_rows"])
    assert {
        (row["family"], row["task_family"]): row["headroom_state"]
        for row in artifact["within_family_cell_rows"]
    } == {
        ("family_a", "alpha-task"): "supported",
        ("family_a", "beta-task"): "supported",
        ("family_b", "alpha-task"): "supported",
        ("family_b", "beta-task"): "supported",
    }

    blocked = _run_fixture(
        tmp_path / "blocked",
        exp6486_style_ids=False,
        mutate_first_raw_after_manifest=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["representation_integrity_ready_score"] == 0.0
    assert "raw_file_hash_mismatch" in blocked["preconditions_checked"]["blocked_reasons"]


def test_scenario_verify_6487_validation_and_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6487-RAW-REPLAY: malformed artifacts fail closed."""

    missing_gate = _run_fixture(tmp_path / "missing-gate", exp6486_style_ids=False, gate_score=0.0)
    assert missing_gate["status"] == "blocked"
    assert "exp6486_gate_not_ready" in missing_gate["gate_check_summary"]["failed_gates"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    assert mod._label_from_row({"label": True}, "unknown") == (True, "raw_label")
    assert mod._label_from_row({}, "unknown") == (None, "unavailable")
    with pytest.raises(ValueError, match="raw vector list required"):
        mod._vector({})

    no_storage_path = tmp_path / "only-vector-entry.json"
    _write_json(
        no_storage_path,
        _raw_payload(
            unit_id="loose-unit",
            candidate_id="loose",
            family="loose-family",
            model_hf_id="loose-model",
            prompt_hash=mod.sha256_json({"prompt": "loose"}),
            candidate_hash=mod.sha256_json({"candidate": "loose"}),
            vector=[0.0],
            exact_label=True,
        ),
    )
    only_vectors = {"raw_vector_manifest": {"vectors": [{"path": str(no_storage_path)}]}}
    paths, entries = mod._manifest_paths(only_vectors)
    assert paths == [no_storage_path]
    storage_only = {"raw_vector_manifest": {"storage_by_split": {"held": [str(no_storage_path)]}}}
    receipts, hashes_match, root_match = mod._raw_hash_receipts(storage_only, paths, entries)
    assert receipts[0]["sha256"].startswith("sha256:")
    assert hashes_match is False
    assert root_match is False
    assert mod.missing_verifier_gaps([{"record_id": "missing"}])[0]["selectable"] is False
    output_block = mod.preconditions_checked(
        root=REPO,
        result_path=tmp_path / "absent" / "artifact.json",
        gate_summary={"failed_gates": []},
        raw_file_hashes=[{"hash_matches_manifest": True}],
    )
    assert "output_path_not_writable" in output_block["blocked_reasons"]

    clean = _run_fixture(tmp_path / "clean", exp6486_style_ids=False)
    missing_field = deepcopy(clean)
    del missing_field["status"]
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(missing_field)
    bad_principles = deepcopy(clean)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)
    bad_provenance = deepcopy(clean)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)
    bad_oracle = deepcopy(clean)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["representation_integrity_ready_score"] = (
        mod.representation_integrity_ready_score(bad_oracle)
    )
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)
    bad_score = deepcopy(clean)
    bad_score["representation_integrity_ready_score"] = 0.0
    with pytest.raises(ValueError, match="representation_integrity_ready_score"):
        mod.validate_artifact(bad_score)
    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["representation_integrity_ready_score"] = (
        mod.representation_integrity_ready_score(bad_substrate)
    )
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_substrate
    )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)
    ready_bad_status = deepcopy(clean)
    ready_bad_status["status"] = "blocked"
    ready_bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(
        ready_bad_status
    )
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(ready_bad_status)
    blocked_bad_verdict = deepcopy(missing_gate)
    blocked_bad_verdict["honest_verdict"] = "wrong"
    blocked_bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_bad_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_bad_verdict)
    disqualified = _run_fixture(tmp_path / "disqualified", exp6486_style_ids=True)
    disqualified_bad_verdict = deepcopy(disqualified)
    disqualified_bad_verdict["honest_verdict"] = "wrong"
    disqualified_bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        disqualified_bad_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(disqualified_bad_verdict)
    unknown_status = deepcopy(disqualified)
    unknown_status["status"] = "strange"
    unknown_status["reproducibility_checksum"] = mod.reproducibility_checksum(
        unknown_status
    )
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(unknown_status)
