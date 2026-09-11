"""Tests for REQ-REPORT-7218 and SCENARIO-REPORT-7218-*.

The tests use temporary output paths. They read frozen repository evidence but
never rewrite prior results or operator-curated files.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7218_v635_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def _publication_receipt() -> dict[str, object]:
    """Supply the unchanged G1-G4 shape without running a child in unit tests."""

    return {
        "paper_ready": False,
        "unmet_gates": ["G2"],
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": False, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
    }


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build one shared terminal fixture because each aggregation reads large ledgers."""

    directory = tmp_path_factory.mktemp("exp7218-artifact")
    output = directory / "results/experiment_7218_v635_capstone.json"
    checkpoint = directory / "results/checkpoints/experiment_7218_v635_capstone.json"
    artifact = capstone.build_artifact(
        ROOT,
        "20260911",
        output,
        checkpoint,
        publication_runner=lambda _root: _publication_receipt(),
    )
    return artifact, output, checkpoint


def test_req_report_7218_spec_and_exact_field_principles() -> None:
    """REQ-REPORT-7218: the requirement and exact field reasons are durable."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7218" in text
    assert "SCENARIO-REPORT-7218-ARTIFACT" in text
    assert capstone.REQUIRED_FIELD_PRINCIPLES == capstone.PROMPT_FIELD_PRINCIPLES


def test_scenario_report_7218_contract_reads_frozen_fourteen_rows() -> None:
    """SCENARIO-REPORT-7218-CONTRACT: frozen YAML and Markdown agree."""

    contract = capstone.load_contract(ROOT)
    assert contract["errors"] == []
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["tasks"]) == 14
    assert contract["tasks"][-1]["id"] == "exp7218-capstone"
    assert all(row["hash_matches_receipt"] for row in contract["source_rows"])


def test_scenario_report_7218_intake_unwraps_only_real_wrapper() -> None:
    """SCENARIO-REPORT-7218-INTAKE: domain dictionaries stay unchanged."""

    wrapped = {"principle": "why", "value": 1}
    domain = {"value": 1, "unit": "joule"}
    assert capstone.unwrap_principle(wrapped) == 1
    assert capstone.unwrap_principle(domain) is domain
    assert capstone.unwrap_principle(3) == 3
    with pytest.raises(ValueError, match="invalid full task id"):
        capstone.task_number("7208")


def test_scenario_report_7218_intake_rejects_nonobject_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7218-INTAKE: malformed roots fail before promotion."""

    source = tmp_path / "array.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root is not an object"):
        capstone._read_json(source)


def test_scenario_report_7218_contract_reports_independent_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7218-CONTRACT: row and byte failures stay explicit."""

    advisory = {
        "raw_source_rows": [
            {"source_type": "yaml", "raw_sha256": "sha256:bad"},
            {"source_type": "markdown", "raw_sha256": "sha256:bad"},
        ]
    }
    advisory_path = tmp_path / capstone.ADVISORY_PATH
    advisory_path.parent.mkdir(parents=True)
    advisory_path.write_text(json.dumps(advisory), encoding="utf-8")
    frozen_yaml = tmp_path / capstone.FROZEN_YAML_PATH
    frozen_design = tmp_path / capstone.FROZEN_DESIGN_PATH
    frozen_yaml.parent.mkdir(parents=True)
    task_id, _title, deliverable = capstone.EXPECTED_CONTRACT[0]
    frozen_yaml.write_text(
        f"tasks:\n  - id: {task_id}\n    title: wrong\n    deliverable: {deliverable}\n",
        encoding="utf-8",
    )
    frozen_design.write_text("mismatched design", encoding="utf-8")
    monkeypatch.setattr(capstone, "evaluate_contract", lambda *_args: {"passed": False})
    contract = capstone.load_contract(tmp_path)
    assert set(contract["errors"]) == {
        "frozen_contract_mismatch",
        "contract_id_order",
        "contract_public_fields",
        "contract_source_hash",
    }


def test_scenario_report_7218_intake_prefers_declared_then_full_id_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7218-INTAKE: similar filenames are not evidence."""

    declared = "results/experiment_7210_v635_span_capture.json"
    fallback = "results/experiment_7210_span_capture.json"
    task = {"id": "exp7210-span-capture", "deliverable": declared}
    (tmp_path / "results").mkdir()
    (tmp_path / fallback).write_text('{"status":"blocked"}', encoding="utf-8")
    evidence = capstone.load_evidence(tmp_path, task, {})
    assert evidence["selected_evidence_path"] == fallback
    assert evidence["evidence_source"] == "conductor_gate_block"

    (tmp_path / declared).write_text('{"status":"complete"}', encoding="utf-8")
    preferred = capstone.load_evidence(tmp_path, task, {})
    assert preferred["selected_evidence_path"] == declared
    assert preferred["evidence_source"] == "declared_deliverable"

    (tmp_path / declared).unlink()
    (tmp_path / fallback).unlink()
    (tmp_path / "results/experiment_7210_v635_capture.json").write_text(
        '{"status":"complete"}', encoding="utf-8"
    )
    missing = capstone.load_evidence(tmp_path, task, {})
    assert missing["selected_evidence_path"] is None
    assert missing["evidence_source"] == "missing"


def test_scenario_report_7218_intake_rejects_quarantine_before_field_gate() -> None:
    """SCENARIO-REPORT-7218-INTAKE: quarantine is an independent rejection."""

    payload = {
        "span_fixture_ready_score": {"principle": "why", "value": 1},
        "flagged_adversarial": True,
    }
    receipt = capstone.quarantine_receipt(
        payload,
        "exp7208-span-fixture",
        "results/experiment_7208_v635_span_fixture.json",
        {},
    )
    gate = capstone.evaluate_gate(payload, "span_fixture_ready_score", 1, receipt)
    assert gate["structured_field_passed"] is True
    assert gate["quarantine_passed"] is False
    assert gate["passed"] is False


def test_scenario_report_7218_arc_deduplicates_ids_and_sessions() -> None:
    """SCENARIO-REPORT-7218-ARC: duplicate receipts count only once."""

    a = {
        "induction_id": "sha256:a",
        "source_session_id": "session-a",
        "seed": 1,
        "source_authenticated": True,
        "model_validity_errors": [],
    }
    b = {
        "induction_id": "sha256:b",
        "source_session_id": "session-b",
        "seed": 2,
        "source_authenticated": True,
        "model_validity_errors": ["bad_model"],
    }
    summary = capstone.deduplicate_arc_rows(
        [[a], [dict(a), b, {"induction_id": 7, "source_authenticated": True}]]
    )
    assert summary["achieved_cumulative_volume"] == 2
    assert summary["distinct_sessions"] == 2
    assert summary["model_valid_receipt_count"] == 1
    assert summary["operational_target"] == 10
    assert summary["target_is_statistical_proof"] is False

    evidence = capstone.load_repository_payloads(ROOT)
    evidence["exp7206-arc-volume-a"]["payload"]["tool_induction_rows"].append(None)
    normalized = capstone._normalized_arc_rows(evidence)
    assert all(isinstance(row, dict) for group in normalized for row in group)


def test_scenario_report_7218_claims_recompute_scientific_boundaries() -> None:
    """SCENARIO-REPORT-7218-CLAIMS: selected headline values derive from rows."""

    payloads = capstone.load_repository_payloads(ROOT)
    claims = capstone.recompute_claims(payloads)
    assert claims
    assert all(row["error"] is None for row in claims)
    assert all(
        {"unit_id", "arm", "seed", "metric", "error", "abstention"} <= row.keys() for row in claims
    )
    values = {(row["task_id"], row["metric"]): row["recomputed_value"] for row in claims}
    assert values[("exp7213-refinement-learning", "refinement_value_score")] == 0
    assert values[("exp7215-down-up-prototype", "down_up_kernel_ready_score")] == 1
    assert values[("exp7216-down-up-quality", "down_up_value_score")] == 0
    assert values[("exp7217-abi-board-readiness", "native_abi_ready_score")] == 1


def test_scenario_report_7218_blocked_builds_complete_matrix(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7218-BLOCKED: external gaps do not make own work partial."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["evidence_matrix"]) == 14
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["upstream"]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert output.is_file()
    assert checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7218_boundaries_and_retirements(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7218-BOUNDARIES: PRD facts and retirements stay separate."""

    artifact = built_artifact[0]
    assert set(artifact["prd_completion"]) == {
        "source_semantics",
        "useful_continual_learning",
        "live_hidden_game_generalization",
        "production_performance",
    }
    assert artifact["scientific_boundaries"]["nfr_01"]["value"] is False
    assert artifact["scope_reduction_compliance"]["retired_boundaries_preserved"] is True
    decisions = artifact["branch_decisions"]
    assert len(decisions) == 14
    assert {row["action"] for row in decisions} <= capstone.BRANCH_ACTIONS
    preserved = artifact["scope_reduction_compliance"]["v634_retirements"]
    assert {row["mechanism"] for row in preserved} == {
        "atomic_prompt",
        "queue_priority_policy",
        "missing_tool_explanation",
        "10x_production_claim",
    }
    expected = {
        7194: "complete_null_no_missing_tool_requested_banked_progress_noncausal",
        7196: "complete_null_atomic_capture_parse_poor_bank_available_for_independent_audit",
        7199: "complete_null: bounded acquisition did not pass the frozen primary-cell gate",
        7202: "complete: all fixed boundary, law, control, and long-chain quality rows were measured. Sample-quality evidence was insufficient. The primary local boundary gate did not pass. The unchanged NFR-01 10x target was not met.",
    }
    for row in preserved:
        number = capstone.task_number(row["prior_id"])
        assert row["prior_verdict"] == expected[number]


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7218_artifact_detects_top_level_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    error: str,
) -> None:
    """SCENARIO-REPORT-7218-ARTIFACT: terminal claims fail closed."""

    artifact = built_artifact[0]
    mutated = deepcopy(artifact)
    mutated[field] = replacement
    assert error in capstone.validate_artifact(mutated, root=ROOT)


def test_scenario_report_7218_artifact_detects_nested_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7218-ARTIFACT: roster and claim mutations fail."""

    artifact = built_artifact[0]
    roster = deepcopy(artifact)
    roster["evidence_matrix"].pop()
    assert "evidence_matrix" in capstone.validate_artifact(roster, root=ROOT)
    claim = deepcopy(artifact)
    claim["rows"][0]["recomputed_value"] = 999
    assert "claim_rows" in capstone.validate_artifact(claim, root=ROOT)
    bad_hash = deepcopy(artifact)
    path = next(iter(bad_hash["source_artifact_hashes"]))
    bad_hash["source_artifact_hashes"][path] = "sha256:bad"
    assert "source_artifact_hashes" in capstone.validate_artifact(bad_hash, root=ROOT)


def test_req_report_7218_publication_gate_runner_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7218: the unchanged G1-G4 subprocess is parsed and bounded."""

    receipt = capstone.run_publication_gate(ROOT)
    assert tuple(receipt["gates"]) == capstone.GATE_IDS
    assert receipt["exit_code"] == 0

    monkeypatch.setattr(
        capstone.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="[]", stderr="wrong shape", returncode=1),
    )
    with pytest.raises(RuntimeError, match="publication gate failed"):
        capstone.run_publication_gate(ROOT)


def test_req_report_7218_build_rejects_invalid_preconditions_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7218: build failures remain terminal and diagnostic."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "results/checkpoints/result.json"
    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT, "20260910", output, checkpoint, lambda _root: _publication_receipt()
        )

    original_contract = capstone.load_contract(ROOT)
    failed_contract = deepcopy(original_contract)
    failed_contract["errors"] = ["test_contract_failure"]
    monkeypatch.setattr(capstone, "load_contract", lambda _root: failed_contract)
    with pytest.raises(RuntimeError, match="essential capstone precondition"):
        capstone.build_artifact(
            ROOT, "20260911", output, checkpoint, lambda _root: _publication_receipt()
        )


def test_req_report_7218_build_rejects_changed_gate_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7218: changed gates and failed parser checks stop writes."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "results/checkpoints/result.json"
    with pytest.raises(RuntimeError, match="changed G1-G4 shape"):
        capstone.build_artifact(ROOT, "20260911", output, checkpoint, lambda _root: {"gates": {}})

    monkeypatch.setattr(capstone, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(RuntimeError, match="capstone validation failed"):
        capstone.build_artifact(
            ROOT, "20260911", output, checkpoint, lambda _root: _publication_receipt()
        )


def test_req_report_7218_build_reloads_final_file_and_main_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7218: the final parser and non-validation CLI path execute."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "results/checkpoints/result.json"
    original_validate = capstone.validate_artifact
    calls = 0

    def fail_second_validation(*args: object, **kwargs: object) -> list[str]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else ["bad_final_file"]

    monkeypatch.setattr(capstone, "validate_artifact", fail_second_validation)
    with pytest.raises(RuntimeError, match="final file-parser validation failed"):
        capstone.build_artifact(
            ROOT, "20260911", output, checkpoint, lambda _root: _publication_receipt()
        )
    monkeypatch.setattr(capstone, "validate_artifact", original_validate)
    invoked: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        capstone,
        "build_artifact",
        lambda *args, **_kwargs: invoked.append(args) or {},
    )
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output)]) == 0
    assert invoked and invoked[0][1] == capstone.RUN_DATE


def test_req_report_7218_cli_validates_existing_artifact(
    tmp_path: Path, built_artifact: tuple[dict[str, object], Path, Path]
) -> None:
    """REQ-REPORT-7218: the public entrypoint supports file-parser validation."""

    output = tmp_path / "result.json"
    artifact = built_artifact[0]
    output.write_text(json.dumps(artifact), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 0
    output.write_text("{}", encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 1
