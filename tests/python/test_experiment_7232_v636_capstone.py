"""Tests for REQ-REPORT-7232 and SCENARIO-REPORT-7232-*.

The tests use temporary output files. They never replace producer evidence or
write to the repository's terminal results directory.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7232_v636_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def _publication_receipt() -> dict[str, object]:
    """Return the stable four-gate shape without a subprocess in unit tests."""

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
    """Build one shared artifact because each build reads all producer rows."""

    directory = tmp_path_factory.mktemp("exp7232-artifact")
    output = directory / "results/experiment_7232_v636_capstone.json"
    checkpoint = directory / "results/checkpoints/experiment_7232_v636_capstone.json"
    artifact = capstone.build_artifact(
        ROOT,
        capstone.RUN_DATE,
        output,
        checkpoint,
        publication_runner=lambda _root: _publication_receipt(),
    )
    return artifact, output, checkpoint


def test_req_report_7232_spec_and_exact_field_principles() -> None:
    """REQ-REPORT-7232: the requirement and exact field reasons are durable."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7232" in text
    assert "SCENARIO-REPORT-7232-ARTIFACT" in text
    assert capstone.REQUIRED_FIELD_PRINCIPLES == capstone.PROMPT_FIELD_PRINCIPLES


def test_scenario_report_7232_contract_reads_frozen_fourteen_rows() -> None:
    """SCENARIO-REPORT-7232-CONTRACT: both frozen plans agree exactly."""

    contract = capstone.load_contract(ROOT)
    assert contract["errors"] == []
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert [row["task_id"] for row in contract["task_contract_rows"]] == list(
        capstone.EXPECTED_TASK_IDS
    )
    assert all(row["hash_matches_receipt"] for row in contract["source_rows"])


def test_scenario_report_7232_intake_unwraps_only_explicit_wrapper() -> None:
    """SCENARIO-REPORT-7232-INTAKE: domain dictionaries stay dictionaries."""

    wrapped = {"principle": "why", "value": 1}
    domain = {"value": 1, "unit": "joule"}
    assert capstone.unwrap_principle(wrapped) == 1
    assert capstone.unwrap_principle(domain) is domain
    assert capstone.unwrap_principle(False) is False


def test_scenario_report_7232_intake_prefers_declared_and_rejects_quarantine(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7232-INTAKE: fallback and quarantine are independent."""

    task = {
        "id": "exp7224-span-capture",
        "deliverable": "results/experiment_7224_v636_span_capture.json",
    }
    fallback = tmp_path / "results/experiment_7224_span_capture.json"
    fallback.parent.mkdir()
    fallback.write_text('{"status":"blocked"}', encoding="utf-8")
    evidence = capstone.load_evidence(tmp_path, task, {})
    assert evidence["evidence_source"] == "conductor_gate_block"

    declared = tmp_path / task["deliverable"]
    declared.write_text('{"status":"complete","flagged_adversarial":true}', encoding="utf-8")
    preferred = capstone.load_evidence(tmp_path, task, {})
    assert preferred["evidence_source"] == "declared_deliverable"
    assert preferred["quarantine_receipt"]["quarantined"] is True
    assert preferred["accepted_for_promoted_evidence"] is False


def test_scenario_report_7232_intake_keeps_legitimate_cascade_without_file() -> None:
    """SCENARIO-REPORT-7232-INTAKE: a skipped sibling needs no invented file."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    gates = capstone.replay_gates(contract["tasks"], evidence)
    row = capstone.matrix_row(
        7,
        contract["tasks"][6],
        evidence["exp7225-semantics-audit"],
        gates,
        [],
    )
    assert row["selected_evidence_path"] is None
    assert row["evidence_source"] == "legitimate_cascade_block"
    assert row["verdict_class"] == "blocked"
    assert row["gate_check_summary"]["upstream"] == "exp7224-span-capture"


def test_scenario_report_7232_claims_recompute_and_abstain_on_quarantine() -> None:
    """SCENARIO-REPORT-7232-CLAIMS: values derive from rows and flags win."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    claims = capstone.recompute_claims(evidence)
    values = {(row["task_id"], row["metric"]): row for row in claims}
    assert (
        values[("exp7219-source-contract", "source_contract_complete_score")]["recomputed_value"]
        == 1
    )
    assert values[("exp7221-arc-session", "cumulative_induction_count")]["recomputed_value"] == 10
    assert values[("exp7221-arc-session", "model_valid_induction_count")]["recomputed_value"] == 0
    assert (
        values[("exp7227-belief-learning", "belief_learning_value_score")]["recomputed_value"] == 0
    )
    assert values[("exp7229-rare-event-audit", "down_up_value_score")]["recomputed_value"] == 0
    assert values[("exp7230-native-belief", "native_cost_value_score")]["abstention"] is True
    assert values[("exp7230-native-belief", "native_cost_value_score")]["promoted"] is False
    assert all(
        {"unit_id", "arm", "seed", "metric", "metric_value", "error", "abstention"} <= row.keys()
        for row in claims
    )


def test_scenario_report_7232_science_keeps_four_questions_independent() -> None:
    """SCENARIO-REPORT-7232-SCIENCE: narrow receipts cannot replace value."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    questions = capstone.scientific_questions(capstone.recompute_claims(evidence), evidence)
    assert set(questions) == {
        "held_out_source_fidelity_and_value",
        "prospective_memory_utility",
        "live_adapter_withheld_progress_with_useful_model_validity",
        "matched_deployment_cost",
    }
    assert questions["held_out_source_fidelity_and_value"]["verdict_class"] == "blocked"
    assert questions["prospective_memory_utility"]["verdict_class"] == "null"
    assert (
        questions["live_adapter_withheld_progress_with_useful_model_validity"]["verdict_class"]
        == "null"
    )
    assert questions["matched_deployment_cost"]["verdict_class"] == "disqualified"


def test_scenario_report_7232_blocked_build_has_complete_scope(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7232-BLOCKED: blocked science is terminal, not partial."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_contract_rows"]) == 14
    assert len(artifact["evidence_matrix"]) == 14
    assert artifact["evidence_matrix"][-1]["task_id"] == "exp7232-capstone"
    assert artifact["evidence_matrix"][-1]["scope_complete"] is True
    assert artifact["rows"] == artifact["recomputed_claim_rows"]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7232_decisions_preserve_exact_prior_failures(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7232-DECISIONS: retirement stays task-specific."""

    artifact = built_artifact[0]
    decisions = artifact["branch_decisions"]
    assert len(decisions) == 14
    assert {row["action"] for row in decisions} <= capstone.BRANCH_ACTIONS
    capture = next(row for row in decisions if row["task_id"] == "exp7224-span-capture")
    assert capture["exact_same_verdict_recurrence"] is True
    assert capture["retire_if_same_verdict_applied"] is True
    assert capture["action"] == "retire"
    assert all(row["prior_failures"] for row in decisions)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7232_artifact_detects_top_level_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    error: str,
) -> None:
    """SCENARIO-REPORT-7232-ARTIFACT: terminal claims fail closed."""

    mutated = deepcopy(built_artifact[0])
    mutated[field] = replacement
    assert error in capstone.validate_artifact(mutated, root=ROOT)


def test_scenario_report_7232_artifact_detects_nested_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7232-ARTIFACT: roster, claims, and hashes are bound."""

    artifact = built_artifact[0]
    roster = deepcopy(artifact)
    roster["task_contract_rows"].pop()
    assert "task_contract_rows" in capstone.validate_artifact(roster, root=ROOT)
    claim = deepcopy(artifact)
    claim["rows"][0]["recomputed_value"] = 999
    assert "claim_rows" in capstone.validate_artifact(claim, root=ROOT)
    source = deepcopy(artifact)
    path = next(iter(source["source_artifact_hashes"]))
    source["source_artifact_hashes"][path] = "sha256:bad"
    assert "source_artifact_hashes" in capstone.validate_artifact(source, root=ROOT)
    action = deepcopy(artifact)
    action["branch_decisions"][0]["action"] = "invent"
    assert "branch_decisions" in capstone.validate_artifact(action, root=ROOT)


def test_req_report_7232_publication_runner_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7232: the unchanged four-gate subprocess fails closed."""

    monkeypatch.setattr(
        capstone.base,
        "run_publication_gate",
        lambda _root: {**_publication_receipt(), "exit_code": 0},
    )
    assert capstone.run_publication_gate(ROOT)["paper_ready"] is False
    monkeypatch.setattr(
        capstone.base,
        "run_publication_gate",
        lambda _root: {"gates": {}},
    )
    with pytest.raises(RuntimeError, match="publication gate"):
        capstone.run_publication_gate(ROOT)


def test_req_report_7232_build_rejects_bad_date_and_gate(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7232: invalid lifecycle inputs stop before terminal output."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT, "20260911", output, checkpoint, lambda _root: _publication_receipt()
        )
    with pytest.raises(RuntimeError, match="publication gate"):
        capstone.build_artifact(ROOT, capstone.RUN_DATE, output, checkpoint, lambda _root: {})


def test_req_report_7232_cli_build_and_validate_paths(
    tmp_path: Path,
    built_artifact: tuple[dict[str, object], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7232: both public CLI paths use explicit temporary files."""

    output = tmp_path / "result.json"
    output.write_text(json.dumps(built_artifact[0]), encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 0
    output.write_text("{}", encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 1

    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        capstone,
        "build_artifact",
        lambda *args, **_kwargs: called.append(args) or {},
    )
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output)]) == 0
    assert called and called[0][1] == capstone.RUN_DATE


def test_req_report_7232_read_json_rejects_non_object(tmp_path: Path) -> None:
    """REQ-REPORT-7232: malformed JSON roots cannot become evidence."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        capstone.read_json(path)


def test_scenario_report_7232_contract_reports_malformed_and_mismatched_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7232-CONTRACT: malformed or changed plans fail closed."""

    yaml_path = tmp_path / capstone.FROZEN_YAML_PATH
    design_path = tmp_path / capstone.FROZEN_DESIGN_PATH
    receipt_path = tmp_path / capstone.CONTRACT_RECEIPT_PATH
    yaml_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("[]", encoding="utf-8")
    design_path.write_text("wrong design", encoding="utf-8")
    receipt_path.write_text('{"raw_source_rows":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root"):
        capstone.load_contract(tmp_path)

    yaml_path.write_text("tasks: []\n", encoding="utf-8")
    monkeypatch.setattr(
        capstone.contract_source,
        "evaluate_contract",
        lambda *_args: {"passed": False, "contract_rows": []},
    )
    contract = capstone.load_contract(tmp_path)
    assert set(contract["errors"]) == {
        "frozen_contract_mismatch",
        "contract_id_order",
        "contract_source_hash",
    }


def test_scenario_report_7232_claim_error_precedence() -> None:
    """SCENARIO-REPORT-7232-CLAIMS: mismatch and authentication stay explicit."""

    unauthenticated = {
        "accepted_for_promoted_evidence": False,
        "quarantine_receipt": {"quarantined": False},
    }
    row = capstone._claim("task", "metric", 1, 1, (), unauthenticated, False)
    assert row["error"] == "unauthenticated_upstream"
    mismatch = capstone._claim("task", "metric", 0, 1, (), unauthenticated, False)
    assert mismatch["error"] == "declared_value_mismatch"


def test_scenario_report_7232_validator_detects_same_id_contract_mutation(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7232-ARTIFACT: same-ID content changes still fail."""

    artifact = deepcopy(built_artifact[0])
    artifact["task_contract_rows"][0]["title"] = "changed"
    artifact["reproducibility_checksum"] = capstone.reproducibility_checksum(artifact)
    assert "task_contract_rows" in capstone.validate_artifact(artifact, root=ROOT)


def test_req_report_7232_build_fails_closed_on_precondition_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7232: internal failures stop before a trusted terminal result."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    failed = {
        "check": "driving_requirement",
        "upstream": "spec",
        "field": "REQ-*",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    monkeypatch.setattr(capstone, "_preconditions", lambda *_args: ([failed], {}))
    with pytest.raises(RuntimeError, match="essential capstone precondition"):
        capstone.build_artifact(
            ROOT, capstone.RUN_DATE, output, checkpoint, lambda _root: _publication_receipt()
        )

    monkeypatch.undo()
    real_validate = capstone.validate_artifact
    monkeypatch.setattr(capstone, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(RuntimeError, match="capstone validation failed"):
        capstone.build_artifact(
            ROOT, capstone.RUN_DATE, output, checkpoint, lambda _root: _publication_receipt()
        )

    calls = 0

    def fail_second(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else ["bad_final"]

    monkeypatch.setattr(capstone, "validate_artifact", fail_second)
    with pytest.raises(RuntimeError, match="final file-parser validation failed"):
        capstone.build_artifact(
            ROOT, capstone.RUN_DATE, output, checkpoint, lambda _root: _publication_receipt()
        )
    monkeypatch.setattr(capstone, "validate_artifact", real_validate)
