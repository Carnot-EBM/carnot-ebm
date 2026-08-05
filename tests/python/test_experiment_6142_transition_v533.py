"""Tests for the Exp6142 V533 transition receipt.

Spec refs: REQ-REPORT-6142,
SCENARIO-REPORT-6142-ACTIVATED-MATRIX,
SCENARIO-REPORT-6142-TERMINAL-CLASSES,
SCENARIO-REPORT-6142-DUPLICATE-ACTIVATION,
SCENARIO-REPORT-6142-RANGE-COLLISION,
SCENARIO-REPORT-6142-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6142_transition_v533 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    fixtures: dict[str, JsonDict] = {
        "exp6138-transition-v532": {
            "schema": "carnot.experiment_6138.transition_v532.v1",
            "status": "complete_with_terminal_receipts",
            "honest_verdict": (
                "complete: archived exactly nine terminal .531 identities into .532"
            ),
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6139-v532-source-delta-ingestion": {
            "schema": "carnot.experiment_6139.v532_source_delta_ingestion.v1",
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V532 source deltas",
            "inference_substrate": "literature_ingestion",
            "accepted_rejected_duplicate_retired_and_abstained_findings": {
                "accepted": [],
                "accepted_count": 0,
            },
            "references_append_receipt": {"appended": False, "accepted_count": 0},
        },
        "exp6140-phase-d-exp6128-option-psychometrics": {
            "schema": "carnot.experiment_6140.phase_d_option_psychometrics.v1",
            "status": "retired",
            "honest_verdict": (
                "retired: saturation plus position-confounded easy families and "
                "typed-choice below-floor fallback leave true inability versus "
                "distractor/position unresolved for Exp6128 source-domain recovery"
            ),
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "retirement_triggered": True,
            "empirical_item_bank_design_ready_score": 0.0,
            "typed_choice_below_floor": True,
            "position_confounded_easy_families": True,
        },
    }
    return fixtures[task_id]


def _v533_task(task_id: str, deliverable: str) -> JsonDict:
    return {
        "id": task_id,
        "milestone": mod.MILESTONE_TO,
        "title": task_id,
        "deliverable": deliverable,
    }


def _active_v532_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_FROM,
        "milestone_title": mod.MILESTONE_FROM_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_FROM,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _active_v533_payload() -> JsonDict:
    task_pairs = [
        ("exp6142-transition-v533", "results/experiment_6142_transition_v533.json"),
        ("exp6143-test-artifact-isolation", "results/experiment_6143_test_artifact_isolation.json"),
        (
            "exp6144-v533-source-delta-ingestion",
            "results/experiment_6144_v533_source_delta_ingestion.json",
        ),
        ("exp6145-constraint-shift-stream", "results/experiment_6145_constraint_shift_stream.json"),
        (
            "exp6146-sota-constraint-event-corpus",
            "results/experiment_6146_sota_constraint_event_corpus.json",
        ),
        (
            "exp6147-task-aware-energy-calibration",
            "results/experiment_6147_task_aware_energy_calibration.json",
        ),
        (
            "exp6148-shifted-family-admission-held",
            "results/experiment_6148_shifted_family_admission_held.json",
        ),
        (
            "exp6149-certified-strategy-schema-fixture",
            "results/experiment_6149_certified_strategy_schema_fixture.json",
        ),
        (
            "exp6150-frozen-qwen-continuous-self-learning-ab",
            "results/experiment_6150_frozen_qwen_continuous_self_learning_ab.json",
        ),
        (
            "exp6151-strategy-memory-shadow-adapter",
            "results/experiment_6151_strategy_memory_shadow_adapter.json",
        ),
        (
            "exp6152-typed-stochastic-constraint-ir",
            "results/experiment_6152_typed_stochastic_constraint_ir.json",
        ),
        (
            "exp6153-thermalized-program-error-audit",
            "results/experiment_6153_thermalized_program_error_audit.json",
        ),
        (
            "exp6154-arc-task-aware-energy-generalization",
            "results/experiment_6154_arc_task_aware_energy_generalization.json",
        ),
        (
            "exp6155-v533-capstone-reconciliation",
            "results/experiment_6155_v533_capstone_reconciliation.json",
        ),
    ]
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [_v533_task(task_id, deliverable) for task_id, deliverable in task_pairs],
    }


def _completion_payload(include_532_blocks: int = 1) -> JsonDict:
    duplicate = {
        "id": "2026.07.510",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    canonical = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-05",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }
    return {
        "milestones": [
            deepcopy(duplicate),
            deepcopy(duplicate),
            *[deepcopy(canonical) for _ in range(include_532_blocks)],
        ]
    }


def _make_root(
    root: Path,
    *,
    include_532_blocks: int = 1,
    include_next: bool = False,
) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == mod.STRUCTURED_GATE_SKIP_TASK_ID:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, "results/experiment_6140_phase_d_exp6128_option_psychometrics.rows.jsonl")
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_v533_payload()))
    if include_next:
        _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_v532_payload()))
        _write_text(root, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(_active_v533_payload()))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT - Milestone 2026.08.533",
                "",
                "**Experiment range:** Exp6142-Exp6155",
                "former proposal-only Exp6142-Exp6151 text is not completion evidence.",
                "Exp6142 transition",
                "Exp6155 capstone",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_532_blocks=include_532_blocks)),
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-05 06:46 UTC | Plan milestone 2026.08.532 | OK | 4 tasks proposed |",
                "| 2026-08-05 06:48 UTC | Milestone 2026.08.532 activated | OK | 4 tasks queued |",
                "| 2026-08-05 10:36 UTC | Exact terminal-boundary handoff from .531 into .53 | OK | cache hit |",
                "| 2026-08-05 10:55 UTC | Reliable dated evidence refresh after the V532 pla | OK | cache hit |",
                "| 2026-08-05 12:50 UTC | Frozen Exp6128 option-aware psychometrics and fami | OK | retired line |",
                "| 2026-08-05 12:52 UTC | Gated on Exp6140 design readiness: exact empirical | GATE_BLOCK | Pre-emptive skip: upstream retired (exp6140-phase-d-exp6128-option-psychometrics) |",
                "| 2026-08-05 13:48 UTC | Plan milestone 2026.08.533 | OK | 14 tasks proposed |",
                "| 2026-08-05 13:50 UTC | Milestone 2026.08.533 activated | OK | 14 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(
        root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\nretired_experiments: []\n"
    )
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
        mod.E2E_PLAN_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6142\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        ],
        "flagged_count": 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id != mod.STRUCTURED_GATE_SKIP_TASK_ID
    }


def _test_receipts() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_6142_transition_v533.py -q --no-cov -n 0",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kinds": [
                "unit",
                "yaml_parse",
                "exact_path",
                "retirement",
                "gate_skip",
                "duplicate_history",
                "activation",
                "exclusion_manifest",
                "range_collision",
                "protected_file",
                "applicable_e2e",
                "no_new_root_clutter",
            ],
        },
        {
            "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6142_transition_v533.py -m pytest tests/python/test_experiment_6142_transition_v533.py -q --no-cov -n 0",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "coverage",
        },
        {
            "command": ".venv/bin/python scripts/adversarial_verify.py --json <present .532 declared deliverables>",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "adversarial_verifier",
        },
        {
            "command": ".venv/bin/python scripts/check_spec_coverage.py",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "spec_coverage",
        },
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 0,
            "ownership_class": "global_suite",
            "phase": "after",
            "failure_node_ids": [],
        },
        {
            "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
            "exit_code": 0,
            "ownership_class": "root_clutter",
            "phase": "before",
            "root_clutter_paths": [],
        },
        {
            "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
            "exit_code": 0,
            "ownership_class": "root_clutter",
            "phase": "after",
            "root_clutter_paths": [],
        },
    ]


def _build(root: Path, rows: list[JsonDict] | None = None) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts() if rows is None else rows,
        duration_s=1.25,
    )


def test_req_report_6142_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6142: OpenSpec names exact identity, retirement, skip, and collision rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6142") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "Exp6138 through Exp6141" in section
    assert "Exp6140 SHALL remain a `retired` scientific line" in section
    assert "Exp6141 SHALL remain a structured" in section
    assert "Exp6142 through Exp6155" in section
    for scenario in (
        "SCENARIO-REPORT-6142-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6142-TERMINAL-CLASSES",
        "SCENARIO-REPORT-6142-DUPLICATE-ACTIVATION",
        "SCENARIO-REPORT-6142-RANGE-COLLISION",
        "SCENARIO-REPORT-6142-SCHEMA",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6142_matrix_retirement_skip_and_proposal_exclusion(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6142-ACTIVATED-MATRIX: four .532 identities classify once."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.08.532",
        "destination_milestone": "2026.08.533",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 4
    assert matrix["exp6140-phase-d-exp6128-option-psychometrics"][
        "same_number_aliases_ignored"
    ] == ["results/experiment_6140_phase_d_exp6128_option_psychometrics.rows.jsonl"]
    assert matrix[mod.STRUCTURED_GATE_SKIP_TASK_ID]["present"] is False
    assert matrix[mod.STRUCTURED_GATE_SKIP_TASK_ID]["terminal_class"] == "structured-gate-skip"

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["complete"] == ["exp6138-transition-v532"]
    assert classes["task_ids_by_terminal_class"]["complete-null"] == [
        "exp6139-v532-source-delta-ingestion"
    ]
    assert classes["task_ids_by_terminal_class"]["retired"] == [
        "exp6140-phase-d-exp6128-option-psychometrics"
    ]

    retirement = report["scientific_retirement_receipt"]
    assert retirement["task_id"] == "exp6140-phase-d-exp6128-option-psychometrics"
    assert retirement["retirement_triggered"] is True
    assert retirement["empirical_item_bank_design_ready_score"] == 0.0
    assert retirement["source_domain_recovery_retired"] is True
    assert retirement["distinct_from_structured_gate_skip"] is True

    skip = report["structured_gate_skip_receipt"]
    assert skip["task_id"] == mod.STRUCTURED_GATE_SKIP_TASK_ID
    assert skip["declared_artifact_present"] is False
    assert skip["reported_as_run"] is False
    assert skip["upstream_retired_task_id"] == mod.SCIENTIFIC_RETIREMENT_TASK_ID

    excluded = report["proposal_only_identities_excluded"]
    assert excluded["former_proposal_only_task_ids"] == [
        "exp6142",
        "exp6143",
        "exp6144",
        "exp6145",
        "exp6146",
        "exp6147",
        "exp6148",
        "exp6149",
        "exp6150",
        "exp6151",
    ]
    assert excluded["old_proposal_carries_completion_credit"] is False
    assert excluded["all_excluded_from_v532_archive"] is True
    assert excluded["canonical_active_range_task_ids"] == [
        f"exp{number}" for number in range(6142, 6156)
    ]

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 3
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_6141_phase_d_empirical_item_bank.json"
    ]
    mod.validate_artifact(report)


def test_scenario_report_6142_append_once_activation_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6142-DUPLICATE-ACTIVATION: history and activation are idempotent."""

    _make_root(tmp_path, include_532_blocks=0)
    first = _build(tmp_path)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    assert first["staged_roadmap_activation_receipt"]["mode"] == "already_active"
    assert first["staged_roadmap_activation_receipt"]["active_roadmap_task_count"] == 14
    assert first["next_task_range"]["start"] == "exp6142"
    assert first["next_task_range"]["end"] == "exp6155"
    assert first["next_task_range"]["reserved_count"] == 14
    assert first["next_range_collision_count"] == 0

    second = _build(tmp_path)
    assert second["research_complete_append_count"] == 0
    assert second["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"

    staged = tmp_path / "staged"
    _make_root(staged, include_next=True)
    staged_report = _build(staged)
    assert staged_report["staged_roadmap_activation_receipt"]["mode"] == "copied_staged_roadmap"
    assert staged_report["staged_roadmap_activation_receipt"]["copied_exactly"] is True
    assert yaml.safe_load((staged / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")) == (
        yaml.safe_load((staged / mod.ROADMAP_NEXT_RELATIVE_PATH).read_text(encoding="utf-8"))
    )

    _write_json(tmp_path, "results/experiment_6155_unowned_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6155_unowned_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6155],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6142_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6142-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6142_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (
            lambda artifact: artifact.update(inference_substrate="live_llm_inference"),
            "inference_substrate",
        ),
        (lambda artifact: artifact.update(honest_verdict="retired: bad"), "honest_verdict"),
        (
            lambda artifact: artifact.update(next_range_collision_count="0"),
            "next_range_collision_count",
        ),
        (
            lambda artifact: artifact.update(research_complete_append_count=2),
            "research_complete_append_count",
        ),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                mod.STRUCTURED_GATE_SKIP_TASK_ID
            ),
            "exactly four",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {mod.STRUCTURED_GATE_SKIP_TASK_ID: []}
            ),
            "exactly four",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                mod.STRUCTURED_GATE_SKIP_TASK_ID
            ].update(identity=["2026.08.532", mod.STRUCTURED_GATE_SKIP_TASK_ID, "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({mod.SCIENTIFIC_RETIREMENT_TASK_ID: "complete"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact["scientific_retirement_receipt"].update(
                retirement_triggered=False
            ),
            "scientific retirement",
        ),
        (
            lambda artifact: artifact["structured_gate_skip_receipt"].update(reported_as_run=True),
            "structured gate skip",
        ),
        (
            lambda artifact: artifact.update(structured_gate_skip_receipt=[]),
            "structured gate skip",
        ),
        (
            lambda artifact: artifact["proposal_only_identities_excluded"].update(
                old_proposal_carries_completion_credit=True
            ),
            "proposal-only",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=2
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"].append([]),
            "adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].update(
                command="python other.py"
            ),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].pop(
                "receipt_hash"
            ),
            "adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["staged_roadmap_activation_receipt"].update(activated=False),
            "activation",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count=1, status="complete"),
            "next_range_collision_count must be zero",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 9.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)

    bad = tmp_path / "bad"
    _make_root(bad)
    _write_text(bad, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no activation\n")
    _write_text(bad, mod.SPEC_RELATIVE_PATH, "missing req\n")
    (bad / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    rows = [row for row in _test_receipts() if row.get("suite_kind") != "coverage"]
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda _root, _before: {
            "files": {"scripts/research_conductor.py": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_atomic_output_receipt",
        lambda _path: {"declared_path": "x", "atomic_probe_write_ok": False, "ok": False},
    )
    blocked = mod.build_report(
        bad, adversarial_receipts=_receipts(), tests_run=rows, duration_s=1.25
    )
    failed = set(blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v532_activation_line_missing_or_not_four",
        "v533_activation_line_missing_or_not_fourteen",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6142_missing",
        "protected_file_modified",
        "atomic_output_unavailable",
    } <= failed

    many = tmp_path / "many"
    _make_root(many)
    bad_roadmap = _active_v533_payload()
    bad_roadmap["milestone"] = "2026.08.999"
    _write_text(many, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(many, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    (many / mod.ROADMAP_DOC_RELATIVE_PATH).unlink()
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS[mod.STRUCTURED_GATE_SKIP_TASK_ID],
        {"status": "complete", "honest_verdict": "complete: invented Exp6141 artifact"},
    )
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS[mod.SCIENTIFIC_RETIREMENT_TASK_ID],
        {
            **_artifact(mod.SCIENTIFIC_RETIREMENT_TASK_ID),
            "status": "complete",
            "honest_verdict": "complete: reopened retired line",
            "retirement_triggered": False,
        },
    )
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp6138-transition-v532")
    sparse_receipts["exp6139-v532-source-delta-ingestion"] = {
        **sparse_receipts["exp6139-v532-source-delta-ingestion"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    many_rows = _test_receipts()
    many_rows[0]["exit_code"] = 1
    for row in many_rows:
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")
    monkeypatch.setattr(
        mod,
        "_proposal_only_identities_excluded",
        lambda _active, _blocks, _matrix: {
            "former_proposal_only_task_ids": list(mod.FORMER_PROPOSAL_ONLY_TASK_IDS),
            "old_proposal_carries_completion_credit": True,
            "all_excluded_from_v532_archive": False,
            "canonical_active_range_task_ids": [],
            "canonical_range_replaces_old_proposal": False,
            "principle": mod.FIELD_PRINCIPLES["proposal_only_identities_excluded"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda _root, _terminal: {
            "append_count": 0,
            "duplicate_history_amplification_count": 1,
        },
    )
    many_blocked = mod.build_report(
        many,
        adversarial_receipts=sparse_receipts,
        tests_run=many_rows,
        duration_s=1.25,
    )
    many_failed = set(many_blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_milestone_mismatch",
        "roadmap_next_unloadable",
        "vnext_proposal_missing",
        "terminal_outcomes_not_preserved",
        "scientific_retirement_not_preserved",
        "structured_gate_skip_not_preserved",
        "proposal_only_identity_included",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "task_owned_gate_failed",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
    } <= many_failed

    mismatched = tmp_path / "mismatched"
    _make_root(mismatched, include_next=True)
    bad_next = _active_v533_payload()
    bad_next["milestone"] = "2026.08.998"
    _write_text(mismatched, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(bad_next))
    _write_text(
        mismatched,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "# Research Roadmap vNEXT - Milestone 2026.08.533\nmissing range\n",
    )
    mismatch_report = _build(mismatched)
    mismatch_failed = set(mismatch_report["preconditions_checked"]["failed_preconditions"])
    assert {
        "staged_roadmap_activation_failed",
        "roadmap_next_milestone_mismatch",
        "vnext_proposal_range_mismatch",
    } <= mismatch_failed


def test_req_report_6142_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6142: helper failures produce explicit blocked receipts."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(output).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

    _, missing_json_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_json_meta["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_json_meta = mod._read_json_mapping(bad_json)
    assert bad_json_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_json_meta = mod._read_json_mapping(list_json)
    assert list_json_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, bad_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"
    assert mod._history_blocks(tmp_path / "missing-root") == []
    assert mod._task_signature({"tasks": "bad"}) == ()
    list_history = tmp_path / "history-list.yaml"
    mod._write_history_blocks(list_history, [], [mod._completion_block_data()])
    assert isinstance(yaml.safe_load(list_history.read_text(encoding="utf-8")), list)
    nonterminal_append = mod._append_completion_if_absent(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"
    malformed = tmp_path / "malformed"
    _make_root(malformed, include_532_blocks=0)
    _write_text(malformed, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    malformed_append = mod._append_completion_if_absent(malformed, terminal=True)
    assert malformed_append["append_count"] == 1

    assert (
        mod._classify_task(
            mod.STRUCTURED_GATE_SKIP_TASK_ID,
            {},
            {"present": False},
            {"latest_status": "GATE_BLOCK", "latest_line": "Pre-emptive skip: upstream retired"},
        )
        == "structured-gate-skip"
    )
    assert mod._classify_task("x", {"status": "retired"}, {"present": True}, {}) == "retired"
    assert (
        mod._classify_task(
            "x", {"honest_verdict": "complete_null: no delta"}, {"present": True}, {}
        )
        == "complete-null"
    )
    assert mod._classify_task("x", {"status": "complete"}, {"present": True}, {}) == "complete"
    assert mod._classify_task("x", {"status": "blocked"}, {"present": True}, {}) == "blocked"
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}, {}) == "missing"
    assert mod._same_number_aliases(tmp_path / "no-results", "exp6140-x", Path("x")) == []
    assert mod._range_number_mentions("Exp6142 and experiment_6155") == {6142, 6155}
    assert mod._range_number_mentions("value 0.6142 is not an experiment id") == set()
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6142})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6142, 6155})
        == "canonical_v533_plan_reference"
    )
    assert (
        mod._allowed_range_reference_kind(
            Path("results/experiment_6138_transition_v532.json"), {6142, 6151}
        )
        == "replaced_v532_proposal_only_reference"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"stdout_json": {"reports": [{"max_severity": None}]}}) == -1
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    assert mod._dirty_worktree_receipt(tmp_path / "not-git")["git_present"] is False
    assert mod._dirty_worktree_receipt(REPO)["git_present"] is True

    verifier = mod._adversarial_receipts_group(
        {
            "exp6138-transition-v532": {
                **_receipt("exp6138-transition-v532"),
                "exit_code": 1,
                "stdout_json": {
                    "reports": [
                        {
                            "artifact": "x",
                            "loaded": True,
                            "flag_count": 1,
                            "max_severity": 3,
                            "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                        }
                    ]
                },
            },
            "exp6139-v532-source-delta-ingestion": {
                **_receipt("exp6139-v532-source-delta-ingestion"),
                "stdout_json": {
                    "reports": [
                        {
                            "artifact": "x",
                            "loaded": True,
                            "flag_count": 1,
                            "max_severity": 1,
                            "flags": [{"kind": "WARN", "severity": "warn"}],
                        }
                    ]
                },
            },
        },
        {
            "exp6138-transition-v532": {
                "present": True,
                "declared_deliverable": "results/experiment_6138_transition_v532.json",
            },
            "exp6139-v532-source-delta-ingestion": {
                "present": True,
                "declared_deliverable": (
                    "results/experiment_6139_v532_source_delta_ingestion.json"
                ),
            },
            **{
                task_id: {"present": False, "declared_deliverable": path.as_posix()}
                for task_id, path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
                if task_id
                not in {
                    "exp6138-transition-v532",
                    "exp6139-v532-source-delta-ingestion",
                }
            },
        },
    )
    assert verifier["failed_receipt_task_ids"] == ["exp6138-transition-v532"]
    assert verifier["warning_receipt_task_ids"] == ["exp6139-v532-source-delta-ingestion"]
    assert verifier["verified_present_declared_deliverable_count"] == 2
    assert (
        mod._root_clutter_delta(
            [{"ownership_class": "root_clutter", "phase": "after", "root_clutter_paths": ["x.py"]}]
        )
        == 0
    )
    assert (
        mod._root_clutter_delta(
            [
                {"ownership_class": "root_clutter", "phase": "before", "root_clutter_paths": []},
                {
                    "ownership_class": "root_clutter",
                    "phase": "after",
                    "root_clutter_paths": ["x.py"],
                },
            ]
        )
        == 1
    )
    assert (
        mod._root_clutter_delta(
            [
                {
                    "ownership_class": "root_clutter",
                    "phase": "before",
                    "root_clutter_paths": ["x.py"],
                }
            ]
        )
        == 0
    )
    assert mod._active_range_task_ids({"tasks": ["bad", {"id": "exp6142-transition-v533"}]}) == [
        "exp6142"
    ]
