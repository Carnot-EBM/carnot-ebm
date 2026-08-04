"""Tests for the Exp6100 V529 transition receipt.

Spec refs: REQ-REPORT-6100,
SCENARIO-REPORT-6100-ACTIVATED-MATRIX,
SCENARIO-REPORT-6100-TERMINAL-CLASSES,
SCENARIO-REPORT-6100-MISSING-AND-GATE-BLOCKS,
SCENARIO-REPORT-6100-DUPLICATE-DEBT-AND-VERIFIER,
SCENARIO-REPORT-6100-RANGE-COLLISION,
SCENARIO-REPORT-6100-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6100_transition_v529 as mod


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
        "exp5962-v528-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V528 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp5963-exact-atom-pair-fixture": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: sealed exact atom-pair fixture is ready",
            "inference_substrate": "deterministic_exact_fixture_generation_no_llm",
        },
        "exp5964-sota-atom-compatibility-corpus": {
            "status": "blocked",
            "honest_verdict": "blocked: insufficient_free_vram",
            "inference_substrate": "live_llm_embedding_extraction",
        },
        "exp5966-discriminative-constraint-acquisition": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [{"upstream": "exp5965-portable-atom-energy-ranker"}],
        },
        "exp5967-delayed-commit-memory-fixture": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_memory_fixture_ready",
            "inference_substrate": "deterministic_delayed_commit_transactional_replay_no_llm",
        },
        "exp5968-delayed-commit-csl-prospective": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_prospective_csl_ready",
            "inference_substrate": "deterministic_delayed_commit_csl_prospective_no_llm",
        },
        "exp5969-csl-poison-drift-abi-audit": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_csl_survives_poison_drift_abi_audit",
            "inference_substrate": "deterministic_csl_poison_drift_abi_audit_no_llm",
        },
        "exp5970-arc-strip-swap-sentinel": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: strip-swap sentinel ready",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "exp5971-arc-strip-swap-battery": {
            "status": "complete_null",
            "honest_verdict": "complete_null: original anchor support is empty",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "exp5972-arc-llm-on-budget2000-feasibility": {
            "status": "complete_feasible",
            "honest_verdict": "complete_feasible: 25-game upper projection fits 12 hours",
            "inference_substrate": "live_llm_inference",
        },
        "exp5973-v528-capstone-reconciliation": {
            "status": "complete_with_blocks",
            "honest_verdict": "complete_with_blocks: .528 reconciled with blocks",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    }
    return fixtures[task_id]


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": "exp6100-transition-v529",
                "milestone": mod.MILESTONE_TO,
                "title": "Exact terminal-boundary handoff from .528 into .529",
                "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
                "prompt": "Archive .528 and prove Exp6100-Exp6111 collision-free.",
            },
            {
                "id": "exp6101-v529-source-delta-ingestion",
                "milestone": mod.MILESTONE_TO,
                "title": "Dated evidence refresh after the V529 planner marker",
                "deliverable": "results/experiment_6101_v529_source_delta_ingestion.json",
            },
            {
                "id": "exp6102-sota-atom-corpus-vram-recovery",
                "milestone": mod.MILESTONE_TO,
                "title": "Checkpointed all-family exact-atom representation corpus VRAM recovery",
                "deliverable": "results/experiment_6102_sota_atom_corpus_vram_recovery.json",
            },
            {
                "id": "exp6103-phase-d-difficulty-ladder-fixture",
                "milestone": mod.MILESTONE_TO,
                "title": "Sealed low-chance Phase-D model-difficulty ladder",
                "deliverable": "results/experiment_6103_phase_d_difficulty_ladder_fixture.json",
                "prompt": "Exp6104 may run only after this fixture is ready.",
            },
        ],
    }


def _completion_payload(include_528_blocks: int = 1) -> JsonDict:
    block = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-04",
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
    old_duplicate = {
        "id": "2026.07.510",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    return {
        "milestones": [
            deepcopy(old_duplicate),
            deepcopy(old_duplicate),
            *[deepcopy(block) for _ in range(include_528_blocks)],
        ]
    }


def _make_root(root: Path, *, include_528_blocks: int = 1) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id in {"exp5961-transition-v528", "exp5965-portable-atom-energy-ranker"}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_5961_gemma31b_placement_decision_corrected.json",
        {"honest_verdict": "complete: alias must not launder the missing transition"},
    )
    _write_json(
        root,
        "results/experiment_5967_delayed_commit_event_fixture.json",
        {"honest_verdict": "complete: prompt-path spelling drift must be ignored"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT",
                "",
                "**Experiment range:** Exp6100-Exp6111",
                "### Exp6100",
                "**Deliverable:** `results/experiment_6100_transition_v529.json`",
                "### Exp6104",
                "**Deliverable:** `results/experiment_6104_phase_d_same_model_candidate_pool.json`",
                "### Exp6111",
                "**Deliverable:** `results/experiment_6111_v529_capstone_reconciliation.json`",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_528_blocks=include_528_blocks)),
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-03 13:37 UTC | Milestone 2026.07.528 activated | OK | 13 tasks queued |",
                "| 2026-08-03 15:01 UTC | Exact terminal-boundary handoff from .527 into .52 | FAIL | cap |",
                "| 2026-08-03 16:25 UTC | Exact terminal-boundary handoff from .527 into .52 | FAIL | cap |",
                "| 2026-08-03 17:50 UTC | Exact terminal-boundary handoff from .527 into .52 | FAIL | cap |",
                "| 2026-08-03 21:55 UTC | Gated on Exp5964 ready: portable exact-atom compat | GATE_BLOCK | skip |",
                "| 2026-08-03 21:55 UTC | Gated on Exp5965 ready: end-to-end discriminative | GATE_BLOCK | failed |",
                "| 2026-08-04 09:00 UTC | Branch-independent .528 capstone and exact reconci | OK | 87 passed |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\nretired_experiments: []\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.DOC_RECONCILE_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6100\n")


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
        if task_id not in {"exp5961-transition-v528", "exp5965-portable-atom-energy-ranker"}
    }


def _test_receipts() -> list[JsonDict]:
    rows = [
        {
            "command": f"task-owned {kind}",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": kind,
        }
        for kind in mod.REQUIRED_TASK_OWNED_GATE_KINDS
    ]
    rows.extend(
        [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "after",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "before",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "after",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["old_probe.py"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "after",
                "root_clutter_paths": ["old_probe.py"],
            },
        ]
    )
    return rows


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
        duration_s=1.25,
    )


def test_req_report_6100_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6100: OpenSpec names the transition contract and field principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6100") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    for scenario in (
        "SCENARIO-REPORT-6100-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6100-TERMINAL-CLASSES",
        "SCENARIO-REPORT-6100-MISSING-AND-GATE-BLOCKS",
        "SCENARIO-REPORT-6100-DUPLICATE-DEBT-AND-VERIFIER",
        "SCENARIO-REPORT-6100-RANGE-COLLISION",
        "SCENARIO-REPORT-6100-SCHEMA",
    ):
        assert scenario in section
    assert "global_suite_failure_delta <= 0" in section
    assert "Exp6100 through Exp6111" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6100_exact_matrix_terminal_missing_and_gate_blocks(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6100-ACTIVATED-MATRIX: thirteen .528 identities classify once."""

    _make_root(tmp_path, include_528_blocks=2)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.528",
        "destination_milestone": "2026.08.529",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 13
    assert matrix["exp5961-transition-v528"]["present"] is False
    assert matrix["exp5961-transition-v528"]["terminal_evidence_source"] == "declared_absence"
    assert matrix["exp5967-delayed-commit-memory-fixture"]["declared_deliverable"] == (
        "results/experiment_5967_delayed_commit_memory_fixture.json"
    )

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["task_ids_by_terminal_class"]["missing"] == ["exp5961-transition-v528"]
    assert classes["task_ids_by_terminal_class"]["conductor-gate-blocked"] == [
        "exp5965-portable-atom-energy-ranker",
        "exp5966-discriminative-constraint-acquisition",
    ]
    assert classes["all_activated_terminal"] is True

    missing = report["missing_artifact_receipt"]
    assert missing["missing_declared_artifact_task_ids"] == ["exp5961-transition-v528"]
    assert missing["same_number_aliases_ignored"]["exp5961-transition-v528"] == [
        "results/experiment_5961_gemma31b_placement_decision_corrected.json"
    ]
    gates = report["conductor_gate_block_receipts"]
    assert gates["gate_blocked_task_ids"] == [
        "exp5965-portable-atom-energy-ranker",
        "exp5966-discriminative-constraint-acquisition",
    ]
    assert gates["executed_experiment_claim_count"] == 0

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 11
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_5961_transition_v528.json",
        "results/experiment_5965_portable_atom_energy_ranker.json",
    ]
    mod.validate_artifact(report)


def test_scenario_report_6100_append_once_debt_delta_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6100-DUPLICATE-DEBT-AND-VERIFIER: inherited debt is delta-gated."""

    _make_root(tmp_path, include_528_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"
    assert report["research_complete_append_receipt"]["before_milestone_block_count"] == 2
    debt = report["inherited_debt_baselines_and_deltas"]
    assert debt["global_suite_failure_delta"] == 0
    assert debt["global_spec_gap_delta"] == 0
    assert debt["root_clutter_delta"] == 0
    assert debt["non_amplification_gate_passed"] is True
    mod.validate_artifact(report)

    absent = tmp_path / "absent"
    _make_root(absent, include_528_blocks=0)
    first = _build(absent)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    second = _build(absent)
    assert second["research_complete_append_count"] == 0

    amplified_rows = deepcopy(_test_receipts())
    for row in amplified_rows:
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
    amplified = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=amplified_rows,
        duration_s=1.25,
    )
    assert amplified["status"] == "blocked"
    assert "global_suite_debt_amplified" in amplified["preconditions_checked"]["failed_preconditions"]

    _write_json(tmp_path, "results/experiment_6107_stale_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6107_stale_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6107],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6100_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6100-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6100_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
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
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="complete_null: bad"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (
            lambda artifact: artifact.update(next_range_collision_count=1, status="complete_with_terminal_receipts"),
            "next_range_collision_count must be zero",
        ),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop("exp5973-v528-capstone-reconciliation"),
            "exactly thirteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5973-v528-capstone-reconciliation": []}
            ),
            "exactly thirteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"]["exp5961-transition-v528"].update(
                identity=["2026.07.528", "exp5961-transition-v528", "wrong.json"]
            ),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"]["terminal_class_by_task_id"].update(
                {"exp5961-transition-v528": "complete"}
            ),
            "terminal classes",
        ),
        (lambda artifact: artifact.update(missing_artifact_receipt=[]), "missing artifact"),
        (
            lambda artifact: artifact["missing_artifact_receipt"].update(
                missing_declared_artifact_task_ids=[]
            ),
            "missing artifact",
        ),
        (lambda artifact: artifact.update(conductor_gate_block_receipts=[]), "gate block"),
        (
            lambda artifact: artifact["conductor_gate_block_receipts"].update(executed_experiment_claim_count=1),
            "gate block",
        ),
        (lambda artifact: artifact.update(adversarial_verifier_receipts=[]), "adversarial verifier"),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=1
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].pop("receipt_hash"),
            "adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].update(command="python other.py"),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["inherited_debt_baselines_and_deltas"].update(
                non_amplification_gate_passed=False
            ),
            "debt non-amplification",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: (
                artifact["protected_files_unchanged"].update(all_unchanged=True),
                next(iter(artifact["protected_files_unchanged"]["files"].values())).update(
                    unchanged=False
                ),
            ),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
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
    rows = _test_receipts()
    rows = [row for row in rows if row.get("suite_kind") != "coverage"]
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda _root, _before: {
            "files": {"research-roadmap.yaml": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(mod, "_resource_receipts", lambda _root: {"disk": {"ok": False}, "memory": {"ok": True}})
    monkeypatch.setattr(mod, "_atomic_output_receipt", lambda _path: {"ok": False})
    blocked = mod.build_report(bad, adversarial_receipts=_receipts(), tests_run=rows, duration_s=1.25)
    failed = set(blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v528_activation_line_missing_or_not_thirteen",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6100_missing",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= failed

    many = tmp_path / "many"
    _make_root(many)
    bad_roadmap = _active_roadmap_payload()
    bad_roadmap["milestone"] = "2026.08.999"
    bad_roadmap["tasks"] = bad_roadmap["tasks"][:-1]
    _write_text(many, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(many, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    (many / mod.ROADMAP_DOC_RELATIVE_PATH).unlink()
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp5963-exact-atom-pair-fixture"],
        {"status": "unknown", "honest_verdict": "unknown"},
    )
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp5962-v528-source-delta-ingestion")
    sparse_receipts["exp5963-exact-atom-pair-fixture"] = {
        **sparse_receipts["exp5963-exact-atom-pair-fixture"],
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
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("tests/python/new/test_spec.py::test_new")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")
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
        "active_roadmap_task_ids_mismatch",
        "roadmap_next_unloadable",
        "vnext_proposal_missing",
        "terminal_outcomes_not_preserved",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "task_owned_gate_failed",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
    } <= many_failed


def test_req_report_6100_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6100: helper failures produce explicit blocked preconditions."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(output).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

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

    assert (
        mod._classify_task(
            "exp5965-portable-atom-energy-ranker",
            {},
            {"present": False},
            {"latest_status": "GATE_BLOCK"},
        )
        == "conductor-gate-blocked"
    )
    assert (
        mod._classify_task(
            "exp5961-transition-v528",
            {},
            {"present": False},
            {"latest_status": "FAIL"},
        )
        == "missing"
    )
    assert (
        mod._classify_task(
            "x",
            {"status": "complete"},
            {"present": True},
            {},
        )
        == "complete"
    )
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}, {}) == "missing"
    staging = mod._optional_staging_roadmap_receipt(
        {"milestone": "2026.08.529"},
        {"present": True, "loadable": True},
    )
    assert staging["reason"] == "present_optional_staging"
    assert mod._same_number_aliases(tmp_path / "no-results", "not-an-exp", Path("x")) == []
    assert mod._range_number_mentions("Exp6100 and experiment_6111") == {6100, 6111}
    assert mod._range_number_mentions("value 0.6107 is not an experiment id") == set()
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6100})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6100, 6111})
        == "allowed_allocation_reference"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []

    after_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {"ownership_class": "global_suite", "phase": "after", "failure_node_ids": ["global_after"]},
            {"ownership_class": "spec_coverage", "phase": "after", "missing_node_ids": ["spec_after"]},
            {"ownership_class": "root_clutter", "phase": "after", "root_clutter_paths": ["root_after.py"]},
        ],
    )
    assert after_only_debt["global_suite_failure_delta"] == 0
    assert after_only_debt["global_spec_gap_delta"] == 0
    assert after_only_debt["root_clutter_delta"] == 0

    before_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {"ownership_class": "global_suite", "phase": "before", "failure_node_ids": ["global_before"]},
            {"ownership_class": "spec_coverage", "phase": "before", "missing_node_ids": ["spec_before"]},
            {"ownership_class": "root_clutter", "phase": "before", "root_clutter_paths": ["root_before.py"]},
        ],
    )
    assert before_only_debt["global_suite_failure_after_node_ids"] == ["global_before"]
    assert before_only_debt["global_spec_gap_after_node_ids"] == ["spec_before"]
    assert before_only_debt["root_clutter_after_paths"] == ["root_before.py"]
    no_debt_rows = mod._debt_baselines_and_deltas(tmp_path, [])
    assert no_debt_rows["root_clutter_delta"] == 0

    _make_root(tmp_path / "receipt_root")
    receipt_report = _build(tmp_path / "receipt_root")
    receipt_matrix = receipt_report["activated_task_and_deliverable_matrix"]
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp5962-v528-source-delta-ingestion")
    sparse_receipts["exp5963-exact-atom-pair-fixture"] = {
        **sparse_receipts["exp5963-exact-atom-pair-fixture"],
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
    }
    sparse_receipts["exp5964-sota-atom-compatibility-corpus"] = {
        **sparse_receipts["exp5964-sota-atom-compatibility-corpus"],
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
    grouped = mod._adversarial_receipts_group(sparse_receipts, receipt_matrix)
    assert grouped["failed_receipt_task_ids"] == ["exp5964-sota-atom-compatibility-corpus"]
    assert grouped["warning_receipt_task_ids"] == ["exp5963-exact-atom-pair-fixture"]
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
