"""Tests for Exp 3097 exact-fixture evaluation protocol audit.

Spec refs: REQ-VERIFY-3097, SCENARIO-VERIFY-3097.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import exact_fixture_eval_protocol_audit_v1 as exp
from carnot.eval import resyn_exact_fixture_bank_generator_v1 as fixture_bank


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_ARTIFACT_FIELDS = {
    "eval_protocol_ready",
    "usable_fixture_count",
    "rejected_fixture_count",
    "stratified_eval_manifest_path",
    "minimum_live_eval_count",
    "fixture_family_counts",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
REQUIRED_MANIFEST_FIELDS = {
    "task_family",
    "expected_answer",
    "solver_label",
    "perturbation_type",
    "verifier_target",
    "repair_target",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fixture_rows() -> list[dict[str, Any]]:
    return fixture_bank.generate_fixture_rows()


def _write_required_sources(root: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(root, exp.FIXTURE_MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        exp.EXP3084_REL_PATH,
        {
            "artifact": "experiment_3084_resyn_exact_fixture_bank_generator_v1",
            "resyn_fixture_bank_ready": True,
            "exact_fixture_count": len(rows),
            "family_counts": {
                family: sum(row["family"] == family for row in rows)
                for family in sorted({row["family"] for row in rows})
            },
            "fixture_manifest_path": exp.FIXTURE_MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: resyn_fixture_bank_ready=true",
        },
    )
    _write_json(
        root,
        exp.EXP3086_REL_PATH,
        {
            "artifact": "experiment_3086_dafny_z3_formal_feedback_pilot_v1",
            "formal_feedback_ready": False,
            "exact_ground_truth_count": 5,
            "honest_verdict": "complete: formal_feedback_ready=false",
        },
    )
    _write_json(
        root,
        exp.EXP3094_REL_PATH,
        {
            "artifact": "experiment_3094_capstone_v288",
            "capstone_ready": True,
            "next_milestone_recommendation": "2026.05.289: clear verifier/repair first",
            "honest_verdict": "complete: capstone_ready=true",
        },
    )
    (root / "CODEX.md").write_text("Spec First\nWrite Tests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text(
        "Adversarial Artifact Verification + Sample-Size Rigor\n", encoding="utf-8"
    )
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "experiment_template.py").write_text(
        "class ExperimentTemplate: ...\n", encoding="utf-8"
    )


def _write_tiny_3085_panel(root: Path, rows: list[dict[str, Any]]) -> None:
    selected: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in rows}):
        family_rows = sorted(
            [row for row in rows if row["family"] == family], key=lambda row: row["fixture_id"]
        )
        selected.extend(family_rows[:3])
    panel_rows: list[dict[str, Any]] = []
    for row in selected:
        target = exp.expected_targets(row)
        for policy in ("baseline", "task_abstention"):
            panel_rows.append(
                {
                    "fixture_id": row["fixture_id"],
                    "family": row["family"],
                    "perturbation_family": row["perturbation_family"],
                    "policy": policy,
                    "expected_answer": target["expected_answer"],
                    "expected_action": target["verifier_target"]["expected_action"],
                }
            )
    _write_jsonl(root, exp.EXP3085_ROWS_REL_PATH, panel_rows)
    _write_json(
        root,
        exp.EXP3085_REL_PATH,
        {
            "artifact": "experiment_3085_icalm_task_abstention_sota_panel_v2",
            "abstention_panel_v2_ready": True,
            "exact_ground_truth_count": 9,
            "panel_row_count": 18,
            "baseline_row_count": 9,
            "task_abstention_row_count": 9,
            "fixture_manifest_path": exp.FIXTURE_MANIFEST_REL_PATH.as_posix(),
            "abstention_precision": 0.0,
            "honest_verdict": "complete_below_gate: abstention_panel_v2_ready=true",
        },
    )


def _config(
    root: Path, *, minimum_live_eval_count: int = exp.MINIMUM_LIVE_EVAL_COUNT
) -> exp.AuditConfig:
    return exp.AuditConfig(
        repo_root=root,
        output_path=root / exp.OUTPUT_REL_PATH,
        stratified_manifest_path=root / exp.STRATIFIED_MANIFEST_REL_PATH,
        minimum_live_eval_count=minimum_live_eval_count,
        started_s=10.0,
        clock=lambda: 12.5,
    )


def test_req_verify_3097_spec_anchor_exists() -> None:
    """REQ-VERIFY-3097: OpenSpec declares the .289 protocol audit contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3097" in spec
    assert "SCENARIO-VERIFY-3097" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "minimum_live_eval_count" in spec
    assert "48 usable exact fixtures" in spec
    assert "honestly skip" in spec


def test_scenario_verify_3097_writes_stratified_manifest_and_tiny_panel_diagnosis(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3097: exact-data scale replaces the 9-case tiny panel."""
    rows = _fixture_rows()
    _write_required_sources(tmp_path, rows)
    _write_tiny_3085_panel(tmp_path, rows)

    artifact = exp.write_artifact(_config(tmp_path))
    manifest_rows = _read_jsonl(tmp_path / artifact["stratified_eval_manifest_path"])

    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["eval_protocol_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["usable_fixture_count"] == 72
    assert artifact["rejected_fixture_count"] == 0
    assert artifact["minimum_live_eval_count"] == 48
    assert artifact["fixture_family_counts"] == {
        "arithmetic_code_assertions": 24,
        "repairable_invalid_candidates": 24,
        "smt_constraints": 24,
    }
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["executes_models"] is False

    diagnosis = artifact["exp3085_tiny_panel_diagnosis"]
    assert diagnosis["exact_ground_truth_count_reported"] == 9
    assert diagnosis["unique_exact_fixtures_in_transcript"] == 9
    assert diagnosis["prompt_policy_rows_per_fixture"] == 2.0
    assert diagnosis["remaining_usable_fixture_count"] == 63
    assert diagnosis["selected_fixture_family_counts"] == {
        "arithmetic_code_assertions": 3,
        "repairable_invalid_candidates": 3,
        "smt_constraints": 3,
    }
    assert "3 fixtures per family" in diagnosis["why_only_9_exact_cases"]

    assert len(manifest_rows) == artifact["usable_fixture_count"]
    assert all(REQUIRED_MANIFEST_FIELDS <= row.keys() for row in manifest_rows)
    assert {row["expected_answer"] for row in manifest_rows} == {
        "INVALID",
        "REPAIRABLE",
        "SAT",
        "UNSAT",
        "VALID",
    }
    repair_rows = [
        row for row in manifest_rows if row["task_family"] == "repairable_invalid_candidates"
    ]
    assert len(repair_rows) == 24
    assert all(row["repair_target"]["repairable"] is True for row in repair_rows)
    assert all(
        row["verifier_target"]["expected_action"] in {"accept", "reject"} for row in manifest_rows
    )

    downstream = artifact["downstream_usage"]
    assert downstream["abstention_sota_panel_v3"]["minimum_unique_fixtures"] == 48
    assert downstream["formal_feedback_v2"]["minimum_unique_repair_fixtures"] == 18
    assert downstream["repair_gating_v2"]["minimum_unique_repair_fixtures"] == 24
    assert downstream["fr11_stress_v2"]["honest_skip_when"] == (
        "skip if fewer than 48 unique exact fixtures span accept, reject, and repair targets"
    )

    source_paths = {source["path"] for source in artifact["source_artifacts"]}
    assert exp.EXP3084_REL_PATH.as_posix() in source_paths
    assert exp.EXP3085_REL_PATH.as_posix() in source_paths
    assert exp.FIXTURE_MANIFEST_REL_PATH.as_posix() in source_paths
    exp.validate_artifact(artifact)


def test_req_verify_3097_rejects_malformed_and_duplicate_fixture_rows() -> None:
    """REQ-VERIFY-3097: malformed and duplicate fixtures are counted visibly."""
    all_rows = _fixture_rows()
    rows = all_rows[:6]
    duplicate = dict(rows[0])
    duplicate["fixture_id"] = rows[1]["fixture_id"]
    duplicate_prompt = dict(rows[3])
    duplicate_prompt["fixture_id"] = "unique-id-with-duplicate-prompt"
    duplicate_prompt["prompt_payload_sha256"] = rows[4]["prompt_payload_sha256"]
    malformed = dict(rows[2])
    del malformed["exact_label"]
    bad_authority = dict(all_rows[6])
    bad_authority["fixture_id"] = "bad-authority"
    wrong_status = "unsat" if bad_authority["exact_label"]["solver_status"] == "sat" else "sat"
    bad_authority["exact_label"] = dict(bad_authority["exact_label"]) | {
        "solver_status": wrong_status
    }

    audit = exp.audit_fixture_rows([*rows, duplicate, duplicate_prompt, malformed, bad_authority])

    assert len(audit.usable_rows) == 6
    assert len(audit.rejected_rows) == 4
    assert {row["reason"] for row in audit.rejected_rows} == {
        "duplicate_fixture_id",
        "duplicate_prompt_payload_sha256",
        "exact_authority_validation_failed",
        "missing_required_fields",
    }


def test_req_verify_3097_fails_closed_below_minimum_live_count(tmp_path: Path) -> None:
    """REQ-VERIFY-3097: tiny exact manifests produce skip guidance, not readiness."""
    rows = _fixture_rows()[:12]
    _write_required_sources(tmp_path, rows)
    _write_tiny_3085_panel(tmp_path, _fixture_rows())

    artifact = exp.write_artifact(_config(tmp_path))

    assert artifact["eval_protocol_ready"] is False
    assert artifact["usable_fixture_count"] == 12
    assert artifact["rejected_fixture_count"] == 0
    assert artifact["honest_verdict"].startswith(
        "blocked_exact_fixture_protocol_precondition_failed"
    )
    assert artifact["downstream_usage"]["abstention_sota_panel_v3"]["ready_for_headline"] is False
    assert artifact["downstream_usage"]["abstention_sota_panel_v3"]["honest_skip_when"] == (
        "skip if fewer than 48 unique exact fixtures are selected"
    )
    exp.validate_artifact(artifact)


def test_req_verify_3097_target_derivation_and_diagnosis_edges() -> None:
    """REQ-VERIFY-3097: target labels and tiny-panel diagnosis fail closed."""
    repair = next(
        row for row in _fixture_rows() if row["family"] == "repairable_invalid_candidates"
    )
    candidate_valid = dict(repair)
    candidate_valid["exact_label"] = dict(candidate_valid["exact_label"]) | {
        "candidate_valid": True,
        "repairable": False,
        "failure_kind": "none",
    }
    unrepairable = dict(repair)
    unrepairable["exact_label"] = dict(unrepairable["exact_label"]) | {
        "candidate_valid": False,
        "repairable": False,
        "failure_kind": "unrepairable",
    }

    assert exp.expected_targets(candidate_valid)["expected_answer"] == "VALID"
    assert exp.expected_targets(candidate_valid)["solver_label"] == "candidate_valid"
    assert exp.expected_targets(unrepairable)["expected_answer"] == "UNREPAIRABLE"
    assert exp.expected_targets(unrepairable)["solver_label"] == "unrepairable"

    unknown = dict(repair) | {"family": "unknown"}
    with pytest.raises(ValueError, match="unknown fixture family"):
        exp.expected_targets(unknown)

    inferred = exp.diagnose_exp3085_tiny_panel(
        exp3085={"exact_ground_truth_count": 5, "panel_row_count": 10},
        panel_rows=[],
        usable_fixture_count=72,
    )
    assert inferred["unique_exact_fixtures_in_transcript"] == 5
    assert "inferred from rows" in inferred["why_only_9_exact_cases"]

    unavailable = exp.diagnose_exp3085_tiny_panel(
        exp3085={}, panel_rows=[], usable_fixture_count=72
    )
    assert unavailable["unique_exact_fixtures_in_transcript"] == 0
    assert "unavailable" in unavailable["why_only_9_exact_cases"]


def test_req_verify_3097_validation_and_loader_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3097: bad artifacts, missing files, and malformed JSON fail closed."""
    rows = _fixture_rows()
    _write_required_sources(tmp_path, rows)
    _write_tiny_3085_panel(tmp_path, rows)
    artifact = exp.write_artifact(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: missing fields"})
    with pytest.raises(ValueError, match="no live model inference"):
        bad_substrate = dict(artifact["inference_substrate"]) | {"no_live_llm_inference": False}
        exp.validate_artifact(artifact | {"inference_substrate": bad_substrate})
    with pytest.raises(ValueError, match="must not execute models"):
        bad_substrate = dict(artifact["inference_substrate"]) | {"executes_models": True}
        exp.validate_artifact(artifact | {"inference_substrate": bad_substrate})
    with pytest.raises(ValueError, match="above minimum"):
        exp.validate_artifact(artifact | {"usable_fixture_count": 1})
    with pytest.raises(ValueError, match="must match rejected_fixtures"):
        exp.validate_artifact(artifact | {"rejected_fixture_count": 1})
    with pytest.raises(ValueError, match="success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready without prefix"})
    with pytest.raises(ValueError, match="precondition failure"):
        exp.validate_artifact(
            artifact | {"eval_protocol_ready": False, "honest_verdict": "blocked"}
        )

    assert exp._safe_load_json(tmp_path / "missing.json") == {}
    malformed_json = tmp_path / "bad.json"
    malformed_json.write_text("{not json", encoding="utf-8")
    assert exp._safe_load_json(malformed_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp._safe_load_json(list_json) == {}

    assert exp._safe_load_jsonl(tmp_path / "missing.jsonl") == []
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{"ok": true}\n[1]\n{bad\n', encoding="utf-8")
    assert exp._safe_load_jsonl(mixed_jsonl) == [
        {"ok": True},
        {"non_object_row": [1]},
        {"malformed_json_line": "{bad"},
    ]

    missing_sources = exp.source_artifacts(tmp_path / "no-such-root")
    assert all(source["exists"] is False and source["sha256"] is None for source in missing_sources)
    outside = exp._relative_path(tmp_path, Path("/definitely/outside"))
    assert outside == "/definitely/outside"
