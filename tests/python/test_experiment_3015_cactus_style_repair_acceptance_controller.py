"""Tests for Exp 3015 offline repair acceptance controller.

Spec: REQ-CODE-3015, SCENARIO-CODE-3015.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import repair_acceptance_controller as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
REQUIRED_FIELDS = {
    "acceptance_controller_ready",
    "controller_config_path",
    "n_candidates_evaluated",
    "false_accept_delta_offline",
    "syntax_failure_delta_offline",
    "schema_failure_delta_offline",
    "pass_at_1_delta_offline",
    "rejected_candidate_table_path",
    "llm_judge_used",
    "honest_verdict",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _candidate_row(
    item_id: str,
    *,
    schema_valid: bool = True,
    syntax_success: bool = True,
    entry_point_present: bool = True,
    original_passed: bool = True,
    metamorphic_passed_all: bool = True,
    false_accept: bool = False,
    intent_drift: bool = False,
    primary_root_cause: str = "passed",
) -> dict[str, Any]:
    failure_modes: list[str] = []
    if not syntax_success:
        failure_modes.append("syntax")
    if not schema_valid:
        failure_modes.append("schema")
    if false_accept:
        failure_modes.append("false_accept")
    if intent_drift:
        failure_modes.append("intent_drift")
    return {
        "row_type": "candidate",
        "item_id": item_id,
        "candidate_sha256": f"{item_id}-sha",
        "schema_valid": schema_valid,
        "syntax_success": syntax_success,
        "entry_point_present": entry_point_present,
        "original_passed": original_passed,
        "metamorphic_passed_all": metamorphic_passed_all,
        "metamorphic_variant_count": 2,
        "metamorphic_pass_count": 2 if metamorphic_passed_all else 0,
        "false_accept": false_accept,
        "intent_drift": intent_drift,
        "failure_modes": failure_modes,
        "failure_mode": failure_modes[0] if failure_modes else "passed",
        "primary_root_cause": primary_root_cause,
        "verifier_log_path": f"results/verifier_transcripts/experiment_3003/{item_id}.json",
        "candidate_patch_path": f"results/raw/experiment_3003/{item_id}.py",
    }


def _write_sources(
    tmp_path: Path,
    *,
    taxonomy_rows: list[dict[str, Any]],
    tautology_ready: bool = True,
) -> None:
    table_path = tmp_path / exp.TAXONOMY_TABLE_REL_PATH
    _write_jsonl(table_path, taxonomy_rows)
    _write_json(
        tmp_path / "results" / exp.EXP3014_FILENAME,
        {
            "artifact": "experiment_3014_repair_syntax_schema_failure_taxonomy_v1",
            "repair_failure_taxonomy_ready": True,
            "taxonomy_table_path": str(exp.TAXONOMY_TABLE_REL_PATH),
            "n_cached_candidates_audited": len(
                [row for row in taxonomy_rows if row.get("row_type") == "candidate"]
            ),
        },
    )
    _write_json(
        tmp_path / "results" / exp.EXP3002_FILENAME,
        {
            "artifact": "experiment_3002_metamorphic_repair_oracle_audit_v1",
            "metamorphic_oracle_ready": True,
            "false_accept_probe_ready": True,
            "tautology_probe_ready": tautology_ready,
            "rejected_variants": [
                {"reason": "tautological_oracle_rejected", "relation_type": "tautology_probe"}
            ]
            if tautology_ready
            else [],
            "metamorphic_manifest_path": str(exp.METAMORPHIC_MANIFEST_REL_PATH),
        },
    )
    _write_jsonl(
        tmp_path / exp.METAMORPHIC_MANIFEST_REL_PATH,
        [
            {
                "source_item_id": row["item_id"],
                "variant_id": f"{row['item_id']}__alpha",
                "relation_type": "alpha_renaming",
            }
            for row in taxonomy_rows
            if row.get("row_type") == "candidate"
        ],
    )
    candidates = []
    for row in taxonomy_rows:
        if row.get("row_type") != "candidate":
            continue
        verifier_path = tmp_path / str(row["verifier_log_path"])
        _write_json(
            verifier_path,
            {
                "item_id": row["item_id"],
                "passed": bool(row.get("original_passed") and row.get("metamorphic_passed_all")),
                "false_accept": bool(row.get("false_accept")),
            },
        )
        candidates.append(
            {
                "item_id": row["item_id"],
                "candidate_sha256": row["candidate_sha256"],
                "verifier_log_path": row["verifier_log_path"],
                "passed": bool(row.get("original_passed") and row.get("metamorphic_passed_all")),
            }
        )
    _write_json(
        tmp_path / "results" / exp.EXP3003_FILENAME,
        {
            "artifact": "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1",
            "candidate_evaluations": candidates,
        },
    )


def test_req_code_3015_spec_anchor_exists() -> None:
    """REQ-CODE-3015: the acceptance controller is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-3015" in spec
    assert "SCENARIO-CODE-3015" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "llm_judge_used=false" in spec


def test_scenario_code_3015_selects_transparent_rule_and_rejects_unsafe_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-3015: cached evidence yields auditable accept/reject decisions."""

    rows = [
        _candidate_row("clean-pass"),
        _candidate_row(
            "syntax-bad",
            syntax_success=False,
            entry_point_present=False,
            original_passed=False,
            metamorphic_passed_all=False,
            primary_root_cause="invalid patch shape",
        ),
        _candidate_row(
            "schema-bad",
            schema_valid=False,
            original_passed=False,
            metamorphic_passed_all=False,
            primary_root_cause="parser/schema mismatch",
        ),
        _candidate_row(
            "false-accept",
            original_passed=True,
            metamorphic_passed_all=False,
            false_accept=True,
            primary_root_cause="false accept",
        ),
        {
            "row_type": "validator",
            "item_id": "tautology-probe",
            "failure_mode": "tautology",
            "failure_modes": ["tautology"],
            "primary_root_cause": "tautology",
        },
    ]
    _write_sources(tmp_path, taxonomy_rows=rows)

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 12.5,
            tests_run=("focused-exp3015",),
        )
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    config_payload = json.loads((tmp_path / artifact["controller_config_path"]).read_text())
    rejected = _read_jsonl(tmp_path / artifact["rejected_candidate_table_path"])

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["acceptance_controller_ready"] is True
    assert artifact["n_candidates_evaluated"] == 4
    assert artifact["false_accept_delta_offline"] == pytest.approx(-0.25)
    assert artifact["syntax_failure_delta_offline"] == pytest.approx(-0.25)
    assert artifact["schema_failure_delta_offline"] == pytest.approx(-0.25)
    assert artifact["pass_at_1_delta_offline"] == pytest.approx(0.75)
    assert artifact["llm_judge_used"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 2.5
    assert artifact["baseline_policy_metrics"]["accept_all"]["pass_at_1"] == pytest.approx(0.25)
    assert artifact["selected_policy_metrics"]["accepted_item_ids"] == ["clean-pass"]
    assert artifact["search_evaluated_rule_count"] > 1
    assert config_payload["policy_type"] == "transparent_grid_rule"
    assert config_payload["selected_rule"]["require_schema_valid"] is True
    assert config_payload["selected_rule"]["require_syntax_success"] is True
    assert config_payload["selected_rule"]["require_tautology_probe_clean"] is True
    assert {row["item_id"] for row in rejected} == {"syntax-bad", "schema-bad", "false-accept"}
    assert any("syntax_success" in row["rejection_reasons"] for row in rejected)
    assert any("schema_valid" in row["rejection_reasons"] for row in rejected)
    assert any("false_accept" in row["rejection_reasons"] for row in rejected)


def test_req_code_3015_blocks_without_cached_taxonomy_rows(tmp_path: Path) -> None:
    """REQ-CODE-3015: missing cached evidence blocks honestly."""

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))

    assert artifact["acceptance_controller_ready"] is False
    assert artifact["n_candidates_evaluated"] == 0
    assert artifact["controller_config_path"] == ""
    assert artifact["rejected_candidate_table_path"] == ""
    assert artifact["honest_verdict"] == "blocked: exp3015 cached evidence unavailable"


def test_req_code_3015_tautology_probe_is_required_for_ready_controller(
    tmp_path: Path,
) -> None:
    """REQ-CODE-3015: tautology exposure keeps an otherwise clean controller blocked."""

    _write_sources(tmp_path, taxonomy_rows=[_candidate_row("clean-pass")], tautology_ready=False)

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))

    assert artifact["acceptance_controller_ready"] is False
    assert artifact["n_candidates_evaluated"] == 1
    assert artifact["tautology_probe_clean"] is False
    assert artifact["selected_policy_metrics"]["accepted_count"] == 0
    assert artifact["honest_verdict"] == "blocked: exp3015 selected controller is not usable"

    explicit_taxonomy = tmp_path / "explicit-taxonomy.jsonl"
    explicit_meta = tmp_path / "explicit-meta.jsonl"
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        taxonomy_table_path=explicit_taxonomy,
        metamorphic_manifest_path=explicit_meta,
    )
    assert config.resolved_taxonomy_table_path({}) == explicit_taxonomy
    assert config.resolved_metamorphic_manifest_path({}) == explicit_meta
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.json").is_absolute()
