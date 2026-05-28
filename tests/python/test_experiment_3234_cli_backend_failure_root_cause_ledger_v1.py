"""Tests for Exp 3234 prompt-injection KAN v4 CLI backend failure ledger.

Spec refs: REQ-REPORT-3234, SCENARIO-REPORT-3234.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cli_backend_failure_root_cause_ledger_3234 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "exp3222_artifact_exists",
    "exp3222_failure_count",
    "repeated_cli_error_signature",
    "monolith_rerun_allowed",
    "split_run_plan_ready",
    "required_next_artifacts",
    "model_spec_gap_found",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conductor_log(*, failures: int = 3, message: str = "backend exited before artifact") -> str:
    return "".join(
        (
            "| 2026-05-28 02:53 UTC | Prompt-Injection KAN Distillation v4 - "
            f"15k Corpus | FAIL | Codex CLI error: {message} |\n"
        )
        for _ in range(failures)
    )


def _capstone_payload(*, v4_outcome: str = mod.PRIOR_V4_OUTCOME) -> dict[str, Any]:
    return {
        "experiment_id": "exp3223",
        "task_id": "exp3223-capstone-v299-single-focus",
        "milestone": "2026.05.299",
        "capstone_v299_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 100,
        "v4_outcome": v4_outcome,
        "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
        "honest_verdict": (
            "complete: capstone_v299_ready=true; paper_ready=false; "
            "v4_outcome=blocked_missing_exp3222_result"
        ),
    }


def _roadmap_yaml() -> str:
    return "\n".join(
        [
            'milestone: "2026.05.300"',
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"',
            "tasks:",
            '  - id: "exp3234-cli-backend-failure-root-cause-ledger-v1"',
            '    deliverable: "results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json"',
            '  - id: "exp3239-prompt-injection-kan-v4-resource-manifest-v1"',
            '    deliverable: "results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json"',
            '  - id: "exp3240-prompt-injection-kan-teacher-label-shard-v1"',
            '    deliverable: "results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json"',
            '  - id: "exp3241-prompt-injection-kan-train-eval-shard-v1"',
            '    deliverable: "results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json"',
            "",
        ]
    )


def _template_text(*, include_cached_sota: bool = True) -> str:
    cached = "cached_sota_pair()" if include_cached_sota else "legacy_pair()"
    return f"""
# MODEL SELECTION - MANDATORY for any live-data or verify-repair experiment:
# Always try `{cached}` first.
from carnot.inference.sota_models import cached_sota_pair
MODEL_SPECS = [
    {{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}},
    {{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}},
    {{"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
]
# Record `models_used` in every artifact with the exact hub IDs.
"""


def _write_sources(tmp_path: Path, *, failures: int = 3, write_exp3222: bool = False) -> None:
    _write_text(tmp_path, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log(failures=failures))
    _write_json(tmp_path, mod.CAPSTONE_V299_REL_PATH, _capstone_payload())
    _write_json(tmp_path, mod.PROMPT_INJECTION_KAN_V2_REL_PATH, {"schema": "kan.v2"})
    _write_json(
        tmp_path,
        mod.PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH,
        {"row-a": {"teacher_label": 0}, "row-b": {"teacher_label": 1}},
    )
    _write_text(tmp_path, mod.EXPERIMENT_TEMPLATE_REL_PATH, _template_text())
    _write_text(tmp_path, mod.ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        tmp_path,
        mod.VNEXT_DOC_REL_PATH,
        "# Research Roadmap vNEXT\n\nexp3234 backend failure ledger\nexp3239 -> exp3241\n",
    )
    if write_exp3222:
        _write_json(
            tmp_path,
            mod.EXP3222_ARTIFACT_REL_PATH,
            {"experiment_id": "exp3222", "honest_verdict": "complete: unexpected v4 labels"},
        )


def test_req_report_3234_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3234: OpenSpec declares the ledger contract before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3234" in spec
    assert "SCENARIO-REPORT-3234" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3234_builds_split_plan_from_repeated_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3234: repeated exp3222 backend failure blocks monolith rerun."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}
    next_paths = {row["path"] for row in artifact["required_next_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3234"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.300"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["exp3222_artifact_exists"] is False
    assert artifact["exp3222_failure_count"] == 3
    assert artifact["repeated_cli_error_signature"] == "Codex CLI error: backend exited before artifact"
    assert artifact["repeated_cli_error_signature_count"] == 3
    assert artifact["monolith_rerun_allowed"] is False
    assert artifact["split_run_plan_ready"] is True
    assert artifact["model_spec_gap_found"] is True
    assert artifact["model_spec_gap_reason"] == "missing_exp3222_artifact_prevents_model_specs_audit"
    assert artifact["capstone_v4_outcome"] == mod.PRIOR_V4_OUTCOME
    assert artifact["experiment_template_model_spec_discipline"]["cached_sota_pair_mentioned"] is True
    assert artifact["experiment_template_model_spec_discipline"]["model_specs_mentioned"] is True
    assert artifact["experiment_template_model_spec_discipline"]["mandated_sota_model_count"] == 3
    assert artifact["prior_prompt_injection_evidence"]["kan_v2_present"] is True
    assert artifact["prior_prompt_injection_evidence"]["teacher_labels_v2_present"] is True
    assert artifact["prior_prompt_injection_evidence"]["teacher_labels_v2_count"] == 2
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_teacher_labeling"] is True
    assert artifact["no_new_kan_training"] is True
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_conductor_execution"] is True
    assert artifact["protected_files_untouched"] == {"scripts/research_conductor.py": True}
    assert artifact["principle_annotations"]["monolith_rerun_allowed"].startswith("False")
    assert len(artifact["exp3222_failure_lines"]) == 3
    assert "backend exited before artifact" in artifact["root_cause_summary"]
    assert "results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json" in next_paths
    assert "results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json" in next_paths
    assert "results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json" in next_paths
    assert "results/experiment_3241_prompt_injection_kan_garak_config_receipts_v1.json" in next_paths
    assert sources["conductor_log"]["sha256"] == _sha256(tmp_path / mod.CONDUCTOR_LOG_REL_PATH)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "v4 training" not in artifact["honest_verdict"].lower()
    assert "v4 labels exist" not in artifact["honest_verdict"].lower()


def test_req_report_3234_writer_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3234: writer persists JSON and malformed evidence is explicit."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["split_run_plan_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._as_mapping([]) == {}

    empty = mod.build_artifact(tmp_path / "empty", started_s=3.0, now_s=2.0)
    assert empty["split_run_plan_ready"] is False
    assert empty["exp3222_failure_count"] == 0
    assert empty["repeated_cli_error_signature"] == ""
    assert "expected three repeated exp3222 CLI failures" in empty["blocked_reasons"]


def test_req_report_3234_detects_unready_split_conditions(tmp_path: Path) -> None:
    """REQ-REPORT-3234: present exp3222 or non-repeated failures keep the split blocked."""

    _write_sources(tmp_path, failures=2, write_exp3222=True)
    _write_text(
        tmp_path,
        mod.EXPERIMENT_TEMPLATE_REL_PATH,
        _template_text(include_cached_sota=False).replace("cached_sota_pair", "legacy_pair"),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["exp3222_artifact_exists"] is True
    assert artifact["exp3222_failure_count"] == 2
    assert artifact["split_run_plan_ready"] is False
    assert artifact["model_spec_gap_found"] is True
    assert (
        mod._model_spec_gap_reason(
            exp3222_exists=True,
            artifact_has_model_specs=True,
            discipline_ready=False,
        )
        == "experiment_template_model_spec_discipline_incomplete"
    )
    assert (
        mod._model_spec_gap_reason(
            exp3222_exists=True,
            artifact_has_model_specs=True,
            discipline_ready=True,
        )
        == "none"
    )
    assert "exp3222 v4 artifact unexpectedly exists" in artifact["blocked_reasons"]
    assert "experiment_template.py does not document cached_sota_pair()" in artifact["blocked_reasons"]
