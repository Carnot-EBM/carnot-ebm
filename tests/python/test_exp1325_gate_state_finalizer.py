"""Tests for the Exp 1325 skeleton and downstream gate-state finalizer.

Spec: REQ-VERIFY-1338,
      SCENARIO-VERIFY-1338
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import exp1325_skeleton_and_gate_state_finalizer as finalizer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_source_files(root: Path, *, refs_include_replacement: bool = True) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json",
        {
            "status": "complete",
            "minimum_parseable_attempts_to_recover": 6,
            "artifact_metadata": {"parse_gate": 0.75},
            "parse_recovery_recommendation": (
                "Recover at least 6 parser failures before rerunning exp1325."
            ),
        },
    )
    _write_json(
        root / "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        {
            "experiment_id": "exp1325",
            "run_date": "20260505",
            "status": "in_progress",
            "models_used": [],
            "certificate_parse_rate": None,
            "certificate_truthfulness_rate": None,
            "honest_verdict": "in_progress",
        },
    )
    _write_json(
        root / "results/experiment_1327_beaver_lite_cactus_safe_prefix_gated_on_validator_pass.json",
        {
            "experiment": 1327,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp1326-satir-nsvif-semantic-validator-gated-on-parse-ge-075",
                    "artifact_field": "validator_execution_pass_rate",
                    "passed": False,
                }
            ],
        },
    )
    _write_json(
        root / "results/operational_retro_2026_04_103.json",
        {
            "bottlenecks_identified": [
                "Disk quota failures hit exp1324/exp1325 in the conductor log.",
                "Dependency pruning was incomplete after stale exp1325.",
            ],
            "improvements_suggested": [
                "Add dependency pruning for semantic validator, safe-prefix, DVI, and GRPO work."
            ],
        },
    )
    (root / "ops/conductor-log.md").write_text(
        "\n".join(
            [
                "| 2026-05-05 07:25 UTC | Triggered Certificate Extraction v5 | FAIL | Codex CLI error: [Errno 122] Disk quota exceeded |",
                "| 2026-05-05 07:31 UTC | SatIR/NSVIF Constraint Index + Semantic Validator | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1325-triggered-certificate-extraction-v5-runtime-fixed-dccd-gbnf) |",
                "| 2026-05-05 07:31 UTC | BEAVER-lite/Cactus Safe-Prefix Acceptance v5 | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp1326-satir-nsvif-semantic-validator-gated-on-parse-ge-075.validator_execution_pass_rate |",
                "| 2026-05-05 07:44 UTC | DVI Certificate-Tail Online Update v2 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1325-triggered-certificate-extraction-v5-runtime-fixed-dccd-gbnf) |",
                "| 2026-05-05 07:44 UTC | GRPO/VPRM v12 Micro-Audit | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1328-continuous-self-learning-memory-promotion-v2) |",
            ]
        ),
        encoding="utf-8",
    )
    reference_text = "No replacement branch has been specified."
    if refs_include_replacement:
        reference_text = (
            "The .104 branch should use trigger-before-constrain certificate tails, "
            "dynamic grammar dispatch, and semantic validation before reopening gates."
        )
    (root / "research-references.md").write_text(reference_text, encoding="utf-8")


def test_exp1338_classifies_stale_skeleton_and_carries_parse_floor_REQ_VERIFY_1338(tmp_path: Path) -> None:
    """REQ-VERIFY-1338-2/3/4/5/7: stale Exp 1325 carries Exp 1324 gates forward."""
    _write_source_files(tmp_path)

    artifact = finalizer.build_exp1325_gate_state_artifact(tmp_path, run_date="20260505")

    assert artifact["status"] == "complete"
    assert artifact["exp1325_terminal_classification"] == "stale_skeleton_environment_failure"
    assert artifact["minimum_parseable_attempts_to_recover"] == 6
    assert artifact["parse_gate_threshold"] == 0.75
    assert artifact["certificate_recovery_ready"] is True
    assert artifact["rerun_is_materially_different"] is True
    assert artifact["stale_artifacts_not_modified"] is True
    assert artifact["honest_verdict"] == "exp1325_stale_environment_failure_gates_closed_recovery_ready"
    assert set(finalizer.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    closed_ids = {task["task_id"] for task in artifact["downstream_tasks_to_keep_closed"]}
    assert {
        "exp1326-satir-nsvif-semantic-validator-gated-on-parse-ge-075",
        "exp1327-beaver-lite-cactus-safe-prefix-gated-on-validator-pass",
        "exp1329-dvi-certificate-tail-online-update-v2-gated-on-parse-and-nonforgetting",
        "exp1330-grpo-vprm-v12-micro-audit-gated-on-dvi-lossless",
    }.issubset(closed_ids)


def test_exp1338_rejects_blind_rerun_when_method_changes_are_missing_REQ_VERIFY_1338(tmp_path: Path) -> None:
    """REQ-VERIFY-1338-6: a replacement is not ready when it is only a blind rerun."""
    _write_source_files(tmp_path, refs_include_replacement=False)

    artifact = finalizer.build_exp1325_gate_state_artifact(tmp_path, run_date="20260505")

    assert artifact["rerun_is_materially_different"] is False
    assert artifact["certificate_recovery_ready"] is False
    assert artifact["required_method_changes"] == []
    assert artifact["honest_verdict"] == "exp1325_stale_environment_failure_gates_closed_recovery_not_ready"


def test_exp1338_writer_persists_in_progress_then_final_without_mutating_sources_REQ_VERIFY_1338(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1338-1/2/7: writer starts in-progress and reads .103 files only."""
    _write_source_files(tmp_path)
    source_paths = [
        tmp_path / "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json",
        tmp_path / "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        tmp_path / "results/experiment_1327_beaver_lite_cactus_safe_prefix_gated_on_validator_pass.json",
        tmp_path / "results/operational_retro_2026_04_103.json",
    ]
    before = {path: path.read_text(encoding="utf-8") for path in source_paths}
    statuses: list[str] = []
    output_path = (
        tmp_path / "results/experiment_1338_exp1325_skeleton_and_gate_state_finalizer.json"
    )

    artifact = finalizer.write_exp1325_gate_state_artifact(
        tmp_path,
        output_path=output_path,
        run_date="20260505",
        write_observer=lambda path, payload: statuses.append(payload["status"]),
    )

    assert statuses == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert {path: path.read_text(encoding="utf-8") for path in source_paths} == before


def test_exp1338_reports_completed_exp1325_without_environment_classification_REQ_VERIFY_1338(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1338-3: substantive metrics are not mislabeled as a stale skeleton."""
    _write_source_files(tmp_path)
    _write_json(
        tmp_path / "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        {
            "status": "complete",
            "certificate_parse_rate": 0.72,
            "certificate_truthfulness_rate": 0.68,
            "honest_verdict": "certificate_parse_gate_still_closed_runtime_fixed_v5",
        },
    )

    artifact = finalizer.build_exp1325_gate_state_artifact(tmp_path, run_date="20260505")

    assert artifact["exp1325_terminal_classification"] == "substantive_exp1325_gate_closed"
    assert artifact["stale_artifacts_not_modified"] is True


def test_exp1338_run_experiment_and_helper_edges_REQ_VERIFY_1338(tmp_path: Path) -> None:
    """REQ-VERIFY-1338-1/3/4/6: direct runner and helper edge cases stay explicit."""
    _write_source_files(tmp_path)
    output_path = tmp_path / "results/experiment_1338.json"
    statuses: list[str] = []

    artifact = finalizer.run_experiment(
        project_root=tmp_path,
        output_path=output_path,
        run_date="20260505",
        write_observer=lambda path, payload: statuses.append(payload["status"]),
    )

    assert statuses == ["in_progress", "complete"]
    assert artifact["status"] == "complete"
    assert finalizer.classify_exp1325_terminal_state({"certificate_parse_rate": False}) == (
        "stale_skeleton_environment_failure"
    )

    direct = finalizer.build_gate_state_artifact(
        exp1324_artifact={
            "status": "complete",
            "parse_gate": 0.75,
            "minimum_parseable_attempts_to_recover": 6,
        },
        exp1325_artifact={"status": "in_progress", "certificate_parse_rate": ""},
        exp1327_artifact={"status": "blocked"},
        retro_artifact={"status": "success"},
        conductor_log="Codex CLI error: [Errno 122] Disk quota exceeded\nGATE_BLOCK\n",
        run_date="20260505",
        project_root=tmp_path,
        proposed_method_changes=None,
    )

    assert direct["parse_gate_threshold"] == 0.75
    assert direct["rerun_is_materially_different"] is True
