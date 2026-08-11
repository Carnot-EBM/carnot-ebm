"""Tests for Exp6316 model-local probe integrity audit.

Spec refs: REQ-KONA-6316, SCENARIO-KONA-6316-REPLAY,
SCENARIO-KONA-6316-MUTATIONS, SCENARIO-KONA-6316-DISAGGREGATED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_6316_model_local_probe_integrity_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/phase3-kona/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6316_model_local_probe_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6316_model_local_probe_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6316_model_local_probe_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6316_model_local_probe_integrity_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6316_model_local_probe_integrity_audit.py"
)
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6316_model_local_probe_integrity_audit "
    "--date 20260811"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6316_model_local_probe_integrity_audit.json"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    RUN_COMMAND,
    ADVERSARIAL_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(mod.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _exact_fixture(root: Path) -> None:
    row = {
        "schema": "test.exp6313.row",
        "pair_id": "pair-0",
        "weakness_family": "path_traversal",
        "source_family": "filesystem",
        "template_id": "tmpl-0",
        "perturbation_id": "pert-0",
        "mutation_group_id": "group-0",
        "split": "held",
        "declared_safety_property": "Paths stay under root.",
        "vulnerable": {"code": "def handle(x):\n    return x\n", "char_length": 27},
        "fixed": {"code": "def handle(x):\n    return x\n", "char_length": 27},
    }
    row["pair_hash"] = mod.exact_row_hash(row)
    sidecar = {
        "schema": "test.exp6313.sidecar",
        "pair_id": row["pair_id"],
        "pair_hash": row["pair_hash"],
        "compile": {"vulnerable": {"ok": True}, "fixed": {"ok": True}},
        "executable_property": {
            "vulnerable_label": "vulnerable",
            "fixed_label": "fixed",
            "vulnerable_property_passed": False,
            "fixed_property_passed": True,
        },
        "ast_or_constraint": {
            "vulnerable_label": "vulnerable",
            "fixed_label": "fixed",
            "vulnerable_property_passed": False,
            "fixed_property_passed": True,
        },
        "targeted_mutation": {"mutation_detected": True},
        "label_receipt": {"validators_agree": True, "llm_labeler_used": False},
    }
    sidecar["sidecar_hash"] = mod.exact_sidecar_hash(sidecar)
    corpus_path = root / mod.EXP6313_CORPUS_RELATIVE_PATH
    sidecar_path = root / mod.EXP6313_SIDECAR_RELATIVE_PATH
    controls_path = root / mod.EXP6313_CONTROL_RELATIVE_PATH
    splits_path = root / mod.EXP6313_SPLIT_RELATIVE_PATH
    _write_jsonl(corpus_path, [row])
    _write_jsonl(sidecar_path, [sidecar])
    _write_json(
        controls_path,
        {
            "aa_duplicates": [{"control_id": "aa-0"}],
            "label_permutations": [{"control_id": "perm-0"}],
            "pair_swaps": [{"control_id": "swap-0"}],
            "evaluator_swaps": [{"control_id": "eval-0"}],
            "held_labels_exposed_to_surface_selection": False,
        },
    )
    _write_json(splits_path, {"held": ["group-0"], "train": [], "validation": []})
    _write_json(
        root / mod.EXP6313_ARTIFACT_RELATIVE_PATH,
        {
            "status": "complete_ready",
            "honest_verdict": "ready: exact fixture",
            "corpus_path_and_hash": {
                "path": str(corpus_path),
                "row_count": 1,
                "sha256": mod.sha256_file(corpus_path),
            },
            "sidecar_path_and_hash": {
                "path": str(sidecar_path),
                "row_count": 1,
                "sha256": mod.sha256_file(sidecar_path),
            },
            "control_manifest_path_and_hash": {
                "path": str(controls_path),
                "sha256": mod.sha256_file(controls_path),
            },
            "split_manifest_path_and_hash": {
                "path": str(splits_path),
                "sha256": mod.sha256_file(splits_path),
            },
            "positive_and_negative_control_results": {
                "all_controls_passed": True,
                "label_permutation_negative_control_caught": True,
                "pair_swap_controls_passed": True,
                "evaluator_swap_controls_passed": True,
                "aa_duplicates_passed": True,
                "held_labels_hidden_from_surface_selection": True,
            },
            "duplicate_and_overlap_checks": {
                "all_checks_passed": True,
                "split_leakage_count": 0,
            },
            "exact_code_safety_fixture_ready_score": 1.0,
        },
    )


def _cell(model_id: str, *, passed: bool) -> dict[str, Any]:
    return {
        "cell_id": f"fold_0|{model_id}|candidate_correctness|finite_domain_csp|low|symbol",
        "row_count": 4,
        "train_anchor_row_count": 4,
        "adequately_powered": True,
        "direction_positive_rate": 1.0 if passed else 0.0,
        "mean_anchor_cosine": 0.4 if passed else -0.4,
        "cell_passed": passed,
    }


def _checkpoint(path: Path, model_id: str, *, metadata_model_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"model_id": metadata_model_id or model_id, "shared_dimension": 2}
    np.savez(
        path,
        encoder_weight=np.eye(2, dtype=np.float32),
        encoder_bias=np.zeros(2, dtype=np.float32),
        metadata_json=np.asarray(mod.canonical_json(metadata)),
    )


def _model_local_artifacts(root: Path, *, swapped_checkpoint: bool = False) -> None:
    model_specs = [{"hf_id": model_id, "family": model_id.rsplit("/", 1)[-1]} for model_id in mod.MANDATED_MODEL_HF_IDS]
    cells = [_cell(mod.MANDATED_MODEL_HF_IDS[0], passed=False)]
    cells.extend(_cell(model_id, passed=True) for model_id in mod.MANDATED_MODEL_HF_IDS[1:])
    for model_id in mod.MANDATED_MODEL_HF_IDS:
        metadata_model_id = "swapped/model" if swapped_checkpoint and model_id == mod.MANDATED_MODEL_HF_IDS[0] else None
        _checkpoint(
            root
            / mod.CHECKPOINT_DIR_RELATIVE_PATH
            / "fold_0"
            / f"{mod.short_model_id(model_id)}.npz",
            model_id,
            metadata_model_id=metadata_model_id,
        )
    _write_json(
        root / mod.FOLD_MANIFEST_RELATIVE_PATH,
        {"schema": "test.fold_manifest", "folds": [{"fold_id": "fold_0"}]},
    )
    failed = [row["cell_id"] for row in cells if row["cell_passed"] is not True]
    _write_json(
        root / mod.EXP6301_ARTIFACT_RELATIVE_PATH,
        {
            "status": "flagged",
            "honest_verdict": "flagged: synthetic failed cell",
            "MODEL_SPECS": model_specs,
            "models_covered": {"models": list(mod.MANDATED_MODEL_HF_IDS)},
            "row_and_checkpoint_reconstruction_receipts": {
                "row_count": 6,
                "source_row_count": 3,
                "shared_item_count": 6,
                "row_hash_mismatch_count": 0,
                "missing_model_item_count": 0,
                "checkpoint_missing_count": 0,
                "checkpoint_metadata_mismatch_count": 0,
                "all_rows_and_checkpoints_reconstructed": True,
            },
            "fold_leakage_checks": {"all_fold_leakage_checks_passed": True},
            "claim_flip_sensitivity": {
                "cell_decisions": cells,
                "held_family_decisions": [],
                "failed_cell_count": len(failed),
                "failed_cells": failed,
                "all_cells_passed": False,
            },
            "pair_swap_controls": {
                "cell_decisions": [{**row, "cell_passed": True} for row in cells],
                "failed_cell_count": 0,
                "all_pair_swap_controls_passed": True,
            },
            "label_permutation_controls": {
                "all_label_permutation_controls_passed": True,
                "label_permutation_collapsed": True,
                "evaluated_pair_count": 3,
            },
            "model_identity_controls": {
                "all_identity_controls_passed": True,
                "matched_semantic_alignment_preserved": True,
                "identity_failed_folds": [],
                "semantic_alignment_failed_pairs": [],
            },
            "norm_length_token_and_truncation_controls": {
                "all_norm_length_token_truncation_controls_passed": False,
                "token_count_pair_mismatch_count": 1,
                "truncation_count": 1,
                "norm_shortcut_cells": [failed[0]],
                "token_count_shortcut_cells": [failed[0]],
                "cell_decisions": cells,
            },
            "duplicate_and_no_information_controls": {
                "all_duplicate_and_no_information_controls_passed": True,
                "no_information_controls": {"no_information_fails_positive_control": True},
            },
            "evaluator_swap_receipts": {
                "all_evaluator_swaps_passed": False,
                "agreement_rate": 0.5,
                "disagreement_cells": failed,
            },
            "disaggregated_cell_decisions": {
                "failed_cells": failed,
                "failed_cell_count": len(failed),
            },
            "failed_cells": failed,
            "surviving_shortcuts": [
                "claim_flip_direction_cell_failure",
                "norm_only_shortcut",
                "evaluator_swap_disagreement",
            ],
            "false_pass_injection_results": {"all_false_pass_injections_blocked": True},
            "activation_bus_integrity_ready_score": 0.0,
            "source_mutation_count": 0,
        },
    )
    _write_json(
        root / mod.EXP5853_ARTIFACT_RELATIVE_PATH,
        {
            "status": "flagged",
            "honest_verdict": "flagged: prior shortcut audit",
            "surviving_shortcuts": ["raw_model_dimension_identity_shortcut"],
        },
    )
    _write_json(
        root / mod.EXP6312_ARTIFACT_RELATIVE_PATH,
        {
            "status": "complete_null",
            "honest_verdict": "complete_null: underpowered norm controls",
            "MODEL_SPECS": model_specs,
            "models_used": list(mod.MANDATED_MODEL_HF_IDS),
            "underpowered_or_missing_cells": [f"{mod.MANDATED_MODEL_HF_IDS[0]}:norm_control"],
            "aa_noise_results_by_model": {
                model_id: {"passed": True, "aa_pair_count": 2}
                for model_id in mod.MANDATED_MODEL_HF_IDS
            },
            "no_shared_adapter_receipt": {"pooled_rescue_allowed": False},
            "source_model_weight_mutation_count": 0,
        },
    )
    _write_json(
        root / mod.EXP6314_ARTIFACT_RELATIVE_PATH,
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "synthetic gate failed",
            "gates_evaluated": [{"passed": False}],
        },
    )


def _fixture(root: Path, *, swapped_checkpoint: bool = False) -> None:
    _exact_fixture(root)
    _model_local_artifacts(root, swapped_checkpoint=swapped_checkpoint)


def _run(root: Path) -> dict[str, Any]:
    return mod.run(
        root=root,
        date="20260811",
        result_path=root / mod.RESULT_RELATIVE_PATH,
        mutation_manifest_path=root / mod.MUTATION_MANIFEST_RELATIVE_PATH,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_kona_6316_spec_declares_required_artifact_contract() -> None:
    """REQ-KONA-6316: OpenSpec anchors fields and replay scenarios."""

    section = SPEC.read_text(encoding="utf-8").split("### REQ-KONA-6316", 1)[1]

    for marker in (
        "SCENARIO-KONA-6316-REPLAY",
        "SCENARIO-KONA-6316-MUTATIONS",
        "SCENARIO-KONA-6316-DISAGGREGATED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "`pooled_rescue_attempt_count`",
        "`source_model_weight_mutation_count`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kona_6316_replays_declared_paths_rows_and_checkpoints(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6316-REPLAY: blocked and missing inputs stay visible."""

    _fixture(tmp_path)
    artifact = _run(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "flagged"
    assert artifact["honest_verdict"].startswith("flagged:")
    assert artifact["model_local_probe_integrity_ready_score"] == 0.0
    assert artifact["pooled_rescue_attempt_count"] == 0
    assert type(artifact["pooled_rescue_attempt_count"]) is int
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    classes = artifact["audited_paths_hashes_and_terminal_classes"]
    assert classes[mod.EXP6314_ARTIFACT_RELATIVE_PATH.as_posix()]["terminal_class"][
        "classification"
    ] == "skipped"
    assert classes[mod.EXP6315_ARTIFACT_RELATIVE_PATH.as_posix()]["terminal_class"][
        "classification"
    ] == "missing"
    exact = artifact["independent_row_checkpoint_and_decision_reconstruction"][
        "exact_fixture_replay"
    ]
    assert exact["row_hash_mismatch_count"] == 0
    assert exact["sidecar_hash_mismatch_count"] == 0
    assert exact["validators_agree"] is True
    checkpoint = artifact["checkpoint_reload_and_score_identity_by_model"]
    assert checkpoint["checkpoint_metadata_mismatch_count"] == 0
    assert checkpoint["score_identity_recomputed_from_cells"] is True
    mutation_receipt = artifact["audit_mutation_manifest_path_and_hash"]
    assert mutation_receipt["sha256"] == mod.sha256_file(mutation_receipt["path"])


def test_scenario_kona_6316_mutations_and_false_ready_attempts_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6316-MUTATIONS: planted failures block readiness."""

    _fixture(tmp_path)
    artifact = _run(tmp_path)

    assert artifact["claim_flip_pair_swap_label_permutation_and_evaluator_swap_results"][
        "all_planted_failures_detected"
    ] is True
    assert artifact["aa_noise_norm_length_final_pool_and_prompt_substitution_results"][
        "final_pool_control"
    ]["aggregate_rescue_rejected"] is True
    assert artifact["aa_noise_norm_length_final_pool_and_prompt_substitution_results"][
        "prompt_verdict_receipt_substitution"
    ]["receipt_overrode_terminal_class"] is False
    assert artifact["truncation_duplicate_missing_checkpoint_swap_and_model_identity_results"][
        "checkpoint_swap_control"
    ]["planted_checkpoint_swap_detected"] is True
    assert artifact["random_label_and_random_pair_controls"][
        "random_label_training_refit_performed"
    ] is False

    wrapped = deepcopy(artifact)
    wrapped["pooled_rescue_attempt_count"] = {"value": 0}
    wrapped["reproducibility_checksum"] = mod.reproducibility_checksum(wrapped)
    assert "pooled_rescue_attempt_count must be bare integer 0" in mod.validate_artifact(
        wrapped
    )

    forced = deepcopy(artifact)
    forced["model_local_probe_integrity_ready_score"] = 1.0
    forced["reproducibility_checksum"] = mod.reproducibility_checksum(forced)
    assert "model_local_probe_integrity_ready_score mismatch" in mod.validate_artifact(
        forced
    )

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("status")
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_principle
    )
    assert "missing field_principles entry: status" in mod.validate_artifact(
        missing_principle
    )

    missing_status = deepcopy(artifact)
    del missing_status["status"]
    missing_status["reproducibility_checksum"] = mod.reproducibility_checksum(missing_status)
    assert "missing required field: status" in mod.validate_artifact(missing_status)

    missing_provenance = deepcopy(artifact)
    missing_provenance["field_provenance"].pop("status")
    missing_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_provenance
    )
    assert "missing field_provenance entry: status" in mod.validate_artifact(
        missing_provenance
    )

    bad_models = deepcopy(artifact)
    bad_models["MODEL_SPECS"] = []
    bad_models["reproducibility_checksum"] = mod.reproducibility_checksum(bad_models)
    assert "MODEL_SPECS must preserve mandated GGUF families" in mod.validate_artifact(
        bad_models
    )

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["model_local_probe_integrity_ready_score"] = (
        mod.model_local_probe_integrity_ready_score(bad_substrate)
    )
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_substrate)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["model_local_probe_integrity_ready_score"] = (
        mod.model_local_probe_integrity_ready_score(bad_oracle)
    )
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    assert "verifier_is_oracle mismatch" in mod.validate_artifact(bad_oracle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "not terminal"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad_verdict)

    bad_exit = deepcopy(artifact)
    bad_exit["test_exit_codes"] = {command: 1 for command in artifact["test_commands"]}
    bad_exit["model_local_probe_integrity_ready_score"] = mod.model_local_probe_integrity_ready_score(
        bad_exit
    )
    bad_exit["reproducibility_checksum"] = mod.reproducibility_checksum(bad_exit)
    assert bad_exit["model_local_probe_integrity_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    missing_checksum = deepcopy(artifact)
    missing_checksum["reproducibility_checksum"] = ""
    assert "reproducibility_checksum missing" in mod.validate_artifact(missing_checksum)


def test_scenario_kona_6316_disaggregates_failures_and_underpowered_cells(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6316-DISAGGREGATED: no cell is hidden by pooled metrics."""

    _fixture(tmp_path, swapped_checkpoint=True)
    artifact = _run(tmp_path)

    energy = artifact["energy_direction_results_by_model_and_fold"]
    failed_model = mod.MANDATED_MODEL_HF_IDS[0]
    assert energy["by_model_and_fold"][failed_model]["fold_0"]["failed_cell_count"] == 1
    assert energy["by_model_and_fold"][failed_model]["fold_0"]["passed"] is False
    interval = artifact["disaggregated_metrics_intervals_and_sample_sizes"][
        "by_model_and_fold"
    ][failed_model]["fold_0"]["wilson_pass_rate_95ci"]
    assert interval[0] <= interval[1]
    failed = artifact["failed_harm_underpowered_missing_and_flagged_cells"]
    assert any(row["kind"] == "underpowered_cell" for row in failed)
    assert any(row["kind"] == "missing_declared_input" for row in failed)
    assert any(row["kind"] == "failed_model_fold_cell" for row in failed)
    assert artifact["checkpoint_reload_and_score_identity_by_model"][
        "checkpoint_metadata_mismatch_count"
    ] == 1
    assert artifact["model_local_probe_integrity_ready_score"] == 0.0


def test_scenario_kona_6316_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-KONA-6316-MUTATIONS: parser and status guards are fail-closed."""

    assert mod.DEFAULT_TEST_EXIT_CODES[mod.FULL_TEST_COMMAND] == 2

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)

    assert mod._read_json_or_empty(tmp_path / "missing.json") == {}
    assert mod._wilson_interval(0, 0) == [0.0, 1.0]
    fallback = mod._models_from_payloads({}, {})
    assert fallback[0]["cached_replay_only"] is True

    ready_status, ready_verdict = mod._status_and_verdict(
        {"model_local_probe_integrity_ready_score": 1.0}
    )
    assert ready_status == "ready"
    assert ready_verdict.startswith("ready:")

    blocked_status, blocked_verdict = mod._status_and_verdict(
        {
            "model_local_probe_integrity_ready_score": 0.0,
            "preconditions_checked": {"preconditions_ready": False, "blocked_reasons": []},
        }
    )
    assert blocked_status == "blocked"
    assert blocked_verdict == "blocked: preconditions_failed"
