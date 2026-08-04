"""Tests for Exp6102 sequential SOTA atom corpus VRAM recovery.

Spec refs: REQ-INFER-SOTA-6102,
SCENARIO-INFER-SOTA-6102-BLOCKED-VRAM,
SCENARIO-INFER-SOTA-6102-RESUME,
SCENARIO-INFER-SOTA-6102-CORPUS.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6102_sota_atom_corpus_vram_recovery as mod
from test_experiment_5964_sota_atom_compatibility_corpus import (
    FakeCompatibilityBackend,
    _fixture,
    _model_specs,
    _ready_preconditions,
)


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-m pytest tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6102_sota_atom_corpus_vram_recovery.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6102_sota_atom_corpus_vram_recovery.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md research-references.md"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions_with_vram(tmp_path: Path, free_mb: int) -> dict[str, Any]:
    preconditions = _ready_preconditions(tmp_path)
    preconditions["gpu"]["devices"] = [
        {
            "index": 0,
            "name": "RTX 3090",
            "memory_total_mb": 24576,
            "memory_free_mb": free_mb,
            "memory_used_mb": 24576 - free_mb,
            "temperature_c": 52,
        },
        {
            "index": 1,
            "name": "RTX 3090",
            "memory_total_mb": 24576,
            "memory_free_mb": free_mb,
            "memory_used_mb": 24576 - free_mb,
            "temperature_c": 53,
        },
    ]
    preconditions["runtime"] = {
        "cuda_build": {"available": True, "stdout": "CUDA 12 fixture"},
        "task_owned_pid_leases": {"current_pid": 1234, "child_pids": []},
    }
    return preconditions


def _prior_vram_block(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_5964_sota_atom_compatibility_corpus.json"
    path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "honest_verdict": "blocked: insufficient_free_vram",
                "preconditions_checked": {"blocked_reasons": ["insufficient_free_vram"]},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_infer_sota_6102_spec_declares_sequential_recovery() -> None:
    """REQ-INFER-SOTA-6102: OpenSpec declares fields, principles, and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-INFER-SOTA-6102") :]

    for marker in (
        "REQ-INFER-SOTA-6102",
        "SCENARIO-INFER-SOTA-6102-BLOCKED-VRAM",
        "SCENARIO-INFER-SOTA-6102-RESUME",
        "SCENARIO-INFER-SOTA-6102-CORPUS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "retirement_triggered=true",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infer_sota_6102_blocked_vram_stops_before_backend(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-6102-BLOCKED-VRAM: first family cannot fit."""

    fixture_artifact_path, context_path, pair_path = _fixture(tmp_path)
    calls: list[str] = []

    def forbidden_backend(
        model_spec: Mapping[str, Any], config: Mapping[str, Any]
    ) -> FakeCompatibilityBackend:
        calls.append(str(model_spec["hf_id"]))
        raise AssertionError("backend must not load when first family cannot fit")

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=12_000),
        embedding_backend_factory=forbidden_backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["retirement_triggered"] is True
    assert artifact["all_family_corpus_ready_score"] == pytest.approx(0.0)
    assert artifact["runtime_cuda_vram_thermal_and_pid_lease_receipts"][
        "unrelated_processes_killed"
    ] == []
    assert artifact["per_family_phase_start_end_and_release_receipts"]["families"][
        mod.MANDATED_MODEL_HF_IDS[0]
    ]["phase_status"] == "blocked_before_load"
    assert mod.validate_artifact(artifact) is True
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert mod.read_model_row_file(tmp_path / mod.model_row_relative_path(hf_id).name) == []


def test_scenario_infer_sota_6102_resume_accepts_exact_and_quarantines_stale(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-6102-RESUME: exact resume and stale quarantine."""

    fixture_artifact_path, context_path, pair_path = _fixture(tmp_path)
    first = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert first["status"] == "complete_ready"

    stale_hf_id = mod.MANDATED_MODEL_HF_IDS[1]
    stale_path = tmp_path / mod.model_row_relative_path(stale_hf_id).name
    stale_path.write_text(stale_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    calls: list[str] = []

    def resume_backend(
        model_spec: Mapping[str, Any], config: Mapping[str, Any]
    ) -> FakeCompatibilityBackend:
        calls.append(str(model_spec["hf_id"]))
        if model_spec["hf_id"] != stale_hf_id:
            raise AssertionError("accepted resume shard should not be re-extracted")
        return FakeCompatibilityBackend(model_spec, config)

    resumed = mod.run(
        result_path=tmp_path / "resumed.json",
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=resume_backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    matrix = resumed["resume_accept_reject_matrix"]["families"]
    assert calls == [stale_hf_id]
    assert matrix[mod.MANDATED_MODEL_HF_IDS[0]]["decision"] == "accepted"
    assert matrix[stale_hf_id]["decision"] == "rejected"
    assert resumed["stale_partial_quarantine_receipt"]["quarantined_count"] == 1
    assert Path(resumed["stale_partial_quarantine_receipt"]["records"][0]["quarantine_path"]).exists()
    assert resumed["status"] == "complete_ready"
    assert mod.validate_artifact(resumed) is True


def test_scenario_infer_sota_6102_corpus_fake_backend_all_families_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-6102-CORPUS: sequential all-family corpus readiness."""

    fixture_artifact_path, context_path, pair_path = _fixture(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["all_family_corpus_ready_score"] == pytest.approx(1.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["python_z3_label_replay"]["all_python_z3_agree"] is True
    assert artifact["python_z3_label_replay"]["semantic_group_count"] >= 4
    assert artifact["shortcut_control_coverage"]["all_required_controls_declared"] is True
    assert artifact["shortcut_control_coverage"]["ranker_trained"] is False
    assert artifact["raw_vs_standardized_feature_schema"]["standardization_train_fold_only"] is True
    assert artifact["row_paths_hashes_and_prefix_chain"]["all_models_have_rows"] is True
    assert artifact["per_family_row_split_and_class_counts"]["all_declared_held_splits_present"] is True
    assert artifact["per_family_row_split_and_class_counts"]["semantic_group_seed_count"] >= 5
    assert mod.validate_artifact(artifact) is True


def test_scenario_infer_sota_6102_later_family_vram_block_is_checkpointed(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-6102: later-family fit failure preserves partial rows."""

    fixture_artifact_path, context_path, pair_path = _fixture(tmp_path)
    specs = _model_specs(tmp_path)
    specs[1]["min_vram_gb"] = 40
    artifact = mod.run(
        result_path=tmp_path / "later-block.json",
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=specs,
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    first, second, third = mod.MANDATED_MODEL_HF_IDS
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["row_paths_hashes_and_prefix_chain"]["row_counts_by_model"][first] > 0
    assert artifact["row_paths_hashes_and_prefix_chain"]["row_counts_by_model"][second] == 0
    assert artifact["per_family_phase_start_end_and_release_receipts"]["families"][first][
        "phase_status"
    ] == "extracted"
    assert artifact["per_family_phase_start_end_and_release_receipts"]["families"][second][
        "phase_status"
    ] == "blocked_before_load"
    assert artifact["per_family_phase_start_end_and_release_receipts"]["families"][third][
        "phase_status"
    ] == "not_attempted_after_block"
    assert mod.validate_artifact(artifact) is True


def test_req_infer_sota_6102_defensive_resume_and_validation_edges(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-6102: stale provenance and malformed artifacts fail closed."""

    fixture_artifact_path, context_path, pair_path = _fixture(tmp_path)
    bad_prior = tmp_path / "bad-prior.json"
    bad_prior.write_text("[]", encoding="utf-8")
    immutable = mod._immutable_hashes(
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=bad_prior,
        row_dir=tmp_path,
    )
    assert immutable["prior_exp5964_artifact"]["status"] is None

    no_gpu = mod._best_fit_device(
        hf_id=mod.MANDATED_MODEL_HF_IDS[0],
        preconditions={"gpu": {"devices": []}},
        estimates={mod.MANDATED_MODEL_HF_IDS[0]: {"estimated_required_mb": 1}},
    )
    assert no_gpu["reason"] == "no_gpu_devices"

    artifact = mod.run(
        result_path=tmp_path / "complete.json",
        row_dir=tmp_path,
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    refreshed = mod.refresh_artifact_test_exit_codes(
        artifact_path=tmp_path / "complete.json",
        test_exit_codes=TEST_EXIT_CODES,
    )
    assert refreshed["reproducibility_checksum"] == mod.reproducibility_checksum(refreshed)

    hf_id = mod.MANDATED_MODEL_HF_IDS[0]
    spec = mod.normalize_model_specs(_model_specs(tmp_path))[0]
    rows = mod.read_model_row_file(tmp_path / mod.model_row_relative_path(hf_id).name)
    pair_rows = mod._read_jsonl(pair_path)
    context_rows = mod._read_jsonl(context_path)
    expected = mod._expected_prompt_hashes(context_rows=context_rows, pair_rows=pair_rows)
    empty_ok, empty_reason, _receipt = mod._verify_resume_candidate(
        hf_id=hf_id,
        rows=[],
        path=tmp_path / "missing.rows.jsonl",
        model_spec=spec,
        pair_rows=pair_rows,
        expected_prompt_hashes=expected,
    )
    assert empty_ok is False
    assert empty_reason == "empty_or_missing_row_shard"

    row_path = tmp_path / mod.model_row_relative_path(hf_id).name
    phase_path = tmp_path / mod.phase_end_relative_path(hf_id).name
    saved_phase = phase_path.read_text(encoding="utf-8")
    phase_path.unlink()
    ok, reason, _receipt = mod._verify_resume_candidate(
        hf_id=hf_id,
        rows=rows,
        path=row_path,
        model_spec=spec,
        pair_rows=pair_rows,
        expected_prompt_hashes=expected,
    )
    assert ok is False
    assert reason == "missing_phase_end_receipt"
    phase_path.write_text("[]", encoding="utf-8")
    ok, reason, _receipt = mod._verify_resume_candidate(
        hf_id=hf_id,
        rows=rows,
        path=row_path,
        model_spec=spec,
        pair_rows=pair_rows,
        expected_prompt_hashes=expected,
    )
    assert ok is False
    assert reason.startswith("phase_end_unreadable:")
    phase_path.write_text(saved_phase, encoding="utf-8")
    bad_expected = dict(expected)
    bad_expected[(0, "context_alone")] = "sha256:bad"
    ok, reason, receipt = mod._verify_resume_candidate(
        hf_id=hf_id,
        rows=rows,
        path=row_path,
        model_spec=spec,
        pair_rows=pair_rows,
        expected_prompt_hashes=bad_expected,
    )
    assert ok is False
    assert "prompt_hash" in reason
    assert receipt["row_count"] == len(rows)

    bad_test_codes = dict(TEST_EXIT_CODES)
    bad_test_codes[TEST_COMMAND] = 1
    downgraded = mod.run(
        result_path=tmp_path / "downgraded.json",
        row_dir=tmp_path / "downgraded",
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=_prior_vram_block(tmp_path),
        model_specs=_model_specs(tmp_path / "downgraded-models"),
        preconditions_checked=_preconditions_with_vram(tmp_path, free_mb=32_000),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=bad_test_codes,
        write=True,
    )
    assert downgraded["status"] == "complete_partial"
    assert downgraded["honest_verdict"].startswith("retired:")

    assert (
        mod.honest_verdict(
            {
                "test_commands": TEST_COMMANDS,
                "test_exit_codes": TEST_EXIT_CODES,
                "preconditions_checked": {"blocked_reasons": []},
                "row_paths_hashes_and_prefix_chain": {
                    "row_counts_by_model": {
                        mod.MANDATED_MODEL_HF_IDS[0]: 1,
                        mod.MANDATED_MODEL_HF_IDS[1]: 0,
                        mod.MANDATED_MODEL_HF_IDS[2]: 0,
                    }
                },
            }
        ).startswith("complete_partial:")
    )
    assert (
        mod._artifact_status(
            blockers=[],
            row_receipt={
                "row_counts_by_model": {
                    mod.MANDATED_MODEL_HF_IDS[0]: 1,
                    mod.MANDATED_MODEL_HF_IDS[1]: 0,
                    mod.MANDATED_MODEL_HF_IDS[2]: 0,
                }
            },
            tentative_ready=False,
        )
        == "complete_partial"
    )
    assert (
        mod._artifact_status(
            blockers=[],
            row_receipt={"row_counts_by_model": {hf_id: 0 for hf_id in mod.MANDATED_MODEL_HF_IDS}},
            tentative_ready=False,
        )
        == "retired"
    )

    malformed_cases = []
    missing = deepcopy(artifact)
    missing.pop("status")
    malformed_cases.append((missing, "missing required fields"))
    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    malformed_cases.append((bad_provenance, "field_provenance"))
    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "wrong"
    bad_substrate["all_family_corpus_ready_score"] = 0.0
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    malformed_cases.append((bad_substrate, "inference_substrate"))
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["all_family_corpus_ready_score"] = 0.0
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    malformed_cases.append((bad_oracle, "verifier_is_oracle"))
    bad_score = deepcopy(artifact)
    bad_score["all_family_corpus_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    malformed_cases.append((bad_score, "all_family_corpus_ready_score"))
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    malformed_cases.append((bad_checksum, "reproducibility_checksum"))
    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "unknown"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    malformed_cases.append((bad_verdict, "honest_verdict"))
    bad_status = deepcopy(artifact)
    bad_status["status"] = "weird"
    bad_status["all_family_corpus_ready_score"] = 0.0
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    malformed_cases.append((bad_status, "status"))
    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"] = "blocked"
    bad_ready_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready_status)
    malformed_cases.append((bad_ready_status, "all_family_corpus_ready_score"))

    for bad_artifact, message in malformed_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    bad_blocked = mod.run(
        result_path=tmp_path / "bad-blocked.json",
        row_dir=tmp_path / "bad-blocked",
        fixture_artifact_path=tmp_path / "missing-fixture.json",
        context_rows_path=tmp_path / "missing-contexts.jsonl",
        pair_rows_path=tmp_path / "missing-pairs.jsonl",
        prior_exp5964_artifact_path=bad_prior,
        model_specs=_model_specs(tmp_path / "bad-blocked-models", missing_index=0),
        preconditions_checked={
            "preconditions_ready": False,
            "blocked_reasons": ["manual"],
            "gpu": {"devices": [], "ok": False},
            "cuda": {"available": False},
            "resources": {"memory": {"ok": False}, "disk": {"ok": False}},
            "runtime": {"task_owned_pid_leases": {"current_pid": 1, "child_pids": []}},
        },
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert {
        "manual",
        "preconditions_not_ready",
        "exp5963_gate_not_ready",
        "exp5963_rows_unavailable",
        "mandated_model_unavailable",
        "embedded_tokenizer_unavailable",
        "no_gpu_devices",
    }.issubset(set(bad_blocked["preconditions_checked"]["blocked_reasons"]))
