"""Tests for Exp5733 sealed finite-choice GGUF proposal channel.

Spec refs: REQ-VERIFY-5733, SCENARIO-VERIFY-5733.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5733_sota_finite_choice_proposal_channel as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5733_sota_finite_choice_proposal_channel.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5733_sota_finite_choice_proposal_channel.py "
    "-m pytest tests/python/test_experiment_5733_sota_finite_choice_proposal_channel.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5733_sota_finite_choice_proposal_channel.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5733_sota_finite_choice_proposal_channel.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"{base['family']}-UD-Q4_K_M.gguf"
        path.write_bytes(b"GGUF-fixture-exp5733-" + bytes([index]) + base["hf_id"].encode())
        spec = dict(base)
        spec["model_path"] = str(path)
        specs.append(spec)
    return mod.normalize_model_specs(specs)


def _fixture_label_tokens() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "label": label,
            "token_ids": [1000 + idx],
            "token_count": 1,
            "unique": True,
            "token_text": label,
        }
        for idx, label in enumerate(mod.LABELS)
    }


def _runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    del controls, random_seeds
    rows = []
    for index, row in enumerate(candidate_rows):
        correct_label = next(item["label"] for item in row["label_mapping"] if item["is_exact"])
        score_vector = {label: -5.0 - offset / 100.0 for offset, label in enumerate(mod.LABELS)}
        # Deliberately choose a wrong proposal on some rows to prove accuracy is
        # descriptive and not a channel qualification gate.
        selected_label = correct_label if index % 5 else mod.LABELS[(mod.LABELS.index(correct_label) + 1) % len(mod.LABELS)]
        score_vector[selected_label] = 2.0
        rows.append(
            {
                "model_hf_id": model_spec["hf_id"],
                "control_id": row["control_id"],
                "prompt_hash": row["prompt_hash"],
                "score_vector": score_vector,
                "label_token_ids": {label: [1000 + idx] for idx, label in enumerate(mod.LABELS)},
                "timing": {"prefill_s": round(0.01 + index / 10000, 6)},
                "error": "",
            }
        )
    return {
        "model_hf_id": model_spec["hf_id"],
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1 | ggml-cuda present",
            "module": "llama_cpp",
        },
        "cuda_device_receipt": {
            "before": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000, "memory_used_mb": 128}],
            "peak": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 18000, "memory_used_mb": 6144}],
            "after": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 23900, "memory_used_mb": 160}],
            "worker_returncode": 0,
        },
        "vocab_only_tokenizer_receipt": {
            "model_hf_id": model_spec["hf_id"],
            "vocab_only": True,
            "load_ok": True,
            "transformers_used": False,
            "label_tokens": _fixture_label_tokens(),
        },
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 40,
        "gpu_memory_before_mb": 128,
        "gpu_memory_peak_mb": 6144,
        "gpu_memory_after_mb": 160,
        "cuda_offload_authenticated": True,
        "offload_log_excerpt": "llama_model_load_tensors: offloaded 40/40 layers to GPU",
        "rows": rows,
    }


def _blocked_cuda_runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    receipt = _runner(model_spec, controls, candidate_rows, random_seeds)
    receipt["n_gpu_layers_offloaded"] = 0
    receipt["gpu_memory_peak_mb"] = receipt["gpu_memory_before_mb"]
    receipt["cuda_offload_authenticated"] = False
    return receipt


def _nonfinite_runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    receipt = _runner(model_spec, controls, candidate_rows, random_seeds)
    receipt["rows"][0]["score_vector"][mod.LABELS[0]] = math.nan
    return receipt


def _run_fixture(tmp_path: Path, runner: mod.ScoreRunner = _runner) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        score_vector_manifest_path=tmp_path / mod.SCORE_VECTOR_MANIFEST_RELATIVE_PATH.name,
        model_specs=_fake_model_specs(tmp_path),
        score_runner=runner,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )


def test_req_verify_5733_spec_declares_finite_choice_contract() -> None:
    """REQ-VERIFY-5733: OpenSpec anchors fields, models, and no-freeform gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5733") : spec.index("### REQ-VERIFY-5615")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5733",
        "SCENARIO-VERIFY-5733",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.SCORE_VECTOR_MANIFEST_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "vocab_only=True",
        "one next-token scoring step",
        "`verifier_is_oracle` SHALL be true",
        "`cuda_offload_authenticated_score` equal to `1.0` only when both flagship",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        principle = mod.FIELD_PRINCIPLES[field]
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5733_control_manifest_domains_and_leakage() -> None:
    """REQ-VERIFY-5733: controls cover required families without answer leakage."""

    controls = mod.freeze_control_manifest()
    candidate_rows = mod.freeze_candidate_rows(controls)
    category_counts = mod.control_category_counts(controls)

    assert len(controls) == 42
    assert sum(row["polarity"] == "positive" for row in controls) == 30
    assert sum(row["polarity"] == "negative" for row in controls) == 12
    for category in mod.REQUIRED_CONTROL_CATEGORIES:
        assert category_counts[category] > 0
    assert all("exp5734" not in json.dumps(row).lower() for row in controls)

    exact_label_counts = {label: 0 for label in mod.LABELS}
    for row in candidate_rows:
        receipt = mod.candidate_domain_receipt(row)
        assert receipt["domain_complete"] is True
        assert receipt["exact_candidate_present"] is True
        assert receipt["candidate_count"] == len(mod.LABELS)
        assert len({item["candidate"] for item in row["label_mapping"]}) == len(mod.LABELS)
        exact_labels = [item["label"] for item in row["label_mapping"] if item["is_exact"]]
        assert len(exact_labels) == 1
        exact_label_counts[exact_labels[0]] += 1
        assert all(item["label"].strip() == item["label"] and len(item["label"]) == 1 for item in row["label_mapping"])
        assert row["leakage_checks"]["label_frequency_balanced"] is True
        assert row["leakage_checks"]["uniform_label_token_length"] is True
    assert set(exact_label_counts.values()) == {7}


def test_scenario_verify_5733_complete_artifact_and_score_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5733: complete score receipts qualify even with wrong proposals."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_score_vector_rows(tmp_path / mod.SCORE_VECTOR_MANIFEST_RELATIVE_PATH.name)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_score_vector_rows(rows, artifact) is True
    assert artifact["MODEL_SPECS"][0]["hf_id"] == mod.QWEN_ID
    assert list(artifact["model_hashes"]) == list(mod.MANDATED_MODEL_IDS)
    assert set(artifact["quantizations"].values()) == {"UD-Q4_K_M"}
    assert artifact["qualified_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["qualified_flagship_model_count"] == 2
    assert artifact["cuda_offload_authenticated_score"] == 1.0
    assert artifact["proposal_channel_ready_score"] == 1.0
    assert artifact["receipt_failure_count"] == 0
    assert artifact["non_finite_score_count"] == 0
    assert artifact["label_collision_count"] == 0
    assert artifact["candidate_omission_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["freeform_generation_used"] is False
    assert artifact["grammar_runtime_used"] is False
    assert artifact["external_scorer_used"] is False
    assert artifact["token_scores_are_semantic_authority"] is False
    assert set(artifact).issubset(set(artifact["field_principles"]))
    assert artifact["random_seed"] == mod.RANDOM_SEEDS["base_seed"]
    assert artifact["model_accuracy"][mod.QWEN_ID] < 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == len(artifact["control_manifest"]) * len(mod.MANDATED_MODEL_IDS)
    assert len(artifact["label_permutation_hashes"]) == len(artifact["control_manifest"])
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact

    previous_break = deepcopy(rows)
    previous_break[1]["previous_row_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="previous_row_hash"):
        mod.verify_score_vector_rows(previous_break, artifact)

    hash_break = deepcopy(rows)
    hash_break[0]["row_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="score_vector_hash"):
        mod.verify_score_vector_rows(hash_break, artifact)

    content_break = deepcopy(rows)
    content_break[0]["selected_label"] = "Z"
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_score_vector_rows(content_break, artifact)

    layer_blocked = deepcopy(artifact)
    layer_blocked["n_gpu_layers_offloaded"][mod.QWEN_ID] = 0
    assert mod.cuda_offload_authenticated_score(layer_blocked) == 0.0
    memory_blocked = deepcopy(artifact)
    memory_blocked["gpu_memory_receipts"][mod.QWEN_ID]["peak_mb"] = memory_blocked["gpu_memory_receipts"][mod.QWEN_ID]["before_mb"]
    assert mod.cuda_offload_authenticated_score(memory_blocked) == 0.0

    invalid = deepcopy(artifact)
    del invalid["honest_verdict"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(invalid)
    invalid = deepcopy(artifact)
    invalid["unprincipled_extra"] = True
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(invalid)
    for mutate, match in (
        (lambda item: item.update({"field_principles": []}), "field_principles"),
        (lambda item: item["field_principles"].update({"MODEL_SPECS": "wrong"}), "field_principles"),
        (lambda item: item.update({"MODEL_SPECS": []}), "MODEL_SPECS"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"freeform_generation_used": True}), "freeform_generation_used"),
        (lambda item: item.update({"cuda_offload_authenticated_score": 0.0}), "cuda_offload_authenticated_score"),
        (lambda item: item.update({"proposal_channel_ready_score": 0.0}), "proposal_channel_ready_score"),
        (lambda item: item.update({"honest_verdict": "blocked: wrong"}), "honest_verdict"),
        (lambda item: item.update({"reproducibility_checksum": "sha256:bad"}), "reproducibility_checksum"),
    ):
        candidate = deepcopy(artifact)
        mutate(candidate)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(candidate)


def test_req_verify_5733_token_receipts_and_validators_are_exact() -> None:
    """REQ-VERIFY-5733: label collisions and validator disagreements fail closed."""

    controls = mod.freeze_control_manifest()
    row = mod.freeze_candidate_rows(controls)[0]
    assert mod.model_family("local/custom-GGUF") == "custom"
    assert mod._selected_from_scores({}) == ("", "missing_score")
    assert mod._free_vram_from_receipt({"before": "not-a-list"}) == 0
    with pytest.raises(ValueError, match="unknown validator kind"):
        mod.exact_answer_by_primary({"validator_payload": {"kind": "mystery"}})

    token_receipt = mod.label_token_receipt(
        model_hf_id=mod.QWEN_ID,
        label_tokens=_fixture_label_tokens(),
    )
    assert token_receipt["label_collision_count"] == 0
    assert token_receipt["all_single_unique_tokens"] is True

    colliding = _fixture_label_tokens()
    colliding["B"] = dict(colliding["B"], token_ids=colliding["A"]["token_ids"])
    bad_receipt = mod.label_token_receipt(model_hf_id=mod.QWEN_ID, label_tokens=colliding)
    assert bad_receipt["label_collision_count"] == 1
    assert bad_receipt["all_single_unique_tokens"] is False

    exact_candidate = next(item["candidate"] for item in row["label_mapping"] if item["is_exact"])
    primary = mod.primary_validate_selection(row, exact_candidate)
    independent = mod.independent_validate_selection(row, exact_candidate)
    enumeration = mod.enumeration_double_check(row, exact_candidate)
    assert primary["selected_is_exact"] is True
    assert independent["selected_is_exact"] is True
    assert enumeration["enumeration_agrees"] is True

    wrong = next(item["candidate"] for item in row["label_mapping"] if not item["is_exact"])
    assert mod.primary_validate_selection(row, wrong)["selected_is_exact"] is False

    tampered = deepcopy(row)
    tampered["expected_answer"] = wrong
    assert mod.validator_disagrees(tampered, exact_candidate) is True


def test_req_verify_5733_cuda_or_nonfinite_failures_block(tmp_path: Path) -> None:
    """REQ-VERIFY-5733: CPU-only receipts or non-finite scores block readiness."""

    blocked = _run_fixture(tmp_path / "cuda", runner=_blocked_cuda_runner)
    assert blocked["cuda_offload_authenticated_score"] == 0.0
    assert blocked["proposal_channel_ready_score"] == 0.0
    assert blocked["receipt_failure_count"] > 0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True
    blocked_verdict = deepcopy(blocked)
    blocked_verdict["honest_verdict"] = "complete: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_verdict)

    nonfinite = _run_fixture(tmp_path / "nan", runner=_nonfinite_runner)
    assert nonfinite["non_finite_score_count"] == len(mod.MANDATED_MODEL_IDS)
    assert nonfinite["receipt_failure_count"] > 0
    assert nonfinite["proposal_channel_ready_score"] == 0.0
    assert nonfinite["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(nonfinite) is True
    assert "retired_runtime_used" in mod._blocked_reasons({"retired_runtime_used": True})

    missing_specs = []
    for spec in mod.MODEL_SPECS:
        missing = dict(spec)
        missing["model_path"] = str(tmp_path / "missing" / f"{spec['family']}.gguf")
        missing_specs.append(missing)
    missing_artifact = mod.run(
        result_path=tmp_path / "missing.json",
        score_vector_manifest_path=tmp_path / "missing.jsonl",
        model_specs=missing_specs,
        score_runner=_runner,
        write=False,
    )
    assert missing_artifact["missing_score_count"] == len(mod.MANDATED_MODEL_IDS) * len(mod.freeze_control_manifest()) * len(mod.LABELS)
    assert missing_artifact["proposal_channel_ready_score"] == 0.0
