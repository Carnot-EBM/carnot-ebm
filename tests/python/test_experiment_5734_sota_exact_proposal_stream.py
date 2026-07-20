"""Tests for Exp5734 sealed chronological exact proposal stream.

Spec refs: REQ-VERIFY-5734, SCENARIO-VERIFY-5734.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5734_sota_exact_proposal_stream as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5734_sota_exact_proposal_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5734_sota_exact_proposal_stream.py "
    "-m pytest tests/python/test_experiment_5734_sota_exact_proposal_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5734_sota_exact_proposal_stream.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5734_sota_exact_proposal_stream.json"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _fixture_label_tokens(offset: int = 1000) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "label": label,
            "token_ids": [offset + index],
            "token_count": 1,
            "unique": True,
            "token_text": label,
        }
        for index, label in enumerate(mod.LABELS)
    }


def _fake_upstream_artifact(
    tmp_path: Path,
    *,
    ready_score: float = 1.0,
    token_collision: bool = False,
    changed_model_hash: bool = False,
) -> dict[str, Any]:
    specs: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}
    resolved: dict[str, dict[str, Any]] = {}
    filenames: dict[str, str] = {}
    quantizations: dict[str, str] = {}
    label_receipts: dict[str, dict[str, Any]] = {}
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"{base['family']}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF-fixture-exp5734-" + bytes([index]) + base["hf_id"].encode())
        digest = _sha256_file(path)
        spec = {
            **base,
            "sequence_index": index,
            "gpu": index,
            "resolved_model_path": str(path),
            "model_path": str(path),
            "gguf_filename": path.name,
            "model_hash": digest,
            "model_size_bytes": path.stat().st_size,
            "quantization": "Q4_K_M",
            "local_model_present": True,
            "headline_eligible": True,
            "legacy_smoke_only": False,
        }
        specs.append(spec)
        model_hashes[base["hf_id"]] = digest
        resolved[base["hf_id"]] = {
            "resolved_model_path": str(path),
            "local_model_present": True,
            "model_size_bytes": path.stat().st_size,
            "model_hash": digest,
        }
        filenames[base["hf_id"]] = path.name
        quantizations[base["hf_id"]] = "Q4_K_M"
        tokens = _fixture_label_tokens(1000 + 100 * index)
        if token_collision and index == 0:
            tokens["B"] = {**tokens["B"], "token_ids": tokens["A"]["token_ids"]}
        label_receipts[base["hf_id"]] = mod.upstream_label_token_receipt(
            model_hf_id=base["hf_id"],
            label_tokens=tokens,
        )
    if changed_model_hash:
        model_hashes[mod.QWEN_ID] = "sha256:" + "0" * 64
    return {
        "schema": "carnot.experiment_5733.sota_finite_choice_proposal_channel.v1",
        "experiment_id": "experiment_5733_sota_finite_choice_proposal_channel",
        "proposal_channel_ready_score": ready_score,
        "qualified_flagship_model_count": 2,
        "cuda_offload_authenticated_score": 1.0,
        "receipt_failure_count": 0,
        "verifier_is_oracle": True,
        "qualified_model_ids": list(mod.HEADLINE_MODEL_IDS),
        "MODEL_SPECS": specs,
        "resolved_model_receipts": resolved,
        "model_hashes": model_hashes,
        "gguf_filenames": filenames,
        "quantizations": quantizations,
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1 | ggml-cuda present",
            "module": "llama_cpp",
        },
        "cuda_device_receipts": {
            hf_id: {
                "before": [{"index": index, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000}],
                "peak": [8000 + 100 * index],
                "after": [{"index": index, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 23900}],
                "worker_returncode": 0,
            }
            for index, hf_id in enumerate(mod.HEADLINE_MODEL_IDS)
        },
        "n_gpu_layers_offloaded": {hf_id: 40 for hf_id in mod.HEADLINE_MODEL_IDS},
        "gpu_memory_receipts": {
            hf_id: {"before_mb": 128, "peak_mb": 6144 + 100 * index, "after_mb": 160}
            for index, hf_id in enumerate(mod.HEADLINE_MODEL_IDS)
        },
        "cuda_offload_authenticated": {hf_id: True for hf_id in mod.HEADLINE_MODEL_IDS},
        "label_token_receipts": label_receipts,
        "freeform_generation_used": False,
        "grammar_runtime_used": False,
        "external_scorer_used": False,
        "token_scores_are_semantic_authority": False,
        "retired_runtime_used": False,
        "inference_substrate": "local_llama_cpp_python_cuda_gguf_finite_choice_proposals",
    }


def _write_upstream(tmp_path: Path, **kwargs: Any) -> Path:
    artifact = _fake_upstream_artifact(tmp_path, **kwargs)
    path = tmp_path / "experiment_5733_sota_finite_choice_proposal_channel.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    del controls, random_seeds
    rows = []
    for local_index, row in enumerate(candidate_rows):
        exact_label = next(item["label"] for item in row["label_mapping"] if item["is_exact"])
        wrong_label = next(item["label"] for item in row["label_mapping"] if not item["is_exact"])
        selected_label = wrong_label if local_index % 11 == 0 else exact_label
        score_vector = {label: -4.0 - offset / 100.0 for offset, label in enumerate(mod.LABELS)}
        score_vector[selected_label] = 3.0 + local_index / 1000.0
        rows.append(
            {
                "model_hf_id": model_spec["hf_id"],
                "control_id": row["row_id"],
                "prompt_hash": row["prompt_hash"],
                "score_vector": score_vector,
                "label_token_ids": {
                    label: [1000 + offset] for offset, label in enumerate(mod.LABELS)
                },
                "timing": {"prefill_s": round(0.01 + local_index / 10000, 6)},
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
        "rows": rows,
    }


def _nonfinite_runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    receipt = _runner(model_spec, controls, candidate_rows, random_seeds)
    receipt["rows"][0]["score_vector"][mod.LABELS[0]] = math.inf
    return receipt


def _missing_row_runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    receipt = _runner(model_spec, controls, candidate_rows, random_seeds)
    receipt["rows"] = receipt["rows"][1:]
    return receipt


def test_req_verify_5734_spec_declares_stream_contract() -> None:
    """REQ-VERIFY-5734: OpenSpec anchors the sealed stream contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5734") : spec.index("### REQ-VERIFY-5615")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5734",
        "SCENARIO-VERIFY-5734",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_MANIFEST_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "at least 96 chronological rows",
        "`verifier_is_oracle` SHALL be true",
        "`sota_proposal_stream_ready_score` SHALL be `1.0` only when both headline",
    ):
        assert marker in section
    for hf_id in mod.HEADLINE_MODEL_IDS:
        assert hf_id in section
    for family in mod.REQUIRED_FAMILIES:
        assert family.replace("_", "-") in section or family in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5734_preregistered_panel_balanced_and_sealed(tmp_path: Path) -> None:
    """REQ-VERIFY-5734: panel rows are balanced and sealed before scores."""

    upstream_path = _write_upstream(tmp_path)
    upstream = mod.load_and_verify_upstream_channel(upstream_path)
    specs = mod.model_specs_from_upstream(upstream)
    panel = mod.preregister_panel(model_specs=specs)

    assert len(panel) == mod.ROW_COUNT
    counts = mod.family_counts(panel)
    assert set(counts) == set(mod.REQUIRED_FAMILIES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert mod.model_counts(panel) == {mod.QWEN_ID: 48, mod.GEMMA31_ID: 48}
    for by_model in mod.model_family_counts(panel).values():
        assert max(by_model.values()) - min(by_model.values()) <= 1

    exact_label_counts = Counter(row["admitted_label"] for row in panel)
    assert set(exact_label_counts.values()) == {16}
    assert mod.split_lengths(panel) == {"prospective_prefix_length": 48, "sealed_suffix_length": 48}

    for row in panel:
        receipt = mod.candidate_domain_receipt(row)
        assert receipt["domain_complete"] is True
        assert receipt["exact_candidate_present"] is True
        assert receipt["candidate_count"] == len(mod.LABELS)
        assert receipt["plausible_hard_distractor_count"] == len(mod.LABELS) - 1
        assert len({item["candidate_id"] for item in row["candidate_domain"]}) == len(mod.LABELS)
        assert len({item["candidate"] for item in row["candidate_domain"]}) == len(mod.LABELS)
        assert len([item for item in row["label_mapping"] if item["is_exact"]]) == 1
        assert row["expected_exact_validator_version"] == mod.EXACT_VALIDATOR_VERSIONS[row["family"]]
        assert row["model_hash"] == upstream["model_hashes"][row["model_hf_id"]]
        assert "FINAL:" not in row["prompt"]
        assert row["prompt"].endswith("Answer label:")

    tampered = deepcopy(panel[0])
    tampered["validator_payload"] = {"kind": "unknown"}
    with pytest.raises(ValueError, match="unknown validator kind"):
        mod.exact_answer_by_primary(tampered)


def test_scenario_verify_5734_complete_artifact_manifest_and_commitments(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5734: complete row receipts qualify and replay exactly."""

    upstream_path = _write_upstream(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_manifest_path=tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name,
        upstream_artifact_path=upstream_path,
        score_runner=_runner,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    rows = mod.read_row_manifest(tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_manifest(rows, artifact) is True
    assert artifact["headline_model_count"] == 2
    assert artifact["sota_proposal_stream_ready_score"] == 1.0
    assert artifact["missing_row_count"] == 0
    assert artifact["non_finite_score_count"] == 0
    assert artifact["label_collision_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["freeform_generation_used"] is False
    assert artifact["grammar_runtime_used"] is False
    assert artifact["external_scorer_used"] is False
    assert artifact["token_scores_are_semantic_authority"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["model_counts"] == {mod.QWEN_ID: 48, mod.GEMMA31_ID: 48}
    assert artifact["proposal_conflict_count"] > 0
    assert len(artifact["proposal_ids"]) == mod.ROW_COUNT
    assert len(artifact["candidate_domain_hashes"]) == mod.ROW_COUNT
    assert len(artifact["label_permutation_hashes"]) == mod.ROW_COUNT
    assert len(artifact["score_vector_hashes"]) == mod.ROW_COUNT
    assert set(artifact).issubset(set(artifact["field_principles"]))
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact

    sampled_families = {
        row["family"] for row in rows if row["enumeration_double_check"]["sampled"] is True
    }
    assert sampled_families == set(mod.REQUIRED_FAMILIES)
    assert any(receipt["proposal_matches_oracle"] is False for receipt in artifact["conflict_receipts"].values())

    previous_break = deepcopy(rows)
    previous_break[1]["previous_row_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="previous_row_hash"):
        mod.verify_row_manifest(previous_break, artifact)

    hash_break = deepcopy(rows)
    hash_break[0]["row_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_row_manifest(hash_break, artifact)

    content_break = deepcopy(rows)
    content_break[0]["selected_label"] = "Z"
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_row_manifest(content_break, artifact)

    score_hash_break = deepcopy(artifact)
    score_hash_break["score_vector_hashes"][rows[0]["row_id"]] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="score_vector_hash"):
        mod.verify_row_manifest(rows, score_hash_break)

    domain_hash_break = deepcopy(artifact)
    domain_hash_break["candidate_domain_hashes"][rows[0]["row_id"]] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="candidate_domain_hash"):
        mod.verify_row_manifest(rows, domain_hash_break)

    label_hash_break = deepcopy(artifact)
    label_hash_break["label_permutation_hashes"][rows[0]["row_id"]] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="label_permutation_hash"):
        mod.verify_row_manifest(rows, label_hash_break)

    proposal_break = deepcopy(artifact)
    proposal_break["proposal_ids"][rows[0]["row_id"]] = "wrong"
    with pytest.raises(mod.ManifestReplayError, match="proposal_id"):
        mod.verify_row_manifest(rows, proposal_break)

    conflict_break = deepcopy(artifact)
    conflict_break["conflict_receipts"][rows[0]["row_id"]] = {"wrong": True}
    with pytest.raises(mod.ManifestReplayError, match="conflict_receipt"):
        mod.verify_row_manifest(rows, conflict_break)

    score_content_break = deepcopy(rows)
    score_content_break[0]["score_vector"][mod.LABELS[0]] += 1.0
    with pytest.raises(mod.ManifestReplayError, match="score_vector_hash"):
        mod.verify_row_manifest(score_content_break, artifact)

    prefix_break = deepcopy(artifact)
    prefix_break["prospective_prefix_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="prospective_prefix_hash"):
        mod.verify_row_manifest(rows, prefix_break)

    suffix_break = deepcopy(artifact)
    suffix_break["sealed_suffix_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="sealed_suffix_hash"):
        mod.verify_row_manifest(rows, suffix_break)

    root_break = deepcopy(artifact)
    root_break["stream_root_commitment"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="stream_root_commitment"):
        mod.verify_row_manifest(rows, root_break)

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
        (lambda item: item.update({"headline_model_count": 1}), "headline_model_count"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"model_weight_mutation": True}), "model_weight_mutation"),
        (lambda item: item.update({"external_scorer_used": True}), "external_scorer_used"),
        (lambda item: item.update({"sota_proposal_stream_ready_score": 0.0}), "sota_proposal_stream_ready_score"),
        (lambda item: item.update({"honest_verdict": "blocked: wrong"}), "honest_verdict"),
        (lambda item: item.update({"reproducibility_checksum": "sha256:bad"}), "reproducibility_checksum"),
    ):
        candidate = deepcopy(artifact)
        mutate(candidate)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(candidate)


def test_req_verify_5734_blockers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5734: upstream, score, row, and validator faults block readiness."""

    blocked_upstream = mod.run(
        result_path=tmp_path / "blocked_upstream.json",
        row_manifest_path=tmp_path / "blocked_upstream.rows.jsonl",
        upstream_artifact_path=_write_upstream(tmp_path / "bad_upstream", token_collision=True),
        score_runner=_runner,
        write=False,
    )
    assert blocked_upstream["sota_proposal_stream_ready_score"] == 0.0
    assert blocked_upstream["missing_row_count"] == mod.ROW_COUNT
    assert "upstream_label_collision" in blocked_upstream["blocked_reasons"]
    assert mod.validate_artifact(blocked_upstream) is True

    blocked_hash = mod.run(
        result_path=tmp_path / "blocked_hash.json",
        row_manifest_path=tmp_path / "blocked_hash.rows.jsonl",
        upstream_artifact_path=_write_upstream(tmp_path / "bad_hash", changed_model_hash=True),
        score_runner=_runner,
        write=False,
    )
    assert "upstream_model_hash_mismatch" in blocked_hash["blocked_reasons"]
    assert blocked_hash["sota_proposal_stream_ready_score"] == 0.0

    nonfinite = mod.run(
        result_path=tmp_path / "nonfinite.json",
        row_manifest_path=tmp_path / "nonfinite.rows.jsonl",
        upstream_artifact_path=_write_upstream(tmp_path / "nonfinite_upstream"),
        score_runner=_nonfinite_runner,
        write=False,
    )
    assert nonfinite["non_finite_score_count"] == len(mod.HEADLINE_MODEL_IDS)
    assert nonfinite["sota_proposal_stream_ready_score"] == 0.0
    assert "non_finite_score_count" in nonfinite["blocked_reasons"]
    assert mod.validate_artifact(nonfinite) is True

    missing = mod.run(
        result_path=tmp_path / "missing.json",
        row_manifest_path=tmp_path / "missing.rows.jsonl",
        upstream_artifact_path=_write_upstream(tmp_path / "missing_upstream"),
        score_runner=_missing_row_runner,
        write=False,
    )
    assert missing["missing_row_count"] == len(mod.HEADLINE_MODEL_IDS)
    assert missing["sota_proposal_stream_ready_score"] == 0.0
    assert "missing_row_count" in missing["blocked_reasons"]
    assert mod.validate_artifact(missing) is True

    panel = mod.preregister_panel(model_specs=mod.model_specs_from_upstream(_fake_upstream_artifact(tmp_path / "panel")))
    wrong_candidate = next(item["candidate"] for item in panel[0]["candidate_domain"] if not item["is_exact"])
    assert mod.primary_validate_selection(panel[0], wrong_candidate)["selected_is_exact"] is False
    tampered = deepcopy(panel[0])
    tampered["validator_payload"] = {**tampered["validator_payload"], "expected_override": wrong_candidate}
    assert mod.validator_disagrees(tampered, wrong_candidate) is True

    with pytest.raises(mod.UpstreamChannelError, match="upstream_channel_not_ready"):
        mod.load_and_verify_upstream_channel(_write_upstream(tmp_path / "not_ready", ready_score=0.0))

    blocked_write_path = tmp_path / "blocked_write.json"
    blocked_rows_path = tmp_path / "blocked_write.rows.jsonl"
    blocked_write = mod.run(
        result_path=blocked_write_path,
        row_manifest_path=blocked_rows_path,
        upstream_artifact_path=_write_upstream(tmp_path / "blocked_write_upstream", token_collision=True),
        score_runner=_runner,
        write=True,
    )
    assert blocked_write_path.exists()
    assert blocked_rows_path.exists()
    assert json.loads(blocked_write_path.read_text(encoding="utf-8")) == blocked_write
    blocked_bad_verdict = deepcopy(blocked_write)
    blocked_bad_verdict["honest_verdict"] = "complete: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_bad_verdict)

    assert "external_scorer_used" in mod._blocked_reasons(
        {
            "external_scorer_used": True,
            "preconditions_checked": {"exp5733_artifact_verified": True},
        }
    )
    with pytest.raises(ValueError, match="unknown family"):
        mod._base_family_row("unknown_family", 0)

    upstream_faults: list[tuple[str, str, Any]] = [
        ("upstream_model_hash_missing", "model_hashes", lambda item: item["model_hashes"].update({mod.QWEN_ID: ""})),
        (
            "upstream_resolved_gguf_missing",
            "resolved_model_receipts",
            lambda item: item["resolved_model_receipts"][mod.QWEN_ID].update({"resolved_model_path": str(tmp_path / "missing.gguf")}),
        ),
        (
            "upstream_model_size_mismatch",
            "resolved_model_receipts",
            lambda item: item["resolved_model_receipts"][mod.QWEN_ID].update({"model_size_bytes": 999999}),
        ),
        (
            "upstream_tokenizer_not_vocab_only",
            "label_token_receipts",
            lambda item: item["label_token_receipts"][mod.QWEN_ID].update({"vocab_only": False}),
        ),
        (
            "upstream_transformers_tokenizer_used",
            "label_token_receipts",
            lambda item: item["label_token_receipts"][mod.QWEN_ID].update({"transformers_used": True}),
        ),
        (
            "upstream_cuda_offload_unauthenticated",
            "cuda_offload_authenticated",
            lambda item: item["cuda_offload_authenticated"].update({mod.QWEN_ID: False}),
        ),
        (
            "upstream_no_gpu_layers_offloaded",
            "n_gpu_layers_offloaded",
            lambda item: item["n_gpu_layers_offloaded"].update({mod.QWEN_ID: 0}),
        ),
        (
            "upstream_no_gpu_memory_delta",
            "gpu_memory_receipts",
            lambda item: item["gpu_memory_receipts"][mod.QWEN_ID].update({"peak_mb": 128}),
        ),
        (
            "upstream_flagship_model_count",
            "qualified_flagship_model_count",
            lambda item: item.update({"qualified_flagship_model_count": 1}),
        ),
        (
            "upstream_cuda_score",
            "cuda_offload_authenticated_score",
            lambda item: item.update({"cuda_offload_authenticated_score": 0.0}),
        ),
        ("upstream_receipt_failure", "receipt_failure_count", lambda item: item.update({"receipt_failure_count": 1})),
        ("upstream_verifier_not_oracle", "verifier_is_oracle", lambda item: item.update({"verifier_is_oracle": False})),
        ("upstream_freeform_generation_used", "freeform_generation_used", lambda item: item.update({"freeform_generation_used": True})),
        ("upstream_headline_model_missing", "qualified_model_ids", lambda item: item.update({"qualified_model_ids": []})),
    ]
    for expected_reason, _field, mutate in upstream_faults:
        upstream_artifact = _fake_upstream_artifact(tmp_path / expected_reason)
        mutate(upstream_artifact)
        errors, _collision_count = mod._upstream_errors(upstream_artifact)
        assert expected_reason in errors
