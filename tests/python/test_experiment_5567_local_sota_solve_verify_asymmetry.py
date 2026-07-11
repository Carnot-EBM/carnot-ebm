"""Tests for Exp5567 local SOTA solve-versus-verify asymmetry.

Spec refs: REQ-VERIFY-5567, SCENARIO-VERIFY-5567.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import pytest

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot import experiment_5567_local_sota_solve_verify_asymmetry as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5567_local_sota_solve_verify_asymmetry.py")
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    qwen = tmp_path / "qwen.gguf"
    gemma = tmp_path / "gemma.gguf"
    qwen.write_bytes(b"qwen")
    gemma.write_bytes(b"gemma")
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": QWEN,
            "family": "qwen",
            "role": "moe",
            "gpu": 0,
            "model_path": str(qwen),
            "headline_eligible": True,
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": GEMMA26,
            "family": "gemma",
            "role": "moe",
            "gpu": 1,
            "model_path": str(gemma),
            "headline_eligible": True,
        },
    ]


def _complete_gate(model_specs: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "cached_sota_pair_called": True,
        "cache_gate_passed": True,
        "blocked_reason": "",
        "cached_pair_hf_ids": [str(row["hf_id"]) for row in model_specs],
        "selected_headline_model_ids": [str(row["hf_id"]) for row in model_specs],
        "legacy_cpu_model_substituted": False,
        "corpus_gate_passed": True,
        "offload_gate_passed": True,
    }


def _complete_device() -> dict[str, object]:
    return {
        "torch_cuda_available": True,
        "torch_device_count": 2,
        "devices": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090"},
        ],
        "llama_cpp_supports_gpu_offload": True,
        "offloaded_layer_count_from_backend_log": 42,
        "gpu_offload_authenticated": True,
    }


def _fake_panel_result(
    pairs: Sequence[dict[str, object]],
    model_specs: Sequence[dict[str, object]],
) -> dict[str, object]:
    solve_records: list[dict[str, object]] = []
    verifier_records: list[dict[str, object]] = []
    raw_hashes: dict[str, str] = {}
    for model_index, spec in enumerate(model_specs):
        hf_id = str(spec["hf_id"])
        for pair_index, pair in enumerate(pairs):
            instance_id = str(pair["instance_id"])
            family = str(pair["family"])
            solve_correct = (pair_index + model_index) % 3 != 0
            solve_hash = mod.sha256_text(f"{hf_id}:{instance_id}:solve")
            raw_hashes[f"{hf_id}:solve:{instance_id}"] = solve_hash
            solve_records.append(
                {
                    "model_hf_id": hf_id,
                    "instance_id": instance_id,
                    "family": family,
                    "parser_ok": True,
                    "exact_accepted": solve_correct,
                    "latency_s": 0.25,
                    "prompt_tokens": 90,
                    "completion_tokens": 18,
                    "response_hash": solve_hash,
                    "error_type": "",
                }
            )
            for candidate_key in ("valid_row", "invalid_row"):
                row = pair[candidate_key]
                assert isinstance(row, dict)
                true_label = str(row["label"])
                for arm in mod.ARMS:
                    if arm == "criteria_decomposition":
                        predicted = true_label
                    elif arm == "granular_score":
                        predicted = "invalid" if pair_index % 5 == 0 else true_label
                    elif arm == "repeated_verdict_3x":
                        predicted = true_label
                    else:
                        predicted = true_label if pair_index % 4 else "invalid"
                    response_hashes = [
                        mod.sha256_text(f"{hf_id}:{instance_id}:{row['row_id']}:{arm}:{repeat}")
                        for repeat in range(3 if arm == "repeated_verdict_3x" else 1)
                    ]
                    for index, value in enumerate(response_hashes):
                        raw_hashes[f"{hf_id}:{instance_id}:{row['row_id']}:{arm}:{index}"] = value
                    verifier_records.append(
                        {
                            "model_hf_id": hf_id,
                            "instance_id": instance_id,
                            "candidate_id": str(row["row_id"]),
                            "family": family,
                            "arm": arm,
                            "true_label": true_label,
                            "predicted_label": predicted,
                            "parser_ok": True,
                            "latency_s": 0.08 * len(response_hashes),
                            "prompt_tokens": 120 * len(response_hashes),
                            "completion_tokens": 8 * len(response_hashes),
                            "response_hashes": response_hashes,
                            "repeat_labels": [predicted] * len(response_hashes),
                            "error_type": "",
                        }
                    )
    return {
        "solve_records": solve_records,
        "verifier_records": verifier_records,
        "raw_response_hash": raw_hashes,
        "inference_duration_s": 123.5,
    }


def test_req_verify_5567_spec_declares_panel_contract() -> None:
    """REQ-VERIFY-5567: OpenSpec anchors cache, offload, statistics, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5567") : spec.index("### REQ-VERIFY-5566")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5567" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(corpus5566.RESULT_RELATIVE_PATH) in section
    assert QWEN in section
    assert GEMMA26 in section
    assert "blocked_missing_sota_cache" in section
    assert "blocked_no_cuda_offload" in section
    assert "SHALL NOT substitute a legacy CPU model" in normalized
    assert "at least 36 independent" in normalized
    assert "`exact_validator_is_oracle` SHALL be `true`" in section
    assert "`verifier_is_oracle` SHALL be `false`" in section
    assert f"`inference_substrate` SHALL equal `{mod.INFERENCE_SUBSTRATE}`" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5567_complete_panel_reports_paired_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5567: complete panels use instance-paired solve/verify metrics."""

    rows = corpus5566.build_corpus_rows(
        json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())
        | {
            "exact_fsm_fixture_extended_ready": True,
            "exact_asp_validator_ready": True,
            "asp_fixture_rows": [],
            "stable_model_reports": [],
        }
    )
    if not rows:
        rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    pairs = mod.sample_independent_pairs(rows, n=mod.MIN_INDEPENDENT_INSTANCES)
    model_specs = _fake_model_specs(tmp_path)
    artifact = mod.build_artifact(
        corpus_rows=rows,
        model_specs=model_specs,
        gate_receipt=_complete_gate(model_specs),
        device_receipt=_complete_device(),
        sampled_pairs=pairs,
        panel_result=_fake_panel_result(pairs, model_specs),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
        bootstrap_iterations=64,
    )

    assert artifact["panel_complete"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["gpu_offload_authenticated"] is True
    assert artifact["exact_validator_is_oracle"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["n_independent_instances"] == 36
    assert artifact["family_counts"] == {
        "contradictions": 9,
        "defaults_exceptions": 9,
        "fsm_transition_consistency": 9,
        "soft_preference_optimality": 9,
    }
    assert artifact["arms"] == list(mod.ARMS)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(artifact["solve_accuracy_by_model"]) == {QWEN, GEMMA26}
    assert artifact["solve_accuracy_by_model"][QWEN]["n"] == 36
    assert (
        artifact["verifier_metrics_by_model_and_arm"][QWEN]["repeated_verdict_3x"][
            "repeat_calls_per_candidate"
        ]
        == 3
    )
    assert (
        artifact["verifier_metrics_by_model_and_arm"][QWEN]["repeated_verdict_3x"][
            "independent_unit"
        ]
        == "instance_id"
    )
    assert (
        artifact["solve_verify_asymmetry"][QWEN]["criteria_decomposition"][
            "solve_minus_verify_balanced_accuracy"
        ]
        < 0.0
    )
    assert artifact["confidence_intervals"][QWEN]["solve_accuracy"]["n_bootstrap"] == 64
    assert (
        artifact["mcnemar_results"][QWEN]["criteria_decomposition"]["paired_unit"] == "instance_id"
    )
    assert artifact["raw_response_hash"]
    assert artifact["parser_failure_count"] == 0

    mod.validate_artifact(artifact)


def test_req_verify_5567_blocked_cache_or_offload_never_promotes_legacy(tmp_path: Path) -> None:
    """REQ-VERIFY-5567: cache and CUDA/offload gates fail closed without CPU fallback."""

    legacy_path = tmp_path / "legacy.gguf"
    legacy_path.write_bytes(b"legacy")
    legacy_pair = [
        {
            "name": "Qwen3.5-0.8B",
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "gpu": 0,
            "model_path": str(legacy_path),
        },
        {
            "name": "Gemma4-E4B-it",
            "hf_id": "google/gemma-4-E4B-it",
            "gpu": 1,
            "model_path": str(legacy_path),
        },
    ]

    specs, gate = mod.resolve_headline_model_specs(pair_resolver=lambda: legacy_pair)
    assert specs == []
    assert gate["blocked_reason"] == "blocked_missing_sota_cache"
    artifact = mod.build_artifact(
        corpus_rows=[],
        model_specs=specs,
        gate_receipt=gate,
        device_receipt={},
        sampled_pairs=[],
        panel_result=None,
    )
    assert artifact["panel_complete"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["legacy_smoke_models_used"] == []
    assert artifact["honest_verdict"].startswith("blocked_missing_sota_cache")
    mod.validate_artifact(artifact)

    model_specs = _fake_model_specs(tmp_path)
    offload_blocked = mod.build_artifact(
        corpus_rows=[],
        model_specs=model_specs,
        gate_receipt=_complete_gate(model_specs) | {"offload_gate_passed": False},
        device_receipt={"gpu_offload_authenticated": False, "devices": []},
        sampled_pairs=[],
        panel_result=None,
    )
    assert offload_blocked["panel_complete"] is False
    assert offload_blocked["gpu_offload_authenticated"] is False
    assert offload_blocked["honest_verdict"].startswith("blocked_no_cuda_offload")
    mod.validate_artifact(offload_blocked)


def test_req_verify_5567_parsers_and_exact_validation_are_fail_closed() -> None:
    """REQ-VERIFY-5567: structured parsing and exact solve validation fail closed."""

    assert mod.parse_verifier_response('{"verdict": "valid"}', "discrete_verdict") == (
        "valid",
        "",
    )
    assert mod.parse_verifier_response(
        '{"criteria": {"schema": true, "constraints": true}}',
        "criteria_decomposition",
    ) == ("valid", "")
    assert mod.parse_verifier_response('{"score": 49}', "granular_score") == ("invalid", "")
    assert mod.parse_verifier_response('```json\n{"score": 81}\n```', "granular_score") == (
        "valid",
        "",
    )
    assert mod.parse_verifier_response("not-json", "discrete_verdict")[0] is None

    rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    pair = mod.sample_independent_pairs(rows, n=4)[0]
    solve = mod.parse_and_score_solve_response(
        json.dumps(
            {
                "candidate_kind": pair["valid_row"]["candidate_kind"],
                "candidate": pair["valid_row"]["candidate"],
            }
        ),
        pair,
    )
    assert solve["parser_ok"] is True
    assert solve["exact_accepted"] is True

    bad = mod.parse_and_score_solve_response("not-json", pair)
    assert bad["parser_ok"] is False
    assert bad["exact_accepted"] is False
    assert bad["error_type"] == "solve_json_parse_failure"


def test_req_verify_5567_statistics_do_not_pool_repeated_calls() -> None:
    """REQ-VERIFY-5567: repeated labels aggregate per candidate before statistics."""

    rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    pairs = mod.sample_independent_pairs(rows, n=4)
    records: list[dict[str, object]] = []
    for pair in pairs:
        for key in ("valid_row", "invalid_row"):
            row = pair[key]
            assert isinstance(row, dict)
            records.append(
                {
                    "model_hf_id": QWEN,
                    "instance_id": pair["instance_id"],
                    "candidate_id": row["row_id"],
                    "family": pair["family"],
                    "arm": "repeated_verdict_3x",
                    "true_label": row["label"],
                    "predicted_label": row["label"],
                    "parser_ok": True,
                    "response_hashes": ["a", "b", "c"],
                    "repeat_labels": [row["label"], row["label"], "invalid"],
                }
            )

    metrics = mod.compute_verifier_metrics(records, [QWEN], ["repeated_verdict_3x"])
    repeated = metrics[QWEN]["repeated_verdict_3x"]
    assert repeated["n_candidates"] == 8
    assert repeated["n_repeated_calls"] == 24
    assert repeated["repeat_calls_per_candidate"] == 3
    assert repeated["balanced_accuracy"] == pytest.approx(1.0)

    result = mod.mcnemar_exact([True, True, False, False], [False, True, True, True])
    assert result["b_solve_correct_verify_wrong"] == 1
    assert result["c_solve_wrong_verify_correct"] == 2
    assert result["p_value_exact"] == pytest.approx(1.0)


def test_req_verify_5567_defensive_branches_and_blocked_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-5567: defensive branches preserve blocked and parser evidence."""

    model_specs = _fake_model_specs(tmp_path)
    resolved, gate = mod.resolve_headline_model_specs(pair_resolver=lambda: model_specs)
    assert [row["hf_id"] for row in resolved] == [QWEN, GEMMA26]
    assert gate["cache_gate_passed"] is True
    failed, failed_gate = mod.resolve_headline_model_specs(
        pair_resolver=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert failed == []
    assert "RuntimeError" in failed_gate["resolver_error"]
    assert mod.model_family("local/other") == "other"

    missing_root = tmp_path / "missing-root"
    assert mod.load_corpus_rows(missing_root) == []
    ready_root = tmp_path / "ready-root"
    ready_path = ready_root / corpus5566.RESULT_RELATIVE_PATH
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(
        json.dumps({"corpus_ready": True, "corpus_rows": [{"row_id": "r0"}]}),
        encoding="utf-8",
    )
    assert mod.load_corpus_rows(ready_root) == [{"row_id": "r0"}]
    unready = tmp_path / corpus5566.RESULT_RELATIVE_PATH
    unready.parent.mkdir(parents=True, exist_ok=True)
    unready.write_text(json.dumps({"corpus_ready": False, "corpus_rows": []}), encoding="utf-8")
    assert mod.load_corpus_rows(tmp_path) == []
    assert (
        mod.sample_independent_pairs(
            [{"label": "invalid", "accepted_by_exact_validator": False, "parent_row_id": "missing"}]
        )
        == []
    )

    assert mod.extract_json_object("[]") == (None, "json_not_object")
    assert mod.extract_json_object('prefix {"a": "{kept}"} suffix') == ({"a": "{kept}"}, "")
    assert mod.extract_json_object('prefix {"a": "quote: \\" ok"} suffix') == (
        {"a": 'quote: " ok'},
        "",
    )
    assert mod.extract_json_object('prefix {"a": } suffix') == (None, "json_parse_failure")
    assert mod.extract_json_object('prefix {"a": 1') == (None, "json_parse_failure")
    assert mod.parse_verifier_response('{"score": "bad"}', "granular_score") == (
        None,
        "verifier_missing_score",
    )
    assert mod.parse_verifier_response('{"score": 101}', "granular_score") == (
        None,
        "verifier_score_out_of_range",
    )
    assert mod.parse_verifier_response('{"criteria": []}', "criteria_decomposition") == (
        None,
        "verifier_missing_label",
    )

    rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    pair = mod.sample_independent_pairs(rows, n=4)[0]
    missing_candidate = mod.parse_and_score_solve_response('{"candidate_kind":"asp_row"}', pair)
    assert missing_candidate["error_type"] == "solve_missing_candidate"
    unknown_kind = mod.parse_and_score_solve_response(
        json.dumps({"candidate_kind": "unknown", "candidate": {}}),
        pair,
    )
    assert str(unknown_kind["error_type"]).startswith("solve_exact_validation_error")

    blocked_corpus = mod.build_artifact(
        corpus_rows=[],
        model_specs=model_specs,
        gate_receipt=_complete_gate(model_specs),
        device_receipt=_complete_device(),
        sampled_pairs=[],
        panel_result=None,
    )
    assert blocked_corpus["honest_verdict"] == "blocked_corpus_unready"
    assert blocked_corpus["panel_complete"] is False

    parser_failure_records = [
        {
            "model_hf_id": QWEN,
            "instance_id": "i0",
            "candidate_id": "c0",
            "arm": "discrete_verdict",
            "true_label": "valid",
            "predicted_label": None,
            "parser_ok": False,
            "response_hashes": [],
            "error_type": "verifier_json_parse_failure",
        },
        {
            "model_hf_id": QWEN,
            "instance_id": "i1",
            "candidate_id": "c1",
            "arm": "discrete_verdict",
            "true_label": "invalid",
            "predicted_label": "valid",
            "parser_ok": True,
            "response_hashes": ["h"],
            "error_type": "",
        },
    ]
    metrics = mod.compute_verifier_metrics(parser_failure_records, [QWEN], ["discrete_verdict"])
    assert metrics[QWEN]["discrete_verdict"]["fn"] == 1
    assert metrics[QWEN]["discrete_verdict"]["fp"] == 1
    assert metrics[QWEN]["discrete_verdict"]["parser_failures"] == 1
    taxonomy = mod._error_taxonomy(
        [{"parser_ok": False, "error_type": "solve_json_parse_failure"}],
        [{"parser_ok": False, "error_type": "verifier_json_parse_failure"}],
    )
    assert taxonomy["solve_json_parse_failure"] == 1
    assert taxonomy["verifier_json_parse_failure"] == 1
    assert taxonomy["parser_failure"] == 2

    assert mod.mcnemar_exact([True, False], [True, False])["p_value_exact"] == pytest.approx(1.0)
    assert mod.honest_verdict(False, "") == "blocked_no_live_panel"
    assert mod._mean([]) == pytest.approx(0.0)
    assert mod._majority_label([]) is None
    assert mod._majority_label(["valid", "invalid"]) is None
    assert mod._majority_label(["invalid", "invalid", "valid"]) == "invalid"
    assert mod._interval([], 7)["n_bootstrap"] == 7
    assert mod._model_specs_have_qwen_and_gemma("bad") is False
    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")
