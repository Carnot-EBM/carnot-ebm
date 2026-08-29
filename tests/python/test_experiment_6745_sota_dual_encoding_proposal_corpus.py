"""Focused tests for the Exp6745 dual-encoding proposal corpus.

Spec refs: REQ-VERIFY-6745 and SCENARIO-VERIFY-6745-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6745_sota_dual_encoding_proposal_corpus as exp
from carnot.verify import dual_certificate_encoder_a as encoder_a
from carnot.verify import dual_certificate_encoder_b as encoder_b


SAT_CNF = {"n_vars": 2, "clauses": [[1], [-2, 1]]}
UNSAT_CNF = {"n_vars": 1, "clauses": [[1], [-1]]}


def _source(row_id: str = "source-1", cnf: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_sha256": f"sha256:{row_id}",
        "pair_id": "pair-1",
        "pair_role": "base",
        "family": "fixture",
        "size_bin": "small",
        "split": "train",
        "seed": 7,
        "cnf": deepcopy(cnf or SAT_CNF),
    }


def _model(family_id: str = "qwen") -> dict[str, Any]:
    return {
        "family_id": family_id,
        "hf_id": f"unsloth/{family_id}-GGUF",
        "role": "flagship_moe",
        "model_path": f"/models/{family_id}.gguf",
        "model_sha256": f"sha256:{family_id}",
        "resolved": True,
        "headline_eligible": True,
    }


def _generation(raw: str, **overrides: Any) -> dict[str, Any]:
    generation = {
        "raw_output": raw,
        "raw_api_response_sha256": "sha256:api",
        "prompt_tokens": 20,
        "generated_tokens": 4,
        "latency_s": 0.25,
        "started_monotonic_ns": 10,
        "finished_monotonic_ns": 20,
        "http_status": 200,
        "finish_reason": "stop",
        "failure_kind": None,
    }
    generation.update(overrides)
    return generation


def _receipt(family_id: str = "qwen") -> dict[str, Any]:
    return {
        "model_family_id": family_id,
        "cuda_offload": True,
        "accelerator_observed": True,
        "memory_recovered": True,
        "authentic": True,
    }


def test_scenario_verify_6745_dsl_parses_certificates_and_abstention() -> None:
    """SCENARIO-VERIFY-6745-DSL accepts only the frozen small DSL."""

    sat = exp.parse_certificate_dsl("SAT x1=1 x2=0")
    unsat = exp.parse_certificate_dsl("UNSAT c1,c2")
    abstain = exp.parse_certificate_dsl("ABSTAIN")
    malformed = exp.parse_certificate_dsl("SAT x1=yes")

    assert sat == {
        "parser_status": "parseable",
        "claim": "SAT",
        "terms": ["x1=1", "x2=0"],
        "parse_failure": None,
        "abstention": False,
    }
    assert unsat["parser_status"] == "parseable"
    assert unsat["terms"] == ["c1", "c2"]
    assert abstain["parser_status"] == "abstention"
    assert abstain["abstention"] is True
    assert malformed["parser_status"] == "malformed"
    assert malformed["claim"] is None


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        ("", "empty_output"),
        ("SAT", "missing_terms"),
        ("UNSAT", "missing_terms"),
        ("MAYBE x1=1", "unknown_claim"),
        ("ABSTAIN later", "abstention_has_terms"),
        ("```SAT x1=1```", "code_fence_not_allowed"),
        ("SAT x0=1", "invalid_sat_term"),
        ("UNSAT c0", "invalid_unsat_term"),
    ],
)
def test_scenario_verify_6745_dsl_rejects_malformed_text(raw: str, failure: str) -> None:
    """SCENARIO-VERIFY-6745-DSL reports a specific syntax failure."""

    parsed = exp.parse_certificate_dsl(raw)

    assert parsed["parser_status"] == "malformed"
    assert parsed["parse_failure"] == failure
    assert parsed["terms"] == []


def test_scenario_verify_6745_independent_encoders_normalize_equally() -> None:
    """SCENARIO-VERIFY-6745-DUAL compares semantic constraints, not text."""

    parsed = exp.parse_certificate_dsl("SAT x2=0 x1=1 x1=1")

    encoded_a = encoder_a.encode_certificate(parsed)
    encoded_b = encoder_b.encode_certificate(parsed)

    assert encoded_a["normalized_constraints"] == encoded_b["normalized_constraints"]
    assert encoded_a["normalized_constraints"] == {
        "claim": "SAT",
        "bindings": [
            {"variable": 1, "values": [True]},
            {"variable": 2, "values": [False]},
        ],
        "core_clause_indices": [],
    }
    assert encoded_a["encoder_id"] != encoded_b["encoder_id"]


def test_scenario_verify_6745_independent_encoders_cover_unsat_and_fail_closed() -> None:
    """SCENARIO-VERIFY-6745-DUAL keeps both encoder implementations defensive."""

    parsed = exp.parse_certificate_dsl("UNSAT c2,c1,c2")

    assert encoder_a.encode_certificate(parsed)["normalized_constraints"][
        "core_clause_indices"
    ] == [1, 2]
    assert encoder_b.encode_certificate(parsed)["normalized_constraints"][
        "core_clause_indices"
    ] == [1, 2]
    with pytest.raises(ValueError, match="not parseable"):
        encoder_a.encode_certificate({"parser_status": "malformed"})
    with pytest.raises(ValueError, match="not parseable"):
        encoder_b.encode_certificate({"parser_status": "malformed"})
    with pytest.raises(ValueError, match="invalid SAT term"):
        encoder_a.encode_certificate(
            {"parser_status": "parseable", "claim": "SAT", "terms": ["bad"]}
        )
    with pytest.raises(ValueError, match="invalid UNSAT term"):
        encoder_a.encode_certificate(
            {"parser_status": "parseable", "claim": "UNSAT", "terms": ["bad"]}
        )
    with pytest.raises(ValueError, match="unsupported claim"):
        encoder_a.encode_certificate(
            {"parser_status": "parseable", "claim": "OTHER", "terms": ["c1"]}
        )
    with pytest.raises(ValueError, match="does not start"):
        encoder_b.encode_certificate(
            {"parser_status": "parseable", "claim": "UNSAT", "terms": ["x1"]}
        )
    with pytest.raises(ValueError, match="positive integer"):
        encoder_b.encode_certificate(
            {"parser_status": "parseable", "claim": "UNSAT", "terms": ["c0"]}
        )
    with pytest.raises(ValueError, match="invalid SAT term"):
        encoder_b.encode_certificate(
            {"parser_status": "parseable", "claim": "SAT", "terms": ["x1=1=0"]}
        )
    with pytest.raises(ValueError, match="invalid SAT value"):
        encoder_b.encode_certificate(
            {"parser_status": "parseable", "claim": "SAT", "terms": ["x1=2"]}
        )
    with pytest.raises(ValueError, match="unsupported claim"):
        encoder_b.encode_certificate(
            {"parser_status": "parseable", "claim": "OTHER", "terms": ["c1"]}
        )


def test_scenario_verify_6745_encoder_disagreement_has_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6745-DUAL labels injected translation disagreement."""

    original = encoder_b.encode_certificate

    def disagree(parsed: dict[str, Any]) -> dict[str, Any]:
        encoded = original(parsed)
        encoded["normalized_constraints"]["bindings"][0]["values"] = [False]
        return encoded

    monkeypatch.setattr(encoder_b, "encode_certificate", disagree)
    row = exp.build_attempt_row(
        _source(),
        _model(),
        _generation("SAT x1=1 x2=0"),
        _receipt(),
        generation_seed=101,
    )

    assert row["diagnosis"] == "translation_disagreement"
    assert row["encoder_agreement"] is False
    assert row["encoder_a"]["exact_check"]["attempted"] is True
    assert row["encoder_b"]["exact_check"]["attempted"] is True


def test_scenario_verify_6745_exact_checker_owns_sat_and_unsat() -> None:
    """SCENARIO-VERIFY-6745-EXACT derives validity only from CNF semantics."""

    sat = encoder_a.encode_certificate(exp.parse_certificate_dsl("SAT x1=1 x2=0"))
    wrong_sat = encoder_a.encode_certificate(exp.parse_certificate_dsl("SAT x1=0 x2=0"))
    unsat = encoder_b.encode_certificate(exp.parse_certificate_dsl("UNSAT c1,c2"))
    weak_core = encoder_b.encode_certificate(exp.parse_certificate_dsl("UNSAT c1"))

    assert exp.exact_check_constraints(SAT_CNF, sat["normalized_constraints"])["valid"] is True
    assert (
        exp.exact_check_constraints(SAT_CNF, wrong_sat["normalized_constraints"])["valid"] is False
    )
    assert exp.exact_check_constraints(UNSAT_CNF, unsat["normalized_constraints"])["valid"] is True
    assert (
        exp.exact_check_constraints(UNSAT_CNF, weak_core["normalized_constraints"])["valid"]
        is False
    )


def test_scenario_verify_6745_exact_checker_rejects_bad_constraint_shapes() -> None:
    """REQ-VERIFY-6745 exact checks fail closed on incomplete or invalid evidence."""

    conflicting = {
        "claim": "SAT",
        "bindings": [{"variable": 1, "values": [False, True]}],
        "core_clause_indices": [],
    }
    bad_core = {
        "claim": "UNSAT",
        "bindings": [],
        "core_clause_indices": [3],
    }

    assert exp.exact_check_constraints(SAT_CNF, conflicting)["reason"] == "assignment_incomplete"
    assert exp.exact_check_constraints(UNSAT_CNF, bad_core)["reason"] == "core_index_out_of_range"
    assert (
        exp.exact_check_constraints(
            SAT_CNF,
            {
                "claim": "SAT",
                "bindings": [
                    {"variable": 1, "values": [1]},
                    {"variable": 2, "values": [False]},
                ],
            },
        )["reason"]
        == "assignment_value_invalid"
    )
    assert (
        exp.exact_check_constraints(
            SAT_CNF,
            {"claim": "SAT", "bindings": [{"variable": 1, "values": [True]}]},
        )["reason"]
        == "assignment_incomplete"
    )
    assert exp.exact_check_constraints(UNSAT_CNF, {"claim": "UNSAT"})["reason"] == "empty_core"
    assert (
        exp.exact_check_constraints(UNSAT_CNF, {"claim": "OTHER"})["reason"] == "unsupported_claim"
    )


def test_scenario_verify_6745_timeout_is_retained() -> None:
    """SCENARIO-VERIFY-6745-RETENTION keeps one complete failed-attempt row."""

    row = exp.build_attempt_row(
        _source(),
        _model(),
        _generation(
            "",
            http_status=124,
            finish_reason="request_timeout",
            failure_kind="TimeoutError: request exceeded 120 seconds",
            latency_s=120.0,
            generated_tokens=0,
        ),
        _receipt(),
        generation_seed=202,
    )

    assert row["diagnosis"] == "malformed_certificate"
    assert row["timed_out"] is True
    assert row["raw_output_sha256"] == exp.sha256_text("")
    assert row["generation_failure_kind"].startswith("TimeoutError")
    assert row["decode_budget"] == exp.DECODE_CONFIG
    assert row["encoder_a"]["attempted"] is False
    assert row["encoder_b"]["attempted"] is False


def test_scenario_verify_6745_runner_bytes_are_decoded_before_parsing() -> None:
    """SCENARIO-VERIFY-6745-RETENTION parses runner bytes without adding a bytes repr."""

    row = exp.build_attempt_row(
        _source(),
        _model(),
        _generation(b"SAT x1=1 x2=0"),
        _receipt(),
        generation_seed=203,
    )

    assert row["raw_output"] == "SAT x1=1 x2=0"
    assert row["parser_status"] == "parseable"
    assert row["diagnosis"] == "exact_valid"


def test_req_verify_6745_row_diagnoses_exact_invalid_and_abstention() -> None:
    """REQ-VERIFY-6745 assigns one closed diagnosis to each diagnosable row."""

    valid = exp.build_attempt_row(
        _source("valid"), _model(), _generation("SAT x1=1 x2=0"), _receipt(), 1
    )
    invalid = exp.build_attempt_row(
        _source("invalid"), _model(), _generation("SAT x1=0 x2=0"), _receipt(), 2
    )
    abstain = exp.build_attempt_row(
        _source("abstain"), _model(), _generation("ABSTAIN"), _receipt(), 3
    )

    assert valid["diagnosis"] == "exact_valid"
    assert invalid["diagnosis"] == "reasoning_error"
    assert abstain["diagnosis"] == "abstention"
    assert all(row["diagnosis"] in exp.DIAGNOSES for row in (valid, invalid, abstain))


def test_req_verify_6745_exact_authority_exception_blocks_without_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6745 does not guess a taxonomy label when exact authority fails."""

    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("authority unavailable")

    monkeypatch.setattr(exp, "exact_check_constraints", unavailable)
    row = exp.build_attempt_row(_source(), _model(), _generation("SAT x1=1 x2=0"), _receipt(), 9)

    assert row["row_blocked"] is True
    assert row["diagnosis"] is None
    assert row["encoder_a"]["exact_check"]["authority_available"] is False
    assert row["encoder_b"]["error"].startswith("RuntimeError")


def test_scenario_verify_6745_completeness_ignores_accuracy() -> None:
    """SCENARIO-VERIFY-6745-COMPLETENESS uses complete attempts, not accuracy."""

    sources = [_source("source-1"), _source("source-2")]
    models = [_model("qwen"), _model("dense"), _model("middle")]
    rows = [
        exp.build_attempt_row(
            source,
            model,
            _generation("not a certificate"),
            _receipt(model["family_id"]),
            generation_seed=index,
        )
        for index, (model, source) in enumerate(
            [(model, source) for model in models for source in sources], start=1
        )
    ]

    reduction = exp.recompute_aggregates(
        rows, sources, models, [_receipt(m["family_id"]) for m in models]
    )

    assert reduction["dual_encoding_corpus_ready"] is True
    assert reduction["planned_row_count"] == 6
    assert reduction["diagnosis_counts"]["malformed_certificate"] == 6
    assert sum(item["exact_valid"] for item in reduction["exact_success_by_model"].values()) == 0


def test_req_verify_6745_missing_or_duplicate_rows_fail_completeness() -> None:
    """REQ-VERIFY-6745 requires each frozen model-instance row exactly once."""

    sources = [_source()]
    models = [_model("qwen"), _model("dense"), _model("middle")]
    rows = [
        exp.build_attempt_row(
            sources[0], model, _generation("ABSTAIN"), _receipt(model["family_id"]), index
        )
        for index, model in enumerate(models, start=1)
    ]

    assert (
        exp.recompute_aggregates(
            rows[:-1], sources, models, [_receipt(m["family_id"]) for m in models]
        )["dual_encoding_corpus_ready"]
        is False
    )
    assert (
        exp.recompute_aggregates(
            rows + [deepcopy(rows[0])], sources, models, [_receipt(m["family_id"]) for m in models]
        )["dual_encoding_corpus_ready"]
        is False
    )


def test_req_verify_6745_encoder_matrix_covers_all_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6745 derives each agreement cell from encoder and checker receipts."""

    model = _model()
    receipt = _receipt()
    rows = [
        exp.build_attempt_row(_source("valid"), model, _generation("SAT x1=1 x2=0"), receipt, 1),
        exp.build_attempt_row(_source("invalid"), model, _generation("SAT x1=0 x2=0"), receipt, 2),
    ]
    original = encoder_b.encode_certificate

    def disagree(parsed: dict[str, Any]) -> dict[str, Any]:
        result = original(parsed)
        result["normalized_constraints"]["bindings"][0]["values"] = [False]
        return result

    monkeypatch.setattr(encoder_b, "encode_certificate", disagree)
    rows.append(
        exp.build_attempt_row(_source("disagree"), model, _generation("SAT x1=1 x2=0"), receipt, 3)
    )
    mixed = deepcopy(rows[0])
    mixed["row_id"] = "qwen|mixed"
    mixed["source_row_id"] = "mixed"
    mixed["encoder_b"]["exact_check"]["valid"] = False
    rows.append(mixed)
    sources = [_source(name) for name in ("valid", "invalid", "disagree", "mixed")]

    matrix = exp.recompute_aggregates(rows, sources, [model], [receipt])["encoder_agreement_matrix"]

    assert matrix == {
        "agree_both_valid": 1,
        "agree_both_invalid": 1,
        "agree_mixed_exact_outcome": 1,
        "disagree": 1,
        "not_applicable": 0,
    }


def test_req_verify_6745_manifest_freezes_prompt_config_and_stream() -> None:
    """REQ-VERIFY-6745 freezes all inference inputs before model output."""

    stream = {
        "hardness_stream_ready": True,
        "deterministic_replay_receipt": {"first_stream_sha256": "sha256:stream"},
        "rows": [_source()],
    }

    manifest = exp.build_frozen_manifest(stream)

    assert manifest["stream_checksum"] == "sha256:stream"
    assert manifest["decode_config"] == exp.DECODE_CONFIG
    assert manifest["planned_row_count"] == 3
    assert manifest["instances"][0]["prompt_sha256"] == exp.sha256_text(
        manifest["instances"][0]["prompt"]
    )
    assert manifest["manifest_sha256"] == exp.manifest_checksum(manifest)


def test_req_verify_6745_artifact_schema_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-6745 keeps required fields, principles, and row-derived totals aligned."""

    source = _source()
    models = [_model("qwen"), _model("dense"), _model("middle")]
    receipts = [_receipt(model["family_id"]) for model in models]
    rows = [
        exp.build_attempt_row(source, model, _generation("ABSTAIN"), receipt, index)
        for index, (model, receipt) in enumerate(zip(models, receipts, strict=True), start=1)
    ]
    manifest = {
        "stream_checksum": "sha256:stream",
        "decode_config": deepcopy(exp.DECODE_CONFIG),
        "instances": [source],
        "planned_row_count": 3,
        "manifest_sha256": "sha256:manifest",
    }
    artifact = exp.build_artifact(
        date="20260829",
        duration_s=61.0,
        manifest=manifest,
        models=models,
        rows=rows,
        gpu_receipts=receipts,
        preconditions={"all_passed": True, "checks": []},
    )

    assert exp.validate_artifact(artifact) == []
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["dual_encoding_corpus_ready"] is True
    path = tmp_path / "artifact.json"
    exp.write_json_atomic(path, artifact)
    assert json.loads(path.read_text()) == artifact

    mutated = deepcopy(artifact)
    mutated["diagnosis_counts"]["abstention"] = 0
    assert "aggregate_recomputation_mismatch" in exp.validate_artifact(mutated)

    invalid_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="aggregate_recomputation_mismatch"):
        exp.write_json_atomic(invalid_path, mutated)


def test_req_verify_6745_hash_and_validation_failure_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-6745 content hashes and schema validation fail closed."""

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"certificate bytes")
    assert exp.sha256_file(payload).startswith("sha256:")
    assert exp.validate_artifact({}) == [
        "missing_required_fields:" + ",".join(sorted(exp.ARTIFACT_FIELDS))
    ]

    artifact = exp.build_blocked_artifact(
        date="20260829",
        duration_s=1.0,
        failed_check="fixture",
        expected=True,
        observed=False,
        models=[],
        manifest={"planned_row_count": 3},
        preconditions={},
    )
    artifact["field_principles"].pop("title")
    artifact["inference_substrate"] = "wrong"
    artifact["verdict_class"] = "wrong"
    artifact["honest_verdict"] = "wrong"
    artifact["rows"] = [{}]
    assert set(exp.validate_artifact(artifact)) == {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "verdict_class_invalid",
        "reproducibility_checksum_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_verdict_prefix_mismatch",
        "blocked_rows_or_gate_invalid",
    }


def test_req_verify_6745_blocked_artifact_is_complete() -> None:
    """REQ-VERIFY-6745 emits the full schema when a precondition fails."""

    artifact = exp.build_blocked_artifact(
        date="20260829",
        duration_s=0.5,
        failed_check="model_cache",
        expected=True,
        observed=False,
        models=[_model()],
        manifest={"planned_row_count": 216, "manifest_sha256": "sha256:manifest"},
        preconditions={"all_passed": False, "checks": []},
    )

    assert artifact["honest_verdict"].startswith("complete_blocked_proposal_corpus")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["planned_row_count"] == 216
    assert artifact["rows"] == []
    assert artifact["dual_encoding_corpus_ready"] is False
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6745_spec_anchor_and_model_policy() -> None:
    """REQ-VERIFY-6745 is present and names all three headline model families."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md").read_text()

    assert "REQ-VERIFY-6745" in spec
    assert "SCENARIO-VERIFY-6745-COMPLETENESS" in spec
    assert [model["hf_id"] for model in exp.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]


def test_req_verify_6745_cli_and_failed_precondition_helpers() -> None:
    """REQ-VERIFY-6745 keeps the planning date and blocked check attributable."""

    assert exp.parse_args([]).date == "20260829"
    assert exp.parse_args(["--date", "20260901"]).date == "20260901"
    assert exp._first_failed(  # noqa: SLF001
        {"checks": [{"check": "cache", "passed": False, "observed": "missing"}]}
    ) == ("cache", "missing")
    assert exp._first_failed({"all_passed": True, "checks": []}) == (  # noqa: SLF001
        "preconditions",
        True,
    )
