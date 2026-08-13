"""Tests for Exp6379 canonical factor-edit transport.

Spec refs: REQ-INFRA-6379, SCENARIO-INFRA-6379-1,
SCENARIO-INFRA-6379-2, SCENARIO-INFRA-6379-3,
SCENARIO-INFRA-6379-4, SCENARIO-INFRA-6379-5.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6379_canonical_factor_edit_transport_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str, text: str) -> dict[str, Any]:
    assert path.endswith(".gguf")
    tokens = max(1, len(text.encode("utf-8")) // 6)
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "token_count": tokens,
        "tokenizer_detail": f"fixture tokenizer counted {tokens} tokens",
        "autotokenizer_used": False,
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        schema_path=tmp_path / (mod.RESULT_RELATIVE_PATH.name + ".canonical_schema.json"),
        exp6366_path=REPO / mod.EXP6366_RELATIVE_PATH,
        data_dir=REPO / mod.EXP6366_DATA_RELATIVE_PATH,
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def test_req_infra_6379_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-INFRA-6379: OpenSpec owns the transport contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6379") :]
    for token in (
        "SCENARIO-INFRA-6379-1",
        "SCENARIO-INFRA-6379-2",
        "SCENARIO-INFRA-6379-3",
        "SCENARIO-INFRA-6379-4",
        "SCENARIO-INFRA-6379-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_infra_6379_freezes_and_labels_exp6366_failures() -> None:
    """SCENARIO-INFRA-6379-1: labels are applied after raw hashes exist."""

    upstream = mod.upstream_exp6366_receipt(REPO / mod.EXP6366_RELATIVE_PATH)
    failures = mod.frozen_raw_failure_receipts(
        REPO / mod.EXP6366_RELATIVE_PATH,
        REPO / mod.EXP6366_DATA_RELATIVE_PATH,
    )

    assert upstream["terminal_class"] == "transport_null"
    assert upstream["zero_parse_valid_objects"] is True
    assert set(failures) == set(mod.MANDATED_MODEL_IDS)
    assert failures[mod.MANDATED_MODEL_IDS[0]]["freeze_before_classification"] is True
    assert "thinking_leakage" in failures[mod.MANDATED_MODEL_IDS[0]]["labels"]
    assert "syntax_failure" in failures[mod.MANDATED_MODEL_IDS[0]]["labels"]
    assert "repetition_collapse" in failures[mod.MANDATED_MODEL_IDS[1]]["labels"]
    assert "syntax_failure" in failures[mod.MANDATED_MODEL_IDS[1]]["labels"]
    assert "truncation" in failures[mod.MANDATED_MODEL_IDS[2]]["labels"]
    assert "structural_failure" in failures[mod.MANDATED_MODEL_IDS[2]]["labels"]


def test_scenario_infra_6379_canonical_object_generates_surfaces() -> None:
    """SCENARIO-INFRA-6379-2: every surface derives from one object."""

    specs = mod.deterministic_model_specs(REPO / "models")
    surfaces = mod.canonical_schema_generated_surfaces(specs)
    schema = mod.canonical_factor_edit_contract()

    assert surfaces["canonical_hash"] == mod.sha256_json(schema)
    assert surfaces["all_surfaces_from_canonical"] is True
    assert surfaces["duplicate_handwritten_surface_count"] == 0
    assert surfaces["field_order"] == mod.CANONICAL_FIELD_ORDER
    assert surfaces["validator_field_list"] == mod.validator_field_list(schema)
    assert surfaces["output_example"] == mod.compact_output_example(schema, specs[0])
    assert "hidden chain" not in surfaces["prompt_fragment"].lower()
    assert "chain of thought" not in json.dumps(surfaces, sort_keys=True).lower()
    assert surfaces["source_binding_checks"] == mod.source_binding_checks(schema)


def test_scenario_infra_6379_mutation_matrix_fails_closed() -> None:
    """SCENARIO-INFRA-6379-3: drift and malformed output are rejected."""

    specs = mod.deterministic_model_specs(REPO / "models")
    schema = mod.canonical_factor_edit_contract()
    example = mod.compact_output_example(schema, specs[0])
    matrix = mod.deterministic_transport_mutation_matrix(schema, specs[0])

    expected = {
        "prompt_schema_conflict",
        "stale_example",
        "missing_fixed_fields",
        "reordered_fields",
        "thinking_prefix",
        "markdown",
        "repeated_tokens",
        "mid_object_truncation",
        "unsupported_source_spans",
        "parse_valid_semantic_corruption",
    }
    assert {row["attack"] for row in matrix["rows"]} == expected
    assert all(row["accepted"] is False for row in matrix["rows"])
    assert all(row["fail_closed"] is True for row in matrix["rows"])
    assert matrix["all_attacks_fail_closed"] is True

    valid_text = mod.canonical_json(example)
    accepted = mod.validate_transport_output(valid_text, schema, specs[0])
    assert accepted["accepted"] is True
    assert accepted["failure_labels"] == []

    repeated = mod.validate_transport_output("own " * 81, schema, specs[0])
    assert repeated["accepted"] is False
    assert "repetition_collapse" in repeated["failure_labels"]
    assert repeated["decision"] == "abstain"


def test_scenario_infra_6379_tokenizer_capacity_is_vocab_only(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6379-4: lower bounds use embedded GGUF tokenizers."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
    )
    surfaces = mod.canonical_schema_generated_surfaces(resolution["MODEL_SPECS"])
    capacity = mod.per_model_minimum_output_tokens_and_capacity_margins(
        resolution["MODEL_SPECS"],
        surfaces,
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert resolution["all_resolved"] is True
    assert [row["hf_id"] for row in resolution["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert set(capacity["by_model"]) == set(mod.MANDATED_MODEL_IDS)
    for receipt in capacity["by_model"].values():
        assert receipt["tokenizer_method"] == mod.TOKENIZER_METHOD
        assert receipt["autotokenizer_used"] is False
        assert receipt["minimum_serialized_output_tokens"] > 0
        assert receipt["required_completion_lower_bound"] > mod.OLD_COMPLETION_BUDGET
        assert receipt["n_ctx_margin"] > 0
    assert capacity["all_three_tokenizer_capacity_receipts_exist"] is True


def test_req_infra_6379_artifact_ready_schema_and_no_generation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6379-5: readiness is transport-only and deterministic."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["canonical_factor_transport_contract_ready_score"] == 1.0
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["live_autoregressive_generation_invoked"] is False
    assert artifact["retired_decoding_mechanism_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_model_quality_or_utility_claim"] is True
    assert artifact["bounded_evidence_summary_variant"]["included_in_json_object"] is True
    assert artifact["prompt_schema_drift_checks"]["all_drift_checks_fail_closed"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])

    source = inspect.getsource(mod)
    for retired in ("outlines", "guidance", "lmql", "grammar_decoder", "parser_retry"):
        assert retired not in source
    assert "from_pretrained" not in source

    bad = deepcopy(artifact)
    bad["canonical_schema_generated_surfaces"]["all_surfaces_from_canonical"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["canonical_factor_transport_contract_ready_score"] == 0.0
    assert bad["status"] == "complete_null"


def test_req_infra_6379_defensive_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-INFRA-6379: helper edges do not promote incomplete evidence."""

    schema = mod.canonical_factor_edit_contract()
    specs = mod.deterministic_model_specs(tmp_path)
    text = mod.canonical_json(mod.compact_output_example(schema, specs[0]))

    missing = mod.build_model_specs(cached_pair_func=lambda **_: None)
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]
    assert "cached_sota_pair_dense_missing" in missing["blocked_reasons"]

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.revision_from_path(Path("/cache/snapshots/rev/model.gguf")) == "rev"
    assert mod.quantization_from_path(Path("model-without-quant.gguf")) == "unknown"
    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.classify_raw_failure("{\"a\": 1") == [
        "truncation",
        "syntax_failure",
        "structural_failure",
    ]

    corrupted = json.loads(text)
    corrupted["evidence_summary"] = "I used hidden chain data."
    receipt = mod.validate_transport_output(mod.canonical_json(corrupted), schema, specs[0])
    assert receipt["accepted"] is False
    assert "semantic_failure" in receipt["failure_labels"]

    bad_artifact = _artifact(tmp_path / "bad")
    bad_artifact["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    mod.refresh_terminal_fields(bad_artifact)
    assert bad_artifact["canonical_factor_transport_contract_ready_score"] == 0.0


def test_req_infra_6379_validation_and_precondition_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6379: uncovered edge branches remain fail-closed."""

    schema = mod.canonical_factor_edit_contract()
    specs = mod.deterministic_model_specs(tmp_path)
    example = mod.compact_output_example(schema, specs[0])

    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert "expected_failure" in str(exc)
    else:
        raise AssertionError("require accepted a false condition")

    assert mod.write_payload_or_hash(tmp_path / "dry.json", {"x": 1}, write=False) == mod.sha256_json(
        {"x": 1}
    )
    assert mod.classify_raw_failure("[1, 2]") == ["structural_failure"]
    assert mod.classify_raw_failure("{\"a\":1}") == ["unknown"]
    assert "structural_failure" in mod.validate_transport_output("[1]", schema, specs[0])[
        "failure_labels"
    ]
    try:
        mod._mutated_text("unknown", example, schema)
    except ValueError as exc:
        assert "unknown_attack" in str(exc)
    else:
        raise AssertionError("unknown mutation was accepted")

    def rejected(payload: dict[str, Any], reason: str) -> dict[str, Any]:
        receipt = mod.validate_transport_output(mod.canonical_json(payload), schema, specs[0])
        assert receipt["accepted"] is False
        assert reason in receipt["reasons"]
        return receipt

    payload = deepcopy(example)
    payload["model_family"] = "wrong"
    rejected(payload, "model_family_mismatch")

    payload = deepcopy(example)
    payload["proposal_id"] = "wrong"
    rejected(payload, "proposal_id_mismatch")

    payload = deepcopy(example)
    payload["hidden_state"] = "blocked"
    receipt = mod.validate_transport_output(json.dumps(payload), schema, specs[0])
    assert "forbidden_fields:hidden_state" in receipt["reasons"]

    payload = deepcopy(example)
    payload["evidence_summary"] = ""
    rejected(payload, "evidence_summary_missing_or_not_string")

    payload = deepcopy(example)
    payload["evidence_summary"] = "x" * (mod.EVIDENCE_SUMMARY_MAX_CHARS + 1)
    rejected(payload, "evidence_summary_too_long")

    payload = deepcopy(example)
    payload["edits"] = {"wrong": 0.5}
    rejected(payload, "edits_not_single_allowed_variable")

    payload = deepcopy(example)
    payload["edits"]["accept_bias"] = "bad"
    rejected(payload, "edit_value_not_number")

    payload = deepcopy(example)
    payload["edits"]["accept_bias"] = 2.0
    rejected(payload, "edit_value_out_of_bounds")

    payload = deepcopy(example)
    payload["selection_score"] = "bad"
    rejected(payload, "selection_score_not_number")

    payload = deepcopy(example)
    payload["selection_score"] = 2.0
    rejected(payload, "selection_score_out_of_bounds")

    payload = deepcopy(example)
    payload["obligations"] = []
    rejected(payload, "obligations_not_singleton")

    payload = deepcopy(example)
    payload["obligations"][0]["source_start"] = 0
    rejected(payload, "unsupported_source_span:obligation")

    data_dir = tmp_path / "exp6366-data"
    raw_dir = data_dir / "sidecars"
    prompt_dir = data_dir / "prompts"
    raw_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    raw = raw_dir / f"{mod.model_slug(mod.MANDATED_MODEL_IDS[0])}.stdout.txt"
    prompt = prompt_dir / f"{mod.model_slug(mod.MANDATED_MODEL_IDS[0])}.prompt.json"
    raw.write_text("own " * 80, encoding="utf-8")
    prompt.write_text("{}", encoding="utf-8")
    exp6366_fixture = tmp_path / "exp6366.json"
    exp6366_fixture.write_text(
        json.dumps(
            {
                "raw_output_before_parse_paths_hashes_and_counts": {
                    "by_model": {
                        mod.MANDATED_MODEL_IDS[0]: {
                            "path": str(tmp_path / "missing.raw"),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fallback = mod.frozen_raw_failure_receipts(exp6366_fixture, data_dir)
    assert fallback[mod.MANDATED_MODEL_IDS[0]]["raw_stdout"]["path"] == str(raw)
    assert "repetition_collapse" in fallback[mod.MANDATED_MODEL_IDS[0]]["labels"]
    assert mod.raw_sidecar_path(data_dir, mod.MANDATED_MODEL_IDS[0]) == raw

    preconditions = mod.preconditions_checked(
        date="20260813",
        upstream={"terminal_class": "unknown"},
        raw_failures={},
        model_resolution={"blocked_reasons": ["missing"], "all_resolved": False},
        capacity={"all_three_tokenizer_capacity_receipts_exist": False},
        surfaces={"all_surfaces_from_canonical": False},
        protected_before={"x": None},
    )
    for reason in (
        "missing",
        "exp6366_not_transport_null",
        "raw_failure_receipts_missing",
        "tokenizer_capacity_receipts_missing",
        "canonical_surface_generation_failed",
        "protected_hash_missing",
    ):
        assert reason in preconditions["blocked_reasons"]

    incomplete_raw = {
        model_id: {"freeze_before_classification": model_id != mod.MANDATED_MODEL_IDS[0]}
        for model_id in mod.MANDATED_MODEL_IDS
    }
    preconditions = mod.preconditions_checked(
        date="20260813",
        upstream={"terminal_class": "transport_null"},
        raw_failures=incomplete_raw,
        model_resolution={"blocked_reasons": [], "all_resolved": True},
        capacity={"all_three_tokenizer_capacity_receipts_exist": True},
        surfaces={"all_surfaces_from_canonical": True},
        protected_before={"x": "sha256:ok"},
    )
    assert "raw_failure_freeze_incomplete" in preconditions["blocked_reasons"]
