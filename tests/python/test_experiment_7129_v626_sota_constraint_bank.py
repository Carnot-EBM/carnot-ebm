"""Tests for the frozen three-family exact constraint bank.

Spec refs: REQ-VERIFY-7129, SCENARIO-VERIFY-7129-EXACT,
SCENARIO-VERIFY-7129-TRANSPORT, SCENARIO-VERIFY-7129-IDENTITY,
SCENARIO-VERIFY-7129-ROWS, SCENARIO-VERIFY-7129-BLOCKED, and
SCENARIO-VERIFY-7129-COMPLETE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7129_v626_sota_constraint_bank as mod


REPO = Path(__file__).resolve().parents[2]


def _resolved_specs(tmp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = {}
    for model_id in mod.REQUIRED_MODEL_IDS:
        path = tmp_path / f"{model_id.rsplit('/', 1)[-1]}-Q4_K_M.gguf"
        path.write_bytes(model_id.encode("utf-8"))
        paths[model_id] = str(path)
    calls: list[dict[str, Any]] = []

    def pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        indices = kwargs["model_indices"]
        ids = [mod.SOTA_REGISTRY_IDS[index] for index in indices]
        return [
            {
                "name": model_id.rsplit("/", 1)[-1],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": paths[model_id],
            }
            for gpu, model_id in zip(kwargs["gpu_indices"], ids, strict=True)
        ]

    return mod.resolve_model_specs(cached_pair_func=pair), calls


def _answer_text(receipt: dict[str, Any]) -> str:
    if receipt["feasible"] is False:
        return '{"status":"UNSAT"}'
    family = receipt["family"]
    key = {
        "sat_logic": "assignment",
        "graph_coloring": "coloring",
        "bounded_scheduling": "starts",
    }[family]
    return json.dumps({"status": "SAT", key: receipt["witness"]}, sort_keys=True)


def _complete_raw_rows(
    specs: list[dict[str, Any]], fixture: dict[str, Any], *, all_parse_fail: bool = False
) -> list[dict[str, Any]]:
    receipt_by_id = {row["instance_id"]: row for row in fixture["solver_receipt_rows"]}
    rows = []
    for scheduled in mod.build_schedule(specs, fixture):
        receipt = receipt_by_id[scheduled["instance_id"]]
        text = "not-json" if all_parse_fail else _answer_text(receipt)
        rows.append(
            mod.raw_output_row(
                scheduled,
                raw_text=text,
                prompt_tokens=17,
                completion_tokens=9,
                duration_s=0.25,
                terminal_state="complete",
            )
        )
    return rows


def _complete_artifact(tmp_path: Path, *, all_parse_fail: bool = False) -> dict[str, Any]:
    specs, _calls = _resolved_specs(tmp_path)
    fixture = mod.build_frozen_fixture()
    raw_rows = _complete_raw_rows(specs, fixture, all_parse_fail=all_parse_fail)
    identity_rows = [
        {
            "model_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "model_sha256": mod.sha256_file(spec["model_path"]),
            "quantization": "Q4_K_M",
            "chat_template_source": "embedded_gguf",
            "chat_template_hash": "sha256:" + "a" * 64,
            "identity_matches": True,
        }
        for spec in specs
    ]
    preconditions = {
        "all_passed": True,
        "checks": [mod.gate_row("all_test_resources", True, True, True)],
    }
    artifact = mod.build_artifact(
        run_date="20260908",
        duration_s=61.0,
        model_specs=specs,
        preconditions=preconditions,
        fixture=fixture,
        raw_rows=raw_rows,
        model_identity_rows=identity_rows,
        raw_trace_manifest=[
            {
                "model_id": spec["hf_id"],
                "path": f"/tmp/{index}.jsonl",
                "sha256": "sha256:" + str(index + 1) * 64,
                "row_count": 36,
            }
            for index, spec in enumerate(specs)
        ],
        gpu_telemetry_rows=[{"phase": "test", "gpu_index": 0, "memory_used_mb": 1}],
        source_artifact_hashes={"test": "sha256:" + "b" * 64},
    )
    assert mod.validate_artifact(artifact) == []
    return artifact


def test_req_verify_7129_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7129 is present before the implementation module."""

    spec = (REPO / "openspec/capabilities/constraint-verification/spec.md").read_text(
        encoding="utf-8"
    )
    for anchor in (
        "REQ-VERIFY-7129",
        "SCENARIO-VERIFY-7129-EXACT",
        "SCENARIO-VERIFY-7129-TRANSPORT",
        "SCENARIO-VERIFY-7129-IDENTITY",
        "SCENARIO-VERIFY-7129-ROWS",
        "SCENARIO-VERIFY-7129-BLOCKED",
        "SCENARIO-VERIFY-7129-COMPLETE",
    ):
        assert anchor in spec


def test_req_verify_7129_resolves_all_paths_through_cached_pair(tmp_path: Path) -> None:
    """REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-IDENTITY."""

    specs, calls = _resolved_specs(tmp_path)

    assert [row["hf_id"] for row in specs] == list(mod.REQUIRED_MODEL_IDS)
    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (1, 0)},
    ]
    assert all(row["resolution_method"] == "cached_sota_pair" for row in specs)
    assert all(row["chat_template_source"] == "embedded_gguf" for row in specs)
    assert mod.model_spec_errors(specs) == []

    changed = deepcopy(specs)
    changed[0]["hf_id"] = "substitute/model"
    changed[1]["quantization"] = "Q2_K"
    changed[2]["model_path"] = ""
    errors = mod.model_spec_errors(changed)
    assert "model_roster_mismatch" in errors
    assert "model_quantization_mismatch:unsloth/gemma-4-31B-it-GGUF" in errors
    assert "model_path_missing:unsloth/gemma-4-26B-A4B-it-GGUF" in errors


def test_req_verify_7129_fixture_has_matched_exact_variants() -> None:
    """REQ-VERIFY-7129 freezes 12 bases and two exact variants per base."""

    fixture = mod.build_frozen_fixture()

    assert len(fixture["base_instance_rows"]) == 12
    assert len(fixture["variant_rows"]) == 24
    assert len(fixture["solver_receipt_rows"]) == 36
    assert mod.fixture_errors(fixture) == []
    assert {row["family"] for row in fixture["base_instance_rows"]} == set(
        mod.CONSTRAINT_FAMILIES
    )
    for family in mod.CONSTRAINT_FAMILIES:
        bases = [row for row in fixture["base_instance_rows"] if row["family"] == family]
        assert len(bases) == 4
        assert len({row["size"] for row in bases}) == 1
        assert len({row["density"] for row in bases}) == 1
        assert {len(row["prompt"]) for row in bases} == {mod.SURFACE_BUDGET_CHARS}
        assert {row["generation_budget_tokens"] for row in bases} == {
            mod.GENERATION_CONFIG["max_tokens"]
        }
    assert all(row["solver_effort_is_model_difficulty"] is False for row in fixture["solver_receipt_rows"])
    assert {row["solver_effort_stratum"] for row in fixture["base_instance_rows"]} == {
        "low",
        "medium",
        "high",
    }


def test_req_verify_7129_wrong_label_relabel_and_paraphrase_drift_fail() -> None:
    """REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-EXACT."""

    fixture = mod.build_frozen_fixture()

    wrong_label = deepcopy(fixture)
    wrong_label["solver_receipt_rows"][0]["feasible"] = not wrong_label[
        "solver_receipt_rows"
    ][0]["feasible"]
    assert "solver_receipt_mismatch:sat-0:canonical" in mod.fixture_errors(wrong_label)

    bad_relabel = deepcopy(fixture)
    relabel = next(row for row in bad_relabel["variant_rows"] if row["variant_kind"] == "relabel")
    relabel["inverse_symbol_map"][next(iter(relabel["inverse_symbol_map"]))] = "not-a-symbol"
    assert any(error.startswith("non_preserving_relabel:") for error in mod.fixture_errors(bad_relabel))

    semantic_drift = deepcopy(fixture)
    paraphrase = next(
        row for row in semantic_drift["variant_rows"] if row["variant_kind"] == "paraphrase"
    )
    paraphrase["formal"]["clauses"][0] = ["A"]
    assert any(error.startswith("semantic_paraphrase_drift:") for error in mod.fixture_errors(semantic_drift))


def test_req_verify_7129_parser_and_exact_instance_scoring() -> None:
    """REQ-VERIFY-7129 parses direct text and keeps exact checks authoritative."""

    fixture = mod.build_frozen_fixture()
    by_id = {row["instance_id"]: row for row in fixture["solver_receipt_rows"]}
    sat = by_id["sat-0:canonical"]
    parsed = mod.parse_model_text(_answer_text(sat), "sat_logic")
    assert parsed["parse_success"] is True
    outcome = mod.verify_direct_answer(sat, parsed["parsed"])
    assert outcome["exact_correct"] is True
    assert outcome["constraint_violation_count"] == 0

    malformed = mod.parse_model_text("prefix only", "sat_logic")
    assert malformed == {"parse_success": False, "parsed": None, "parse_error": "json_object_missing"}
    wrong_shape = mod.parse_model_text('{"status":"SAT","answer_id":3}', "sat_logic")
    assert wrong_shape["parse_success"] is False
    assert wrong_shape["parse_error"] == "finite_answer_id_transport_forbidden"
    wrong = mod.verify_direct_answer(sat, {"status": "SAT", "assignment": {}})
    assert wrong["exact_correct"] is False
    assert wrong["constraint_violation_count"] > 0


def test_req_verify_7129_complete_poor_bank_is_positive_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-COMPLETE."""

    artifact = _complete_artifact(tmp_path, all_parse_fail=True)

    assert artifact["planned_cell_count"] == 108
    assert artifact["completed_cell_count"] == 108
    assert artifact["sota_constraint_bank_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive_")
    assert all(row["accuracy"] == 0.0 for row in artifact["family_rows"])
    assert all(row["parse_rate"] == 0.0 for row in artifact["family_rows"])
    assert artifact["finite_answer_id_transport_used"] is False
    assert artifact["schema_constraintir_reprompt_used"] is False


def test_req_verify_7129_transport_substitution_aggregate_and_partial_fail(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7129 transport, identity, row, and partial scenarios."""

    complete = _complete_artifact(tmp_path)

    transported = deepcopy(complete)
    transported["finite_answer_id_transport_used"] = True
    transported["reproducibility_checksum"] = mod.artifact_checksum(transported)
    assert "finite_answer_id_transport_used" in mod.validate_artifact(transported)

    reprompted = deepcopy(complete)
    reprompted["schema_constraintir_reprompt_used"] = True
    reprompted["reproducibility_checksum"] = mod.artifact_checksum(reprompted)
    assert "schema_constraintir_reprompt_used" in mod.validate_artifact(reprompted)

    substituted = deepcopy(complete)
    substituted["model_output_rows"][0]["model_id"] = "substitute/model"
    substituted["reproducibility_checksum"] = mod.artifact_checksum(substituted)
    assert "model_output_projection_mismatch" in mod.validate_artifact(substituted)

    aggregate_only = deepcopy(complete)
    for key in ("rows", "model_output_rows", "parse_rows", "exact_outcome_rows"):
        aggregate_only[key] = []
    aggregate_only["completed_cell_count"] = 108
    aggregate_only["reproducibility_checksum"] = mod.artifact_checksum(aggregate_only)
    assert "aggregate_only_or_cell_key_mismatch" in mod.validate_artifact(aggregate_only)

    fixture = {
        "base_instance_rows": complete["base_instance_rows"],
        "variant_rows": complete["variant_rows"],
        "solver_receipt_rows": complete["solver_receipt_rows"],
    }
    fixture["fixture_hash"] = mod.sha256_text(mod.canonical_json(fixture))
    raw_partial = complete["rows"][:-1]
    counts = {
        model_id: sum(row["model_id"] == model_id for row in raw_partial)
        for model_id in mod.REQUIRED_MODEL_IDS
    }
    partial = mod.build_artifact(
        run_date="20260908",
        duration_s=61.0,
        model_specs=complete["MODEL_SPECS"],
        preconditions=complete["preconditions_checked"],
        fixture=fixture,
        raw_rows=raw_partial,
        model_identity_rows=complete["model_identity_confound_rows"],
        raw_trace_manifest=[
            {
                "model_id": model_id,
                "path": f"/tmp/{index}.jsonl",
                "sha256": "sha256:" + str(index + 1) * 64,
                "row_count": counts[model_id],
            }
            for index, model_id in enumerate(mod.REQUIRED_MODEL_IDS)
        ],
    )
    assert mod.validate_artifact(partial) == []


def test_req_verify_7129_blocked_artifact_is_schema_complete(tmp_path: Path) -> None:
    """REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-BLOCKED."""

    specs, _calls = _resolved_specs(tmp_path)
    checks = [mod.gate_row("idle_rtx_3090_count", 2, 1, False)]
    artifact = mod.build_artifact(
        run_date="20260908",
        duration_s=0.1,
        model_specs=specs,
        preconditions={"all_passed": False, "checks": checks},
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "idle_rtx_3090_count"
    assert artifact["gate_check_summary"]["expected_value"] == 2
    assert artifact["gate_check_summary"]["observed_value"] == 1
    assert mod.validate_artifact(artifact) == []


def test_req_verify_7129_raw_rows_persist_and_resume_without_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7129 persists each direct invocation before parsing."""

    specs, _calls = _resolved_specs(tmp_path)
    fixture = mod.build_frozen_fixture()
    scheduled = mod.build_schedule(specs, fixture)[0]
    raw = mod.raw_output_row(
        scheduled,
        raw_text="not-json",
        prompt_tokens=2,
        completion_tokens=1,
        duration_s=0.01,
        terminal_state="complete",
    )
    path = tmp_path / "raw.jsonl"

    assert mod.persist_raw_row(path, raw)["written"] is True
    assert mod.persist_raw_row(path, raw)["written"] is False
    assert mod.load_raw_rows(path) == [raw]
    changed = deepcopy(raw)
    changed["raw_text"] = "changed"
    with pytest.raises(ValueError, match="raw_row_mismatch"):
        mod.persist_raw_row(path, changed)

    path.write_text(path.read_text(encoding="utf-8") + json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_cell_key"):
        mod.load_raw_rows(path)


def test_req_verify_7129_checksum_and_projection_mutations_fail(tmp_path: Path) -> None:
    """REQ-VERIFY-7129 cold validation rejects aggregate and checksum drift."""

    artifact = _complete_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["rows"][0]["raw_output_hash"] = "sha256:" + "0" * 64
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "combined_row_projection_mismatch" in mod.validate_artifact(changed)

    checksum = deepcopy(artifact)
    checksum["duration_s"] += 1
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(checksum)


def test_req_verify_7129_defensive_fixture_and_parser_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7129 rejects malformed formal, surface, and direct-text inputs."""

    specs, _calls = _resolved_specs(tmp_path)
    invalid_specs = deepcopy(specs)
    invalid_specs[0]["model_path"] = str(tmp_path / "mmproj-Q4_K_M.gguf")
    invalid_specs[0]["chat_template_source"] = "manual"
    invalid_specs[0]["resolution_method"] = "direct"
    invalid_specs[0]["gpu_indices"] = [0]
    errors = mod.model_spec_errors(invalid_specs)
    assert f"model_path_not_language_gguf:{mod.REQUIRED_MODEL_IDS[0]}" in errors
    assert f"chat_template_source_mismatch:{mod.REQUIRED_MODEL_IDS[0]}" in errors
    assert f"model_resolution_method_mismatch:{mod.REQUIRED_MODEL_IDS[0]}" in errors
    assert f"model_execution_policy_mismatch:{mod.REQUIRED_MODEL_IDS[0]}" in errors

    monkeypatch.setattr(mod, "SURFACE_BUDGET_CHARS", 1)
    with pytest.raises(ValueError, match="surface_budget_exceeded"):
        mod._padded_prompt("sat_logic", mod._sat_formals()[0], paraphrase=False)
    monkeypatch.setattr(mod, "SURFACE_BUDGET_CHARS", 1_200)
    with pytest.raises(ValueError, match="unknown_constraint_family"):
        mod.solve_formal("unknown", {})

    schedule_formal = mod._schedule_formals()[0]
    assert mod._schedule_violations(schedule_formal, {}) > 0
    graph_formal = mod._graph_formals()[0]
    assert mod._graph_violations(graph_formal, {}) > 0

    assert mod._contains_answer_id([{"choice_id": 2}]) is True
    invalid_json = mod.parse_model_text("{bad} then no object", "sat_logic")
    assert invalid_json["parse_error"] == "json_object_missing"
    assert mod.parse_model_text('{"status":"MAYBE"}', "sat_logic")["parse_error"] == "status_invalid"
    assert (
        mod.parse_model_text('{"status":"SAT","assignment":{}}', "unknown")["parse_error"]
        == "direct_assignment_shape_invalid"
    )

    fixture = mod.build_frozen_fixture()
    receipts = {row["instance_id"]: row for row in fixture["solver_receipt_rows"]}
    assert mod.verify_direct_answer(receipts["sat-0:canonical"], {"status": "MAYBE"})[
        "exact_correct"
    ] is False
    graph_wrong = {
        "status": "SAT",
        "coloring": {node: 1 for node in receipts["graph-0:canonical"]["formal"]["nodes"]},
    }
    assert mod.verify_direct_answer(receipts["graph-0:canonical"], graph_wrong)[
        "constraint_violation_count"
    ] > 0
    schedule_wrong = {
        "status": "SAT",
        "starts": {job: 0 for job in receipts["schedule-0:canonical"]["formal"]["jobs"]},
    }
    assert mod.verify_direct_answer(receipts["schedule-0:canonical"], schedule_wrong)[
        "constraint_violation_count"
    ] > 0

    raw_path = tmp_path / "bad-hash.jsonl"
    raw_path.write_text(
        json.dumps({"cell_key": "one", "raw_text": "x", "raw_output_hash": "wrong"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw_output_hash_mismatch"):
        mod.load_raw_rows(raw_path)
    assert mod.load_raw_rows(tmp_path / "absent.jsonl") == []
    assert mod._response_fields({}) == ("", "", 0, 0)
    assert mod._response_fields(
        {
            "choices": [{"message": {"content": "x", "reasoning": "r"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
    ) == ("x", "r", 3, 4)


def test_req_verify_7129_fixture_mutation_matrix() -> None:
    """SCENARIO-VERIFY-7129-EXACT covers every stored proof and surface gate."""

    fixture = mod.build_frozen_fixture()

    count_drift = deepcopy(fixture)
    count_drift["base_instance_rows"].append(deepcopy(count_drift["base_instance_rows"][0]))
    count_drift["variant_rows"].append(deepcopy(count_drift["variant_rows"][0]))
    count_drift["solver_receipt_rows"].append(deepcopy(count_drift["solver_receipt_rows"][0]))
    count_errors = mod.fixture_errors(count_drift)
    assert "base_instance_count_or_family_mismatch" in count_errors
    assert "variant_count_mismatch" in count_errors
    assert "solver_receipt_count_mismatch" in count_errors

    def errors_after(change: Any) -> list[str]:
        changed = deepcopy(fixture)
        change(changed)
        changed["fixture_hash"] = mod.sha256_text(
            mod.canonical_json({key: value for key, value in changed.items() if key != "fixture_hash"})
        )
        return mod.fixture_errors(changed)

    assert any(
        error.startswith("base_label_mismatch:")
        for error in errors_after(
            lambda value: value["base_instance_rows"][0].__setitem__("objective", 99)
        )
    )
    assert any(
        error.startswith("base_surface_budget_mismatch:")
        for error in errors_after(
            lambda value: value["base_instance_rows"][0].__setitem__("prompt", "short")
        )
    )
    assert any(
        error.startswith("base_prompt_hash_mismatch:")
        for error in errors_after(
            lambda value: value["base_instance_rows"][0].__setitem__("prompt_hash", "wrong")
        )
    )
    paraphrase_index = next(
        index
        for index, row in enumerate(fixture["variant_rows"])
        if row["variant_kind"] == "paraphrase"
    )
    assert any(
        error.startswith("paraphrase_recheck_missing:")
        for error in errors_after(
            lambda value: value["variant_rows"][paraphrase_index].__setitem__(
                "independent_exact_recheck", False
            )
        )
    )
    assert any(
        error.startswith("unknown_variant_kind:")
        for error in errors_after(
            lambda value: value["variant_rows"][paraphrase_index].__setitem__(
                "variant_kind", "unknown"
            )
        )
    )
    assert any(
        error.startswith("variant_exact_semantics_mismatch:")
        for error in errors_after(
            lambda value: value["variant_rows"][0].__setitem__("proof_preserving", False)
        )
    )
    assert any(
        error.startswith("variant_surface_budget_mismatch:")
        for error in errors_after(
            lambda value: value["variant_rows"][0].__setitem__("prompt", "short")
        )
    )
    assert any(
        error.startswith("variant_prompt_hash_mismatch:")
        for error in errors_after(
            lambda value: value["variant_rows"][0].__setitem__("prompt_hash", "wrong")
        )
    )
    hash_drift = deepcopy(fixture)
    hash_drift["fixture_hash"] = "wrong"
    assert "fixture_hash_mismatch" in mod.fixture_errors(hash_drift)


def test_req_verify_7129_validator_defensive_mutation_matrix(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7129-ROWS recomputes every terminal and row guard."""

    complete = _complete_artifact(tmp_path)

    def validation_errors(change: Any) -> list[str]:
        changed = deepcopy(complete)
        change(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed)

    missing = deepcopy(complete)
    missing.pop("rows")
    assert mod.validate_artifact(missing)[0].startswith("required_fields_missing:")
    assert "field_principles_incomplete" in validation_errors(
        lambda value: value["field_principles"].__setitem__("rows", "")
    )
    assert "venue_or_oracle_mismatch" in validation_errors(
        lambda value: value.__setitem__("execution_venue", "remote")
    )
    assert "verdict_class_invalid" in validation_errors(
        lambda value: value.__setitem__("verdict_class", "mystery")
    )
    assert "honest_verdict_prefix_mismatch" in validation_errors(
        lambda value: value.__setitem__("honest_verdict", "blocked_wrong")
    )
    assert "planned_cell_count_mismatch" in validation_errors(
        lambda value: value.__setitem__("planned_cell_count", 107)
    )
    assert "live_substrate_class_mismatch" in validation_errors(
        lambda value: value.__setitem__("inference_substrate_class", "blocked_no_run")
    )

    blocked = mod.build_artifact(
        run_date="20260908",
        duration_s=0.1,
        model_specs=complete["MODEL_SPECS"],
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("blocked_test", True, False, False)],
        },
    )

    def blocked_errors(change: Any) -> list[str]:
        changed = deepcopy(blocked)
        change(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed)

    assert "model_roster_mismatch" in blocked_errors(
        lambda value: value["MODEL_SPECS"].pop()
    )
    assert "blocked_substrate_class_mismatch" in blocked_errors(
        lambda value: value.__setitem__("inference_substrate_class", "model_bounded_generation")
    )
    assert "blocked_gate_detail_missing" in blocked_errors(
        lambda value: value.__setitem__("gate_check_summary", {})
    )
    assert "blocked_ready_score_nonzero" in blocked_errors(
        lambda value: value.__setitem__("sota_constraint_bank_ready_score", 1)
    )

    def row_change(field: str, value: Any) -> list[str]:
        changed = deepcopy(complete)
        changed["rows"][0][field] = value
        changed["model_output_rows"][0][field] = value
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed)

    assert "proposal_source_not_direct_text" in row_change("proposal_source", "answer_menu")
    assert "row_answer_id_transport_used" in row_change("finite_answer_id_transport_used", True)
    assert "row_constraintir_reprompt_used" in row_change("schema_constraintir_reprompt_used", True)
    assert "raw_not_persisted_before_parse" in row_change("raw_persisted_before_parse", False)
    assert "raw_output_hash_mismatch" in row_change("raw_text", "changed without new hash")
    assert "model_substitution" in row_change("model_id", "substitute/model")

    duplicate = deepcopy(complete)
    duplicate["rows"].append(deepcopy(duplicate["rows"][0]))
    duplicate["model_output_rows"].append(deepcopy(duplicate["model_output_rows"][0]))
    duplicate["parse_rows"].append(deepcopy(duplicate["parse_rows"][0]))
    duplicate["exact_outcome_rows"].append(deepcopy(duplicate["exact_outcome_rows"][0]))
    duplicate["reproducibility_checksum"] = mod.artifact_checksum(duplicate)
    assert "aggregate_only_or_cell_key_mismatch" in mod.validate_artifact(duplicate)

    ready_score = validation_errors(
        lambda value: value.__setitem__("sota_constraint_bank_ready_score", 0)
    )
    assert "ready_score_mismatch" in ready_score

    wrong_complete_class = deepcopy(complete)
    wrong_complete_class["verdict_class"] = "partial"
    wrong_complete_class["honest_verdict"] = "partial_complete_but_wrong_class"
    wrong_complete_class["reproducibility_checksum"] = mod.artifact_checksum(wrong_complete_class)
    assert "complete_verdict_not_positive" in mod.validate_artifact(wrong_complete_class)

    fixture = {
        "base_instance_rows": complete["base_instance_rows"],
        "variant_rows": complete["variant_rows"],
        "solver_receipt_rows": complete["solver_receipt_rows"],
    }
    fixture["fixture_hash"] = mod.sha256_text(mod.canonical_json(fixture))
    partial_raw = complete["rows"][:-1]
    counts = {
        model_id: sum(row["model_id"] == model_id for row in partial_raw)
        for model_id in mod.REQUIRED_MODEL_IDS
    }
    partial = mod.build_artifact(
        run_date="20260908",
        duration_s=61.0,
        model_specs=complete["MODEL_SPECS"],
        preconditions=complete["preconditions_checked"],
        fixture=fixture,
        raw_rows=partial_raw,
        model_identity_rows=complete["model_identity_confound_rows"],
        raw_trace_manifest=[
            {
                "model_id": model_id,
                "path": f"/tmp/{index}.jsonl",
                "sha256": "sha256:" + str(index + 1) * 64,
                "row_count": counts[model_id],
            }
            for index, model_id in enumerate(mod.REQUIRED_MODEL_IDS)
        ],
    )
    assert mod.validate_artifact(partial) == []

    incomplete_positive = deepcopy(partial)
    incomplete_positive["verdict_class"] = "positive"
    incomplete_positive["honest_verdict"] = "positive_incomplete_wrong_class"
    incomplete_positive["reproducibility_checksum"] = mod.artifact_checksum(incomplete_positive)
    assert "incomplete_verdict_not_partial" in mod.validate_artifact(incomplete_positive)

    bad_partial_gate = deepcopy(partial)
    bad_partial_gate["gate_check_summary"] = mod.gate_summary(
        [mod.gate_row("different", True, False, False)]
    )
    bad_partial_gate["reproducibility_checksum"] = mod.artifact_checksum(bad_partial_gate)
    assert "partial_gate_detail_mismatch" in mod.validate_artifact(bad_partial_gate)
