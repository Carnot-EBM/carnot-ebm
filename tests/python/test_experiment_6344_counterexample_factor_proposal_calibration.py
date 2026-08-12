"""Tests for Exp6344 counterexample factor proposal calibration.

Spec refs: REQ-LEARN-6344, REQ-LEARN-6344-SCHEMA,
REQ-LEARN-6344-ISOLATION, REQ-LEARN-6344-MATCHING,
REQ-LEARN-6344-SINGLE-OPEN, REQ-LEARN-6344-ORACLE-BOUNDARY,
REQ-LEARN-6344-PROVENANCE, SCENARIO-LEARN-6344-LOCALITY,
SCENARIO-LEARN-6344-ISOLATION, SCENARIO-LEARN-6344-MATCHED-BUDGETS,
SCENARIO-LEARN-6344-SINGLE-OPEN, SCENARIO-LEARN-6344-READY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6344_counterexample_factor_proposal_calibration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".gguf")
        path.write_bytes((model_id + "\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _fake_cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def _cached_pair(
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
        if model_indices == (0, 2):
            return [
                {
                    "name": "Qwen3.6-35B-A3B",
                    "hf_id": mod.MANDATED_MODEL_IDS[0],
                    "gpu": gpu_indices[0],
                    "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
                },
                {
                    "name": "Gemma4-31B-it",
                    "hf_id": mod.MANDATED_MODEL_IDS[1],
                    "gpu": gpu_indices[1],
                    "model_path": str(paths[mod.MANDATED_MODEL_IDS[1]]),
                },
            ]
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": mod.MANDATED_MODEL_IDS[0],
                "gpu": gpu_indices[0],
                "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": mod.MANDATED_MODEL_IDS[2],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.MANDATED_MODEL_IDS[2]]),
            },
        ]

    return _cached_pair


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        cached_pair_func=_fake_cached_pair(paths, calls),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
        host_checks_func=mod.deterministic_host_receipts,
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def _read_json(receipt: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def test_req_learn_6344_spec_declares_contract_and_principles() -> None:
    """REQ-LEARN-6344-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6344") :]
    for token in (
        "REQ-LEARN-6344-SCHEMA",
        "REQ-LEARN-6344-ISOLATION",
        "REQ-LEARN-6344-MATCHING",
        "REQ-LEARN-6344-SINGLE-OPEN",
        "REQ-LEARN-6344-ORACLE-BOUNDARY",
        "SCENARIO-LEARN-6344-LOCALITY",
        "SCENARIO-LEARN-6344-ISOLATION",
        "SCENARIO-LEARN-6344-MATCHED-BUDGETS",
        "SCENARIO-LEARN-6344-SINGLE-OPEN",
        "SCENARIO-LEARN-6344-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_6344_model_specs_use_cached_pair_and_embedded_tokenizer(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6344: model specs come from cached GGUF helper calls."""

    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair(paths, calls),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
    )

    assert resolution["all_resolved"] is True
    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolution["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in resolution["MODEL_SPECS"])
    assert all(row["tokenizer_loadable"] is True for row in resolution["MODEL_SPECS"])
    assert all(row["model_path"].endswith(".gguf") for row in resolution["MODEL_SPECS"])
    assert mod.AUTOTOKENIZER_USAGE_COUNT == 0

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "not checked"),
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_missing" in missing["blocked_reasons"]


def test_scenario_learn_6344_counterexample_minimality_and_information_isolation(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6344-ISOLATION: proposer payloads reveal no labels."""

    artifact = _artifact(tmp_path)
    schema = _read_json(artifact["factor_edit_schema_path_and_hash"])
    manifest = _read_json(artifact["development_event_manifest_path_and_hash"])
    minimizer = _read_json(artifact["counterexample_minimizer_path_hash_and_exactness"])
    event = mod.development_events()[0]
    exposed = mod.exposed_event_payload(event)

    assert schema["schema"] == mod.FACTOR_EDIT_SCHEMA
    assert manifest["event_count"] == len(mod.development_events())
    assert artifact["information_exposure_contract"]["protected_outcomes_visible_before_selection"] is False
    assert artifact["protected_validation_leak_count"] == 0
    assert minimizer["all_counterexamples_minimal"] is True
    assert all(row["minimal"] is True and row["exact"] is True for row in minimizer["counterexamples"])
    assert set(exposed) == {
        "event_id",
        "changed_factor",
        "minimized_exact_counterexample",
        "allowed_variables",
        "edit_bounds",
    }
    assert "protected_outcome" not in exposed
    assert "exact_label" not in json.dumps(exposed)


def test_scenario_learn_6344_budget_parity_and_factor_locality(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6344-LOCALITY: invalid edits reject before selection."""

    artifact = _artifact(tmp_path)
    budgets = artifact["matched_call_token_candidate_time_and_checker_budgets"]
    locality = artifact["schema_validity_and_factor_locality_results"]
    event = mod.development_events()[0]
    valid = mod.proposal_record(event, mod.ARMS[-1], 0, mod.MANDATED_MODEL_IDS[0])

    assert budgets["budget_parity"] is True
    first_budget = budgets["by_arm"][mod.ARMS[0]]
    for arm in mod.ARMS:
        assert budgets["by_arm"][arm] == first_budget
    assert locality["all_selected_factor_local"] is True
    assert locality["all_selected_schema_valid"] is True
    assert locality["invalid_fixture_results"]["wrong_factor"]["valid"] is False
    assert locality["invalid_fixture_results"]["forbidden_variable"]["valid"] is False
    assert locality["invalid_fixture_results"]["out_of_bounds"]["valid"] is False

    wrong_factor = {**valid, "changed_factor": "repair_factor"}
    forbidden_variable = {**valid, "edits": {"accept_bias": 0.2, "repair_bias": 0.1}}
    out_of_bounds = {**valid, "edits": {"accept_bias": 9.0}}
    assert mod.validate_proposal(wrong_factor, event, mod.factor_edit_schema())["valid"] is False
    assert mod.validate_proposal(forbidden_variable, event, mod.factor_edit_schema())["valid"] is False
    assert mod.validate_proposal(out_of_bounds, event, mod.factor_edit_schema())["valid"] is False


def test_scenario_learn_6344_single_open_cost_accounting_and_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6344-READY: protected validation opens once after selection."""

    artifact = _artifact(tmp_path)
    protected = artifact["protected_outcome_seal_and_single_open_receipt"]
    exact = artifact["exact_proposal_success_cost_and_movement_by_model_family_arm"]
    deltas = artifact["paired_deltas_intervals_and_sample_sizes"]
    costs = artifact["verification_calls_time_cost_and_error_table"]

    assert protected["open_count"] == 1
    assert protected["opened_after_selection"] is True
    assert protected["protected_visible_before_selection"] is False
    assert artifact["counterexample_proposal_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert costs["all_costs_accounted"] is True
    assert costs["checker_error_count"] == 0
    assert costs["total_checker_calls"] == sum(
        row["exact_checker_calls"] for row in exact["rows"]
    )
    for family in mod.REQUIRED_MODEL_FAMILIES:
        repeated = exact["by_family_arm"][family]["repeated_temperature_sampling"]
        directed = exact["by_family_arm"][family]["counterexample_directed_proposals"]
        delta = deltas["by_family"][family]
        assert directed["success_per_cost"] > repeated["success_per_cost"]
        assert delta["delta_success_per_cost"] > 0
        assert delta["lower"] > 0
        assert delta["n"] == len(mod.development_events())


def test_req_learn_6344_cli_schema_checksum_and_oracle_boundary(tmp_path: Path) -> None:
    """REQ-LEARN-6344-ORACLE-BOUNDARY: CLI writes a valid terminal artifact."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    data = tmp_path / "data"
    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    assert (
        mod.main(
            [
                "--date",
                "20260812",
                "--output",
                str(output),
                "--data-dir",
                str(data),
                "--validate",
            ],
            cached_pair_func=_fake_cached_pair(paths, calls),
            tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
            host_checks_func=mod.deterministic_host_receipts,
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    for field in (
        "protected_validation_leak_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        assert artifact[field] == 0
        assert type(artifact[field]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_oracle_claim_boundary"]["verifier_is_oracle"] is True
    assert artifact["exact_oracle_claim_boundary"]["release_authority"] == "exact_checker_only"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["generated_label_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="generated_label_count"):
        mod.validate_artifact(bad_zero)

    failed_delta = json.loads(json.dumps(artifact))
    failed_delta["paired_deltas_intervals_and_sample_sizes"]["all_required_families_positive"] = False
    _refresh(failed_delta)
    assert failed_delta["counterexample_proposal_ready_score"] == 0.0

    bad_status = json.loads(json.dumps(failed_delta))
    bad_status["status"] = "complete_positive"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6344_helper_error_paths(tmp_path: Path) -> None:
    """REQ-LEARN-6344-SCHEMA: helpers fail closed on malformed inputs."""

    artifact = _artifact(tmp_path, write=False)

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.rounded(1.2345678912349) == 1.234567891235
    assert mod.as_mapping([]) == {}
    assert mod.revision_from_path(Path("/cache/models--x/snapshots/rev123/model.gguf")) == "rev123"
    assert mod.quantization_from_path(Path("model-UD-Q4_K_M.gguf")) == "UD-Q4_K_M"
    assert mod.model_family_for_id("unknown/model") == "unknown"
    assert mod.terminal_class("complete_null", "") == "terminal_null"
    assert mod.terminal_class("blocked", "") == "terminal_blocked"
    assert mod.terminal_class("other", "") == "terminal_unknown"
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")
    with pytest.raises(ValueError, match="unknown_arm"):
        mod.proposal_record(mod.development_events()[0], "bad_arm", 0, mod.MANDATED_MODEL_IDS[0])
    with pytest.raises(ValueError, match="unknown_factor"):
        mod.exact_success({"changed_factor": "missing", "edits": {}})
    with pytest.raises(ValueError, match="unknown_event"):
        mod.exact_success(
            {
                **mod.proposal_record(
                    mod.development_events()[0],
                    mod.ARMS[-1],
                    0,
                    mod.MANDATED_MODEL_IDS[0],
                ),
                "event_id": "missing",
            }
        )

    malformed = json.loads(json.dumps(artifact))
    malformed["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    _refresh(malformed)
    assert malformed["counterexample_proposal_ready_score"] == 0.0

    event = mod.development_events()[0]
    valid = mod.proposal_record(event, mod.ARMS[-1], 0, mod.MANDATED_MODEL_IDS[0])
    assert mod.validate_proposal({}, event, mod.factor_edit_schema())["reason"] == "missing_fields"
    assert mod.validate_proposal({**valid, "edits": {}}, event, mod.factor_edit_schema())["reason"] == "empty_edits"
    assert mod.validate_proposal(
        {**valid, "edits": {"accept_bias": 0.7, "extra_bias": 0.7}},
        event,
        mod.factor_edit_schema(),
    )["reason"] == "variable_locality"
    assert mod.validate_proposal(
        {**valid, "edits": {"accept_bias": 0.7, "protected_outcome": 0.7}},
        event,
        mod.factor_edit_schema(),
    )["reason"] == "variable_locality"
    assert mod.validate_proposal(
        {**valid, "forbidden_probe": True},
        event,
        {**mod.factor_edit_schema(), "forbidden_fields": ["forbidden_probe"]},
    )["reason"] == "forbidden_fields"
    invalid = {**valid, "edits": {"accept_bias": 0.0}}
    invalid_schema = {**valid, "edits": {}}
    assert mod.exact_success(invalid_schema) is False
    assert mod.exact_success(invalid) is False
    big_schema = mod.factor_edit_schema()
    big_schema["edit_bounds"] = {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0}
    two_var_event = {**event, "allowed_variables": ["accept_bias", "repair_bias"]}
    assert mod.validate_proposal(
        {**valid, "edits": {"accept_bias": 0.7, "repair_bias": 0.7}},
        two_var_event,
        big_schema,
    )["reason"] == "movement_bounds"

    harm = mod.harm_summary(
        {"models_used": []},
        {
            "schema_validity_and_factor_locality_results": {"all_selected_factor_local": False},
            "paired_deltas_intervals_and_sample_sizes": {
                "by_family": {mod.REQUIRED_MODEL_FAMILIES[0]: {"n": 0}}
            },
            "verification_calls_time_cost_and_error_table": {"checker_error_count": 1},
        },
    )
    assert harm["harm_detected"] is True
    assert "factor_locality" in harm["flagged_cells"]
    assert "checker_errors" in harm["flagged_cells"]
    assert mod.REQUIRED_MODEL_FAMILIES[0] in harm["underpowered_cells"]

    for field in (
        "schema_validity_and_factor_locality_results",
        "protected_outcome_seal_and_single_open_receipt",
        "verification_calls_time_cost_and_error_table",
        "protected_files_unchanged",
    ):
        damaged = json.loads(json.dumps(artifact))
        damaged[field] = []
        assert mod.ready_score(damaged) == 0.0
