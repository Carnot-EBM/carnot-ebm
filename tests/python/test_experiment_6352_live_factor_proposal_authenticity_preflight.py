"""Tests for Exp6352 live factor proposal authenticity preflight.

Spec refs: REQ-LEARN-6352, SCENARIO-LEARN-6352-PREFLIGHT,
SCENARIO-LEARN-6352-GENERATION, SCENARIO-LEARN-6352-EVENTS,
SCENARIO-LEARN-6352-ISOLATION, SCENARIO-LEARN-6352-READY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_6352_live_factor_proposal_authenticity_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".Q4_K_M.gguf")
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


def _fake_generation(
    *,
    spec: dict[str, Any],
    event: dict[str, Any],
    raw_path: Path,
    prompt_payload: dict[str, Any],
    seed: int,
    sampling: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    variable = event["allowed_variables"][0]
    proposal = {
        "schema": mod.FACTOR_EDIT_SCHEMA,
        "proposal_id": f"{mod.model_slug(spec['hf_id'])}:{event['event_id']}:0",
        "event_id": event["event_id"],
        "model_hf_id": spec["hf_id"],
        "arm": mod.LIVE_ARM,
        "candidate_index": 0,
        "changed_factor": event["changed_factor"],
        "edits": {variable: 0.5},
        "selection_score": 0.5,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(json.dumps(proposal, sort_keys=True).encode("utf-8"))
    raw_hash = mod.sha256_file(raw_path)
    return {
        "hf_id": spec["hf_id"],
        "model_family": spec["model_family"],
        "event_id": event["event_id"],
        "raw_output_path": str(raw_path),
        "raw_output_sha256": raw_hash,
        "raw_output_bytes": raw_path.stat().st_size,
        "pid": 12345,
        "command_path": "fake-llama-cpp-python",
        "argv_sha256": mod.sha256_json({"hf_id": spec["hf_id"], "seed": seed}),
        "seed": seed,
        "sampling": sampling,
        "timeout_s": timeout_s,
        "exit_state": {"returncode": 0, "timed_out": False},
        "token_counts": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36},
        "timing": {"started_ns": 10, "raw_written_ns": 20, "ended_ns": 30, "duration_s": 0.02},
        "cuda": {"gpu": spec["gpu"], "n_gpu_layers": -1, "main_gpu": spec["gpu"]},
        "prompt_sha256": mod.sha256_json(prompt_payload),
        "live_autoregressive_generation_invoked": True,
    }


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _fake_model_paths(tmp_path / "models")
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
        generation_func=_fake_generation,
        write=write,
    )


def _read_json(receipt: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def test_req_learn_6352_spec_declares_contract_and_principles() -> None:
    """REQ-LEARN-6352: OpenSpec owns fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6352") :]
    for token in (
        "SCENARIO-LEARN-6352-PREFLIGHT",
        "SCENARIO-LEARN-6352-GENERATION",
        "SCENARIO-LEARN-6352-EVENTS",
        "SCENARIO-LEARN-6352-ISOLATION",
        "SCENARIO-LEARN-6352-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in " ".join(section.split())


def test_scenario_learn_6352_model_specs_use_cached_pair_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6352-PREFLIGHT: model rows use required GGUF helpers."""

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
    assert mod.AUTOTOKENIZER_USAGE_COUNT == 0

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "not checked"),
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_missing" in missing["blocked_reasons"]


def test_scenario_learn_6352_event_manifest_is_fresh_balanced_and_unlabelled(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6352-EVENTS: families, structures, and surfaces balance."""

    artifact = _artifact(tmp_path)
    manifest = _read_json(artifact["generated_event_manifest_path_and_hash"])
    balance = artifact["event_family_structure_and_surface_balance"]
    exposed = json.dumps(manifest["events_exposed_to_proposer"], sort_keys=True)

    assert manifest["fresh_for_exp6352"] is True
    assert manifest["event_count"] == len(mod.generated_events())
    assert balance["family_count"] >= 2
    assert balance["balanced"] is True
    assert all(count == 2 for count in balance["events_by_family"].values())
    assert "protected_outcome" not in exposed
    assert "exact_label" not in exposed
    assert "hidden_state" not in exposed


def test_scenario_learn_6352_generation_raw_before_parse_and_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6352-GENERATION: live outputs are frozen before parsing."""

    artifact = _artifact(tmp_path)

    assert artifact["live_factor_proposal_authenticity_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["live_autoregressive_generation_invoked"] is True
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    for field in (
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
        "protected_validation_leak_count",
    ):
        assert type(artifact[field]) is int
        assert artifact[field] == 0

    raw = artifact["raw_model_output_paths_hashes_and_counts"]
    before_parse = artifact["raw_output_before_parse_receipts"]
    parse_counts = artifact["parse_valid_invalid_and_timeout_counts_by_model"]
    assert raw["model_count"] == len(mod.MANDATED_MODEL_IDS)
    assert before_parse["all_raw_outputs_frozen_before_parse"] is True
    for model_id in mod.MANDATED_MODEL_IDS:
        assert raw["by_model"][model_id]["raw_output_count"] == 1
        assert parse_counts["by_model"][model_id]["valid"] == 1
        assert parse_counts["by_model"][model_id]["invalid"] == 0
        assert parse_counts["by_model"][model_id]["timeouts"] == 0


def test_scenario_learn_6352_isolation_laundering_and_oracle_boundary(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6352-ISOLATION: attacks fail closed after raw freeze."""

    artifact = _artifact(tmp_path)
    isolation = artifact["same_step_read_write_isolation_results"]
    laundering = artifact["deterministic_replay_laundering_checks"]
    checker = artifact["exact_checker_calls_time_cost_and_error_table"]

    assert isolation["same_step_read_after_write_attempted"] is True
    assert isolation["proposal_read_root_unchanged"] is True
    assert isolation["unapproved_write_visible_to_same_step"] is False
    assert all(row["fail_closed"] is True for row in laundering["checks"])
    assert checker["exact_checker_calls"] == len(mod.MANDATED_MODEL_IDS)
    assert checker["checker_error_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_checker_paths_hashes_and_versions"]["proposal_quality_oracle"] is False


def test_req_learn_6352_artifact_schema_fail_closed_and_checksum(tmp_path: Path) -> None:
    """REQ-LEARN-6352: required fields, checksum, and blocked path validate."""

    artifact = _artifact(tmp_path)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["protected_validation_leak_count"] == 0
    mod.validate_artifact(artifact)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blocked = mod.run(
        date="20260812",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "not checked"),
        host_checks_func=mod.deterministic_host_receipts,
        generation_func=_fake_generation,
        write=False,
    )
    assert blocked["live_factor_proposal_authenticity_ready_score"] == 0.0
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["live_autoregressive_generation_invoked"] is False
    assert blocked["models_used"] == []


def test_req_learn_6352_defensive_helpers_and_receipts(tmp_path: Path) -> None:
    """REQ-LEARN-6352: helper edge cases stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert str(exc) == "expected_failure"
    assert mod.quantization_from_path(Path("model-without-token.gguf")) == "unknown"
    assert mod.terminal_class("blocked_precondition", "") == "terminal_blocked"
    assert mod.terminal_class("complete_null", "") == "terminal_null"
    assert mod.terminal_class("other", "") == "terminal_unknown"
    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}

    rows = mod.parse_gpu_query("0, Test GPU, 24576, 128, 24448\n")
    assert rows == [
        {"index": 0, "name": "Test GPU", "total_mb": 24576, "used_mb": 128, "free_mb": 24448}
    ]
    summary = mod.harm_summary(
        {"MODEL_SPECS": [{"hf_id": "missing", "exists": False, "tokenizer_loadable": False}]},
        {"model_count": 0},
        {"all_raw_outputs_frozen_before_parse": False},
        {"by_model": {"missing": {"valid": 0, "invalid": 1, "timeouts": 1}}},
    )
    assert summary["missing_model_cells"] == ["missing"]
    assert summary["flagged_cells"] == [
        "raw_before_parse",
        "invalid_parse:missing",
        "timeout:missing",
    ]
    assert summary["harm_detected"] is True

    assert mod.extract_json_payload("prefix {\"a\": 1} suffix") == {"a": 1}
    assert mod.extract_json_payload("no json here") is None
    assert mod.extract_json_payload("{bad}") is None
    assert mod.extract_json_payload("[]") is None

    events = mod.generated_events()
    invalid_path = tmp_path / "invalid.raw"
    invalid_path.write_text("no json", encoding="utf-8")
    timeout_path = tmp_path / "timeout.raw"
    timeout_path.write_text(json.dumps({"event_id": events[0]["event_id"]}), encoding="utf-8")
    parsed = mod.parse_raw_outputs(
        {
            "receipts": {
                mod.MANDATED_MODEL_IDS[0]: {
                    "raw_output_path": str(invalid_path),
                    "raw_output_sha256": mod.sha256_file(invalid_path),
                    "exit_state": {"timed_out": False},
                    "timing": {"raw_written_ns": 1},
                },
                mod.MANDATED_MODEL_IDS[1]: {
                    "raw_output_path": str(timeout_path),
                    "raw_output_sha256": mod.sha256_file(timeout_path),
                    "exit_state": {"timed_out": True},
                    "timing": {"raw_written_ns": 1},
                },
            }
        },
        events,
    )
    assert parsed["counts"]["by_model"][mod.MANDATED_MODEL_IDS[0]]["invalid"] == 1
    assert parsed["counts"]["by_model"][mod.MANDATED_MODEL_IDS[1]]["timeouts"] == 1
    assert mod.status({"preconditions_checked": {"all_preconditions_passed": True}}) == "complete_null"
    assert mod.honest_verdict({"status": "complete_null"}).startswith("complete_null:")
    assert mod.exact_oracle_boundary()["proposal_quality_oracle"] is False
