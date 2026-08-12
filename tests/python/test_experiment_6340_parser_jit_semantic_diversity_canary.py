"""Tests for Exp6340 parser-JIT semantic diversity canary.

Spec refs: REQ-KONA-6340, SCENARIO-KONA-6340-GATE-REPLAY,
SCENARIO-KONA-6340-MATCHED-ARMS, SCENARIO-KONA-6340-SEMANTIC-DEDUP,
SCENARIO-KONA-6340-ORACLE-BOUNDARY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6340_parser_jit_semantic_diversity_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6340_parser_jit_semantic_diversity_canary "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-m pytest tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6340_parser_jit_semantic_diversity_canary.py "
    "--fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6340_parser_jit_semantic_diversity_canary.json"
)
TEST_COMMANDS = [
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def test_req_kona_6340_spec_declares_required_canary_contract() -> None:
    """REQ-KONA-6340: the OpenSpec entry declares the full artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6340") :]

    for marker in (
        "SCENARIO-KONA-6340-GATE-REPLAY",
        "SCENARIO-KONA-6340-MATCHED-ARMS",
        "SCENARIO-KONA-6340-SEMANTIC-DEDUP",
        "SCENARIO-KONA-6340-ORACLE-BOUNDARY",
        "results/experiment_6340_parser_jit_semantic_diversity_canary.json",
        "unconstrained sampling, grammar masking, deterministic parser-state",
        "`semantic_diversity_gain_score` SHALL be bare `1.0` only",
    ):
        assert marker in section


def test_scenario_kona_6340_gate_replay_uses_cached_pair_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6340-GATE-REPLAY: model specs come from cached GGUFs."""

    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair(paths, calls),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
    )

    assert resolution["all_resolved"] is True
    assert calls == [
        {"gpu_indices": (0, 1), "model_indices": None},
        {"gpu_indices": (0, 1), "model_indices": (0, 2)},
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


def test_scenario_kona_6340_matched_arm_budgets_and_contract() -> None:
    """SCENARIO-KONA-6340-MATCHED-ARMS: every arm has identical budgets."""

    fixtures = mod.development_fixtures()
    contract = mod.prompt_decoder_and_prefix_contract(fixtures, grammar_receipt={"path": "x", "sha256": "y"})
    budgets = mod.matched_budgets()

    assert {fixture.split for fixture in fixtures} == {"development"}
    assert contract["development_family_count"] == 2
    assert contract["embedded_gguf_tokenizer_only"] is True
    assert contract["hf_tokenizer_loader_forbidden"] is True
    assert contract["predeclared_primary_endpoint"] == "unique_valid_normalized_semantics"
    assert budgets["budget_parity"] is True
    baseline = budgets["by_arm"][mod.ARMS[0]]
    for arm in mod.ARMS:
        assert budgets["by_arm"][arm] == baseline


def test_scenarios_kona_6340_artifact_dedup_fallback_and_oracle_boundary(
    tmp_path: Path,
) -> None:
    """REQ-KONA-6340: the terminal artifact carries every required receipt."""

    artifact = _run_fake_positive(tmp_path)

    assert artifact["status"] == "complete_ready"
    assert artifact["semantic_diversity_gain_score"] == 1.0
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_oracle_claim_boundary"]["model_supplies_safety_authority"] is False
    assert artifact["exact_oracle_claim_boundary"]["exact_compiler_counts_as_model_verification"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    model_id = mod.MANDATED_MODEL_IDS[0]
    family = "access_gate"
    unique = artifact["unique_valid_normalized_semantics_by_model_family_arm"][model_id][family]
    assert unique[mod.PREDECLARED_PREFIX_ARM]["unique_valid_count"] == 2
    assert unique["grammar_masking"]["unique_valid_count"] == 1
    assert unique["unconstrained_sampling"]["unique_valid_count"] == 1

    metrics = artifact["exact_utility_fallback_latency_and_cost_by_model_family_arm"]
    assert metrics[model_id]["incident_response"]["unconstrained_sampling"]["fallback_used"] is True
    assert metrics[model_id][family][mod.PREDECLARED_PREFIX_ARM]["fallback_used"] is False
    assert artifact["parse_normalization_and_contract_results"]["deduplicated_after_normalization"] is True
    assert artifact["verification_calls_time_cost_and_error_table"]["accepted_contract_violation_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    raw_receipts = artifact["raw_generation_paths_hashes_and_counts"][model_id]
    for arm, receipt in raw_receipts.items():
        path = Path(receipt["path"])
        assert path.exists(), arm
        assert mod.sha256_file(path) == receipt["sha256"]

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert artifact["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert artifact["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]


def test_scenario_kona_6340_intervention_logs_preserve_prefix_rejections(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6340-SEMANTIC-DEDUP: rejected prefixes remain visible."""

    artifact = _run_fake_positive(tmp_path)
    logs = artifact["parser_state_and_jit_intervention_logs"]
    rows = logs["rows"]

    assert logs["jit_rejected_prefix_count"] > 0
    assert logs["parser_state_correction_count"] > 0
    assert any(row["action"] == "jit_prefix_reject_and_completion" for row in rows)
    assert any(row["rejected_prefixes"] for row in rows)
    assert all("final_candidate_sha256" in row for row in rows)


def test_scenario_kona_6340_validator_rejects_false_readiness(tmp_path: Path) -> None:
    """SCENARIO-KONA-6340-ORACLE-BOUNDARY: false positive readiness fails."""

    artifact = _run_fake_positive(tmp_path)

    missing = deepcopy(artifact)
    missing.pop("parser_state_and_jit_intervention_logs")
    with pytest.raises(ValueError, match="parser_state_and_jit_intervention_logs"):
        mod.validate_artifact(missing)

    bad_count = deepcopy(artifact)
    bad_count["generated_label_count"] = False
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="generated_label_count"):
        mod.validate_artifact(bad_count)

    bad_budget = deepcopy(artifact)
    bad_budget["matched_call_token_candidate_time_and_checker_budgets"]["budget_parity"] = False
    bad_budget["reproducibility_checksum"] = mod.payload_checksum(bad_budget)
    with pytest.raises(ValueError, match="semantic_diversity_gain_score"):
        mod.validate_artifact(bad_budget)

    bad_models = deepcopy(artifact)
    bad_models["models_used"] = []
    bad_models["reproducibility_checksum"] = mod.payload_checksum(bad_models)
    with pytest.raises(ValueError, match="semantic_diversity_gain_score"):
        mod.validate_artifact(bad_models)

    bad_violation = deepcopy(artifact)
    bad_violation["verification_calls_time_cost_and_error_table"][
        "accepted_contract_violation_count"
    ] = 1
    bad_violation["reproducibility_checksum"] = mod.payload_checksum(bad_violation)
    with pytest.raises(ValueError, match="accepted_contract_violation_count"):
        mod.validate_artifact(bad_violation)

    blocked = deepcopy(artifact)
    blocked["semantic_diversity_paired_deltas_intervals_and_sample_sizes"][
        "all_required_cells_positive"
    ] = False
    blocked["semantic_diversity_gain_score"] = 0.0
    blocked["status"] = "complete_no_value"
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    assert mod.validate_artifact(blocked) is True


def test_req_kona_6340_defensive_and_blocked_paths(tmp_path: Path) -> None:
    """REQ-KONA-6340: blocked and defensive receipts remain explicit."""

    paths = _fake_model_paths(tmp_path)
    blocked = mod.run(
        date="20260812",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked_data",
        duration_s=0.5,
        cached_pair_func=_fake_cached_pair(paths, []),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
        generation_func=_fake_generation,
        host_checks_func=lambda: {"cuda_devices": {"available": False, "count": 0}},
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["models_used"] == []
    assert blocked["raw_generation_paths_hashes_and_counts"][mod.MANDATED_MODEL_IDS[0]][
        "unconstrained_sampling"
    ]["written_atomically"] is False

    no_write_hash = mod.write_payload_or_hash(tmp_path / "not_written.json", {"a": 1}, write=False)
    assert no_write_hash.startswith("sha256:")
    assert not (tmp_path / "not_written.json").exists()
    assert mod.grammar_sidecar_receipt(tmp_path, mod.development_fixtures())["path"].endswith(
        mod.GRAMMAR_FILE_NAME
    )
    assert mod.artifact_receipt(Path("results/no_such_6340.json"), score_field="missing_score")[
        "terminal_class"
    ] == "missing"

    fixture = mod.development_fixtures()[0]
    valid = mod.valid_zero_energy_programs(fixture, limit=1)[0]
    final, log = mod.apply_arm_intervention(
        arm="deterministic_parser_state_correction",
        raw_source=valid,
        raw_body=valid,
        fixture=fixture,
        candidate_index=0,
    )
    assert final == valid
    assert log["action"] == "parser_state_noop"
    with pytest.raises(ValueError, match="unknown_arm"):
        mod.apply_arm_intervention(
            arm="unknown",
            raw_source="",
            raw_body="",
            fixture=fixture,
            candidate_index=0,
        )
    assert mod.parser_status("not a policy\n") == "unknown_syntax:too_short"

    domain_row = mod.parse_candidate_source(
        model_id=mod.MANDATED_MODEL_IDS[0],
        family=fixture.family,
        split=fixture.split,
        arm="unconstrained_sampling",
        seed=1,
        candidate_index=0,
        raw_source="",
        final_source="policy x\nstates: nope;\nactions: nope;\nrule nope -> nope;\nend\n",
        fixture=fixture,
    )
    assert domain_row["final_parse_status"] == "domain_mismatch"

    bad_paths = {model_id: tmp_path / f"missing_{index}.gguf" for index, model_id in enumerate(mod.MANDATED_MODEL_IDS)}
    bad_resolution = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair(bad_paths, []),
        tokenizer_func=lambda _: (False, "bad"),
    )
    assert bad_resolution["all_resolved"] is False
    assert any(reason.startswith("model_path_missing") for reason in bad_resolution["blocked_reasons"])

    positive = _run_fake_positive(tmp_path / "positive")
    for mutate in (
        lambda art: art["preconditions_checked"].__setitem__("all_passed", False),
        lambda art: art.__setitem__("models_used", []),
        lambda art: art["matched_call_token_candidate_time_and_checker_budgets"].__setitem__(
            "budget_parity", False
        ),
        lambda art: art["protected_files_unchanged"].__setitem__("all_unchanged", False),
        lambda art: art["test_exit_codes"].__setitem__(FOCUSED_TEST_COMMAND, 1),
        lambda art: art["verification_calls_time_cost_and_error_table"].__setitem__(
            "accepted_contract_violation_count", 1
        ),
        lambda art: art["semantic_diversity_paired_deltas_intervals_and_sample_sizes"].__setitem__(
            "all_required_cells_positive", False
        ),
    ):
        candidate = deepcopy(positive)
        mutate(candidate)
        assert mod.expected_gain_score(candidate) == 0.0

    no_value = deepcopy(positive)
    no_value["semantic_diversity_gain_score"] = 0.0
    assert mod.status_from_artifact(no_value) == "complete_no_value"
    assert mod.honest_verdict(no_value).startswith("complete_null:")

    bad_principle = deepcopy(positive)
    bad_principle["field_principles"].pop("status")
    with pytest.raises(ValueError, match="field_principles:status"):
        mod.validate_artifact(bad_principle)

    bad_provenance = deepcopy(positive)
    bad_provenance["field_provenance"].pop("status")
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_oracle = deepcopy(positive)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_boundary = deepcopy(positive)
    bad_boundary["exact_oracle_claim_boundary"]["exact_compiler_counts_as_model_verification"] = True
    bad_boundary["reproducibility_checksum"] = mod.payload_checksum(bad_boundary)
    with pytest.raises(ValueError, match="exact_oracle_claim_boundary"):
        mod.validate_artifact(bad_boundary)

    bad_delta = deepcopy(positive)
    bad_delta["semantic_diversity_paired_deltas_intervals_and_sample_sizes"][
        "all_required_cells_positive"
    ] = False
    bad_delta["reproducibility_checksum"] = mod.payload_checksum(bad_delta)
    with pytest.raises(ValueError, match="semantic_diversity_gain_score:deltas"):
        mod.validate_artifact(bad_delta)

    bad_checksum = deepcopy(positive)
    bad_checksum["duration_s"] = 999.0
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    metrics = deepcopy(positive["exact_utility_fallback_latency_and_cost_by_model_family_arm"])
    metrics[mod.MANDATED_MODEL_IDS[0]]["access_gate"]["unconstrained_sampling"][
        "contract_violation_count"
    ] = 1
    harm = mod.harm_summary(
        {"models_used": []},
        {"exact_utility_fallback_latency_and_cost_by_model_family_arm": metrics},
    )
    assert harm["flagged_contract_violation_cells"]


def test_req_kona_6340_cli_writes_requested_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KONA-6340: the module command writes the requested artifact."""

    paths = _fake_model_paths(tmp_path)
    monkeypatch.setattr(mod, "cached_sota_pair", _fake_cached_pair(paths, []))
    monkeypatch.setattr(mod, "gguf_tokenizer_loadable", lambda path: (path.endswith(".gguf"), "embedded ok"))
    monkeypatch.setattr(mod, "generate_with_llama_cli", _fake_generation)
    monkeypatch.setattr(mod, "host_environment_receipts", _fake_host)

    result_path = tmp_path / "cli_artifact.json"
    data_dir = tmp_path / "cli_data"

    assert mod.main(
        [
            "--date",
            "20260812",
            "--result-path",
            str(result_path),
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_ready"
    assert payload["semantic_diversity_gain_score"] == 1.0


def _run_fake_positive(tmp_path: Path) -> dict[str, Any]:
    paths = _fake_model_paths(tmp_path)
    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        data_dir=tmp_path / "data",
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        cached_pair_func=_fake_cached_pair(paths, []),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
        generation_func=_fake_generation,
        host_checks_func=_fake_host,
        write=True,
    )
    assert mod.validate_artifact(artifact) is True
    return artifact


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "hf_cache"
    rows = {
        mod.MANDATED_MODEL_IDS[0]: (
            base
            / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
            / "snapshots"
            / "qwenrev"
            / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
        ),
        mod.MANDATED_MODEL_IDS[1]: (
            base
            / "models--unsloth--gemma-4-31B-it-GGUF"
            / "snapshots"
            / "dense"
            / "gemma-4-31B-it-Q4_K_M.gguf"
        ),
        mod.MANDATED_MODEL_IDS[2]: (
            base
            / "models--unsloth--gemma-4-26B-A4B-it-GGUF"
            / "snapshots"
            / "moe"
            / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
        ),
    }
    for path in rows.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake gguf")
    return rows


def _fake_cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def _cached_pair(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        del preferred_quant
        calls.append({"gpu_indices": gpu_indices, "model_indices": model_indices})
        qwen = {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": mod.MANDATED_MODEL_IDS[0],
            "gpu": gpu_indices[0],
            "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
        }
        gemma26 = {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": mod.MANDATED_MODEL_IDS[2],
            "gpu": gpu_indices[1],
            "model_path": str(paths[mod.MANDATED_MODEL_IDS[2]]),
        }
        gemma31 = {
            "name": "Gemma4-31B-it",
            "hf_id": mod.MANDATED_MODEL_IDS[1],
            "gpu": gpu_indices[1],
            "model_path": str(paths[mod.MANDATED_MODEL_IDS[1]]),
        }
        if model_indices == (0, 2):
            return [qwen, gemma31]
        return [qwen, gemma26]

    return _cached_pair


def _fake_generation(
    model_spec: dict[str, Any],
    prompt: str,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    del prompt
    fixtures = mod.development_fixtures()
    arm = str(model_spec["arm"])
    blocks = []
    for fixture in fixtures:
        variants = mod.valid_zero_energy_programs(fixture, limit=2)
        for candidate_index in range(mod.CANDIDATE_COUNT):
            if arm == mod.PREDECLARED_PREFIX_ARM:
                source = (
                    variants[0]
                    if candidate_index == 0
                    else "not a policy\n"
                )
            elif arm == "deterministic_parser_state_correction":
                source = "not a policy\n"
            elif arm == "grammar_masking":
                source = variants[0] if candidate_index == 0 else ""
            elif fixture.family == "incident_response":
                source = "not a policy\n"
            else:
                source = variants[0] if candidate_index == 0 else "not a policy\n"
            blocks.append(
                "BEGIN_CANDIDATE family={family} candidate={index}\n{source}END_CANDIDATE".format(
                    family=fixture.family,
                    index=candidate_index,
                    source=source,
                )
            )
    raw_text = "\n".join(blocks)
    return {
        "raw_text": raw_text,
        "receipt": {
            "mode": "fake_llama_cli",
            "arm": arm,
            "seed": seed,
            "exit_code": 0,
            "latency_s": 0.01,
            "stdout_sha256": "sha256:" + mod.sha256_text(raw_text),
            "prompt_tokens_estimated": 10,
            "generated_tokens_estimated": len(raw_text.split()),
            "memory_before_mb": {int(model_spec["gpu"]): 100},
            "memory_after_release_mb": {int(model_spec["gpu"]): 100},
            "cuda_layer_offload": {"cuda_layer_offload_confirmed": True},
            "cuda_layer_offload_confirmed": True,
            "release_within_512mb": True,
        },
    }


def _fake_host() -> dict[str, Any]:
    return {
        "cuda_devices": {
            "available": True,
            "count": 2,
            "devices": [
                {"index": 0, "name": "fake0", "memory_total_mb": 24576, "memory_free_mb": 24000},
                {"index": 1, "name": "fake1", "memory_total_mb": 24576, "memory_free_mb": 24000},
            ],
        },
        "vram": {
            "0": {"index": 0, "memory_total_mb": 24576, "memory_free_mb": 24000},
            "1": {"index": 1, "memory_total_mb": 24576, "memory_free_mb": 24000},
        },
        "ram": {"available_gb": 128.0},
        "disk": {"available_gb": 512.0},
        "llama_cpp_cli": {"exists": True, "path": str(mod.LLAMA_CPP_CLI_PATH)},
        "llama_cpp_gpu_offload": {"available": True, "detail": "fake"},
    }
