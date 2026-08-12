"""Tests for Exp6329 prospective held-family guarded policy A/B.

Spec refs: REQ-KONA-6329, SCENARIO-KONA-6329-GATE-REPLAY,
SCENARIO-KONA-6329-SEAL-CHRONOLOGY, SCENARIO-KONA-6329-MATCHED-ARMS,
SCENARIO-KONA-6329-ORACLE-BOUNDARY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6329_prospective_held_family_guarded_policy_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6329_prospective_held_family_guarded_policy_ab "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-m pytest tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6329_prospective_held_family_guarded_policy_ab.json"
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


def _fake_model_file(tmp_path: Path, hf_id: str, filename: str) -> Path:
    revision = "rev-" + mod.model_slug(hf_id)
    path = tmp_path / "hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots" / revision
    path.mkdir(parents=True)
    model_path = path / filename
    model_path.write_bytes((hf_id + "\n").encode("utf-8"))
    return model_path


def _fake_cached_pair_factory(tmp_path: Path):
    qwen = _fake_model_file(
        tmp_path, mod.MANDATED_MODEL_IDS[0], "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    dense = _fake_model_file(
        tmp_path, mod.MANDATED_MODEL_IDS[1], "gemma-4-31B-it-Q4_K_M.gguf"
    )
    middle = _fake_model_file(
        tmp_path, mod.MANDATED_MODEL_IDS[2], "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    )

    def fake_cached_pair(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        assert gpu_indices == (0, 1)
        assert preferred_quant == "Q4_K_M"
        if model_indices == (0, 2):
            return [
                {
                    "name": "Qwen3.6-35B-A3B",
                    "hf_id": mod.MANDATED_MODEL_IDS[0],
                    "gpu": 0,
                    "model_path": str(qwen),
                },
                {
                    "name": "Gemma4-31B-it",
                    "hf_id": mod.MANDATED_MODEL_IDS[1],
                    "gpu": 1,
                    "model_path": str(dense),
                },
            ]
        assert model_indices is None
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": mod.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": str(qwen),
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": mod.MANDATED_MODEL_IDS[2],
                "gpu": 1,
                "model_path": str(middle),
            },
        ]

    return fake_cached_pair


def _wrong_program(fixture: exp6326.PolicyFixture) -> str:
    contract = exp6326.validate_contract(fixture.contract)
    return exp6326.program_text(
        name=f"{fixture.family}_bad",
        states=contract.states,
        actions=contract.actions,
        mapping={state: contract.actions[-1] for state in contract.states},
    )


def _fake_generation_output(
    model_spec: dict[str, Any],
    prompt: str,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    assert "BEGIN_CANDIDATE" in prompt
    assert budget["candidate_count"] == 2
    blocks: list[str] = []
    for fixture in mod.build_held_families():
        blocks.append(
            f"BEGIN_CANDIDATE family={fixture.family} candidate=0\n"
            f"{_wrong_program(fixture)}END_CANDIDATE\n"
        )
        blocks.append(
            f"BEGIN_CANDIDATE family={fixture.family} candidate=1\n"
            f"{fixture.fallback_program}END_CANDIDATE\n"
        )
    text = "\n".join(blocks)
    return {
        "raw_text": text,
        "receipt": {
            "mode": "fake_generation",
            "model_hf_id": model_spec["hf_id"],
            "seed": seed,
            "latency_s": 0.01,
            "prompt_tokens_estimated": len(prompt.split()),
            "generated_tokens_estimated": len(text.split()),
            "exit_code": 0,
            "cuda_layer_offload_confirmed": True,
            "memory_before_mb": {"0": 4, "1": 4},
            "memory_after_release_mb": {"0": 4, "1": 4},
            "release_within_512mb": True,
        },
    }


def _host_ok() -> dict[str, Any]:
    return {
        "cuda_devices": {"available": True, "count": 2},
        "vram": {"gpu0_total_mb": 24576, "gpu1_total_mb": 24576},
        "ram": {"available_gb": 100},
        "disk": {"available_gb": 1000},
        "llama_cpp_cli": {"path": str(mod.LLAMA_CPP_CLI_PATH), "exists": True},
        "llama_cpp_gpu_offload": {"available": True},
    }


def _clock():
    stamps = iter(
        [
            "2026-08-12T00:00:00Z",
            "2026-08-12T00:00:01Z",
            "2026-08-12T00:00:02Z",
            "2026-08-12T00:00:03Z",
            "2026-08-12T00:00:04Z",
            "2026-08-12T00:00:05Z",
        ]
    )
    return lambda: next(stamps)


def _build_artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        data_dir=tmp_path / "sidecars",
        duration_s=120.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        cached_pair_func=_fake_cached_pair_factory(tmp_path),
        tokenizer_func=lambda path: (True, f"embedded tokenizer ok for {Path(path).name}"),
        generation_func=_fake_generation_output,
        host_checks_func=_host_ok,
        clock_func=_clock(),
        write=True,
    )


def test_req_kona_6329_spec_declares_prospective_contract() -> None:
    """REQ-KONA-6329: OpenSpec declares the prospective held-family contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6329") :]

    for marker in (
        "SCENARIO-KONA-6329-GATE-REPLAY",
        "SCENARIO-KONA-6329-SEAL-CHRONOLOGY",
        "SCENARIO-KONA-6329-MATCHED-ARMS",
        "SCENARIO-KONA-6329-ORACLE-BOUNDARY",
        "results/experiment_6329_prospective_held_family_guarded_policy_ab.json",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "AutoTokenizer",
        "predecision",
    ):
        assert marker in section


def test_scenario_kona_6329_gate_replay_model_specs_and_tokenizer(tmp_path: Path) -> None:
    """SCENARIO-KONA-6329-GATE-REPLAY: cached GGUF specs gate the run."""

    records = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair_factory(tmp_path),
        tokenizer_func=lambda path: (True, f"embedded tokenizer ok for {Path(path).name}"),
    )

    assert [record["hf_id"] for record in records["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert records["all_resolved"] is True
    assert records["cached_sota_pair_calls"] == [
        "cached_sota_pair(gpu_indices=(0,1))",
        "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))",
    ]
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in records["MODEL_SPECS"])
    assert all(row["model_path"].endswith(".gguf") for row in records["MODEL_SPECS"])
    assert "AutoTokenizer" not in (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    missing = mod.build_model_specs(cached_pair_func=lambda **_: None, tokenizer_func=lambda _: (False, "missing"))
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_missing" in missing["blocked_reasons"]


def test_scenario_kona_6329_holdout_overlap_and_seal_chronology(tmp_path: Path) -> None:
    """SCENARIO-KONA-6329-SEAL-CHRONOLOGY: outcomes open after raw receipts."""

    artifact = _build_artifact(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["prospective_guarded_policy_ready_score"] == 1.0
    assert artifact["overlap_receipts"]["declared_no_overlap"] is True
    assert artifact["overlap_receipts"]["total_overlap_count"] == 0

    manifest = artifact["sealed_holdout_manifest_path_and_hash"]
    registration = artifact["prospective_registration_path_hash_and_timestamp"]
    predecision = artifact["immutable_predecision_and_raw_candidate_paths_hashes"]
    outcome = artifact["outcome_seal_and_open_receipts"]
    for receipt in (manifest, registration, predecision["predecision_receipt"], outcome["open_receipt"]):
        path = Path(receipt["path"])
        assert path.exists()
        assert mod.sha256_file(path) == receipt["sha256"]

    chronology = outcome["chronology"]
    assert chronology["registration_at"] < chronology["holdout_sealed_at"]
    assert chronology["holdout_sealed_at"] < chronology["raw_candidates_immutable_at"]
    assert chronology["raw_candidates_immutable_at"] < chronology["predecision_immutable_at"]
    assert chronology["predecision_immutable_at"] < chronology["exact_outcomes_opened_at"]
    assert outcome["opened_after_predecision_immutable"] is True
    assert outcome["open_receipt"]["opened_against_predecision_sha256"] == predecision[
        "predecision_receipt"
    ]["sha256"]

    for receipt in predecision["raw_candidate_paths_hashes"].values():
        path = Path(receipt["path"])
        assert path.exists()
        assert mod.sha256_file(path) == receipt["sha256"]


def test_scenario_kona_6329_budget_parity_fallback_and_cell_metrics(tmp_path: Path) -> None:
    """SCENARIO-KONA-6329-MATCHED-ARMS: matched arms charge fallback exactly."""

    artifact = _build_artifact(tmp_path)
    budgets = artifact["matched_call_token_candidate_time_and_fallback_budgets"]
    for arm in mod.ARMS:
        assert budgets["by_arm"][arm]["candidate_count"] == budgets["candidate_count"]
        assert budgets["by_arm"][arm]["max_tokens"] == budgets["max_tokens"]
        assert budgets["by_arm"][arm]["time_budget_s"] == budgets["time_budget_s"]
        assert budgets["by_arm"][arm]["fallback_utility"] == budgets["fallback_utility"]
        assert budgets["by_arm"][arm]["fallback_cost"] == budgets["fallback_cost"]

    metrics = artifact[
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed"
    ]
    deltas = artifact["fallback_adjusted_delta_over_guard_only_by_model_and_family"]
    for model_id in mod.MANDATED_MODEL_IDS:
        assert set(metrics[model_id]) == set(mod.HELD_FAMILY_ORDER)
        for family in mod.HELD_FAMILY_ORDER:
            seed = str(mod.RANDOM_SEEDS[mod.MANDATED_MODEL_IDS.index(model_id)])
            arms = metrics[model_id][family][seed]
            assert arms["guard_plus_fallback"]["fallback_used"] is True
            assert arms["bounded_exact_factor_energy_search_plus_fallback"]["fallback_used"] is False
            assert arms["guard_plus_fallback"]["full_fallback_cost_charged"] == mod.FALLBACK_COST
            assert arms["bounded_exact_factor_energy_search_plus_fallback"]["contract_violation_count"] == 0
            assert deltas[model_id][family]["delta"] > 0

    harm = artifact["harm_underpowered_missing_and_flagged_cells"]
    assert harm["flagged_cell_count"] == 0
    assert harm["guarded_accepted_contract_violation_count"] == 0


def test_scenario_kona_6329_oracle_boundary_and_false_readiness_rejection(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6329-ORACLE-BOUNDARY: false readiness is rejected."""

    artifact = _build_artifact(tmp_path)

    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_oracle_claim_boundary"]["model_supplies_safety_authority"] is False
    assert set(artifact["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    bad_hidden = deepcopy(artifact)
    bad_hidden["hidden_state_access_count"] = False
    bad_hidden["reproducibility_checksum"] = mod.payload_checksum(bad_hidden)
    with pytest.raises(ValueError, match="hidden_state_access_count"):
        mod.validate_artifact(bad_hidden)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_command = deepcopy(artifact)
    bad_command["test_exit_codes"][FOCUSED_TEST_COMMAND] = 1
    bad_command["prospective_guarded_policy_ready_score"] = 1.0
    bad_command["status"] = "complete_ready"
    bad_command["honest_verdict"] = mod.honest_verdict(bad_command)
    bad_command["reproducibility_checksum"] = mod.payload_checksum(bad_command)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_command)


def test_req_kona_6329_cli_and_blocked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KONA-6329: CLI writes artifacts and blocked paths fail closed."""

    result_path = tmp_path / "cli_artifact.json"
    data_dir = tmp_path / "cli_data"
    assert mod.utc_now().endswith("Z")
    monkeypatch.setattr(mod, "cached_sota_pair", _fake_cached_pair_factory(tmp_path))
    monkeypatch.setattr(mod, "gguf_tokenizer_loadable", lambda path: (True, "embedded tokenizer ok"))
    monkeypatch.setattr(mod, "generate_with_llama_cli", _fake_generation_output)
    monkeypatch.setattr(mod, "host_environment_receipts", _host_ok)
    monkeypatch.setattr(mod, "utc_now", _clock())

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
    assert payload["prospective_guarded_policy_ready_score"] == 1.0

    blocked = mod.run(
        date="20260812",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked_data",
        duration_s=12.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "missing"),
        generation_func=_fake_generation_output,
        host_checks_func=lambda: {"cuda_devices": {"available": False, "count": 0}},
        clock_func=_clock(),
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["models_used"] == []
    assert all(
        receipt["candidate_count"] == 0
        for receipt in blocked["immutable_predecision_and_raw_candidate_paths_hashes"][
            "raw_candidate_paths_hashes"
        ].values()
    )

    assert mod.extract_program_source("text only") == ""
    assert mod.paired_interval([])["sample_size"] == 0
    assert mod.paired_interval([0.5])["ci95"] == [0.5, 0.5]
    assert mod.extract_quantization(Path("model-BF16.gguf")) == "unknown"
    assert mod.extract_revision(Path("/tmp/model-Q4_K_M.gguf")) == "unknown"
    assert mod.honest_verdict({"status": "complete_no_value"}).startswith("complete_null:")

    original_ids = mod.MANDATED_MODEL_IDS

    def missing_path_pair(**kwargs: Any) -> list[dict[str, Any]]:
        model_indices = kwargs.get("model_indices")
        if model_indices == (0, 2):
            return [
                {
                    "name": "qwen",
                    "hf_id": original_ids[0],
                    "gpu": 0,
                    "model_path": str(tmp_path / "missing-qwen.gguf"),
                },
                {
                    "name": "dense",
                    "hf_id": original_ids[1],
                    "gpu": 1,
                    "model_path": str(tmp_path / "missing-dense.gguf"),
                },
            ]
        return [
            {
                "name": "qwen",
                "hf_id": original_ids[0],
                "gpu": 0,
                "model_path": str(tmp_path / "missing-qwen.gguf"),
            },
            {
                "name": "middle",
                "hf_id": original_ids[2],
                "gpu": 1,
                "model_path": str(tmp_path / "missing-middle.gguf"),
            },
        ]

    missing_records = mod.build_model_specs(
        cached_pair_func=missing_path_pair,
        tokenizer_func=lambda _: (False, "no tokenizer"),
    )
    assert any(reason.startswith("model_path_missing") for reason in missing_records["blocked_reasons"])
    assert any(
        reason.startswith("embedded_tokenizer_not_loadable")
        for reason in missing_records["blocked_reasons"]
    )

    monkeypatch.setattr(mod, "MANDATED_MODEL_IDS", tuple(reversed(original_ids)))
    order_records = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair_factory(tmp_path / "order"),
        tokenizer_func=lambda _: (True, "ok"),
    )
    assert "mandated_model_order" in order_records["blocked_reasons"]
    monkeypatch.setattr(mod, "MANDATED_MODEL_IDS", original_ids)

    fixture = mod.build_held_families()[0]
    raw_outputs = {
        original_ids[0]: {
            "seed": mod.RANDOM_SEEDS[0],
            "raw_text": (
                "BEGIN_CANDIDATE family=triage_lane candidate=0\n"
                + fixture.fallback_program
                + "END_CANDIDATE\n"
            ),
        }
    }
    monkeypatch.setattr(mod.exp6326, "factor_energy", lambda *_: 99)
    mismatch = mod.parse_and_score_candidates(raw_outputs, [fixture])
    assert mismatch["exact_factor_energies_by_candidate"]["mismatch_count"] == 1

    sparse = mod.evaluate_arms([], [fixture])
    seed = str(mod.RANDOM_SEEDS[0])
    assert sparse["metrics"][original_ids[0]][fixture.family][seed]["guard_plus_fallback"][
        "fallback_used"
    ] is True

    underpowered = mod.harm_summary(
        generation={"models_used": list(original_ids)},
        parsed={"parse_and_normalization_results": {"parser_failure_count": 0}},
        arm_results={"guard_receipts": {"guarded_accepted_contract_violation_count": 0}},
        fallback_delta={
            original_ids[0]: {
                fixture.family: {
                    "adequately_powered": False,
                    "positive": False,
                }
            }
        },
    )
    assert underpowered["underpowered_cells"][0]["reason"] == "underpowered"
