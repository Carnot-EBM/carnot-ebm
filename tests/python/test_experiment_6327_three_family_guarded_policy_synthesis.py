"""Tests for Exp6327 guarded policy synthesis.

Spec refs: REQ-KONA-6327, SCENARIO-KONA-6327-GATE,
SCENARIO-KONA-6327-MATCHED-ARMS, SCENARIO-KONA-6327-ORACLE-BOUNDARY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6327_three_family_guarded_policy_synthesis as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6327_three_family_guarded_policy_synthesis "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py "
    "-m pytest tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6327_three_family_guarded_policy_synthesis.json"
)
TEST_COMMANDS = [
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
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
                {"name": "Qwen3.6-35B-A3B", "hf_id": mod.MANDATED_MODEL_IDS[0], "gpu": 0, "model_path": str(qwen)},
                {"name": "Gemma4-31B-it", "hf_id": mod.MANDATED_MODEL_IDS[1], "gpu": 1, "model_path": str(dense)},
            ]
        assert model_indices is None
        return [
            {"name": "Qwen3.6-35B-A3B", "hf_id": mod.MANDATED_MODEL_IDS[0], "gpu": 0, "model_path": str(qwen)},
            {"name": "Gemma4-26B-A4B-it", "hf_id": mod.MANDATED_MODEL_IDS[2], "gpu": 1, "model_path": str(middle)},
        ]

    return fake_cached_pair


def _fake_generation_output(
    model_spec: dict[str, Any],
    prompt: str,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    assert "BEGIN_CANDIDATE" in prompt
    assert budget["candidate_count"] == 2
    blocks: list[str] = []
    for fixture in exp6326.build_fixture_manifest():
        if fixture.family == "access_gate":
            first = "policy broken\nstates: guest;\nactions: deny;\nrule guest -> deny;\n"
        else:
            first = fixture.fallback_program
        second = fixture.fallback_program
        blocks.append(
            f"BEGIN_CANDIDATE family={fixture.family} candidate=0\n"
            f"{first}END_CANDIDATE\n"
        )
        blocks.append(
            f"BEGIN_CANDIDATE family={fixture.family} candidate=1\n"
            f"{second}END_CANDIDATE\n"
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
        },
    }


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
        host_checks_func=lambda: {
            "cuda_devices": {"available": True, "count": 2},
            "vram": {"gpu0_total_mb": 24576, "gpu1_total_mb": 24576},
            "ram": {"available_gb": 100},
            "disk": {"available_gb": 1000},
            "llama_cpp_cli": {"path": str(mod.LLAMA_CPP_CLI_PATH), "exists": True},
            "llama_cpp_gpu_offload": {"available": True},
        },
        write=True,
    )


def test_req_kona_6327_spec_declares_three_family_artifact_contract() -> None:
    """REQ-KONA-6327: OpenSpec anchors the three-model synthesis artifact."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6327") :]

    for marker in (
        "SCENARIO-KONA-6327-GATE",
        "SCENARIO-KONA-6327-MATCHED-ARMS",
        "SCENARIO-KONA-6327-ORACLE-BOUNDARY",
        "results/experiment_6327_three_family_guarded_policy_synthesis.json",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section


def test_scenario_kona_6327_gate_uses_cached_pair_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6327-GATE: specs come from cached pairs and GGUF tokenizers."""

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
    assert all(row["tokenizer_method"] == "llama_cpp_embedded_gguf_vocab_only" for row in records["MODEL_SPECS"])
    assert all(row["model_path"].endswith(".gguf") for row in records["MODEL_SPECS"])
    assert "AutoTokenizer" not in (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    missing = mod.build_model_specs(cached_pair_func=lambda **_: None, tokenizer_func=lambda _: (False, "missing"))
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_missing" in missing["blocked_reasons"]


def test_scenario_kona_6327_matched_arms_hash_parse_and_fallback_accounting(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6327-MATCHED-ARMS: budgets, raw hashes, and fallbacks match."""

    artifact = _build_artifact(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["guarded_policy_synthesis_ready_score"] == 1.0
    assert set(artifact["models_used"]) == set(mod.MANDATED_MODEL_IDS)
    assert set(artifact["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)

    budgets = artifact["matched_call_token_candidate_and_time_budgets"]
    for arm in mod.ARMS:
        assert budgets["by_arm"][arm]["candidate_count"] == budgets["candidate_count"]
        assert budgets["by_arm"][arm]["max_tokens"] == budgets["max_tokens"]
        assert budgets["by_arm"][arm]["time_budget_s"] == budgets["time_budget_s"]

    raw = artifact["raw_candidate_paths_hashes_and_counts"]
    assert set(raw) == set(mod.MANDATED_MODEL_IDS)
    for receipt in raw.values():
        path = Path(receipt["path"])
        assert path.exists()
        assert mod.sha256_file(path) == receipt["sha256"]
        assert receipt["candidate_count"] == len(exp6326.build_fixture_manifest()) * 2

    parse_results = artifact["parse_and_normalization_results"]
    assert parse_results["parser_failure_count"] == len(mod.MANDATED_MODEL_IDS)
    assert parse_results["by_family"]["access_gate"]["parser_failure_count"] == len(
        mod.MANDATED_MODEL_IDS
    )

    metrics = artifact[
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed"
    ]
    for model_id in mod.MANDATED_MODEL_IDS:
        access = metrics[model_id]["access_gate"]
        assert access["exact_guard_plus_hash_pinned_fallback"]["fallback_used"] is True
        assert access["bounded_exact_factor_energy_guided_candidate_search_plus_fallback"]["fallback_used"] is False
        assert (
            access["bounded_exact_factor_energy_guided_candidate_search_plus_fallback"]["utility"]
            > access["exact_guard_plus_hash_pinned_fallback"]["utility"]
        )


def test_scenario_kona_6327_oracle_boundary_and_false_readiness_rejection(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-6327-ORACLE-BOUNDARY: exact guard is the only authority."""

    artifact = _build_artifact(tmp_path)

    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_oracle_claim_boundary"]["model_supplies_safety_authority"] is False
    assert artifact["guard_accept_reject_and_fallback_receipts"]["guarded_accepted_contract_violation_count"] == 0
    assert artifact["exact_factor_energies_by_candidate"]["mismatch_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    bad_count = deepcopy(artifact)
    bad_count["hidden_state_access_count"] = False
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="hidden_state_access_count"):
        mod.validate_artifact(bad_count)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_ready = deepcopy(artifact)
    bad_ready["guard_accept_reject_and_fallback_receipts"]["guarded_accepted_contract_violation_count"] = 1
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_ready)

    blocked = deepcopy(artifact)
    blocked["MODEL_SPECS"][0]["exists"] = False
    blocked["guarded_policy_synthesis_ready_score"] = 0.0
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = mod._honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    assert mod.validate_artifact(blocked) is True


def test_req_kona_6327_cli_writes_requested_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KONA-6327: the required module command writes the artifact."""

    result_path = tmp_path / "cli_artifact.json"
    data_dir = tmp_path / "cli_data"

    monkeypatch.setattr(mod, "cached_sota_pair", _fake_cached_pair_factory(tmp_path))
    monkeypatch.setattr(mod, "gguf_tokenizer_loadable", lambda path: (True, "embedded tokenizer ok"))
    monkeypatch.setattr(mod, "generate_with_llama_cli", _fake_generation_output)
    monkeypatch.setattr(
        mod,
        "host_environment_receipts",
        lambda: {
            "cuda_devices": {"available": True, "count": 2},
            "vram": {"gpu0_total_mb": 24576, "gpu1_total_mb": 24576},
            "ram": {"available_gb": 100},
            "disk": {"available_gb": 1000},
            "llama_cpp_cli": {"path": str(mod.LLAMA_CPP_CLI_PATH), "exists": True},
            "llama_cpp_gpu_offload": {"available": True},
        },
    )

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
    assert payload["guarded_policy_synthesis_ready_score"] == 1.0


def test_req_kona_6327_fail_closed_and_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KONA-6327: blocked paths and helper edge cases stay explicit."""

    fake_cached_pair = _fake_cached_pair_factory(tmp_path)
    pair = fake_cached_pair(gpu_indices=(0, 1))
    Path(pair[0]["model_path"]).unlink()
    records = mod.build_model_specs(
        cached_pair_func=lambda **kwargs: pair if kwargs.get("model_indices") is None else [],
        tokenizer_func=lambda path: (False, f"bad tokenizer for {path}"),
    )
    assert records["all_resolved"] is False
    assert any(reason.startswith("model_path_missing") for reason in records["blocked_reasons"])
    assert any(reason.startswith("embedded_tokenizer_not_loadable") for reason in records["blocked_reasons"])

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
        host_checks_func=lambda: {
            "cuda_devices": {"available": False, "count": 0},
            "llama_cpp_cli": {"path": str(mod.LLAMA_CPP_CLI_PATH), "exists": False},
            "llama_cpp_gpu_offload": {"available": False},
        },
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["models_used"] == []
    assert all(row["candidate_count"] == 0 for row in blocked["raw_candidate_paths_hashes_and_counts"].values())

    artifact = _build_artifact(tmp_path / "ready")
    no_write_generation = mod.generate_raw_outputs(
        artifact["MODEL_SPECS"],
        exp6326.build_fixture_manifest(),
        data_dir=tmp_path / "no_write",
        prompt=mod.build_prompt(exp6326.build_fixture_manifest()),
        budget=mod.matched_budgets(),
        generation_func=_fake_generation_output,
        write=False,
    )
    assert all(receipt["sha256"].startswith("sha256:") for receipt in no_write_generation["raw_candidate_paths_hashes_and_counts"].values())

    fixture = exp6326.build_fixture_manifest()[0]
    contract = exp6326.validate_contract(fixture.contract)
    factors = exp6326.compile_contract_to_factors(contract)
    missing = mod.parse_candidate(
        model_id=mod.MANDATED_MODEL_IDS[0],
        family=fixture.family,
        split=fixture.split,
        seed=1,
        candidate_index=0,
        source="no program here",
        contract=contract,
        factors=factors,
    )
    assert missing["parse_status"] == "missing_block"

    mismatch_source = (
        "policy p\nstates: only;\nactions: deny;\nrule only -> deny;\nend\n"
    )
    mismatch = mod.parse_candidate(
        model_id=mod.MANDATED_MODEL_IDS[0],
        family=fixture.family,
        split=fixture.split,
        seed=1,
        candidate_index=0,
        source=mismatch_source,
        contract=contract,
        factors=factors,
    )
    assert mismatch["parse_status"] == "domain_mismatch"

    assert mod.best_candidate([missing]) == missing
    sparse = mod.evaluate_arms([missing], exp6326.build_fixture_manifest())
    assert sparse["guard_receipts"]["fallback_count_by_arm"]["exact_guard_plus_hash_pinned_fallback"] >= 1

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod.upstream_receipt()["terminal_class"] == "missing"

    parsed_cuda = mod.parse_cuda_offload("llama_model_load: offloaded 41/42 layers to GPU")
    assert parsed_cuda["cuda_layer_offload_confirmed"] is True
    assert mod.parse_cuda_offload("no offload")["cuda_layer_offload_confirmed"] is False
    assert mod.release_with({0: 4}, {0: 100}, 512) is True
    assert mod.release_with({0: 4}, {0: 600}, 512) is False
    assert mod.paired_interval([])["sample_size"] == 0
    assert mod.paired_interval([0.5])["ci95"] == [0.5, 0.5]
    assert mod.extract_program_source("text only") == ""
    assert mod.extract_quantization(Path("model-BF16.gguf")) == "unknown"
    assert mod.extract_revision(Path("/tmp/model-Q4_K_M.gguf")) == "unknown"
    assert mod._honest_verdict({"status": "complete_no_value"}).startswith("complete_null:")

    original_ids = mod.MANDATED_MODEL_IDS

    def original_order_pair(**kwargs: Any) -> list[dict[str, Any]]:
        model_indices = kwargs.get("model_indices")
        if model_indices == (0, 2):
            return [
                {"name": "qwen", "hf_id": original_ids[0], "gpu": 0, "model_path": str(_fake_model_file(tmp_path / "order", original_ids[0], "qwen-Q4_K_M.gguf"))},
                {"name": "dense", "hf_id": original_ids[1], "gpu": 1, "model_path": str(_fake_model_file(tmp_path / "order", original_ids[1], "dense-Q4_K_M.gguf"))},
            ]
        return [
            {"name": "qwen", "hf_id": original_ids[0], "gpu": 0, "model_path": str(_fake_model_file(tmp_path / "order2", original_ids[0], "qwen-Q4_K_M.gguf"))},
            {"name": "middle", "hf_id": original_ids[2], "gpu": 1, "model_path": str(_fake_model_file(tmp_path / "order2", original_ids[2], "middle-Q4_K_M.gguf"))},
        ]

    monkeypatch.setattr(mod, "MANDATED_MODEL_IDS", tuple(reversed(original_ids)))
    order = mod.build_model_specs(
        cached_pair_func=original_order_pair,
        tokenizer_func=lambda _: (True, "ok"),
    )
    assert "mandated_model_order" in order["blocked_reasons"]
    monkeypatch.setattr(mod, "MANDATED_MODEL_IDS", original_ids)

    mismatch_fixture = exp6326.build_fixture_manifest()[0]
    monkeypatch.setattr(mod.exp6326, "factor_energy", lambda *_: 99)
    mismatch_summary = mod.parse_and_score_candidates(
        {
            original_ids[0]: {
                "seed": 1,
                "raw_text": (
                    "BEGIN_CANDIDATE family=access_gate candidate=0\n"
                    + mismatch_fixture.fallback_program
                    + "END_CANDIDATE\n"
                ),
            }
        },
        [mismatch_fixture],
    )
    assert mismatch_summary["exact_factor_energies_by_candidate"]["mismatch_count"] == 1
