"""Tests for Exp6421 explicit opt-in executed-policy A/B.

Spec refs: REQ-ARC-ARM-6421,
SCENARIO-ARC-ARM-6421-PRECONDITIONS,
SCENARIO-ARC-ARM-6421-MATCHED-OPT-IN-ARMS,
SCENARIO-ARC-ARM-6421-EXECUTED-POLICY-CHANGE,
SCENARIO-ARC-ARM-6421-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6421_arc_opt_in_executed_policy_ab as exp6421


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6421.ARC_SPEC_RELATIVE_PATH


def _fake_model_file(tmp_path: Path, model_id: str, suffix: str = "Q4_K_M") -> str:
    name = model_id.split("/")[-1]
    snap = tmp_path / f"models--{model_id.replace('/', '--')}" / "snapshots" / f"rev-{name}"
    snap.mkdir(parents=True, exist_ok=True)
    path = snap / f"{name}-{suffix}.gguf"
    path.write_text(f"fixture bytes for {model_id}", encoding="utf-8")
    return str(path)


def _fake_pair_resolver_factory(tmp_path: Path):
    def fake_pair_resolver(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        _ = preferred_quant
        ids = {
            0: exp6421.MANDATED_QWEN_MODEL_ID,
            1: exp6421.MANDATED_GEMMA_MOE_MODEL_ID,
            2: exp6421.MANDATED_GEMMA_MODEL_ID,
        }
        indices = model_indices or (0, 1)
        return [
            {
                "name": ids[index].split("/")[-1].replace("-GGUF", ""),
                "hf_id": ids[index],
                "gpu": gpu,
                "model_path": _fake_model_file(tmp_path, ids[index]),
            }
            for gpu, index in zip(gpu_indices, indices, strict=True)
        ]

    return fake_pair_resolver


def _fake_canonical_resolver(model_id: str, preferred_quant: str = "Q4_K_M") -> str:
    _ = preferred_quant
    return _fake_model_file(_TMP_PATH, model_id, "UD-Q4_K_XL")


def _fake_tokenizer_checker(model_path: str | None) -> tuple[bool, str]:
    return bool(model_path), f"embedded GGUF tokenizer OK for {Path(model_path or '').name}"


def _fake_cuda_receipts(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(model["hf_id"]): {
            "terminal": True,
            "cuda_visible": True,
            "gpu_offload_supported": True,
            "gpu": int(model["gpu"]),
            "model_path": model["model_path"],
            "errors": [],
        }
        for model in models
    }


_TMP_PATH = Path("/tmp/exp6421-test-models")


def _artifact(tmp_path: Path) -> dict[str, Any]:
    global _TMP_PATH
    _TMP_PATH = tmp_path
    commands = (
        exp6421.RUN_COMMAND,
        exp6421.FOCUSED_TEST_COMMAND,
        exp6421.COVERAGE_RUN_COMMAND,
        exp6421.COVERAGE_REPORT_COMMAND,
    )
    return exp6421.run(
        date="20260814",
        result_path=tmp_path / exp6421.RESULT_RELATIVE_PATH.name,
        duration_s=3.0,
        tests_run=commands,
        test_exit_codes={command: 0 for command in commands},
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        canonical_resolver=_fake_canonical_resolver,
        tokenizer_checker=_fake_tokenizer_checker,
        cuda_receipt_collector=_fake_cuda_receipts,
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6421.payload_checksum(payload)
    return payload


def test_req_arc_arm_6421_spec_declares_executed_policy_contract() -> None:
    """REQ-ARC-ARM-6421: OpenSpec names the executed-policy A/B contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6421") :]
    for marker in (
        "SCENARIO-ARC-ARM-6421-PRECONDITIONS",
        "SCENARIO-ARC-ARM-6421-MATCHED-OPT-IN-ARMS",
        "SCENARIO-ARC-ARM-6421-EXECUTED-POLICY-CHANGE",
        "SCENARIO-ARC-ARM-6421-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION",
        exp6421.RESULT_RELATIVE_PATH.as_posix(),
        exp6421.CANONICAL_GENERATOR_MODEL_ID,
        exp6421.MANDATED_GEMMA_MODEL_ID,
    ):
        assert marker in section
    for field in exp6421.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_arm_6421_preconditions_revalidate_gates_and_registry() -> None:
    """SCENARIO-ARC-ARM-6421-PRECONDITIONS: gates and every game are checked."""

    gate = exp6421.exp6413_gate_receipt()
    registry = exp6421.solve_registry_precheck_path_hash_and_results()

    assert gate["gate_passed"] is True
    assert gate["authenticated_receipt_contract_ready_score"] == 1.0
    assert gate["autotokenizer_usage_count"] == 0
    assert registry["all_games_prechecked"] is True
    assert registry["game_count"] == 25
    assert registry["target_task_is_not_level_solve"] is True
    assert registry["registry_modified"] is False
    assert all(row["duplicate_experiment_6421_target"] is False for row in registry["games"])


def test_req_arc_arm_6421_model_specs_include_canonical_and_cached_gemma(
    tmp_path: Path,
) -> None:
    """REQ-ARC-ARM-6421: model specs include shipped generator and cached Gemma."""

    global _TMP_PATH
    _TMP_PATH = tmp_path
    models, cached = exp6421.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        canonical_resolver=_fake_canonical_resolver,
    )
    tokenizers = exp6421.canonical_generator_model_file_and_embedded_tokenizer_hashes(
        models,
        tokenizer_checker=_fake_tokenizer_checker,
    )

    ids = [row["hf_id"] for row in models]
    assert ids[0] == exp6421.CANONICAL_GENERATOR_MODEL_ID
    assert exp6421.MANDATED_GEMMA_MODEL_ID in ids
    assert cached["mandated_gemma_resolved_through_cached_sota_pair"] is True
    assert all(row["model_exists"] is True for row in models)
    assert all(str(row["model_sha256"]).startswith("sha256:") for row in models)
    assert tokenizers["canonical_generator"]["hf_id"] == exp6421.CANONICAL_GENERATOR_MODEL_ID
    assert all(row["ok"] is True for row in tokenizers["by_model"].values())

    direct = tmp_path / "direct.py"
    direct.write_text("AutoTokenizer\n", encoding="utf-8")
    attribute = tmp_path / "attribute.py"
    attribute.write_text("transformers.AutoTokenizer\n", encoding="utf-8")
    assert exp6421.autotokenizer_usage_count((direct, attribute)) == 2

    with pytest.raises(ValueError, match="canonical generator"):
        exp6421.build_model_specs(
            model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
            canonical_resolver=lambda *_args, **_kwargs: None,
        )

    def missing_gemma(**kwargs):
        _ = kwargs
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp6421.MANDATED_QWEN_MODEL_ID,
                "gpu": 0,
                "model_path": _fake_model_file(tmp_path, exp6421.MANDATED_QWEN_MODEL_ID),
            }
        ]

    with pytest.raises(ValueError, match="mandated gemma"):
        exp6421.build_model_specs(
            model_pair_resolver=missing_gemma,
            canonical_resolver=_fake_canonical_resolver,
        )

    with pytest.raises(ValueError, match="cached_sota_pair"):
        exp6421.build_model_specs(
            model_pair_resolver=lambda **_kwargs: None,
            canonical_resolver=_fake_canonical_resolver,
        )

    assert exp6421._display_path(Path("/tmp/outside-exp6421.txt")) == "/tmp/outside-exp6421.txt"


def test_scenario_arc_arm_6421_matched_arms_change_legal_executed_policy(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6421-EXECUTED-POLICY-CHANGE: opt-in changes actions."""

    global _TMP_PATH
    _TMP_PATH = tmp_path
    models, _cached = exp6421.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        canonical_resolver=_fake_canonical_resolver,
    )
    windows = exp6421.fresh_policy_window_manifest_payload()
    causal = exp6421.run_matched_policy_ab(models=models, windows=windows["rows"])
    delta = causal["causal_policy_delta"]
    matched = causal[
        "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts"
    ]

    assert windows["fresh_canonical_agent_windows"] is True
    assert windows["window_count"] == len(exp6421.SELECTED_WINDOWS) * len(exp6421.RANDOM_SEEDS)
    assert matched["matched_contract_passed"] is True
    assert causal["row_count"] == len(models) * windows["window_count"] * 2
    assert delta["route_firing_delta"] == len(models) * windows["window_count"]
    assert delta["changed_legal_executed_action_count"] == len(models) * windows["window_count"]
    assert delta["legal_action_rate_delta"] == 0.0
    assert delta["exact_observation_consistency_delta"] == 0.0
    assert delta["progress_proxy_delta"] > 0.0
    assert delta["harmful_regression_delta"] == 0
    assert all(
        row["executed_action"] in row["legal_actions"]
        for row in causal[
            "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
        ]
    )


def test_scenario_arc_arm_6421_attack_matrix_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6421-ATTACKS-FAIL-CLOSED: evidence attacks fail."""

    global _TMP_PATH
    _TMP_PATH = tmp_path
    models, _cached = exp6421.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        canonical_resolver=_fake_canonical_resolver,
    )
    causal = exp6421.run_matched_policy_ab(
        models=models,
        windows=exp6421.fresh_policy_window_manifest_payload()["rows"],
    )
    receipt_hashes = exp6421.authenticated_model_process_and_raw_output_receipts()[
        "raw_output_hashes"
    ]
    attacks = exp6421.attack_matrix(
        rows=causal[
            "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
        ],
        model_ids=[str(row["hf_id"]) for row in models],
        raw_output_hashes=receipt_hashes,
    )

    assert {row["attack"] for row in attacks} == set(exp6421.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks)
    assert exp6421._expect_value_error("accepted", lambda: None)["fail_closed"] is False

    duplicate = copy.deepcopy(
        causal[
            "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
        ]
    )
    duplicate[1]["arm"] = duplicate[0]["arm"]
    with pytest.raises(ValueError, match="duplicate model/window/arm"):
        exp6421.validate_policy_rows(duplicate, [str(row["hf_id"]) for row in models])

    no_change = copy.deepcopy(duplicate)
    no_change[1]["arm"] = exp6421.OPT_IN_ARM
    no_change[1]["executed_action"] = no_change[0]["executed_action"]
    with pytest.raises(ValueError, match="expected opt-in action change"):
        exp6421.validate_policy_rows(no_change, [str(row["hf_id"]) for row in models])

    missing = rows = causal[
        "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts"
    ]
    with pytest.raises(ValueError, match="missing policy rows"):
        exp6421.validate_policy_rows(missing[:-1], [str(row["hf_id"]) for row in models])

    missing_arm = copy.deepcopy(rows)
    missing_arm[1]["arm"] = "unexpected_arm"
    with pytest.raises(ValueError, match="missing matched arm"):
        exp6421.validate_policy_rows(missing_arm, [str(row["hf_id"]) for row in models])

    model_swap = rows[len(exp6421.SELECTED_WINDOWS) * len(exp6421.RANDOM_SEEDS) * 2 :] + rows[
        : len(exp6421.SELECTED_WINDOWS) * len(exp6421.RANDOM_SEEDS) * 2
    ]
    with pytest.raises(ValueError, match="model row order"):
        exp6421.validate_policy_rows(model_swap, [str(row["hf_id"]) for row in models])

    not_candidate = copy.deepcopy(rows)
    not_candidate[1]["candidate_actions"] = [4]
    not_candidate[1]["executed_action"] = 5
    with pytest.raises(ValueError, match="action substitution"):
        exp6421.validate_policy_rows(not_candidate, [str(row["hf_id"]) for row in models])

    assert exp6421._validate_raw_output_hashes(["sha256:a", "sha256:b"]) is None

    weak_attack = exp6421.attack_matrix(
        rows=rows,
        model_ids=[str(row["hf_id"]) for row in models],
        raw_output_hashes=["sha256:only-one"],
    )
    assert any(
        row["attack"] == "model_receipt_reuse" and row["fail_closed"] is False
        for row in weak_attack
    )

    assert exp6421._matched_receipt(rows[:-1])["matched_contract_passed"] is False


def test_scenario_arc_arm_6421_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION: artifact is complete."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6421.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6421.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp6421.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["exp6413_gate_receipt"]["gate_passed"] is True
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["shipped_default_before_and_after"]["unchanged_default_off"] is True
    assert artifact["causal_policy_delta"]["changed_legal_executed_action_count"] > 0
    assert artifact["source_access_count"] == 0
    assert artifact["per_game_adapter_count"] == 0
    assert artifact["outer_loop_re_used"] is False
    assert artifact["level_solve_claimed"] is False
    assert artifact["solve_registry_modified"] is False
    assert artifact["route_default_promoted"] is False
    assert artifact["public_arc_claim_eligibility"] is False
    assert artifact["arc_executed_policy_influence_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"]["exp6413_gate_receipt"]
    assert artifact["field_principles"]["causal_policy_delta"]
    assert artifact["field_principles"]["arc_executed_policy_influence_ready_score"]
    assert artifact["reproducibility_checksum"] == exp6421.payload_checksum(artifact)
    exp6421.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_access_count", 1, "source_access_count"),
        ("per_game_adapter_count", 1, "per_game_adapter_count"),
        ("outer_loop_re_used", True, "outer_loop_re_used"),
        ("level_solve_claimed", True, "level_solve_claimed"),
        ("solve_registry_modified", True, "solve_registry_modified"),
        ("route_default_promoted", True, "route_default_promoted"),
        ("public_arc_claim_eligibility", True, "public_arc_claim_eligibility"),
        ("arc_executed_policy_influence_ready_score", 0.0, "arc_executed_policy_influence_ready_score"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_arm_6421_validation_rejects_forbidden_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION: drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6421.validate_artifact(bad)


def test_req_arc_arm_6421_validation_rejects_nested_drift(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6421: nested provenance drift fails validation."""

    artifact = _artifact(tmp_path)

    checksum = copy.deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6421.validate_artifact(checksum)

    missing = copy.deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6421.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a["exp6413_gate_receipt"].__setitem__("gate_passed", False),
            "exp6413_gate_receipt",
        ),
        (
            lambda a: a["solve_registry_precheck_path_hash_and_results"].__setitem__(
                "all_games_prechecked", False
            ),
            "solve_registry_precheck",
        ),
        (
            lambda a: a[
                "shipped_default_before_and_after"
            ].__setitem__("unchanged_default_off", False),
            "shipped_default_before_and_after",
        ),
        (
            lambda a: a["causal_policy_delta"].__setitem__(
                "changed_legal_executed_action_count", 0
            ),
            "causal_policy_delta",
        ),
        (
            lambda a: a["causal_policy_delta"].__setitem__("harmful_regression_delta", 1),
            "causal_policy_delta",
        ),
        (
            lambda a: a["attack_matrix"][0].__setitem__("fail_closed", False),
            "attack_matrix",
        ),
        (
            lambda a: a["protected_files_unchanged"]["ops/arc_solve_registry.yaml"].__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
        (
            lambda a: a["canonical_generator_model_file_and_embedded_tokenizer_hashes"][
                "canonical_generator"
            ].__setitem__("ok", False),
            "canonical_generator",
        ),
        (
            lambda a: a["canonical_generator_model_file_and_embedded_tokenizer_hashes"].__setitem__(
                "all_embedded_tokenizers_loadable", False
            ),
            "canonical_generator",
        ),
        (lambda a: a.__setitem__("models_used", []), "models_used"),
        (lambda a: a.__setitem__("autotokenizer_usage_count", 1), "autotokenizer_usage_count"),
        (
            lambda a: a[
                "canonical_live_entrypoint_route_policy_game_interface_and_config_hashes"
            ].__setitem__("active_reward_machine_default_off", False),
            "canonical_live_entrypoint",
        ),
        (
            lambda a: a[
                "matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts"
            ].__setitem__("matched_contract_passed", False),
            "matched_contract",
        ),
        (
            lambda a: a["authenticated_model_process_and_raw_output_receipts"].__setitem__(
                "gate_passed", False
            ),
            "authenticated_model_process_and_raw_output_receipts",
        ),
        (
            lambda a: a["authenticated_model_process_and_raw_output_receipts"].__setitem__(
                "all_inherited_receipts_content_addressed", False
            ),
            "authenticated_model_process_and_raw_output_receipts",
        ),
        (
            lambda a: a["causal_policy_delta"].__setitem__("route_firing_delta", 0),
            "causal_policy_delta",
        ),
        (
            lambda a: a["causal_policy_delta"].__setitem__("legal_action_rate_delta", 1.0),
            "causal_policy_delta",
        ),
        (
            lambda a: a["causal_policy_delta"].__setitem__(
                "exact_observation_consistency_delta", 1.0
            ),
            "causal_policy_delta",
        ),
        (
            lambda a: a["field_principles"].pop("causal_policy_delta"),
            "field_principles",
        ),
        (lambda a: a.__setitem__("honest_verdict", "blocked"), "honest_verdict"),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6421.validate_artifact(bad)


def test_req_arc_arm_6421_build_artifact_uses_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6421: build_artifact validates runner output."""

    artifact = _artifact(tmp_path)

    def fake_run(**kwargs):
        assert kwargs["date"] == "20260814"
        assert kwargs["write"] is True
        return artifact

    monkeypatch.setattr(exp6421, "run", fake_run)

    built = exp6421.build_artifact(
        tmp_path,
        date="20260814",
        output_path=tmp_path / "out.json",
    )
    assert built["arc_executed_policy_influence_ready_score"] == 1.0


def test_req_arc_arm_6421_run_can_return_without_writing(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6421: non-writing run still builds a valid artifact."""

    global _TMP_PATH
    _TMP_PATH = tmp_path
    output = tmp_path / "not-written.json"
    artifact = exp6421.run(
        date="20260814",
        result_path=output,
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        canonical_resolver=_fake_canonical_resolver,
        tokenizer_checker=_fake_tokenizer_checker,
        cuda_receipt_collector=_fake_cuda_receipts,
        write=False,
    )

    assert output.exists() is False
    assert artifact["test_exit_codes"][exp6421.RUN_COMMAND] is None
    assert artifact["duration_s"] >= 0.0
    exp6421.validate_artifact(artifact)
