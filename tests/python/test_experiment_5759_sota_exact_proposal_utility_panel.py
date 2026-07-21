"""Tests for Exp5759 SOTA exact proposal utility panel.

Spec refs: REQ-VERIFY-5759, REQ-BENCH-5759, SCENARIO-VERIFY-5759,
SCENARIO-VERIFY-5759-BLOCKED, SCENARIO-BENCH-5759,
SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5759_sota_exact_proposal_utility_panel as mod


REPO = Path(__file__).resolve().parents[2]
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5759_sota_exact_proposal_utility_panel.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py "
    "-m pytest tests/python/test_experiment_5759_sota_exact_proposal_utility_panel.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5759_sota_exact_proposal_utility_panel.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _fixture_runtime(
    model_spec: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    """SCENARIO-VERIFY-5759: fixture logits rank exact-optimal labels first."""

    del random_seeds
    rows = []
    for index, row in enumerate(candidate_rows):
        optimal_ids = set(row["exact_optimum_candidate_ids"])
        selected_label = next(
            item["label"] for item in row["label_mapping"] if item["candidate_id"] in optimal_ids
        )
        score_vector = {
            item["label"]: -5.0 - offset / 100.0 for offset, item in enumerate(row["label_mapping"])
        }
        score_vector[selected_label] = 10.0 + index / 1000.0
        rows.append(
            {
                "model_hf_id": model_spec["hf_id"],
                "instance_id": row["instance_id"],
                "prompt_hash": row["prompt_hash"],
                "score_vector": score_vector,
                "label_token_ids": {
                    item["label"]: [1000 + offset]
                    for offset, item in enumerate(row["label_mapping"])
                },
                "prompt_token_count": 128 + index,
                "timing": {"prefill_s": round(0.002 + index / 100000, 6)},
                "error": "",
            }
        )
    return {
        "model_hf_id": model_spec["hf_id"],
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1 | ggml-cuda fixture",
        },
        "gpu_assignment": model_spec["gpu"],
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 40,
        "gpu_memory_before_mb": 128,
        "gpu_memory_peak_mb": 4096,
        "gpu_memory_after_mb": 160,
        "cuda_offload_authenticated": True,
        "rows": rows,
    }


def _run_fixture(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / "checkpoint.json",
        preconditions_checked=mod.fixture_preconditions(tmp_path),
        score_runner=_fixture_runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_5759_specs_declare_panel_contract() -> None:
    """REQ-VERIFY-5759 and REQ-BENCH-5759: OpenSpec anchors Exp5759."""

    bench = BENCH_SPEC.read_text(encoding="utf-8")
    verify = VERIFY_SPEC.read_text(encoding="utf-8")
    bench_section = bench[bench.index("### REQ-BENCH-5759") : bench.index("### REQ-BENCH-3389")]
    verify_section = verify[
        verify.index("### REQ-VERIFY-5759") : verify.index("### REQ-VERIFY-5615")
    ]

    for marker in (
        "REQ-BENCH-5759",
        "SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS",
        str(mod.RESULT_RELATIVE_PATH),
        "`proposal_utility_lcb`",
        "`flagship_nonregression_count`",
        "`proposal_utility_ready_score`",
    ):
        assert marker in bench_section
    for marker in (
        "REQ-VERIFY-5759",
        "SCENARIO-VERIFY-5759-BLOCKED",
        "`verifier_is_oracle=true`",
        "`generated_text_scoring_used=false`",
        "`token_scores_are_semantic_authority=false`",
    ):
        assert marker in verify_section


def test_scenario_5759_freezes_science_panel_and_label_bijections(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5759: science rows, labels, budgets, and hashes are sealed."""

    preconditions = mod.fixture_preconditions(tmp_path)
    rows = mod.load_science_rows(preconditions)
    panel = mod.freeze_science_panel(rows)

    assert len(rows) == 60
    assert {row["split"] for row in rows} == {"science"}
    assert mod.science_split_hash(rows) == preconditions["science_split_hash"]
    assert mod.science_row_count(rows) == 60
    assert mod.family_counts(rows) == {family: 15 for family in mod.REQUIRED_FAMILIES}

    for row in panel:
        assert row["candidate_order_frozen_before_model_access"] is True
        assert row["label_bijection_complete"] is True
        assert row["matched_budget"]["candidate_count"] == len(row["candidate_ids"])
        assert row["matched_budget"]["top_k"] == mod.TOP_K
        assert row["matched_budget"]["exact_validator_call_budget"] == len(row["candidate_ids"])
        assert len({item["label"] for item in row["label_mapping"]}) == len(row["candidate_ids"])
        assert {item["candidate_id"] for item in row["label_mapping"]} == set(row["candidate_ids"])
        assert row["prompt_hash"] == mod.sha256_text(row["prompt"])


def test_scenario_5759_complete_fixture_artifact_and_gate_scalars(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5759: fixture proposal utility is exact-validator grounded."""

    artifact = _run_fixture(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.HEADLINE_MODEL_IDS)
    assert artifact["models_used"] == list(mod.HEADLINE_MODEL_IDS)
    assert artifact["science_row_count"] == 60
    assert artifact["verifier_is_oracle"] is True
    assert artifact["llm_judge_used"] is False
    assert artifact["generated_text_scoring_used"] is False
    assert artifact["token_scores_are_semantic_authority"] is False
    assert artifact["model_weight_mutation"] is False
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["authority_violation_count"] == 0
    assert artifact["proposal_utility_delta_overall"] > 0.0
    assert artifact["proposal_utility_lcb"] >= 0.0
    assert artifact["flagship_nonregression_count"] == 2
    assert artifact["proposal_utility_ready_score"] == pytest.approx(1.0)
    assert artifact["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)
    for model_id in mod.HEADLINE_MODEL_IDS:
        assert artifact["per_model_metrics"][model_id]["top_1_feasible_discovery"] == pytest.approx(
            1.0
        )
        assert artifact["per_model_metrics"][model_id][
            "top_k_exact_optimum_discovery"
        ] == pytest.approx(1.0)
        assert artifact["cuda_offload_authenticated"][model_id] is True
        assert artifact["confidence_intervals"]["by_model"][model_id]["lcb"] >= 0.0
    assert set(artifact["confidence_intervals"]) == {"by_row", "by_family", "by_model", "overall"}
    assert artifact["model_identity_shortcut_residual"]["max_abs_residual"] >= 0.0


def test_scenario_5759_blocked_preconditions_do_not_call_runner(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS: blocks before inference."""

    def forbidden_runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("score runner must not be called")

    preconditions = mod.fixture_preconditions(tmp_path)
    preconditions["preconditions_ready"] = False
    preconditions["blocked_reasons"] = ["mandated_gguf_missing"]
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "blocked.checkpoint.json",
        preconditions_checked=preconditions,
        score_runner=forbidden_runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["models_used"] == []
    assert artifact["proposal_utility_delta_overall"] == pytest.approx(0.0)
    assert artifact["proposal_utility_lcb"] == pytest.approx(-1.0)
    assert artifact["proposal_utility_ready_score"] == pytest.approx(0.0)
    assert artifact["preconditions_checked"]["blocked_reasons"] == ["mandated_gguf_missing"]
    assert mod.validate_artifact(artifact) is True


def test_scenario_5759_checkpoint_resume_skips_completed_model(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5759: checkpointed model receipts are reused exactly."""

    preconditions = mod.fixture_preconditions(tmp_path)
    rows = mod.freeze_science_panel(mod.load_science_rows(preconditions))
    first_model = dict(mod.MODEL_SPECS[0])
    first_model.update(preconditions["resolved_model_receipts"][first_model["hf_id"]])
    first_receipt = _fixture_runtime(first_model, rows, dict(mod.RANDOM_SEEDS))
    checkpoint = tmp_path / "resume.json"
    mod.write_checkpoint(
        {
            "panel_hash": mod.sha256_json([row["panel_row_hash"] for row in rows]),
            "runtime_receipts": [first_receipt],
        },
        checkpoint,
    )
    called: list[str] = []

    def counting_runner(
        model_spec: dict[str, Any],
        candidate_rows: list[dict[str, Any]],
        random_seeds: dict[str, int],
    ) -> dict[str, Any]:
        called.append(model_spec["hf_id"])
        return _fixture_runtime(model_spec, candidate_rows, random_seeds)

    artifact = mod.run(
        result_path=tmp_path / "resumed.json",
        checkpoint_path=checkpoint,
        preconditions_checked=preconditions,
        score_runner=counting_runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert called == list(mod.HEADLINE_MODEL_IDS[1:])
    assert artifact["models_used"] == list(mod.HEADLINE_MODEL_IDS)
    assert artifact["checkpoint_resume_receipt"]["resumed_model_ids"] == [mod.HEADLINE_MODEL_IDS[0]]
    assert artifact["checkpoint_resume_receipt"]["checkpoint_reused"] is True


def test_scenario_5759_validation_and_authority_negative_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5759-BLOCKED: schema and authority violations fail closed."""

    artifact = _run_fixture(tmp_path)

    wrapped_gate = deepcopy(artifact)
    wrapped_gate["proposal_utility_lcb"] = {"value": artifact["proposal_utility_lcb"]}
    with pytest.raises(ValueError, match="proposal_utility_lcb"):
        mod.validate_artifact(wrapped_gate)

    judge_used = deepcopy(artifact)
    judge_used["llm_judge_used"] = True
    judge_used["authority_violation_count"] = mod.authority_violation_count(judge_used)
    judge_used["proposal_utility_ready_score"] = mod.proposal_utility_ready_score(judge_used)
    judge_used["honest_verdict"] = mod.honest_verdict(judge_used)
    judge_used["reproducibility_checksum"] = mod.reproducibility_checksum(judge_used)
    assert judge_used["authority_violation_count"] > 0
    assert judge_used["proposal_utility_ready_score"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="llm_judge_used"):
        mod.validate_artifact(judge_used)

    wrong_models = deepcopy(artifact)
    wrong_models["MODEL_SPECS"] = wrong_models["MODEL_SPECS"][:-1]
    wrong_models["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_models)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(wrong_models)

    missing = deepcopy(artifact)
    del missing["verifier_is_oracle"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"].pop("status")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    verifier_false = deepcopy(artifact)
    verifier_false["verifier_is_oracle"] = False
    verifier_false["authority_violation_count"] = mod.authority_violation_count(verifier_false)
    verifier_false["proposal_utility_ready_score"] = mod.proposal_utility_ready_score(
        verifier_false
    )
    verifier_false["honest_verdict"] = mod.honest_verdict(verifier_false)
    verifier_false["reproducibility_checksum"] = mod.reproducibility_checksum(verifier_false)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(verifier_false)

    wrong_substrate = deepcopy(artifact)
    wrong_substrate["inference_substrate"] = "cpu"
    wrong_substrate["authority_violation_count"] = mod.authority_violation_count(wrong_substrate)
    wrong_substrate["proposal_utility_ready_score"] = mod.proposal_utility_ready_score(
        wrong_substrate
    )
    wrong_substrate["honest_verdict"] = mod.honest_verdict(wrong_substrate)
    wrong_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(wrong_substrate)

    wrong_gate_list = deepcopy(artifact)
    wrong_gate_list["producer_gate_fields"] = []
    wrong_gate_list["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_gate_list)
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(wrong_gate_list)

    wrong_authority_count = deepcopy(artifact)
    wrong_authority_count["authority_violation_count"] = 1
    wrong_authority_count["proposal_utility_ready_score"] = mod.proposal_utility_ready_score(
        wrong_authority_count
    )
    wrong_authority_count["honest_verdict"] = mod.honest_verdict(wrong_authority_count)
    wrong_authority_count["reproducibility_checksum"] = mod.reproducibility_checksum(
        wrong_authority_count
    )
    with pytest.raises(ValueError, match="authority_violation_count"):
        mod.validate_artifact(wrong_authority_count)

    wrong_ready = deepcopy(artifact)
    wrong_ready["proposal_utility_ready_score"] = 0.0
    wrong_ready["honest_verdict"] = mod.honest_verdict(wrong_ready)
    wrong_ready["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_ready)
    with pytest.raises(ValueError, match="proposal_utility_ready_score"):
        mod.validate_artifact(wrong_ready)

    wrong_verdict = deepcopy(artifact)
    wrong_verdict["honest_verdict"] = "partial"
    wrong_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(wrong_verdict)

    wrong_checksum = deepcopy(artifact)
    wrong_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(wrong_checksum)


def test_scenario_5759_helper_negative_controls(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS: helper faults are explicit."""

    preconditions = mod.fixture_preconditions(tmp_path)
    rows = mod.load_science_rows(preconditions)
    panel = mod.freeze_science_panel(rows)

    bad_hash = deepcopy(preconditions)
    bad_hash["science_split_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="science_split_hash"):
        mod.load_science_rows(bad_hash)

    bad_count = deepcopy(preconditions)
    bad_count["science_row_count"] = 59
    with pytest.raises(ValueError, match="science_row_count"):
        mod.load_science_rows(bad_count)

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json_object(list_json)

    first = panel[0]
    assert mod._selected_order_from_scores(first, {}) == ([], "missing_score")
    nonfinite = {item["label"]: 0.0 for item in first["label_mapping"]}
    nonfinite[first["label_mapping"][0]["label"]] = float("nan")
    assert mod._selected_order_from_scores(first, nonfinite) == ([], "non_finite_score")
    assert mod._first_position(["not-a-target"], {"target"}) == 2
    assert mod._aggregate_metric_rows([]) == {"row_count": 0}
    assert mod._bootstrap_interval([0.25], 1)["lcb"] == pytest.approx(0.25)

    missing_eval = deepcopy(first)
    missing_eval["solution_evaluations"] = {}
    assert mod._validator_disagreement_count([missing_eval]) == len(first["candidate_ids"])
