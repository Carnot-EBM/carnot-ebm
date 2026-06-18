"""Tests for Exp 4359 Prism-hardened free-form verifier-guided search.

Spec: REQ-VERIFY-4359, SCENARIO-VERIFY-4359.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from carnot import experiment_4359_prism_hardened_verifier_guided_search as exp


class TinyTokenizer:
    mask_token_id = exp.MASK_TOKEN_ID

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        if not text:
            return [0]
        return [max(1, ord(ch) % 251) for ch in text[:64]]

    def detokenize(self, token_ids: list[int]) -> bytes:
        return "".join(chr(max(32, min(126, int(token_id)))) for token_id in token_ids).encode(
            "utf-8"
        )


class KeywordScorer:
    mask_token_id = exp.MASK_TOKEN_ID

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        del step
        text = "".join(
            chr(int(token_id))
            for token_id in canvas_ids
            if int(token_id) != self.mask_token_id and 0 <= int(token_id) <= 0x10FFFF
        )
        return 0.05 if "answer is" in text else 5.0


def _binary(tmp_path: Path, payload: bytes = b"binary") -> Path:
    path = tmp_path / "llama-diffusion-gemma-eval"
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    return tmp_path


def _loader_result() -> exp.VocabLoadResult:
    return exp.VocabLoadResult(
        ok=True,
        backend="test",
        mode="embedded_vocab_metadata",
        elapsed_s=0.001,
        token_count=1,
        token_ids=(exp.MASK_TOKEN_ID,),
        detail="test loader",
        tokenizer=TinyTokenizer(),
    )


def _scorer_artifact(path: Path, scorer_path: Path, *, audit_passed: bool = True) -> None:
    payload = {
        "honest_verdict": (
            "complete: leak_robust_partial_state_scorer_built"
            if audit_passed
            else "complete_no_leak_robust_partial_state_scorer"
        ),
        "scorer_leak_audit_passed": audit_passed,
        "masked_answer_recovery_auroc": 0.559682 if audit_passed else 0.9,
        "process_ranking_auroc": 0.704633 if audit_passed else 0.5,
        "scorer_module_path": str(scorer_path),
        "verifier_is_oracle": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path.write_bytes(b"test scorer placeholder")


def _corpus_items(n: int = 120) -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"free_math_{index:03d}",
            "family": "math",
            "prompt": f"Return only the integer result of {index} + 7.",
            "expected_answer": str(index + 7),
            "answer_cells": [0],
        }
        for index in range(n)
    ]


def _passing_leak_recheck() -> dict[str, Any]:
    return {
        "status": "measured",
        "fresh_heldout_n": 160,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.83,
        "scorer_leak_recheck_passed": True,
    }


def _failing_leak_recheck() -> dict[str, Any]:
    return {
        "status": "measured",
        "fresh_heldout_n": 160,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.51,
        "scorer_leak_recheck_passed": False,
    }


def _completion(text: str, score: float) -> dict[str, Any]:
    return {
        "completion": text,
        "mean_logit": score,
        "intrinsic_svf_score": score / 10.0,
        "external_energy": 1.0 / max(score, 1.0),
        "nfe_used": 16,
        "uses_external_scorer": False,
    }


def _winning_proposals(
    *,
    task: exp.FreeFormTask,
    global_index: int,
    config: exp.PrismSearchConfig,
    **_: Any,
) -> dict[str, Any]:
    answer = task.expected_answer
    wrong = str(int(answer) + 1)
    best_correct = global_index < 90
    svf_correct = global_index < 110
    prism_correct = global_index < 170
    arms = {
        "unguided": _completion(f"answer is {wrong} via unguided", 8.0),
        "best_of_n": _completion(
            f"answer is {answer if best_correct else wrong} via best", 9.0
        ),
        "intrinsic_svf": _completion(
            f"answer is {answer if svf_correct else wrong} via svf", 10.0
        ),
        "prism_carnot": _completion(
            f"answer is {answer if prism_correct else wrong} via prism", 11.0
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    arms["prism_carnot"]["frontier_completions"] = [
        f"answer is {answer} branch {branch}" for branch in range(config.frontier_width)
    ]
    return {"status": "generated", "arms": arms}


def _identical_proposals(
    *,
    task: exp.FreeFormTask,
    **_: Any,
) -> dict[str, Any]:
    text = f"answer is {task.expected_answer}"
    return {
        "status": "generated",
        "arms": {arm: _completion(text, 10.0) for arm in exp.ARM_KEYS},
    }


def _tautology_proposals(
    *,
    task: exp.FreeFormTask,
    global_index: int,
    **_: Any,
) -> dict[str, Any]:
    answer = task.expected_answer
    wrong = str(int(answer) + 1)
    prism_correct = global_index < 120
    control_correct = global_index < 60
    arms = {
        "unguided": _completion(
            f"answer is {answer if control_correct else wrong} via unguided", 8.0
        ),
        "best_of_n": _completion(
            f"answer is {answer if control_correct else wrong} via best", 9.0
        ),
        "intrinsic_svf": _completion(
            f"answer is {answer if control_correct else wrong} via svf", 10.0
        ),
        "prism_carnot": _completion(
            f"answer is {answer if prism_correct else wrong} via prism", 11.0
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    return {"status": "generated", "arms": arms}


def _null_proposals(
    *,
    task: exp.FreeFormTask,
    global_index: int,
    **_: Any,
) -> dict[str, Any]:
    answer = task.expected_answer
    wrong = str(int(answer) + 1)
    unguided_correct = global_index < 20
    best_correct = global_index < 120
    svf_correct = global_index < 110
    prism_correct = global_index < 100
    arms = {
        "unguided": _completion(
            f"answer is {answer if unguided_correct else wrong} via unguided", 8.0
        ),
        "best_of_n": _completion(
            f"answer is {answer if best_correct else wrong} via best", 9.0
        ),
        "intrinsic_svf": _completion(
            f"answer is {answer if svf_correct else wrong} via svf", 10.0
        ),
        "prism_carnot": _completion(
            f"answer is {answer if prism_correct else wrong} via prism", 11.0
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    arms["prism_carnot"]["frontier_completions"] = [
        f"answer is {answer} branch {branch}" for branch in range(4)
    ]
    return {"status": "generated", "arms": arms}


def _proposal_failure(**_: Any) -> dict[str, Any]:
    return {"status": "blocked_test_proposal_failure"}


def _clean_adversarial_verify(_path: Path) -> dict[str, Any]:
    return {"status": "clean", "critical_flags": [], "warn_flags": [], "returncode": 0}


def _common_run_kwargs(tmp_path: Path) -> dict[str, Any]:
    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4337.json"
    scorer_path = tmp_path / "dina_lrm_partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)
    return {
        "pr_binary_path": _binary(tmp_path),
        "cache_root": cache_root,
        "scorer_artifact_path": scorer_artifact,
        "scorer_path": scorer_path,
        "resolve_gguf_fn": lambda **_: str(gguf_path),
        "vocab_loader_fn": lambda _path, _probe: _loader_result(),
        "process_rows_fn": lambda: [],
        "scorer_loader_fn": lambda _path: KeywordScorer(),
        "leak_recheck_fn": lambda **_: _passing_leak_recheck(),
        "search_corpus_items_fn": _corpus_items,
        "max_tasks_per_seed": 80,
        "seeds": (4359, 4360, 4361),
        "minimum_duration_s": 0.0,
        "adversarial_verify_fn": _clean_adversarial_verify,
    }


def test_req_verify_4359_spec_declares_free_form_prism_contract() -> None:
    """REQ-VERIFY-4359: OpenSpec declares the Prism free-form generation gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4359",
        "SCENARIO-VERIFY-4359",
        "experiment_4359_prism_hardened_verifier_guided_search.py",
        "free-form",
        "controls_not_differentiable",
        "scorer_disagreement_rate",
        "branch_diversity",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4359_missing_pr_binary_blocks_before_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: missing PR binary stops before scorer/corpus work."""

    def fail_resolve(**_: Any) -> str:
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        scorer_artifact_path=tmp_path / "missing-4337.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        resolve_gguf_fn=fail_resolve,
        search_corpus_items_fn=pytest.fail,
        minimum_duration_s=0.0,
        adversarial_verify_fn=_clean_adversarial_verify,
    )

    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4359_leak_robust_scorer_must_pass_and_load(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: failed Exp 4337 audit blocks Prism search."""

    common = _common_run_kwargs(tmp_path)
    _scorer_artifact(
        Path(common["scorer_artifact_path"]),
        Path(common["scorer_path"]),
        audit_passed=False,
    )
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-scorer.json",
        proposal_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_leak_robust_scorer_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "leak_robust_scorer_gate"
    assert artifact["preconditions_checked"][-1]["scorer_leak_audit_passed"] is False
    assert artifact["benchmark_n"] == 0


def test_scenario_4359_leak_recheck_blocks_search(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: answer-masked collapse stops before generation."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = lambda **_: _failing_leak_recheck()
    artifact = exp.run(
        artifact_path=tmp_path / "leaky.json",
        proposal_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "scorer_leaky_in_search_corpus"
    assert artifact["scorer_leak_recheck_passed"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4359_search_corpus_unavailable_blocks_before_recheck(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4359: undersized free-form corpus is terminal."""

    common = _common_run_kwargs(tmp_path)
    common["search_corpus_items_fn"] = lambda: _corpus_items(79)
    common["leak_recheck_fn"] = pytest.fail
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-corpus.json",
        proposal_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_search_corpus_unavailable"
    assert artifact["benchmark_n"] == 0
    assert artifact["preconditions_checked"][-1]["resource"] == "prism_free_form_search_corpus"


def test_scenario_4359_noop_guard_rejects_identical_free_form_arms(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: bit-identical generated text rejects utility reporting."""

    artifact = exp.run(
        artifact_path=tmp_path / "controls.json",
        proposal_fn=_identical_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["benchmark_n"] == 0
    assert artifact["control_noop_guard"]["bit_identical_completion_pairs"]


def test_scenario_4359_proposal_failures_are_partial(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: generation failures do not become fabricated data."""

    artifact = exp.run(
        artifact_path=tmp_path / "partial.json",
        proposal_fn=_proposal_failure,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "partial: prism_search_generation_incomplete"
    assert artifact["benchmark_n"] == 0
    assert artifact["benchmark_failures"]


def test_scenario_4359_tautology_guard_rejects_equal_deltas(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: >5-sig-fig delta agreement is terminal."""

    artifact = exp.run(
        artifact_path=tmp_path / "tautology.json",
        proposal_fn=_tautology_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["control_noop_guard"]["tautology_delta_pairs"]
    assert artifact["s3_minus_best_of_n_delta"] == pytest.approx(
        artifact["s3_minus_intrinsic_svf_delta"]
    )


def test_scenario_4359_clean_powered_null_is_distinct(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: a differentiated null is reported without a gain."""

    artifact = exp.run(
        artifact_path=tmp_path / "null.json",
        proposal_fn=_null_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "complete: clean_powered_null_prism_carnot"
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["controls_differentiated"] is True
    assert artifact["s3_minus_best_of_n_delta"] < 0.0
    assert artifact["s3_minus_intrinsic_svf_delta"] < 0.0


def test_scenario_4359_complete_path_reports_prism_gain(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4359: complete run reports Prism-Carnot utility at fixed NFE."""

    artifact_path = tmp_path / "artifact.json"
    artifact = exp.run(
        artifact_path=artifact_path,
        proposal_fn=_winning_proposals,
        bootstrap_resamples=2500,
        **_common_run_kwargs(tmp_path),
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: prism_carnot_guided_beats_control"
    assert artifact["s3_guided_beats_control"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["scorer_leak_recheck_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["seed_count"] == 3
    assert artifact["benchmark_n_per_seed"] == 80
    assert artifact["benchmark_n"] == 240
    assert artifact["nfe_budget"] == 16
    assert artifact["condition_pass_counts"]["best_of_n"] == 90
    assert artifact["condition_pass_counts"]["intrinsic_svf"] == 110
    assert artifact["condition_pass_counts"]["prism_carnot"] == 170
    assert artifact["s3_minus_best_of_n_delta"] == pytest.approx(0.333333)
    assert artifact["s3_minus_intrinsic_svf_delta"] == pytest.approx(0.25)
    assert artifact["s3_gain_ci95"][0] > 0.0
    assert artifact["branch_diversity"]["mean_unique_completions"] == pytest.approx(4.0)
    assert artifact["scorer_disagreement_rate"] > 0.0
    assert artifact["model_specs"]["prism_hts_config"]["nfe_budget"] == 16
    assert artifact["model_specs"]["intrinsic_svf_baseline"]["uses_external_scorer"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_req_verify_4359_validation_and_artifact_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-4359: validators enforce bare decision fields."""

    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_prism_rows([])

    controls = exp.assess_prism_control_differentiation(
        [
            {
                "task_id": "t0",
                "unguided": False,
                "best_of_n": True,
                "intrinsic_svf": True,
                "prism_carnot": True,
            },
            {
                "task_id": "t1",
                "unguided": False,
                "best_of_n": False,
                "intrinsic_svf": True,
                "prism_carnot": True,
            },
        ],
        [
            {
                "unguided_completion": "answer is 1 via unguided",
                "best_of_n_completion": "answer is 1 via best",
                "intrinsic_svf_completion": "answer is 2 via svf",
                "prism_carnot_completion": "answer is 3 via prism",
            },
            {
                "unguided_completion": "answer is 4 via unguided",
                "best_of_n_completion": "answer is 5 via best",
                "intrinsic_svf_completion": "answer is 6 via svf",
                "prism_carnot_completion": "answer is 7 via prism",
            },
        ],
    )
    assert controls["controls_differentiated"] is False

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True},
            {"resource": "leak_robust_scorer_gate", "ok": True, "scorer_path": str(tmp_path / "s.pkl")},
            {"resource": "prism_free_form_search_corpus", "ok": True},
        ]
    }
    summary = exp.summarize_prism_rows(
        [
            {
                "task_id": f"t{i}",
                "unguided": False,
                "best_of_n": i < 10,
                "intrinsic_svf": i < 20,
                "prism_carnot": i < 40,
            }
            for i in range(80)
        ],
        resamples=2500,
        seed=4359,
    )
    artifact = exp.build_artifact(
        honest_verdict="complete: clean_powered_null_prism_carnot",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary | {"s3_guided_beats_control": False},
        leak_recheck=_passing_leak_recheck(),
        controls=controls | {"controls_differentiated": True, "tautology_delta_pairs": []},
        scorer_gate={"ok": True, "scorer_path": str(tmp_path / "s.pkl")},
        corpus_check={"ok": True, "name": "free_form_math_code_v1", "checksum": "abc"},
        corpus_items=_corpus_items(),
        config=exp.PrismSearchConfig(),
        adversarial_verify={"status": "clean", "critical_flags": []},
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("s3_guided_beats_control", lambda a: a.update({"s3_guided_beats_control": 1})),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        ("s3_minus_best_of_n_delta", lambda a: a.update({"s3_minus_best_of_n_delta": "0"})),
        (
            "s3_minus_intrinsic_svf_delta",
            lambda a: a.update({"s3_minus_intrinsic_svf_delta": "0"}),
        ),
        ("s3_gain_ci95", lambda a: a.update({"s3_gain_ci95": [0.0]})),
        ("scorer_disagreement_rate", lambda a: a.update({"scorer_disagreement_rate": "0"})),
        ("scorer_leak_recheck_passed", lambda a: a.update({"scorer_leak_recheck_passed": 1})),
        ("nfe_budget", lambda a: a.update({"nfe_budget": "16"})),
        ("benchmark_n", lambda a: a.update({"benchmark_n": "80"})),
        ("branch_diversity", lambda a: a.update({"branch_diversity": []})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "Prism fixed-NFE gain cannot be true",
            lambda a: a.update({"s3_guided_beats_control": True, "s3_gain_ci95": [0.0, 0.2]}),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)


def test_req_verify_4359_utility_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-VERIFY-4359: small helpers cover deterministic non-live branches."""

    defaults = exp.default_free_form_search_items(3)
    assert {item["family"] for item in defaults} == {"math", "code"}

    tasks = exp.build_free_form_search_tasks(defaults, max_tasks=2, seed=4359)
    assert len(tasks) == 2
    assert exp.evaluate_free_form_completion(tasks[0], f"return {tasks[0].expected_answer}")
    assert not exp.evaluate_free_form_completion(tasks[0], "return -999")

    leak_rows = exp._leak_recheck_items(
        [
            {"prompt": "", "expected_answer": "1"},
            {"task_id": "text", "prompt": "p", "expected_answer": "abc"},
        ]
    )
    assert len(leak_rows) == 2
    assert "abc_wrong" in leak_rows[1]["step_text"]

    assert exp._branch_diversity([])["status"] == "not_run"
    fallback_diversity = exp._branch_diversity([{"task_id": "x"}])
    assert fallback_diversity["mean_unique_completions"] == 1.0
    assert exp._scorer_disagreement_rate([]) == 0.0
    assert exp._intrinsic_svf_score([]) == 0.0
    assert exp._intrinsic_svf_score([1.0, 3.0]) == pytest.approx(1.9)
    assert exp._tautology_delta_pairs({"s3_minus_best_of_n_delta": 0.1}) == []
    assert exp._significant_digits_match(0.0, 0.1, 5) is False
    assert exp._normalize_completion_text(" a\n\x00 b ") == "a b"
    assert exp._preview("abcdef", limit=3) == "abc..."
    assert exp._seed_checkpoint_path(None, 1) is None
    exp._checkpoint(None, rows=[], records=[], failures=[])

    sleeps: list[float] = []
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 0.01)
    assert sleeps and sleeps[0] > 0.0

    path = exp._seed_checkpoint_path(tmp_path / "artifact.checkpoint.json", 4359)
    assert path is not None and path.name.endswith("seed4359.json")
