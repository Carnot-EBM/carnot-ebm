"""Tests for Exp 4374 DiffusionGemma scorer repair-or-retire.

Spec: REQ-VERIFY-4374, SCENARIO-VERIFY-4374.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from carnot import experiment_4374_diffusiongemma_scorer_repair_or_retire as exp


class TinyTokenizer:
    mask_token_id = exp.MASK_TOKEN_ID

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        return [max(1, ord(ch) % 251) for ch in text[:64]] or [0]

    def detokenize(self, token_ids: list[int]) -> bytes:
        return "".join(chr(max(32, min(126, int(token_id)))) for token_id in token_ids).encode(
            "utf-8"
        )


class KeywordScorer:
    mask_token_id = exp.MASK_TOKEN_ID

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        del step
        text = "".join(
            chr(int(token_id) - exp.TOKEN_OFFSET)
            for token_id in canvas_ids
            if int(token_id) != self.mask_token_id
            and 0 <= int(token_id) - exp.TOKEN_OFFSET <= 0x10FFFF
        )
        return 0.05 if ("answer is" in text or "return" in text) else 5.0


class LeakSequence:
    def __init__(self, *passed: bool) -> None:
        self._passed = list(passed)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        passed = self._passed.pop(0) if self._passed else False
        return {
            "status": "measured",
            "fresh_heldout_n": 160,
            "unmasked_auroc": 0.9 if passed else 0.91,
            "answer_masked_auroc": 0.82 if passed else 0.49,
            "answer_masked_cells": 80,
            "scorer_leak_recheck_passed": passed,
        }


def _binary(tmp_path: Path) -> Path:
    path = tmp_path / "llama-diffusion-gemma-eval"
    path.write_bytes(b"binary")
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
        "unguided": _completion(f"raw {wrong} ???", 8.0),
        "best_of_n": _completion(f"answer is {answer if best_correct else wrong} via best", 9.0),
        "intrinsic_svf": _completion(
            f"answer is {answer if svf_correct else wrong} with local check", 10.0
        ),
        "prism_carnot": _completion(
            f"return answer is {answer if prism_correct else wrong} after coherent blocks", 11.0
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    arms["prism_carnot"]["frontier_completions"] = [
        f"return answer is {answer} branch {branch}" for branch in range(config.frontier_width)
    ]
    return {"status": "generated", "arms": arms}


def _null_proposals(
    *,
    task: exp.FreeFormTask,
    global_index: int,
    config: exp.PrismSearchConfig,
    **_: Any,
) -> dict[str, Any]:
    answer = task.expected_answer
    wrong = str(int(answer) + 1)
    arms = {
        "unguided": _completion(f"raw {answer if global_index < 20 else wrong}", 8.0),
        "best_of_n": _completion(f"answer is {answer if global_index < 120 else wrong} via best", 9.0),
        "intrinsic_svf": _completion(
            f"answer is {answer if global_index < 110 else wrong} with local check", 10.0
        ),
        "prism_carnot": _completion(
            f"return answer is {answer if global_index < 100 else wrong} after coherent blocks",
            11.0,
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    arms["prism_carnot"]["frontier_completions"] = [
        f"return answer is {answer} branch {branch}" for branch in range(config.frontier_width)
    ]
    return {"status": "generated", "arms": arms}


def _identical_proposals(*, task: exp.FreeFormTask, **_: Any) -> dict[str, Any]:
    text = f"answer is {task.expected_answer}"
    return {"status": "generated", "arms": {arm: _completion(text, 10.0) for arm in exp.ARM_KEYS}}


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
            f"raw {answer if control_correct else wrong} via unguided {global_index}", 8.0
        ),
        "best_of_n": _completion(
            f"answer is {answer if control_correct else wrong} via best {global_index}", 9.0
        ),
        "intrinsic_svf": _completion(
            f"answer is {answer if control_correct else wrong} via svf {global_index}", 10.0
        ),
        "prism_carnot": _completion(
            f"return answer is {answer if prism_correct else wrong} via prism {global_index}",
            11.0,
        ),
    }
    arms["prism_carnot"]["uses_external_scorer"] = True
    return {"status": "generated", "arms": arms}


def _proposal_failure(**_: Any) -> dict[str, Any]:
    return {"status": "blocked_test_proposal_failure"}


def _repair_ok(tmp_path: Path):
    def _repair(**_: Any) -> dict[str, Any]:
        path = tmp_path / "repaired_scorer.pkl"
        path.write_bytes(b"repaired")
        return {
            "ok": True,
            "status": "repaired",
            "scorer": KeywordScorer(),
            "scorer_path": str(path),
            "process_ranking_auroc": 0.86,
            "masked_answer_recovery_auroc": 0.5,
            "train_records": 120,
            "heldout_records": 40,
        }

    return _repair


def _repair_fail(**_: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "repair_failed",
        "error": "test repair failed",
        "process_ranking_auroc": 0.5,
        "masked_answer_recovery_auroc": 0.9,
    }


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
        "repaired_scorer_path": tmp_path / "exp4374_repaired.pkl",
        "resolve_gguf_fn": lambda **_: str(gguf_path),
        "vocab_loader_fn": lambda _path, _probe: _loader_result(),
        "process_rows_fn": lambda: [],
        "scorer_loader_fn": lambda _path: KeywordScorer(),
        "leak_recheck_fn": LeakSequence(True),
        "repair_scorer_fn": _repair_ok(tmp_path),
        "search_corpus_items_fn": _corpus_items,
        "max_tasks_per_seed": 80,
        "seeds": (4374, 4375, 4376),
        "minimum_duration_s": 0.0,
        "adversarial_verify_fn": _clean_adversarial_verify,
    }


def test_req_verify_4374_spec_declares_repair_or_retire_contract() -> None:
    """REQ-VERIFY-4374: OpenSpec declares scorer repair, CoDiLA, and retirement."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4374",
        "SCENARIO-VERIFY-4374",
        "experiment_4374_diffusiongemma_scorer_repair_or_retire.py",
        "scorer_requalified_leak_clean",
        "codila_control_differentiates",
        "retired_in_generation_conversion_unmeasurable",
        "verifier_is_oracle=false",
        "arXiv:2603.20216",
    ):
        assert marker in spec


def test_scenario_4374_missing_pr_binary_blocks_before_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: missing PR binary stops before scorer/corpus work."""

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
    assert artifact["scorer_requalified_leak_clean"] is False
    assert artifact["codila_control_differentiates"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"


def test_scenario_4374_missing_scorer_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: missing leak-robust scorer stops before CoDiLA."""

    common = _common_run_kwargs(tmp_path)
    Path(common["scorer_path"]).unlink()
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-scorer.json",
        proposal_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_leak_robust_scorer_unavailable"
    assert artifact["benchmark_n"] == 0
    assert artifact["preconditions_checked"][-1]["resource"] == "leak_robust_scorer_gate"
    assert artifact["preconditions_checked"][-1]["ok"] is False


def test_scenario_4374_undersized_generation_corpus_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: undersized free-form corpus blocks measurement."""

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


def test_scenario_4374_repairs_scorer_and_reports_generation_gain(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: failed recheck can repair before a fixed-NFE gain."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = LeakSequence(False, True)
    artifact_path = tmp_path / "gain.json"
    artifact = exp.run(
        artifact_path=artifact_path,
        proposal_fn=_winning_proposals,
        bootstrap_resamples=2500,
        **common,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: diffusiongemma_repair_or_retire_generation_gain"
    assert artifact["s3_guided_beats_control"] is True
    assert artifact["scorer_requalified_leak_clean"] is True
    assert artifact["scorer_requalification"]["repair_attempted"] is True
    assert artifact["codila_control_differentiates"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["benchmark_n"] == 240
    assert artifact["s3_minus_best_of_n_delta"] == pytest.approx(0.333333)
    assert artifact["fixed_nfe_summary"]["s3_minus_intrinsic_svf_delta"] == pytest.approx(0.25)
    assert artifact["s3_gain_ci95"][0] > 0.0
    assert artifact["model_specs"]["guidance_source"]["kind"] == "requalified_scorer"
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_scenario_4374_generation_failures_are_partial(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: proposal failures are reported as partial data."""

    artifact = exp.run(
        artifact_path=tmp_path / "partial.json",
        proposal_fn=_proposal_failure,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "partial: diffusiongemma_repair_or_retire_incomplete"
    assert artifact["benchmark_n"] == 0
    assert artifact["retirement_gate"]["reason"] == "generation_incomplete"
    assert artifact["benchmark_failures"]


def test_scenario_4374_irreparable_scorer_and_codila_degenerate_retires(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4374: leaky scorer plus non-differentiating CoDiLA retires."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = LeakSequence(False, False)
    common["repair_scorer_fn"] = _repair_fail
    artifact = exp.run(
        artifact_path=tmp_path / "retired.json",
        proposal_fn=_identical_proposals,
        **common,
    )

    assert artifact["honest_verdict"] == "retired_in_generation_conversion_unmeasurable"
    assert artifact["scorer_requalified_leak_clean"] is False
    assert artifact["codila_control_differentiates"] is False
    assert artifact["controls_differentiated"] is False
    assert artifact["benchmark_n"] == 0
    assert artifact["retirement_gate"]["reason"] == "scorer_leaky_and_codila_not_differentiating"


def test_scenario_4374_noop_guard_retires_repeated_control_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: bit-identical generated arms trigger retirement."""

    artifact = exp.run(
        artifact_path=tmp_path / "controls.json",
        proposal_fn=_identical_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "retired_in_generation_conversion_unmeasurable"
    assert artifact["scorer_requalified_leak_clean"] is True
    assert artifact["controls_differentiated"] is False
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["control_noop_guard"]["bit_identical_completion_pairs"]
    assert artifact["retirement_gate"]["reason"] == "controls_not_differentiable"


def test_scenario_4374_full_run_tautology_retires(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: full-run delta tautology repeats the control failure."""

    artifact = exp.run(
        artifact_path=tmp_path / "tautology.json",
        proposal_fn=_tautology_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "retired_in_generation_conversion_unmeasurable"
    assert artifact["benchmark_n"] == 240
    assert artifact["controls_differentiated"] is False
    assert artifact["control_noop_guard"]["tautology_delta_pairs"]
    assert artifact["retirement_gate"]["reason"] == "controls_not_differentiable"


def test_scenario_4374_clean_powered_null_is_decision_grade(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4374: differentiated controls can still retire on a clean null."""

    artifact = exp.run(
        artifact_path=tmp_path / "null.json",
        proposal_fn=_null_proposals,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "complete: clean_powered_null_in_generation_conversion"
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["controls_differentiated"] is True
    assert artifact["codila_control_differentiates"] is True
    assert artifact["s3_minus_best_of_n_delta"] < 0.0
    assert artifact["fixed_nfe_summary"]["s3_minus_intrinsic_svf_delta"] < 0.0


def test_req_verify_4374_codila_requalification_and_validation_helpers(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4374: helper APIs enforce CoDiLA and bare artifact fields."""

    codila = exp.CodilaLocalCoherenceScorer()
    coherent = codila.score_completion("return answer is 42 after coherent local blocks")
    degenerate = codila.score_completion("???? ????? ?????")
    assert coherent < degenerate
    assert codila.score_partial_state([exp.MASK_TOKEN_ID] * exp.CANVAS_LEN, 0) > coherent
    assert codila.score_partial_state([ord("a") + exp.TOKEN_OFFSET, exp.MASK_TOKEN_ID], 0) > 0.0
    assert exp._local_transition_score(["only"], 4) == 0.0

    records = [
        {
            "unguided_completion": "????",
            "best_of_n_completion": "answer answer answer",
            "intrinsic_svf_completion": "answer is 8",
            "prism_carnot_completion": "return answer is 8 after coherent local blocks",
        }
    ]
    codila_control = exp.assess_codila_control_differentiation(records, codila)
    assert codila_control["codila_control_differentiates"] is True
    assert exp.assess_codila_control_differentiation([], codila)[
        "codila_control_differentiates"
    ] is False

    rows = exp.generation_items_to_labeled_rows(
        [
            {"prompt": "Return x", "expected_answer": "abc", "task_id": "text"},
            {"expected_answer": "1", "task_id": "missing_prompt"},
        ]
    )
    assert len(rows) == 2
    assert "<<answer>>abc_wrong" in rows[1]["step_text"]

    repair = exp.repair_generation_corpus_scorer(
        corpus_items=_corpus_items(24),
        scorer_path=tmp_path / "real_repair.pkl",
        seed=4374,
        max_features=200,
    )
    assert repair["ok"] is True
    assert Path(repair["scorer_path"]).exists()
    assert repair["train_records"] > 0

    common = _common_run_kwargs(tmp_path)
    artifact = exp.run(
        artifact_path=tmp_path / "valid.json",
        proposal_fn=_winning_proposals,
        **common,
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("s3_guided_beats_control", lambda a: a.update({"s3_guided_beats_control": 1})),
        (
            "scorer_requalified_leak_clean",
            lambda a: a.update({"scorer_requalified_leak_clean": "true"}),
        ),
        (
            "codila_control_differentiates",
            lambda a: a.update({"codila_control_differentiates": 1}),
        ),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        ("s3_minus_best_of_n_delta", lambda a: a.update({"s3_minus_best_of_n_delta": "0"})),
        ("s3_gain_ci95", lambda a: a.update({"s3_gain_ci95": [0.0]})),
        ("nfe_budget", lambda a: a.update({"nfe_budget": "16"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "s3_guided_beats_control cannot be true",
            lambda a: a.update({"s3_guided_beats_control": True, "s3_gain_ci95": [0.0, 0.2]}),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)
