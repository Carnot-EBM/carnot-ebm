"""Tests for Exp 4264 code oracle-distinct replication retry.

Spec refs: REQ-VERIFY-4264, SCENARIO-VERIFY-4264.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import code_oracle_distinct_replication_retry_4264 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _adversarial_clean(_path: Path) -> dict:
    return {
        "returncode": 0,
        "reports": [
            {
                "flags": [],
                "flag_count": 0,
                "max_severity": 0,
            }
        ],
    }


def _write_exp4233_source(root: Path) -> None:
    source_rel = Path("results/exp4233_source_fixture.jsonl")
    _write_jsonl(
        root / source_rel,
        [
            {
                "task_id": "HumanEval/source",
                "completion": "def source_fixture(x):\n    return x\n",
                "hidden_pass": True,
                "source_draw_index": 0,
            }
        ],
    )
    _write_json(
        root / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json",
        {
            "honest_verdict": "complete: code_oracle_distinct_beats_vote",
            "candidate_pool": {
                "source_id": "exp4233_fixture",
                "source_paths": [str(source_rel)],
            },
        },
    )


def _write_evalplus_checkpoint_fixture(root: Path, rel: str, *, task_count: int = 6) -> Path:
    evaluations_by_task: dict[str, list[dict]] = {}
    for task_index in range(task_count):
        task_id = f"HumanEval/{task_index}"
        function_name = f"solve_second_{task_index}"
        correct = (
            f"def {function_name}(x):\n"
            "    # PASS_PATTERN stable invariant candidate\n"
            "    return x + 1\n"
        )
        wrong = (
            f"def {function_name}(x):\n"
            "    # WRONG_PATTERN brittle shortcut candidate\n"
            "    return x - 1\n"
        )
        evaluations_by_task[task_id] = [
            {
                "draw_index": 0,
                "status": "ok",
                "code": wrong,
                "visible_passes": [True],
                "hidden_passes": [False, True],
                "generation_seconds": 0.2,
                "truncated": False,
                "error": None,
            },
            {
                "draw_index": 1,
                "status": "ok",
                "code": wrong,
                "visible_passes": [True],
                "hidden_passes": [False, True],
                "generation_seconds": 0.2,
                "truncated": False,
                "error": None,
            },
            {
                "draw_index": 2,
                "status": "ok",
                "code": correct,
                "visible_passes": [True],
                "hidden_passes": [True, True],
                "generation_seconds": 0.2,
                "truncated": False,
                "error": None,
            },
        ]
    path = root / rel
    _write_json(
        path,
        {
            "experiment": "experiment_4057_offarc_power_evalplus_checkpoint",
            "schema": "fixture",
            "evaluation_corpus": "EvalPlus HumanEval+/MBPP+ hidden tests",
            "k_candidates_per_task": 3,
            "completed_task_ids": sorted(evaluations_by_task),
            "evaluations_by_task": evaluations_by_task,
            "model_specs": {"local_generator": "unsloth/gemma-4-12B-it-GGUF"},
        },
    )
    return path


def test_req_4264_spec_declares_retry_or_retire_contract() -> None:
    """REQ-VERIFY-4264: OpenSpec declares the retry/retire artifact contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4264",
        "SCENARIO-VERIFY-4264",
        "python/carnot/reporting/code_oracle_distinct_replication_retry_4264.py",
        "results/experiment_4264_code_oracle_distinct_replication_retry.py",
        "blocked_code_gen_model_not_cached",
        "code_replication_retired",
        "code_replication_beats_vote",
        "code_predictor_minus_vote_delta",
        "code_predictor_minus_vote_ci95",
        "oracle_at_k",
        "replication_read",
        "verifier_is_oracle=false",
        "calibrated imbalance-aware",
        "concrete `.gguf` files",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_evalplus_checkpoint_second_corpus_replicates_without_execution(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4264: cached EvalPlus pool can replicate the learned win."""

    _write_exp4233_source(tmp_path)
    _write_evalplus_checkpoint_fixture(tmp_path, "results/evalplus_checkpoint.json")

    artifact = mod.run(
        tmp_path,
        pool_specs=(
            mod.PoolSpec(
                "evalplus_checkpoint_fixture",
                (Path("results/evalplus_checkpoint.json"),),
            ),
        ),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_oracle_distinct_replication_replicates"
    assert artifact["code_replication_beats_vote"] is True
    assert artifact["replication_read"] == "replicates"
    assert artifact["code_replication_retired"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 6
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.0)
    assert artifact["pass_rates"]["predictor_at_1"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_delta"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_ci95"] == [1.0, 1.0]
    assert artifact["off_fold_auroc"] > 0.9
    assert artifact["bootstrap_resamples"] >= 2000
    assert artifact["candidate_pool"]["source_id"] == "evalplus_checkpoint_fixture"
    assert artifact["candidate_pool"]["source_schema"] == "evalplus_checkpoint"
    assert artifact["model_specs"]["generation_model"]["hf_id"] == "unsloth/gemma-4-12B-it-GGUF"
    assert artifact["model_specs"]["second_corpus_id"] == "evalplus_checkpoint_fixture"
    assert artifact["model_specs"]["verifier_is_oracle"] is False
    assert "hidden_pass" not in artifact["model_specs"]["feature_names"]
    assert "test_execution" in artifact["model_specs"]["forbidden_inference_signals"]
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (
        tmp_path
        / "results"
        / "experiment_4264_code_oracle_distinct_replication_retry.json"
    ).exists()


def test_missing_pool_and_missing_sota_model_blocks_before_generation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4264: no cached GGUF stops before any generation claim."""

    _write_exp4233_source(tmp_path)

    artifact = mod.run(
        tmp_path,
        pool_specs=(mod.PoolSpec("missing", (Path("results/missing.jsonl"),)),),
        gguf_resolver=lambda _root: None,
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_code_gen_model_not_cached"
    assert artifact["code_replication_beats_vote"] is False
    assert artifact["code_replication_retired"] is True
    assert artifact["replication_read"] == "code_replication_retired"
    assert artifact["held_out_task_n"] == 0
    assert artifact["oracle_at_k"] == 0.0
    assert artifact["model_specs"]["generation_model"]["available"] is False
    assert artifact["acceptance_gate"] is True


def test_generation_infeasible_retires_after_cached_model_precondition(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4264: cached model plus no buildable pool retires honestly."""

    _write_exp4233_source(tmp_path)

    artifact = mod.run(
        tmp_path,
        pool_specs=(mod.PoolSpec("missing", (Path("results/missing.jsonl"),)),),
        gguf_resolver=lambda _root: {
            "hf_id": "unsloth/gemma-4-12B-it-GGUF",
            "model_path": "/tmp/gemma-4-12b-it-Q4_K_M.gguf",
            "available": True,
        },
        generation_runner=lambda *_args, **_kwargs: None,
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_replication_retired"
    assert artifact["code_replication_beats_vote"] is False
    assert artifact["code_replication_retired"] is True
    assert artifact["replication_read"] == "code_replication_retired"
    assert artifact["model_specs"]["generation_model"]["available"] is True
    assert artifact["methodology_note"].startswith("Retired")


def test_checkpoint_loader_defensive_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4264: malformed cached pools are audited, not accepted."""

    missing = Path("results/missing_checkpoint.json")
    bad = tmp_path / "results" / "bad.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{bad", encoding="utf-8")
    _write_json(tmp_path / "results" / "plain.json", {"not_evaluations": True})
    _write_json(
        tmp_path / "results" / "mixed.json",
        {
            "evaluations_by_task": {
                "NotAList": "skip",
                "Mixed/0": [
                    None,
                    {"code": "", "hidden_passes": [True]},
                    {
                        "code": "def f(x):\n    return x\n",
                        "hidden_pass": True,
                        "draw_index": "bad",
                    },
                    {
                        "code": "def f(x):\n    return x + 1\n",
                        "hidden_passes": "bad",
                        "draw_index": 2,
                    },
                ],
            }
        },
    )

    rows, report = mod._load_evalplus_checkpoint_rows(
        tmp_path,
        mod.PoolSpec(
            "mixed_fixture",
            (
                missing,
                Path("results/bad.json"),
                Path("results/plain.json"),
                Path("results/mixed.json"),
            ),
        ),
    )

    assert len(rows) == 0
    assert report["candidate_rows"] == 1
    assert report["viable_candidate_rows"] == 0
    assert report["paths"][0]["exists"] is False
    assert "JSONDecodeError" in report["paths"][1]["error"]
    assert report["paths"][2]["candidate_rows"] == 0
    assert report["paths"][3]["candidate_rows"] == 1


def test_duplicate_cached_pool_and_missing_exp4233_are_rejected(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4264: source-distinctness rejects Exp 4233 duplicates."""

    checkpoint = _write_evalplus_checkpoint_fixture(tmp_path, "results/duplicate_checkpoint.json")
    _write_json(
        tmp_path / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json",
        {
            "candidate_pool": {
                "source_id": "duplicate_source",
                "source_paths": [str(checkpoint.relative_to(tmp_path))],
            }
        },
    )

    with pytest.raises(mod.MissingDistinctPool) as duplicate_exc:
        mod.load_second_candidate_pool(
            tmp_path,
            pool_specs=(
                mod.PoolSpec(
                    "duplicate_checkpoint",
                    (Path("results/duplicate_checkpoint.json"),),
                ),
            ),
        )
    assert (
        duplicate_exc.value.attempted_sources[0]["skip_reason"]
        == "candidate_source_not_distinct_from_exp4233"
    )

    with pytest.raises(mod.MissingDistinctPool) as missing_exc:
        mod.load_second_candidate_pool(tmp_path / "missing-root", pool_specs=())
    assert missing_exc.value.attempted_sources == []
    assert missing_exc.value.exp4233_source["artifact_exists"] is False


def test_gguf_resolver_and_small_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4264: model precondition resolves concrete GGUF paths."""

    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: tmp_path))
    assert mod.resolve_cached_sota_gguf(tmp_path) is None

    gguf = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--unsloth--gemma-4-12B-it-GGUF"
        / "snapshots"
        / "fixture"
        / "gemma-4-12b-it-Q4_K_M.gguf"
    )
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_text("fixture", encoding="utf-8")
    resolved = mod.resolve_cached_sota_gguf(tmp_path)
    assert resolved is not None
    assert resolved["hf_id"] == "unsloth/gemma-4-12B-it-GGUF"
    assert resolved["model_path"] == str(gguf)

    assert mod._replication_read({"headroom_exists": False})[0] == "no_headroom"
    assert (
        mod._replication_read({"headroom_exists": True, "code_oracle_distinct_beats_vote": False})[0]
        == "corpus_specific"
    )

    pool = mod.CandidatePool(
        source_id="selected",
        rows=[],
        source_paths=[],
        source_sha256={},
        task_n=0,
        candidate_n=0,
        positive_n=0,
        pass_rate=0.0,
        attempted_sources=[],
        vote_signature_source="normalized_code_text_signature",
    )
    assert mod._selected_report(pool) == {}

    pool_with_bad_model = mod.CandidatePool(
        source_id="selected",
        rows=[],
        source_paths=[],
        source_sha256={},
        task_n=0,
        candidate_n=0,
        positive_n=0,
        pass_rate=0.0,
        attempted_sources=[{"source_id": "selected", "generation_model": "bad"}],
        vote_signature_source="normalized_code_text_signature",
    )
    assert mod._pool_generation_model(pool_with_bad_model) == {"hf_id": "", "available": None}


def test_validate_artifact_rejects_4264_schema_drift() -> None:
    """REQ-VERIFY-4264: artifact schema rejects non-bare or inconsistent fields."""

    valid = {
        "honest_verdict": "complete: code_replication_retired",
        "code_replication_beats_vote": False,
        "code_predictor_minus_vote_delta": 0.0,
        "code_predictor_minus_vote_ci95": [0.0, 0.0],
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "replication_read": "code_replication_retired",
        "code_replication_retired": True,
        "verifier_is_oracle": False,
        "model_specs": {"verifier_is_oracle": False},
        "random_seed": 4264,
        "reproducibility_checksum": "checksum",
        "field_principles": mod.FIELD_PRINCIPLES,
        "spec_refs": mod.SPEC_REFS,
        "acceptance_gate": True,
    }
    mod.validate_artifact(valid)
    cases = [
        ("missing", lambda d: d.pop("code_replication_retired")),
        ("bad_verdict", lambda d: d.__setitem__("honest_verdict", "pending")),
        ("bad_win_bool", lambda d: d.__setitem__("code_replication_beats_vote", 1)),
        ("bad_retired_bool", lambda d: d.__setitem__("code_replication_retired", 1)),
        ("bad_float", lambda d: d.__setitem__("oracle_at_k", True)),
        ("bad_ci", lambda d: d.__setitem__("code_predictor_minus_vote_ci95", [0.0])),
        ("bad_n", lambda d: d.__setitem__("held_out_task_n", 0.0)),
        ("bad_read", lambda d: d.__setitem__("replication_read", "maybe")),
        ("win_read_mismatch", lambda d: d.__setitem__("code_replication_beats_vote", True)),
        ("retired_read_mismatch", lambda d: d.__setitem__("replication_read", "replicates")),
        (
            "win_and_retired",
            lambda d: (
                d.__setitem__("code_replication_beats_vote", True),
                d.__setitem__("replication_read", "replicates"),
            ),
        ),
        ("oracle_true", lambda d: d.__setitem__("verifier_is_oracle", True)),
        ("bad_seed", lambda d: d.__setitem__("random_seed", 4264.0)),
        ("missing_specs", lambda d: d.__setitem__("model_specs", None)),
        ("specs_oracle_true", lambda d: d.__setitem__("model_specs", {"verifier_is_oracle": True})),
        ("bad_principles", lambda d: d.__setitem__("field_principles", {})),
        ("bad_refs", lambda d: d.__setitem__("spec_refs", [])),
    ]
    for _name, mutate in cases:
        drifted = dict(valid)
        mutate(drifted)
        with pytest.raises(ValueError):
            mod.validate_artifact(drifted)
