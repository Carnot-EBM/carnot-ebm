"""Tests for Exp 4319 off-ARC execution-verifier accumulation.

Spec: REQ-VERIFY-4319, SCENARIO-VERIFY-4319.
"""

from __future__ import annotations

import json
import types
import builtins
from pathlib import Path

import pytest

import carnot.verify.off_arc_execution_verifier_transfer_accumulate as exp4319
from carnot.verify.off_arc_execution_verifier_transfer_accumulate import (
    REQUIRED_FIELDS,
    blocker_from_preconditions,
    blocked_artifact,
    bootstrap_ci95,
    build_accumulation_artifact,
    check_preconditions,
    load_task_outcomes,
    resolve_gemma_gguf,
    run,
    validate_artifact,
)


def _write_source(path: Path, rows: list[tuple[str, bool, bool]], *, corpus: str) -> Path:
    """REQ-VERIFY-4319: synthetic source with vote/demo hidden-test outcomes."""
    payload = {
        "experiment": path.stem,
        "schema": "synthetic.exp4319.source.v1",
        "corpus": corpus,
        "model_specs": {"local_generator": "unsloth/gemma-4-12B-it-GGUF"},
        "random_seed": 7,
        "reproducibility_checksum": path.stem,
        "n_tasks": len(rows),
        "per_task": [
            {
                "task_id": task_id,
                "corpus": corpus,
                "armA_vote_pass1": vote_pass,
                "armB_demofit_pass1": demofit_pass,
                "armB_demo_perfect_count": 1 if demofit_pass else 0,
                "n_candidates": 5,
                "n_visible_tests": 2,
                "n_hidden_tests": 3,
                "oracle_hidden_pass": vote_pass or demofit_pass,
            }
            for task_id, vote_pass, demofit_pass in rows
        ],
        "candidate_pool": {
            task_id: [
                {
                    "draw_index": 0,
                    "code_sha256": f"{task_id}-code",
                    "visible_passes": [True, True],
                    "hidden_passes": [demofit_pass],
                }
            ]
            for task_id, _vote_pass, demofit_pass in rows
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _preconditions() -> list[dict[str, object]]:
    return [
        {
            "resource": "generator_gguf_cached",
            "available": True,
            "path": "/models/gemma-4-12b-it-Q4_K_M.gguf",
        },
        {"resource": "mbpp_evalplus_corpus_loadable", "available": True},
        {"resource": "prior_accumulation_checkpoint_readable", "available": True},
        {"resource": "restricted_exec_sandbox_available", "available": True},
        {"resource": "trm_training_stood_down", "available": True},
    ]


def test_build_accumulation_artifact_reports_bare_gate_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4319: accumulated vote/demo metrics use bare gate fields."""
    prior = _write_source(
        tmp_path / "prior.json",
        [("MBPP:1", True, True), ("MBPP:2", False, True)],
        corpus="mbpp",
    )
    window = _write_source(
        tmp_path / "window.json",
        [("HumanEval/1", True, False), ("HumanEval/2", False, True)],
        corpus="evalplus",
    )

    artifact = build_accumulation_artifact(
        prior_paths=[prior],
        window_paths=[window],
        preconditions_checked=_preconditions(),
        model_specs={
            "generator_hf_id": "unsloth/gemma-4-12B-it-GGUF",
            "generator_gguf_path": "/models/gemma-4-12b-it-Q4_K_M.gguf",
            "execution_verifier": "visible demo-fit exact-output selector",
        },
        seed=4319,
        bootstrap_resamples=2000,
        started_s=0.0,
        ended_s=1.0,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_FIELDS).issubset(artifact)
    assert artifact["spec_refs"] == ["REQ-VERIFY-4319", "SCENARIO-VERIFY-4319"]
    assert artifact["accumulated_n"] == 4
    assert artifact["accumulation_window_added"] == 2
    assert artifact["hidden_test_vote_at_1"] == 0.5
    assert artifact["hidden_test_demofit_accuracy"] == 0.75
    assert artifact["off_arc_demofit_minus_vote_delta"] == 0.25
    assert len(artifact["off_arc_delta_ci95"]) == 2
    assert artifact["bootstrap_resamples"] == 2000
    assert type(artifact["off_arc_demofit_beats_vote"]) is bool
    assert artifact["verifier_is_oracle"] is True
    assert artifact["source_artifacts"][1]["tasks_used"] == 2


def test_positive_ci_sets_beats_vote_true(tmp_path: Path) -> None:
    """REQ-VERIFY-4319: the gate is true only when the paired CI excludes zero."""
    window = _write_source(
        tmp_path / "all_win.json",
        [(f"HumanEval/{idx}", False, True) for idx in range(8)],
        corpus="evalplus",
    )

    artifact = build_accumulation_artifact(
        prior_paths=[],
        window_paths=[window],
        preconditions_checked=_preconditions(),
        model_specs={"generator_hf_id": "unsloth/gemma-4-12B-it-GGUF"},
        seed=4319,
        bootstrap_resamples=2000,
        started_s=0.0,
        ended_s=1.0,
    )

    validate_artifact(artifact)
    assert artifact["off_arc_demofit_minus_vote_delta"] == 1.0
    assert artifact["off_arc_delta_ci95"] == [1.0, 1.0]
    assert artifact["off_arc_demofit_beats_vote"] is True
    assert artifact["honest_verdict"].startswith("success:")


def test_blocked_generator_not_cached_artifact_is_complete() -> None:
    """SCENARIO-VERIFY-4319: missing GGUF emits blocked_generator_not_cached."""
    preconditions = [
        {"resource": "generator_gguf_cached", "available": False, "path": None},
        {"resource": "mbpp_evalplus_corpus_loadable", "available": True},
    ]

    artifact = blocked_artifact(
        "blocked_generator_not_cached",
        preconditions_checked=preconditions,
        seed=4319,
        started_s=0.0,
        ended_s=0.25,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_generator_not_cached"
    assert artifact["accumulated_n"] == 0
    assert artifact["accumulation_window_added"] == 0
    assert artifact["off_arc_demofit_beats_vote"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "precondition_check_no_inference"


def test_preconditions_resolve_cache_and_prior_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-4319: preconditions record cache, corpus, sandbox, and TRM state."""
    cache = tmp_path / "models--unsloth--gemma-4-12B-it-GGUF"
    gguf = cache / "snapshots" / "abc" / "gemma-4-12b-it-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    source = _write_source(
        tmp_path / "evalplus.json",
        [("HumanEval/7", True, True)],
        corpus="evalplus_humaneval",
    )
    mbpp_manifest = tmp_path / "mbpp.jsonl"
    mbpp_manifest.write_text('{"task_id": 1}\n', encoding="utf-8")

    assert resolve_gemma_gguf(cache) == gguf
    checks = check_preconditions(
        prior_paths=[],
        window_paths=[source],
        cache_dir=cache,
        mbpp_manifest=mbpp_manifest,
    )

    by_resource = {row["resource"]: row for row in checks}
    assert by_resource["generator_gguf_cached"]["available"] is True
    assert by_resource["mbpp_evalplus_corpus_loadable"]["available"] is True
    assert by_resource["prior_accumulation_checkpoint_readable"]["available"] is True
    assert by_resource["trm_training_stood_down"]["available"] is True
    assert blocker_from_preconditions(checks) is None

    fallback_cache = tmp_path / "fallback-cache"
    fallback_gguf = fallback_cache / "flat-model.gguf"
    fallback_cache.mkdir()
    fallback_gguf.write_bytes(b"gguf")
    assert resolve_gemma_gguf(fallback_cache) == fallback_gguf

    missing_checks = check_preconditions(
        prior_paths=[tmp_path / "missing-source.json"],
        window_paths=[],
        cache_dir=cache,
        mbpp_manifest=tmp_path / "missing-mbpp.jsonl",
    )
    missing_by_resource = {row["resource"]: row for row in missing_checks}
    assert missing_by_resource["prior_accumulation_checkpoint_readable"]["available"] is False
    assert missing_by_resource["mbpp_evalplus_corpus_loadable"]["available"] is False
    assert blocker_from_preconditions(missing_checks) == "blocked_mbpp_evalplus_corpus_loadable"

    bad_manifest_checks = check_preconditions(
        prior_paths=[],
        window_paths=[source],
        cache_dir=cache,
        mbpp_manifest=tmp_path,
    )
    bad_by_resource = {row["resource"]: row for row in bad_manifest_checks}
    assert bad_by_resource["mbpp_evalplus_corpus_loadable"]["available"] is False


def test_source_loading_skips_duplicates_and_empty_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-4319: accumulation is corpus+task keyed, not exp-id keyed."""
    first = _write_source(
        tmp_path / "first.json",
        [("MBPP:1", True, False), ("MBPP:2", False, True)],
        corpus="mbpp",
    )
    second = _write_source(
        tmp_path / "second.json",
        [("MBPP:1", False, True), ("HumanEval/3", False, True)],
        corpus="mbpp",
    )
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["per_task"].append("not-a-row")
    payload["per_task"].append({"task_id": 123})
    payload["per_task"].append({"task_id": "missing-fields"})
    payload["per_task"].append(
        {
            "task_id": "MBPP:9",
            "armA_vote_pass1": True,
            "armB_demofit_pass1": True,
            "oracle_hidden_pass": True,
        }
    )
    payload["per_task"].append(
        {
            "task_id": "HumanEval/4",
            "armA_vote_pass1": False,
            "armB_demofit_pass1": True,
            "oracle_hidden_pass": True,
        }
    )
    payload["per_task"].append(
        {
            "task_id": "custom-task",
            "armA_vote_pass1": False,
            "armB_demofit_pass1": False,
            "oracle_hidden_pass": False,
        }
    )
    second.write_text(json.dumps(payload), encoding="utf-8")

    outcomes, summaries = load_task_outcomes([first, second])

    assert [row.task_id for row in outcomes] == [
        "MBPP:1",
        "MBPP:2",
        "HumanEval/3",
        "MBPP:9",
        "HumanEval/4",
        "custom-task",
    ]
    assert summaries[0]["tasks_used"] == 2
    assert summaries[0]["declared_n"] == 2
    assert summaries[1]["tasks_used"] == 4
    assert bootstrap_ci95([], seed=4319) == [0.0, 0.0]
    assert exp4319._declared_n({}) is None


def test_run_writes_replay_artifact_and_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4319: run writes replay JSON or blocked JSON honestly."""
    cache = tmp_path / "cache"
    gguf = cache / "snapshots" / "abc" / "gemma-4-12b-it-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    source = _write_source(
        tmp_path / "window.json",
        [("HumanEval/10", True, True), ("HumanEval/11", True, False)],
        corpus="evalplus_humaneval",
    )
    replay_out = tmp_path / "replay.json"

    replay = run(
        output_path=replay_out,
        prior_paths=[],
        window_paths=[source],
        cache_dir=cache,
        seed=4319,
    )

    assert replay_out.exists()
    validate_artifact(replay)
    assert replay["accumulated_n"] == 2
    assert replay["accumulation_window_added"] == 2
    assert replay["adversarial_verify"]["critical_count"] == 0

    blocked_out = tmp_path / "blocked.json"
    blocked = run(
        output_path=blocked_out,
        prior_paths=[],
        window_paths=[source],
        cache_dir=tmp_path / "missing-cache",
        seed=4319,
    )

    assert blocked_out.exists()
    assert blocked["honest_verdict"] == "blocked_generator_not_cached"
    assert blocker_from_preconditions(blocked["preconditions_checked"]) == (
        "blocked_generator_not_cached"
    )


def test_negative_ci_sets_scope_boundary_verdict(tmp_path: Path) -> None:
    """REQ-VERIFY-4319: powered negative CI gets a scope-boundary verdict."""
    source = _write_source(
        tmp_path / "negative.json",
        [(f"HumanEval/{idx}", True, False) for idx in range(6)],
        corpus="evalplus_humaneval",
    )

    artifact = build_accumulation_artifact(
        prior_paths=[],
        window_paths=[source],
        preconditions_checked=_preconditions(),
        model_specs={"generator_hf_id": "unsloth/gemma-4-12B-it-GGUF"},
        seed=4319,
        bootstrap_resamples=2000,
        started_s=None,
        ended_s=None,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: off_arc_demofit_powered_negative_scope_boundary"
    assert artifact["duration_s"] == 0.0001


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("accumulated_n"), "missing required field"),
        (lambda artifact: artifact.__setitem__("honest_verdict", "bad"), "terminal-prefixed"),
        (
            lambda artifact: artifact.__setitem__("off_arc_demofit_beats_vote", 1),
            "bare bool",
        ),
        (lambda artifact: artifact.__setitem__("verifier_is_oracle", False), "bare bool true"),
        (lambda artifact: artifact.__setitem__("accumulated_n", True), "bare int"),
        (
            lambda artifact: artifact.__setitem__("off_arc_demofit_minus_vote_delta", True),
            "bare float",
        ),
        (lambda artifact: artifact.__setitem__("off_arc_delta_ci95", [0.0]), "two-element"),
        (lambda artifact: artifact.__setitem__("bootstrap_resamples", 1999), "at least 2000"),
        (lambda artifact: artifact.__setitem__("preconditions_checked", {}), "must be a list"),
        (lambda artifact: artifact.__setitem__("model_specs", []), "must be an object"),
        (lambda artifact: artifact.__setitem__("reproducibility_checksum", ""), "non-empty"),
    ],
)
def test_validate_artifact_rejects_bad_schema(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-VERIFY-4319: validator rejects non-bare or missing gate fields."""
    source = _write_source(
        tmp_path / "source.json",
        [("HumanEval/5", False, True)],
        corpus="evalplus_humaneval",
    )
    artifact = build_accumulation_artifact(
        prior_paths=[],
        window_paths=[source],
        preconditions_checked=_preconditions(),
        model_specs={"generator_hf_id": "unsloth/gemma-4-12B-it-GGUF"},
        seed=4319,
        bootstrap_resamples=2000,
        started_s=0.0,
        ended_s=1.0,
    )

    mutate(artifact)
    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact)


def test_process_and_adversarial_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4319: preflight helpers fail closed on process/verifier errors."""
    class RaisesRun:
        def __call__(self, *args, **kwargs):
            raise OSError("ps failed")

    monkeypatch.setattr(exp4319.subprocess, "run", RaisesRun())
    assert exp4319._trm_training_stood_down() is False

    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(
            stdout=(
                "\n"
                "not-a-pid malformed\n"
                f"{exp4319.os.getpid()} current-process\n"
                "99999 unrelated\n"
                "10000 python train.py --out results/trm_runs/demo\n"
            )
        )

    monkeypatch.setattr(exp4319.subprocess, "run", fake_run)
    assert exp4319._trm_training_stood_down() is False

    def fake_run_trm(*args, **kwargs):
        return types.SimpleNamespace(stdout="10001 python trm_train.py\n")

    monkeypatch.setattr(exp4319.subprocess, "run", fake_run_trm)
    assert exp4319._trm_training_stood_down() is False

    fake_module = types.ModuleType("adversarial_verify")

    def raise_verify(_path):
        raise RuntimeError("verify failed")

    fake_module.verify_artifact = raise_verify
    monkeypatch.setitem(exp4319.sys.modules, "adversarial_verify", fake_module)
    summary = exp4319._adversarial_verify_summary(Path("artifact.json"))
    assert summary["status"] == "error"

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "carnot.verify" and "sandbox" in fromlist:
            raise ImportError("sandbox unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert exp4319._sandbox_importable() is False
