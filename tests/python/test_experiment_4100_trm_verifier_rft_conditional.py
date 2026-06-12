"""Tests for Exp 4100 conditional TRM verifier-RFT.

Spec refs: REQ-LEARN-4100, SCENARIO-LEARN-4100-SMOKE,
SCENARIO-LEARN-4100-RFT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic import arc_exp4100_trm_verifier_rft_conditional as exp4100


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_repo_ready(tmp_path: Path, *, verifier_beats: bool) -> Path:
    trm_cache = tmp_path / "hf_cache" / "models--arcprize--trm_arc_prize_verification"
    (trm_cache / "snapshots").mkdir(parents=True)
    nano_src = tmp_path / "nano-trm" / "src"
    nano_src.mkdir(parents=True)
    (nano_src / "arc_evaluator.py").write_text("# substrate marker\n", encoding="utf-8")
    (nano_src / "baseline.py").write_text("# substrate marker\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    probe = {
        "experiment": "experiment_4099_trm_pool_verifier_discrimination_probe",
        "honest_verdict": "complete: fixture",
        "verifier_beats_trm_vote": verifier_beats,
        "best_reranker": "K_OF_N_AGREEMENT",
        "captured_pp_directional": 0.125 if verifier_beats else 0.0,
        "per_reranker": {
            "TRM_VOTE": {"pass@1": 0.2, "pass@2": 0.27, "captured_pp": 0.0, "captured_pp_ci95": [0.0, 0.0]},
            "K_OF_N_AGREEMENT": {
                "pass@1": 0.3,
                "pass@2": 0.39 if verifier_beats else 0.27,
                "captured_pp": 0.125 if verifier_beats else 0.0,
                "captured_pp_ci95": [0.05, 0.2] if verifier_beats else [0.0, 0.0],
            },
        },
        "random_seed": 4099,
        "reproducibility_checksum": "exp4099-fixture-checksum",
    }
    (results / "experiment_4099_trm_pool_verifier_discrimination_probe.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )
    return trm_cache


def _cuda_ok() -> tuple[bool, str]:
    return True, "cuda fixture"


def test_req_learn_4100_spec_declares_contract() -> None:
    """REQ-LEARN-4100: OpenSpec declares branch and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4100" in spec
    assert "SCENARIO-LEARN-4100-SMOKE" in spec
    assert "SCENARIO-LEARN-4100-RFT" in spec
    for field in exp4100.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "blocked_trm_weights_not_cached" in spec


def test_req_learn_4100_preconditions_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-4100: missing resources map to explicit blocked verdicts."""

    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=tmp_path / "missing-hf",
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_trm_weights_not_cached"
    assert checks[0].resource == "trm_weights_cached"
    assert checks[0].available is False

    empty_cache = tmp_path / "empty-hf"
    empty_cache.mkdir()
    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=empty_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_trm_weights_not_cached"
    assert checks[0].detail.startswith("empty directory:")

    trm_cache = tmp_path / "hf"
    trm_cache.mkdir()
    (trm_cache / "weights.marker").write_text("cached\n", encoding="utf-8")
    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_trm_substrate_missing"
    assert any(check.resource == "nano_trm_substrate" and not check.available for check in checks)

    nano_src = tmp_path / "nano-trm" / "src"
    nano_src.mkdir(parents=True)
    (nano_src / "arc_evaluator.py").write_text("", encoding="utf-8")
    (nano_src / "baseline.py").write_text("", encoding="utf-8")
    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=lambda: (False, "no cuda"),
    )
    assert blocker == "blocked_cuda_unavailable"
    assert any(check.resource == "cuda_available" and check.detail == "no cuda" for check in checks)

    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_exp4099_probe_missing"


def test_req_learn_4100_precondition_json_and_cuda_exceptions(tmp_path: Path) -> None:
    """REQ-LEARN-4100: malformed probes and CUDA checker errors block honestly."""

    trm_cache = _make_repo_ready(tmp_path, verifier_beats=False)
    probe = tmp_path / "results" / "experiment_4099_trm_pool_verifier_discrimination_probe.json"
    probe.write_text("{", encoding="utf-8")

    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_exp4099_probe_missing"
    assert any("JSONDecodeError" in check.detail for check in checks)

    probe.write_text(json.dumps({"verifier_beats_trm_vote": {"wrapped": False}}), encoding="utf-8")
    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_exp4099_probe_missing"
    assert any("not a bare bool" in check.detail for check in checks)

    probe.write_text(json.dumps({"verifier_beats_trm_vote": False}), encoding="utf-8")
    checks, blocker, _path, _probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda probe broke")),
    )
    assert blocker == "blocked_cuda_unavailable"
    assert any("RuntimeError: cuda probe broke" in check.detail for check in checks)

    available, detail = exp4100._default_cuda_checker()
    assert isinstance(available, bool)
    assert "torch.cuda.is_available()" in detail


def test_scenario_learn_4100_smoke_branch_writes_checkpoint_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4100-SMOKE: no verifier signal runs native checkpoint smoke."""

    trm_cache = _make_repo_ready(tmp_path, verifier_beats=False)
    checkpoint = tmp_path / "results" / "native-smoke" / "checkpoints" / "last.ckpt"

    def fake_smoke(config: exp4100.NativeSmokeConfig) -> exp4100.SmokeRunResult:
        assert config.max_steps >= 1
        return exp4100.SmokeRunResult(
            checkpoint_ok=True,
            checkpoint_reload_ok=True,
            checkpoint_path=checkpoint,
            duration_s=12.5,
            command=["python", "src/nn/train.py"],
            stdout_tail=["[exp4100:native-train] step=1 epoch=0"],
        )

    output_path = tmp_path / "results" / exp4100.RESULT_FILENAME
    artifact = exp4100.run_experiment(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        output_path=output_path,
        cuda_checker=_cuda_ok,
        smoke_runner=fake_smoke,
    )

    assert artifact["branch_taken"] == "smoke"
    assert artifact["trm_native_trainer_checkpoint_ok"] is True
    assert artifact["rft_vs_ablation_delta"]["status"] == "not_run_no_verifier_signal"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 12.5
    assert "live_model" not in json.dumps(artifact)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert exp4100.artifact_schema_errors(artifact) == []


def test_scenario_learn_4100_rft_branch_records_a_vs_b_delta(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4100-RFT: RFT branch reports verifier-label delta."""

    trm_cache = _make_repo_ready(tmp_path, verifier_beats=True)

    def fake_rft(config: exp4100.RftConfig) -> exp4100.RftRunResult:
        assert config.best_reranker == "K_OF_N_AGREEMENT"
        return exp4100.RftRunResult(
            trm_native_trainer_checkpoint_ok=True,
            duration_s=91.0,
            rft_vs_ablation_delta={
                "metric": "heldout_pass@2",
                "delta": 0.14,
                "ci95": [0.04, 0.23],
                "status": "ci95_excludes_zero",
            },
            arm_metrics={
                "A_verifier_certified": {"pass@1": 0.42, "pass@2": 0.58},
                "B_vote_certified": {"pass@1": 0.36, "pass@2": 0.44},
                "cold": {"pass@1": 0.31, "pass@2": 0.38},
            },
            corpus_summary={"train_n": 24, "heldout_n": 12, "certifier": "K_OF_N_AGREEMENT"},
        )

    artifact = exp4100.run_experiment(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
        rft_runner=fake_rft,
    )

    assert artifact["branch_taken"] == "rft"
    assert artifact["trm_native_trainer_checkpoint_ok"] is True
    assert artifact["rft_vs_ablation_delta"]["ci95"][0] > 0.0
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["arm_metrics"]["A_verifier_certified"]["pass@2"] > artifact["arm_metrics"]["B_vote_certified"]["pass@2"]
    assert exp4100.artifact_schema_errors(artifact) == []


def test_req_learn_4100_checksum_is_deterministic_and_drift_sensitive(tmp_path: Path) -> None:
    """REQ-LEARN-4100: checksum covers branch inputs and corpus summary."""

    trm_cache = _make_repo_ready(tmp_path, verifier_beats=False)
    result = exp4100.SmokeRunResult(
        checkpoint_ok=True,
        checkpoint_reload_ok=True,
        checkpoint_path=tmp_path / "last.ckpt",
        duration_s=70.0,
        command=["train"],
        stdout_tail=[],
    )
    checks, blocker, probe_path, probe = exp4100.check_preconditions(
        repo_root=tmp_path,
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
    )
    assert blocker is None

    first = exp4100.build_smoke_artifact(
        exp4099_artifact=probe,
        exp4099_path=probe_path,
        preconditions_checked=checks,
        smoke_result=result,
        smoke_plan={"split": "tiny", "max_steps": 200},
    )
    second = exp4100.build_smoke_artifact(
        exp4099_artifact=probe,
        exp4099_path=probe_path,
        preconditions_checked=checks,
        smoke_result=result,
        smoke_plan={"split": "tiny", "max_steps": 200},
    )
    drifted = exp4100.build_smoke_artifact(
        exp4099_artifact=probe,
        exp4099_path=probe_path,
        preconditions_checked=checks,
        smoke_result=result,
        smoke_plan={"split": "tiny", "max_steps": 201},
    )

    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["reproducibility_checksum"] != drifted["reproducibility_checksum"]


def test_req_learn_4100_schema_errors_and_native_command_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4100: schema guard and native command make regressions visible."""

    artifact = exp4100.build_blocked_artifact(
        "blocked_cuda_unavailable",
        preconditions_checked=[],
        duration_s=0.1,
    )
    bad = dict(artifact)
    del bad["branch_taken"]
    bad["honest_verdict"] = "maybe"
    bad["trm_native_trainer_checkpoint_ok"] = "yes"
    bad["preconditions_checked"] = {}
    bad["rft_vs_ablation_delta"] = {"ci95": [0.0]}
    bad["random_seed"] = True
    bad["reproducibility_checksum"] = 123

    errors = exp4100.artifact_schema_errors(bad)

    assert "missing required field branch_taken" in errors
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "trm_native_trainer_checkpoint_ok must be a bare bool" in errors
    assert "preconditions_checked must be a list" in errors
    assert "rft_vs_ablation_delta must include two-element ci95" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be a string" in errors

    bad["honest_verdict"] = 123
    bad["rft_vs_ablation_delta"] = "not-dict"
    errors = exp4100.artifact_schema_errors(bad)
    assert "honest_verdict must be a string" in errors
    assert "rft_vs_ablation_delta must be a dict" in errors

    bad["honest_verdict"] = "complete: fixture"
    bad["rft_vs_ablation_delta"] = {"ci95": [0.0, "bad"]}
    errors = exp4100.artifact_schema_errors(bad)
    assert "rft_vs_ablation_delta ci95 values must be numeric" in errors

    config = exp4100.NativeSmokeConfig(repo_root=tmp_path)
    command = exp4100.build_native_smoke_command(config)
    env = exp4100.build_native_smoke_env(config)

    assert str(tmp_path / "nano-trm" / "src" / "nn" / "train.py") in command
    assert "experiment=trm_sudoku_4x4" in command
    assert any(item.startswith("+callbacks.exp4100_progress._target_=") for item in command)
    assert str(tmp_path / "nano-trm" / "src") in env["PYTHONPATH"]

    venv_python = tmp_path / "nano-trm" / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert exp4100.NativeSmokeConfig(repo_root=tmp_path).python_executable == str(venv_python)


def test_req_learn_4100_defensive_builder_and_runner_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-4100: defensive branches remain schema-checked and deterministic."""

    blocked = exp4100.run_experiment(
        repo_root=tmp_path,
        trm_weights_dir=tmp_path / "missing",
        cuda_checker=_cuda_ok,
        output_path=tmp_path / "blocked.json",
    )
    assert blocked["branch_taken"] == "blocked"

    dict_check_artifact = exp4100.build_blocked_artifact(
        "blocked_cuda_unavailable",
        preconditions_checked=[{"resource": "cuda_available", "available": False, "detail": "fixture"}],
    )
    assert dict_check_artifact["preconditions_checked"][0]["resource"] == "cuda_available"

    trm_cache = _make_repo_ready(tmp_path / "smoke-error", verifier_beats=False)
    smoke_error = exp4100.run_experiment(
        repo_root=tmp_path / "smoke-error",
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
        smoke_runner=lambda _config: (_ for _ in ()).throw(RuntimeError("smoke broke")),
    )
    assert smoke_error["branch_taken"] == "smoke"
    assert smoke_error["trm_native_trainer_checkpoint_ok"] is False

    trm_cache = _make_repo_ready(tmp_path / "rft-error", verifier_beats=True)
    rft_error = exp4100.run_experiment(
        repo_root=tmp_path / "rft-error",
        trm_weights_dir=trm_cache,
        cuda_checker=_cuda_ok,
        rft_runner=lambda _config: (_ for _ in ()).throw(RuntimeError("rft broke")),
    )
    assert rft_error["branch_taken"] == "rft"
    assert rft_error["rft_vs_ablation_delta"]["status"] == "RuntimeError: rft broke"

    default_rft = exp4100.default_rft_runner(
        exp4100.RftConfig(
            repo_root=tmp_path,
            exp4099_artifact={"verifier_beats_trm_vote": True},
            exp4099_path=tmp_path / "probe.json",
            best_reranker="fixture",
        )
    )
    assert default_rft.corpus_summary["status"] == "not_run_default_rft_runner_unimplemented"

    no_lift = exp4100.build_rft_artifact(
        exp4099_artifact={"reproducibility_checksum": "x"},
        exp4099_path=tmp_path / "probe.json",
        preconditions_checked=[],
        rft_result=exp4100.RftRunResult(
            trm_native_trainer_checkpoint_ok=True,
            duration_s=1.0,
            rft_vs_ablation_delta={"metric": "heldout_pass@2", "delta": 0.0, "ci95": [-0.1, 0.1]},
            arm_metrics={},
            corpus_summary={},
        ),
    )
    assert no_lift["honest_verdict"].startswith("complete: verifier_rft_no_ci")

    checkpoint_failed = exp4100.build_rft_artifact(
        exp4099_artifact={"reproducibility_checksum": "x"},
        exp4099_path=tmp_path / "probe.json",
        preconditions_checked=[],
        rft_result=exp4100.RftRunResult(
            trm_native_trainer_checkpoint_ok=False,
            duration_s=1.0,
            rft_vs_ablation_delta={"metric": "heldout_pass@2", "delta": 0.0, "ci95": [0.0, 0.0]},
            arm_metrics={},
            corpus_summary={},
        ),
    )
    assert checkpoint_failed["honest_verdict"] == "complete: verifier_rft_native_trainer_checkpoint_failed"

    smoke_failed = exp4100.build_smoke_artifact(
        exp4099_artifact={"verifier_beats_trm_vote": False, "captured_pp_directional": 0.0},
        exp4099_path=tmp_path / "probe.json",
        preconditions_checked=[],
        smoke_result=exp4100.SmokeRunResult(False, False, None, 1.0, [], []),
        smoke_plan={"max_steps": 1},
    )
    assert smoke_failed["honest_verdict"].startswith("complete: trm_native_trainer_checkpoint_failed")

    monkeypatch.setattr(exp4100, "artifact_schema_errors", lambda _artifact: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        exp4100.build_blocked_artifact("blocked_cuda_unavailable", preconditions_checked=[])
    with pytest.raises(ValueError, match="boom"):
        exp4100.build_smoke_artifact(
            exp4099_artifact={"verifier_beats_trm_vote": False},
            exp4099_path=None,
            preconditions_checked=[],
            smoke_result=exp4100.SmokeRunResult(True, True, tmp_path / "x.ckpt", 1.0, [], []),
            smoke_plan={},
        )
    with pytest.raises(ValueError, match="boom"):
        exp4100.build_rft_artifact(
            exp4099_artifact={},
            exp4099_path=None,
            preconditions_checked=[],
            rft_result=exp4100.RftRunResult(
                True,
                1.0,
                {"metric": "heldout_pass@2", "delta": 0.0, "ci95": [0.0, 0.0]},
                {},
                {},
            ),
        )


def test_req_learn_4100_path_env_and_checkpoint_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-4100: nano-trm path and checkpoint helpers are explicit."""

    nano_root = tmp_path / "nano-trm"
    expected_src = str(nano_root / "src")
    if expected_src in sys.path:
        sys.path.remove(expected_src)
    assert exp4100.ensure_nano_trm_src_on_path(nano_root) == nano_root / "src"
    assert sys.path[0] == expected_src
    sys.path.remove(expected_src)

    monkeypatch.setenv("PYTHONPATH", "existing-path")
    env = exp4100.build_native_smoke_env(exp4100.NativeSmokeConfig(repo_root=tmp_path))
    assert env["PYTHONPATH"].endswith("existing-path")

    assert exp4100._latest_checkpoint(tmp_path / "missing") is None
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    other = ckpt_dir / "epoch.ckpt"
    other.write_text("not torch", encoding="utf-8")
    assert exp4100._latest_checkpoint(ckpt_dir) == other
    last = ckpt_dir / "last.ckpt"
    last.write_text("not torch", encoding="utf-8")
    assert exp4100._latest_checkpoint(ckpt_dir) == last

    missing_ok, missing_detail = exp4100._load_torch_checkpoint(tmp_path / "absent.ckpt")
    assert missing_ok is False
    assert "FileNotFoundError" in missing_detail

    torch = pytest.importorskip("torch")
    good = tmp_path / "good.ckpt"
    torch.save({"state_dict": {}}, good)
    assert exp4100._load_torch_checkpoint(good) == (True, "torch.load ok")

    bad = tmp_path / "bad.ckpt"
    torch.save(["not", "a", "mapping"], bad)
    bad_ok, bad_detail = exp4100._load_torch_checkpoint(bad)
    assert bad_ok is False
    assert "unexpected checkpoint payload" in bad_detail
