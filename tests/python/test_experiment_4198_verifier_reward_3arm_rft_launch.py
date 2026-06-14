"""Tests for Exp 4198 detached verifier-reward LoRA-RFT launch.

Spec refs: REQ-CODE-4198, SCENARIO-CODE-4198-GATED-LAUNCH,
SCENARIO-CODE-4198-HONEST-DEFERRAL.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_4198_verifier_reward_3arm_rft_launch as exp4198


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


def _write_a1_artifact(
    path: Path,
    *,
    precision: float = 0.956,
    youden_j: float = 0.414,
    harness_ready: bool = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "phase0_precision": precision,
                "youden_j": youden_j,
                "harness_ready": harness_ready,
                "operating_point": {
                    "base": "google/gemma-4-E4B-it",
                    "corpus": "fixture",
                    "K": 5,
                    "max_new_tokens": 512,
                    "base_passrate": 0.6,
                    "own_visible_perfect_rate": 0.6,
                    "truncation_rate": 0.0,
                    "no_answer_rate": 0.0,
                },
                "model_specs": {
                    "trainable_base": "google/gemma-4-E4B-it",
                    "trainable_base_is_non_qwen": True,
                    "certification_reference": "same-generator-gemma",
                    "runner": "scripts/experiments/verifier_reward_code_lora_rft_3arm.py",
                },
                "random_seed": 4197,
                "reproducibility_checksum": "sha256:a1",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_checkpoint(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "draw_index": 0,
                            "code": "def add_one(x):\n    return x + 1\n",
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "status": "ok",
                        },
                        {
                            "draw_index": 1,
                            "code": "def add_one(x):\n    return x\n",
                            "visible_passes": [False],
                            "hidden_passes": [False],
                            "status": "ok",
                        },
                        {
                            "draw_index": 2,
                            "code": "def add_one(x):\n    return x + 2\n",
                            "visible_passes": [False],
                            "hidden_passes": [True],
                            "status": "ok",
                        },
                    ],
                    "HumanEval/1": [
                        {
                            "draw_index": 0,
                            "code": "def sub_one(x):\n    return x - 1\n",
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "status": "ok",
                        },
                        {
                            "draw_index": 1,
                            "code": "def sub_one(x):\n    return x\n",
                            "visible_passes": [False],
                            "hidden_passes": [False],
                            "status": "ok",
                        },
                        {
                            "draw_index": 2,
                            "code": "def sub_one(x):\n    return x - 2\n",
                            "visible_passes": [True],
                            "hidden_passes": [False],
                            "status": "ok",
                        },
                        {
                            "draw_index": 3,
                            "code": "def sub_one(x):\n    return 0\n",
                            "visible_passes": [False],
                            "hidden_passes": [False],
                            "status": "ok",
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_unmatched_checkpoint(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "draw_index": 0,
                            "code": "def f(x):\n    return x\n",
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "status": "ok",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_req_code_4198_spec_declares_launch_contract() -> None:
    """REQ-CODE-4198: OpenSpec declares the detached launch artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4198" in spec
    assert "SCENARIO-CODE-4198-GATED-LAUNCH" in spec
    assert "SCENARIO-CODE-4198-HONEST-DEFERRAL" in spec
    for field in exp4198.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4198.FIELD_PRINCIPLES


def test_scenario_code_4198_honest_deferral_does_not_launch(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-CODE-4198-HONEST-DEFERRAL: failed A1 gate skips launch."""

    a1 = _write_a1_artifact(tmp_path / "a1.json", precision=0.4, harness_ready=False)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.json")
    launched = False

    def fail_launch(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("deferral must not launch training")

    monkeypatch.setattr(exp4198, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(exp4198, "_start_detached_process", fail_launch)

    artifact = exp4198.run(
        output_path=tmp_path / "out.json",
        a1_artifact_path=a1,
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
    )

    assert artifact["honest_verdict"] == "complete_verifier_reward_train_deferred_no_clean_operating_point"
    assert artifact["training_launched"] is False
    assert artifact["stable_checkpoint_path"] == ""
    assert artifact["arm_corpus_sizes"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert launched is False


def test_req_code_4198_helper_edges_are_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-4198: helper edge cases are deterministic and auditable."""

    class ToDict:
        def to_dict(self):
            return {"path": tmp_path}

    class Item:
        def item(self):
            return 7

    assert exp4198._jsonable(tmp_path) == str(tmp_path)
    assert exp4198._jsonable(ToDict()) == {"path": str(tmp_path)}
    assert exp4198._jsonable(Item()) == 7
    assert exp4198._float(True) == 0.0
    assert exp4198._float("bad") == 0.0

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp4198.load_json(bad_json)

    assert exp4198._pid_is_alive(0) is False
    monkeypatch.setattr(exp4198.os, "kill", lambda _pid, _sig: None)
    assert exp4198._pid_is_alive(123) is True

    def missing(_pid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(exp4198.os, "kill", missing)
    assert exp4198._pid_is_alive(123) is False

    def forbidden(_pid, _sig):
        raise PermissionError

    monkeypatch.setattr(exp4198.os, "kill", forbidden)
    assert exp4198._pid_is_alive(123) is True


def test_req_code_4198_cuda_blocker_is_terminal(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-4198: CUDA must be available before the long LoRA launch."""

    a1 = _write_a1_artifact(tmp_path / "a1.json")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.json")
    monkeypatch.setattr(exp4198, "_cuda_is_available", lambda: False)

    artifact = exp4198.run(
        output_path=tmp_path / "out.json",
        a1_artifact_path=a1,
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["training_launched"] is False
    assert artifact["preconditions"]["cuda_available"] is False


def test_req_code_4198_unmatched_corpora_block_before_launch(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-4198: A/B N-matching is a precondition for detached training."""

    a1 = _write_a1_artifact(tmp_path / "a1.json")
    checkpoint = _write_unmatched_checkpoint(tmp_path / "checkpoint.json")

    monkeypatch.setattr(exp4198, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(exp4198, "_seed_torch", lambda seed: None)
    monkeypatch.setattr(
        exp4198,
        "_start_detached_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
    )

    artifact = exp4198.run(
        output_path=tmp_path / "out.json",
        a1_artifact_path=a1,
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
    )

    assert artifact["honest_verdict"] == "blocked_3arm_corpus_unmatched"
    assert artifact["training_launched"] is False
    assert artifact["arm_corpus_sizes"]["A"] == 1
    assert artifact["arm_corpus_sizes"]["B"] == 0


def test_scenario_code_4198_gated_launch_writes_stable_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-CODE-4198-GATED-LAUNCH: clean gate launches into stable path."""

    a1 = _write_a1_artifact(tmp_path / "a1.json")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.json")
    captured: dict[str, object] = {}

    def fake_start(command, *, cwd, log_path, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["log_path"] = log_path
        captured["env"] = env
        Path(log_path).write_text("launched\n", encoding="utf-8")
        return exp4198.DetachedProcess(pid=12345, returncode=None, log_path=Path(log_path))

    monkeypatch.setattr(exp4198, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(exp4198, "_seed_torch", lambda seed: None)
    monkeypatch.setattr(exp4198, "_start_detached_process", fake_start)
    monkeypatch.setattr(exp4198, "_pid_is_alive", lambda pid: pid == 12345)

    artifact = exp4198.run(
        output_path=tmp_path / "out.json",
        a1_artifact_path=a1,
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
        random_seed=4198,
    )

    stable_path = Path(artifact["stable_checkpoint_path"])
    command = captured["command"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["training_launched"] is True
    assert artifact["arm_corpus_sizes"]["A"] == 3
    assert artifact["arm_corpus_sizes"]["B"] == 3
    assert artifact["arm_corpus_sizes"]["C"] == 3
    assert artifact["gold_control_early_read"]["status"] == "pending_training_checkpoint"
    assert artifact["model_specs"]["trainable_base"] == "google/gemma-4-E4B-it"
    assert artifact["model_specs"]["trainable_base_is_non_qwen"] is True
    assert artifact["random_seed"] == 4198
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert "4198" not in stable_path.name
    assert (stable_path / "launch_state.json").is_file()
    assert (stable_path / "corpora" / "arm_A.jsonl").is_file()
    assert "--train" in command
    assert "--train-root" in command
    assert str(stable_path / "arms") in command

    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_code_4198_existing_state_and_gold_read_edges(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-4198: corrupt/dead state is ignored and gold reads load when present."""

    stable = tmp_path / "stable"
    stable.mkdir()
    assert exp4198._existing_live_run(stable) is None

    (stable / "launch_state.json").write_text("{", encoding="utf-8")
    assert exp4198._existing_live_run(stable) is None

    (stable / "launch_state.json").write_text(json.dumps({"pid": 999}), encoding="utf-8")
    monkeypatch.setattr(exp4198, "_pid_is_alive", lambda pid: False)
    assert exp4198._existing_live_run(stable) is None

    gold = {"available": True, "status": "pass", "arm_c_minus_base": 0.1}
    (stable / "gold_control_early_read.json").write_text(json.dumps(gold), encoding="utf-8")
    assert exp4198._gold_control_early_read(stable, 0.6) == gold


def test_req_code_4198_existing_live_run_is_resumed_not_restarted(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-4198: a live stable checkpoint is resumed without a new spawn."""

    a1 = _write_a1_artifact(tmp_path / "a1.json")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.json")
    monkeypatch.setattr(exp4198, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(exp4198, "_seed_torch", lambda seed: None)
    monkeypatch.setattr(exp4198, "_pid_is_alive", lambda pid: pid == 67890)

    first = exp4198.prepare_launch(
        a1_payload=exp4198.load_json(a1),
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
        random_seed=4198,
    )
    first.stable_checkpoint_path.mkdir(parents=True, exist_ok=True)
    (first.stable_checkpoint_path / "launch_state.json").write_text(
        json.dumps({"pid": 67890, "command": ["already", "running"]}),
        encoding="utf-8",
    )

    artifact = exp4198.run(
        output_path=tmp_path / "out.json",
        a1_artifact_path=a1,
        generation_checkpoint=checkpoint,
        checkpoint_root=tmp_path / "stable",
        random_seed=4198,
    )

    assert artifact["training_launched"] is True
    assert artifact["launch_status"]["status"] == "existing_live_run"
    assert artifact["launch_status"]["pid"] == 67890


def test_req_code_4198_a1_runner_accepts_stable_train_root(tmp_path: Path) -> None:
    """REQ-CODE-4198: the A1 runner can checkpoint arms under a stable root."""

    runner_path = REPO / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"
    spec = importlib.util.spec_from_file_location("verifier_reward_3arm_runner", runner_path)
    assert spec is not None
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    checkpoint = _write_checkpoint(tmp_path / "checkpoint.json")
    train_root = tmp_path / "stable" / "arms"
    artifact = runner.run(
        checkpoint=checkpoint,
        seed=4198,
        smoke=True,
        train=False,
        output_path=tmp_path / "runner.json",
        train_root=train_root,
    )

    assert artifact["training"]["arm_a"]["output_dir"] == str(train_root / "arm_a_certified")
    assert artifact["training"]["arm_a"]["random_seed"] == 4198
    assert artifact["arm_sizes"]["arm_a_certified"] == artifact["arm_sizes"]["arm_b_random_control"]


def test_req_code_4198_result_script_delegates_to_module() -> None:
    """REQ-CODE-4198: requested results script is executable by path."""

    script = REPO / "results" / "experiment_4198_verifier_reward_3arm_rft_launch.py"
    spec = importlib.util.spec_from_file_location("exp4198_result_script", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main is exp4198.main
