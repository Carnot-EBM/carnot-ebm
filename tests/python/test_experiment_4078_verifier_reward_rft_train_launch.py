"""Tests for Exp 4078 verifier-reward RFT detached train launch.

Spec refs: REQ-LEARN-4078, SCENARIO-LEARN-4078-BLOCKED,
SCENARIO-LEARN-4078-LAUNCH.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.agentic import arc_exp4078_verifier_reward_rft_train_launch as exp4078


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _available_checks() -> list[exp4078.PreconditionCheck]:
    return [
        exp4078.PreconditionCheck("hf_safetensors_qwen_qwen3_5_0_8b", True, "test"),
        exp4078.PreconditionCheck("hf_safetensors_openbmb_minicpm5_1b", True, "test"),
        exp4078.PreconditionCheck("trl_peft_trainers", True, "test"),
        exp4078.PreconditionCheck("cuda_visible", True, "test"),
        exp4078.PreconditionCheck("exp4077_corpora", True, "test"),
    ]


def test_req_learn_4078_spec_declares_launch_contract() -> None:
    """REQ-LEARN-4078: OpenSpec declares resources, launch fields, and paths."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4078" in spec
    assert "SCENARIO-LEARN-4078-BLOCKED" in spec
    assert "SCENARIO-LEARN-4078-LAUNCH" in spec
    assert exp4078.RESULT_FILENAME in spec
    assert "Qwen/Qwen3.5-0.8B" in spec
    assert "openbmb/MiniCPM5-1B" in spec
    for field in exp4078.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4078_blocked_when_exp4077_corpora_missing(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4078-BLOCKED: missing corpora block before launch."""

    output_path = tmp_path / "artifact.json"
    artifact = exp4078.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        preconditions_checker=lambda **_: [
            *_available_checks()[:-1],
            exp4078.PreconditionCheck("exp4077_corpora_missing", False, "missing rft_correct"),
        ],
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert artifact["honest_verdict"] == "blocked_exp4077_corpora_missing"
    assert artifact["train_launched"] is False
    assert artifact["launched_workers"] == []
    assert artifact["checkpoint_paths"]
    assert set(artifact["checkpoint_paths"]) == set(artifact["epochs_completed"])
    assert all(value == 0 for value in artifact["epochs_completed"].values())
    assert exp4078.artifact_schema_errors(artifact) == []


def test_req_learn_4078_corpora_precondition_requires_complete_4077_and_jsonl(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4078-1: Exp 4077 sidecars must exist and be non-empty."""

    results = tmp_path / "results"
    results.mkdir()
    blocked = exp4078.check_exp4077_corpora(repo_root=tmp_path)
    assert blocked.available is False
    assert "missing artifact" in blocked.detail

    (results / "experiment_4077_verifier_reward_rft_corpus_build.json").write_text(
        json.dumps({"honest_verdict": "blocked_precision_gate_unmet_0.1_1.0"}),
        encoding="utf-8",
    )
    assert exp4078.check_exp4077_corpora(repo_root=tmp_path).available is False
    assert "not complete" in exp4078.check_exp4077_corpora(repo_root=tmp_path).detail

    (results / "experiment_4077_verifier_reward_rft_corpus_build.json").write_text(
        json.dumps({"honest_verdict": "complete: rft_corpus_built"}),
        encoding="utf-8",
    )
    assert exp4078.check_exp4077_corpora(repo_root=tmp_path).available is False
    assert "missing or empty" in exp4078.check_exp4077_corpora(repo_root=tmp_path).detail

    for arm in ("rft_correct", "rft_ablation", "gold_sft"):
        (results / f"experiment_4077_{arm}.jsonl").write_text(
            json.dumps({"text": f"{arm} row"}) + "\n",
            encoding="utf-8",
        )

    ready = exp4078.check_exp4077_corpora(repo_root=tmp_path)
    assert ready.available is True
    assert "rft_correct=1" in ready.detail

    empty = results / "experiment_4077_rft_correct.jsonl"
    empty.write_text("", encoding="utf-8")
    assert exp4078.check_exp4077_corpora(repo_root=tmp_path).available is False

    (results / "experiment_4077_verifier_reward_rft_corpus_build.json").write_text(
        "{bad json",
        encoding="utf-8",
    )
    malformed = exp4078.check_exp4077_corpora(repo_root=tmp_path)
    assert malformed.available is False
    assert "malformed artifact" in malformed.detail


def test_scenario_learn_4078_launches_detached_workers_with_stable_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4078-LAUNCH: workers are base+arm keyed and detached."""

    launched_specs: list[exp4078.WorkerSpec] = []

    def launcher(spec: exp4078.WorkerSpec) -> exp4078.LaunchedWorker:
        launched_specs.append(spec)
        spec.checkpoint_path.mkdir(parents=True, exist_ok=True)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("launch ok\n", encoding="utf-8")
        return exp4078.LaunchedWorker(
            base_key=spec.base.key,
            arm=spec.arm,
            pid=1000 + len(launched_specs),
            command=spec.command,
            checkpoint_path=str(spec.checkpoint_path),
            log_path=str(spec.log_path),
            detached=True,
            started=True,
        )

    artifact = exp4078.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "launch.json",
        preconditions_checker=lambda **_: _available_checks(),
        worker_launcher=launcher,
        bases=(exp4078.BASE_MODELS[0],),
    )

    assert artifact["honest_verdict"].startswith("complete: rft_3arm_train_launched_detached")
    assert artifact["train_launched"] is True
    assert len(artifact["launched_workers"]) == 3
    assert [spec.arm for spec in launched_specs] == ["rft_correct", "rft_ablation", "gold_sft"]
    assert len({spec.checkpoint_path for spec in launched_specs}) == 3
    assert all(worker["detached"] is True for worker in artifact["launched_workers"])
    assert all("--checkpoint-path" in worker["command"] for worker in artifact["launched_workers"])
    assert exp4078.artifact_schema_errors(artifact) == []

    config = artifact["training_config"]
    assert config["lora_rank"] == exp4078.DEFAULT_TRAINING_CONFIG.lora_rank
    assert config["learning_rate"] == exp4078.DEFAULT_TRAINING_CONFIG.learning_rate
    assert config["random_seed"] == exp4078.DEFAULT_TRAINING_CONFIG.random_seed


def test_req_learn_4078_epoch_scan_and_schema_validation(tmp_path: Path) -> None:
    """REQ-LEARN-4078-3: progress accounting and artifact schema are defensive."""

    checkpoint_root = tmp_path / "checkpointed"
    (checkpoint_root / "checkpoint-2").mkdir(parents=True)
    (checkpoint_root / "checkpoint-2" / "trainer_state.json").write_text(
        json.dumps({"epoch": 1.0}),
        encoding="utf-8",
    )
    (checkpoint_root / "checkpoint-7").mkdir()
    (checkpoint_root / "checkpoint-7" / "trainer_state.json").write_text(
        json.dumps({"epoch": 2.75}),
        encoding="utf-8",
    )

    assert exp4078.epochs_completed_from_checkpoint(checkpoint_root) == 2
    assert exp4078.epochs_completed_from_checkpoint(tmp_path / "missing") == 0

    no_state = tmp_path / "no-state"
    (no_state / "checkpoint-abc").mkdir(parents=True)
    assert exp4078.epochs_completed_from_checkpoint(no_state) == 1

    bad_state = tmp_path / "bad-state"
    (bad_state / "checkpoint-1").mkdir(parents=True)
    (bad_state / "checkpoint-1" / "trainer_state.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    assert exp4078.epochs_completed_from_checkpoint(bad_state) == 1

    paths = exp4078.stable_checkpoint_paths(
        repo_root=tmp_path,
        bases=(exp4078.BASE_MODELS[0],),
        arms=("rft_correct", "rft_ablation"),
    )
    assert sorted(paths) == ["qwen35_08b:rft_ablation", "qwen35_08b:rft_correct"]
    assert all("experiment_4078_verifier_reward_rft_train" in str(path) for path in paths.values())

    bad: dict[str, Any] = {
        "honest_verdict": "bad",
        "train_launched": "yes",
        "checkpoint_paths": [],
        "epochs_completed": {"x": "0"},
        "inference_substrate": "wrong",
    }
    errors = exp4078.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "train_launched must be a bare bool" in errors
    assert "checkpoint_paths must be a dict" in errors
    assert "epochs_completed values must be bare ints" in errors
    assert "inference_substrate must declare the Exp 4078 substrate" in errors

    missing_errors = exp4078.artifact_schema_errors({})
    assert "missing required field honest_verdict" in missing_errors
    assert "honest_verdict must be a string" in missing_errors
    assert "epochs_completed must be a dict" in missing_errors

    path_type_errors = exp4078.artifact_schema_errors(
        {
            "honest_verdict": "blocked_test",
            "train_launched": False,
            "checkpoint_paths": {"x": 1},
            "epochs_completed": {"x": 0},
            "inference_substrate": exp4078.INFERENCE_SUBSTRATE,
        }
    )
    assert "checkpoint_paths keys and values must be strings" in path_type_errors

    mismatch_errors = exp4078.artifact_schema_errors(
        {
            "honest_verdict": "blocked_test",
            "train_launched": False,
            "checkpoint_paths": {"x": "/tmp/x"},
            "epochs_completed": {"y": 0},
            "inference_substrate": exp4078.INFERENCE_SUBSTRATE,
        }
    )
    assert "checkpoint_paths and epochs_completed keys must match" in mismatch_errors

    worker_errors = exp4078.artifact_schema_errors(
        {
            "honest_verdict": "complete: test",
            "train_launched": True,
            "checkpoint_paths": {},
            "epochs_completed": {},
            "inference_substrate": exp4078.INFERENCE_SUBSTRATE,
            "launched_workers": [],
        }
    )
    assert "train_launched artifacts must include launched_workers" in worker_errors
    assert "launched_workers entries must be dicts" in exp4078.artifact_schema_errors(
        {
            "honest_verdict": "complete: test",
            "train_launched": True,
            "checkpoint_paths": {},
            "epochs_completed": {},
            "inference_substrate": exp4078.INFERENCE_SUBSTRATE,
            "launched_workers": ["bad"],
        }
    )
    assert "launched workers must be detached" in exp4078.artifact_schema_errors(
        {
            "honest_verdict": "complete: test",
            "train_launched": True,
            "checkpoint_paths": {},
            "epochs_completed": {},
            "inference_substrate": exp4078.INFERENCE_SUBSTRATE,
            "launched_workers": [{"detached": False}],
        }
    )


def test_req_learn_4078_launch_worker_uses_session_detach(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-LEARN-4078-2: the real launcher asks Popen for a new POSIX session."""

    calls: list[dict[str, Any]] = []

    class FakeProcess:
        pid = 4242

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append({"command": command, **kwargs})
        return FakeProcess()

    monkeypatch.setattr(exp4078.subprocess, "Popen", fake_popen)

    spec = exp4078.build_worker_specs(
        repo_root=tmp_path,
        bases=(exp4078.BASE_MODELS[0],),
        arms=("rft_correct",),
        python_executable=Path("/venv/bin/python"),
    )[0]
    worker = exp4078.launch_worker(spec)

    assert worker.pid == 4242
    assert worker.detached is True
    assert worker.started is True
    assert calls[0]["start_new_session"] is True
    assert calls[0]["stdin"] is exp4078.subprocess.DEVNULL
    assert calls[0]["cwd"] == str(tmp_path)
