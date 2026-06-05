"""Tests for Exp 3825 offline verifier-oracle Q-head distillation.

Spec refs: REQ-3825, SCENARIO-3825-SKIP, SCENARIO-3825-TRAIN,
SCENARIO-3825-ABLATION.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Pre-warm torch at collection time so the repo RSS watchdog does not charge
# PyTorch's one-time import/allocation cost to the first test case.
_warm_layer = torch.nn.Linear(1, 1)
_warm_optim = torch.optim.Adam(_warm_layer.parameters(), lr=0.01)

from scripts.experiments import experiment_3825_distillation_oracle_qhead as mod


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _corpus(n: int = 60) -> list[dict[str, object]]:
    return [
        {"id": item_id, "difficulty": "hard" if item_id % 2 == 0 else "extreme"}
        for item_id in range(n)
    ]


def test_spec_declares_req_3825() -> None:
    """REQ-3825: OpenSpec declares Exp 3825 before implementation."""
    spec = Path("openspec/capabilities/phase3-kona/spec.md").read_text(encoding="utf-8")

    assert "REQ-3825" in spec
    assert "SCENARIO-3825-SKIP" in spec
    assert "SCENARIO-3825-TRAIN" in spec
    assert "SCENARIO-3825-ABLATION" in spec


def test_closed_headroom_gate_writes_skipped_artifact(tmp_path: Path) -> None:
    """SCENARIO-3825-SKIP: closed headroom gate skips distillation cleanly."""
    headroom_path = _write_json(tmp_path / "results/experiment_3824.json", {"headroom_confirmed": False})

    artifact = mod.build_artifact(headroom_artifact_path=headroom_path)

    assert artifact["honest_verdict"] == "complete: distillation_skipped_headroom_not_confirmed"
    assert mod.field_value(artifact["n_train_trajectories"]) == 0
    assert mod.field_value(artifact["n_heldout_trajectories"]) == 0
    assert mod.field_value(artifact["preconditions_checked"])["headroom_confirmed"] is False
    for field_name, principle in mod.REQUIRED_PRINCIPLES.items():
        assert artifact[field_name]["principle"] == principle


def test_verifier_oracle_labels_final_trajectory_constraints() -> None:
    """REQ-3825: verifier oracle produces correctness and constraint-count labels."""
    refiner = mod.PrototypeRecursiveRefiner(latent_dim=6, n_steps=6)
    correct_latents = refiner.unroll({"id": 0, "difficulty": "hard"}, variant=0)
    wrong_latents = refiner.unroll({"id": 1, "difficulty": "hard"}, variant=0)

    correct_label = mod.verify_final_trajectory(correct_latents)
    wrong_label = mod.verify_final_trajectory(wrong_latents)

    assert correct_label["correct"] is True
    assert wrong_label["correct"] is False
    assert correct_label["constraints_satisfied"] > wrong_label["constraints_satisfied"]
    assert correct_label["constraint_count"] == wrong_label["constraint_count"]


def test_training_reports_heldout_ablated_auroc_and_monotonic_calibration() -> None:
    """SCENARIO-3825-TRAIN: Q-head learns held-out signal from continuous latents."""
    refiner = mod.PrototypeRecursiveRefiner(latent_dim=6, n_steps=6)
    trajectories = mod.build_trajectory_dataset(
        _corpus(),
        refiner=refiner,
        variants_per_record=2,
    )
    train_rows, heldout_rows = mod.split_trajectories(
        trajectories,
        train_fraction=0.6,
        random_seed=3825,
    )

    qhead, report = mod.train_and_evaluate_qhead(
        train_rows,
        heldout_rows,
        random_seed=3825,
        epochs=60,
        lr=0.05,
    )

    assert qhead is not None
    assert report["heldout_auroc"] > 0.9
    assert report["ablated_auroc"] > 0.9
    assert report["calibration_monotonic"] is True
    assert len(report["per_step_calibration_curve"]) == refiner.n_steps
    assert report["per_step_calibration_curve"][-1]["correct_mean_score"] > (
        report["per_step_calibration_curve"][0]["correct_mean_score"]
    )


def test_ablation_zeros_step_and_identity_conditioning_features() -> None:
    """SCENARIO-3825-ABLATION: ablated features exclude step/id crutches."""
    refiner = mod.PrototypeRecursiveRefiner(latent_dim=6, n_steps=4)
    trajectories = mod.build_trajectory_dataset(
        [{"id": 0, "difficulty": "hard"}],
        refiner=refiner,
        variants_per_record=2,
    )

    full_features = mod.latent_feature_tensor(trajectories, final_only=True, ablated=False)
    ablated_features = mod.latent_feature_tensor(trajectories, final_only=True, ablated=True)

    assert full_features.shape == ablated_features.shape
    assert float(full_features[:, 2].abs().sum()) > 0.0
    assert float(ablated_features[:, 2].abs().sum()) == 0.0
    assert float(ablated_features[:, 4:].abs().sum()) == 0.0


def test_open_headroom_builds_principled_training_artifact(tmp_path: Path) -> None:
    """REQ-3825: open headroom gate writes the required principle-bearing metrics."""
    corpus_path = _write_json(tmp_path / "data/headroom_corpus_exp3824.json", _corpus())
    headroom_path = _write_json(
        tmp_path / "results/experiment_3824_headroom_gate_corpus.json",
        {"headroom_confirmed": True, "corpus_path": {"value": str(corpus_path)}},
    )

    artifact = mod.build_artifact(
        headroom_artifact_path=headroom_path,
        refiner=mod.PrototypeRecursiveRefiner(latent_dim=6, n_steps=6),
        random_seed=3825,
        training_epochs=60,
    )

    assert artifact["honest_verdict"].startswith(
        "complete: distillation_oracle_qhead_feasible_auroc"
    )
    assert mod.field_value(artifact["qhead_heldout_auroc"]) > 0.9
    assert mod.field_value(artifact["qhead_ablated_auroc"]) > 0.9
    assert mod.field_value(artifact["per_step_calibration_monotonic"]) is True
    assert mod.field_value(artifact["n_train_trajectories"]) == 72
    assert mod.field_value(artifact["n_heldout_trajectories"]) == 48
    assert "constraint_oracle" in mod.field_value(artifact["verifier_oracle_label_source"])
    assert len(mod.field_value(artifact["reproducibility_checksum"])) == 16
    for field_name, principle in mod.REQUIRED_PRINCIPLES.items():
        assert artifact[field_name]["principle"] == principle


def test_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-3825: terminal artifact writer persists JSON for conductor audit."""
    artifact = mod.skipped_artifact(
        preconditions={"torch_available": True, "headroom_confirmed": False},
        duration_s=0.01,
        random_seed=3825,
    )
    output_path = tmp_path / "experiment_3825.json"

    mod.write_artifact(artifact, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == mod.SCHEMA
    assert loaded["honest_verdict"] == "complete: distillation_skipped_headroom_not_confirmed"


def test_field_value_rejects_unprincipled_metric() -> None:
    """REQ-3825: required artifact metric reads must be principle-bearing."""
    with pytest.raises(TypeError):
        mod.field_value({"value": 1.0})


def test_low_level_edge_cases_and_source_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-3825: defensive helpers fail closed and cover source-loader outcomes."""
    assert mod._import_available("definitely_missing_module_for_3825") is False

    with pytest.raises(ValueError, match="latent_dim"):
        mod.PrototypeRecursiveRefiner(latent_dim=5)
    with pytest.raises(ValueError, match="n_steps"):
        mod.PrototypeRecursiveRefiner(n_steps=1)

    wide = mod.PrototypeRecursiveRefiner(latent_dim=8, n_steps=2)
    assert len(wide.unroll({"id": 0, "difficulty": "hard"}, variant=0)[0]) == 8

    missing_ok, missing_msg = mod.check_nano_trm_source_loadable(tmp_path)
    assert missing_ok is False
    assert "missing" in missing_msg

    fake_root = tmp_path / "fake"
    fake_file = fake_root / "nano-trm/src/nn/models/trm.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("class NotTRM: pass\n", encoding="utf-8")
    fake_ok, fake_msg = mod.check_nano_trm_source_loadable(fake_root)
    assert fake_ok is False
    assert "ImportError" in fake_msg

    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]
    real_ok, real_msg = mod.check_nano_trm_source_loadable()
    assert real_ok is True
    assert "TRMModule" in real_msg

    preconditions: dict[str, object] = {}
    monkeypatch.setattr(mod, "check_nano_trm_source_loadable", lambda: (False, "no source"))
    with pytest.raises(RuntimeError, match="no source"):
        mod.load_default_refiner(preconditions)
    assert preconditions["trm_source_loadable"] is False

    monkeypatch.setattr(mod, "check_nano_trm_source_loadable", lambda: (True, "ok"))
    loaded = mod.load_default_refiner(preconditions)
    assert "nano-trm source importable" in loaded.source_label


def test_verifier_and_split_edge_cases() -> None:
    """REQ-3825: verifier and split helpers reject invalid inputs."""
    with pytest.raises(ValueError, match="empty"):
        mod.verify_final_trajectory([])

    refiner = mod.PrototypeRecursiveRefiner()
    rows = mod.build_trajectory_dataset(_corpus(2), refiner=refiner, variants_per_record=1)
    with pytest.raises(ValueError, match="train_fraction"):
        mod.split_trajectories(rows, train_fraction=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        mod.split_trajectories([], train_fraction=0.5)


def test_auroc_ties_single_class_and_bounded_verdict() -> None:
    """SCENARIO-3825-TRAIN: AUROC and verdict gates handle bounded outcomes."""
    assert mod.compute_auroc([1, 1], [0.7, 0.8]) == 0.5
    assert mod.compute_auroc([1, 0], [0.5, 0.5]) == 0.5
    assert mod.classify_verdict(
        heldout_auroc=0.51,
        ablated_auroc=0.51,
        calibration_monotonic=True,
    ) == "complete: distillation_oracle_qhead_bounded_no_signal_auroc0.510"


def test_json_loaders_and_path_resolution_edge_cases(tmp_path: Path) -> None:
    """REQ-3825: malformed artifacts and corpus paths are rejected deterministically."""
    explicit = tmp_path / "explicit.json"
    assert mod._resolve_corpus_path({}, explicit_corpus_path=explicit, headroom_artifact_path=tmp_path) == explicit
    assert mod._resolve_corpus_path({}, explicit_corpus_path=None, headroom_artifact_path=tmp_path) is None

    unresolved = mod._resolve_corpus_path(
        {"corpus_path": {"value": "missing.json"}},
        explicit_corpus_path=None,
        headroom_artifact_path=tmp_path / "results/experiment_3824.json",
    )
    assert unresolved == Path("missing.json")

    list_path = _write_json(tmp_path / "list.json", [])
    with pytest.raises(ValueError, match="headroom artifact"):
        mod._load_headroom(list_path)

    dict_path = _write_json(tmp_path / "dict.json", {})
    with pytest.raises(ValueError, match="corpus"):
        mod._load_corpus(dict_path)


def test_build_artifact_blocked_precondition_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-3825-SKIP: blocked_* verdicts carry precondition evidence."""
    missing = tmp_path / "missing.json"
    assert mod.build_artifact(headroom_artifact_path=missing)["honest_verdict"] == (
        "blocked_exp3824_headroom_artifact_missing"
    )

    headroom_path = _write_json(tmp_path / "headroom.json", {"headroom_confirmed": True})
    monkeypatch.setattr(mod, "_import_available", lambda name: False if name == "torch" else True)
    assert mod.build_artifact(headroom_artifact_path=headroom_path)["honest_verdict"] == (
        "blocked_torch_unavailable"
    )
    monkeypatch.setattr(mod, "_import_available", lambda name: True)

    malformed_headroom = _write_json(tmp_path / "bad_headroom.json", [])
    assert mod.build_artifact(headroom_artifact_path=malformed_headroom)["honest_verdict"] == (
        "blocked_exp3824_headroom_artifact_malformed"
    )

    assert mod.build_artifact(headroom_artifact_path=headroom_path)["honest_verdict"] == (
        "blocked_exp3824_corpus_unavailable"
    )

    bad_corpus = _write_json(tmp_path / "bad_corpus.json", {})
    headroom_with_bad_corpus = _write_json(
        tmp_path / "headroom_bad_corpus.json",
        {"headroom_confirmed": True, "corpus_path": {"value": str(bad_corpus)}},
    )
    assert mod.build_artifact(headroom_artifact_path=headroom_with_bad_corpus)["honest_verdict"] == (
        "blocked_exp3824_corpus_malformed"
    )

    corpus_path = _write_json(tmp_path / "corpus.json", _corpus(6))
    headroom_with_corpus = _write_json(
        tmp_path / "headroom_with_corpus.json",
        {"headroom_confirmed": True, "corpus_path": {"value": str(corpus_path)}},
    )
    monkeypatch.setattr(mod, "load_default_refiner", lambda preconditions: (_ for _ in ()).throw(RuntimeError("no trm")))
    assert mod.build_artifact(headroom_artifact_path=headroom_with_corpus)["honest_verdict"] == (
        "blocked_trm_source_not_loadable"
    )

    monkeypatch.setattr(mod, "VARIANTS_PER_RECORD", 1)
    one_class_path = _write_json(tmp_path / "one_class.json", [{"id": i * 2, "difficulty": "hard"} for i in range(6)])
    one_class_headroom = _write_json(
        tmp_path / "headroom_one_class.json",
        {"headroom_confirmed": True, "corpus_path": {"value": str(one_class_path)}},
    )
    assert mod.build_artifact(
        headroom_artifact_path=one_class_headroom,
        refiner=mod.PrototypeRecursiveRefiner(),
    )["honest_verdict"] == "blocked_qhead_training_no_label_balance"


def test_main_writes_artifact_and_prints_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-3825: command entrypoint writes the terminal artifact and prints verdict."""
    calls: dict[str, object] = {}
    monkeypatch.setattr(mod, "build_artifact", lambda: {"honest_verdict": "complete: demo"})
    monkeypatch.setattr(mod, "write_artifact", lambda artifact, output_path=mod.OUTPUT_PATH: calls.update(artifact=artifact))
    monkeypatch.setattr("builtins.print", lambda value: calls.update(printed=value))

    mod.main()

    assert calls["artifact"] == {"honest_verdict": "complete: demo"}
    assert calls["printed"] == "complete: demo"
