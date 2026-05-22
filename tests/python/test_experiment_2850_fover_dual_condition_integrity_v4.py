"""Tests for Exp 2850 FoVer dual-condition integrity rerun.

Spec: REQ-VERIFY-2850,
      SCENARIO-VERIFY-2850,
      SCENARIO-VERIFY-2850-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fover_dual_condition_integrity_v4 as mod
from carnot.eval.fover_dual_condition_integrity_v4 import (
    ExperimentConfig,
    build_reproducibility_checksum,
    run_experiment,
)
from carnot.eval.fover_memory_leakage_v3 import (
    CONDITION_ARCHITECTURE_ONLY,
    CONDITION_PRODUCTION,
    ConditionMeasurement,
    PreconditionCheck,
    discover_fr11_state_files,
)


SEEDS = (42, 137)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fover_rows(path: Path, n_per_class: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(n_per_class):
        rows.append(
            {
                "question_id": f"ok_{idx}",
                "step_text": f"Compute {idx} + {idx} = {2 * idx}.",
                "label": "correct",
            }
        )
        rows.append(
            {
                "question_id": f"bad_{idx}",
                "step_text": f"Compute {idx} + {idx} = {2 * idx + 1}.",
                "label": "incorrect",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_state(root: Path) -> None:
    _write_json(root / "results" / "nexus_constraint_memory_v2.json", {"rules": []})
    _write_json(
        root / "results" / "session_memory_1447" / "run" / "session_state.json",
        {
            "case_memory": {
                "entries": [
                    {
                        "key": {
                            "benchmark_slice": "fover:bad_0",
                            "violation_families": ["fr11_v7_dvi_verified_incorrect"],
                            "prompt_sketch": "compute mismatch",
                        },
                        "prompt_tokens": ["compute", "mismatch"],
                        "violation_types": ["fr11_v7_dvi_verified_incorrect"],
                    }
                ]
            }
        },
    )
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fr11_zenil_distill_v2.jsonl").write_text(
        json.dumps({"question_id": "bad_1", "is_correct": False}) + "\n",
        encoding="utf-8",
    )


def _minimal_repo(root: Path) -> None:
    _write_fover_rows(root / "data" / "fover_corpus.jsonl")
    _write_state(root)


def _clean_adversarial_report(_path: Path) -> dict[str, Any]:
    return {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}


def test_scenario_verify_2850_blocked_fover_dataset_writes_required_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2850-BLOCKED: missing FoVer data blocks without metrics."""

    _write_state(tmp_path)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=SEEDS,
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        adversarial_verify_runner=_clean_adversarial_report,
        write=True,
    )

    assert artifact["artifact"] == "experiment_2850_fover_dual_condition_integrity_v4"
    assert artifact["honest_verdict"] == "blocked_fover_dataset"
    assert artifact["run_date"] == "20260522"
    assert artifact["random_seed"] == 42
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["n_examples"] == 4
    assert artifact["n_seeds"] == 2
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_learning_contribution"] == {}
    assert artifact["per_seed_results"] == []
    assert artifact["live_model_invoked"] is False
    assert artifact["compute_bound_claim"] is False
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["adversarial_verify_flags"] == []
    assert "model_specs" not in artifact
    encoded = json.dumps(artifact)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    checks = {row["resource"]: row for row in artifact["preconditions_checked"]}
    assert checks["fover_corpus"]["available"] is False
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_verify_2850_success_summary_restores_state_without_overclaim(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2850: dataset-only A/B scoring restores state and stays honest."""

    _minimal_repo(tmp_path)
    calls: list[tuple[int, str, int]] = []

    def fake_condition_runner(
        config: ExperimentConfig,
        seed: int,
        condition: str,
        require_no_state: bool,
    ) -> ConditionMeasurement:
        visible = len(discover_fr11_state_files(config.repo_root))
        calls.append((seed, condition, visible))
        if require_no_state:
            assert visible == 0
        offset = 0.01 if seed == 137 else 0.0
        production = condition == CONDITION_PRODUCTION
        return ConditionMeasurement(
            seed=seed,
            condition=condition,
            auroc=(0.91 + offset) if production else (0.84 + offset),
            per_verifier_auroc={"tier0r_curry_howard": 0.88 + offset},
            n_examples=4,
            state_visible_count=visible,
            fr11_state_loaded=production and visible > 0,
            subset_sha256=f"subset-{seed}",
            python_executable="in-process",
        )

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=SEEDS,
            started_at=1.0,
            clock=lambda: 9.0,
        ),
        condition_runner=fake_condition_runner,
        adversarial_verify_runner=_clean_adversarial_report,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.915)
    assert artifact["condition_a_production_auroc_std"] == pytest.approx(0.005)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.845)
    assert artifact["condition_b_architecture_only_auroc_std"] == pytest.approx(0.005)
    assert artifact["learning_contribution"] == pytest.approx(0.07)
    assert artifact["per_verifier_learning_contribution"] == {
        "tier0r_curry_howard": pytest.approx(0.0)
    }
    assert artifact["state_files_restored_sha_match"] is True
    assert all(row["condition_b_state_visible_count"] == 0 for row in artifact["per_seed_results"])
    assert artifact["live_model_invoked"] is False
    assert artifact["compute_bound_claim"] is False
    assert "model_specs" not in artifact
    assert calls == [
        (42, CONDITION_PRODUCTION, 3),
        (42, CONDITION_ARCHITECTURE_ONLY, 0),
        (137, CONDITION_PRODUCTION, 3),
        (137, CONDITION_ARCHITECTURE_ONLY, 0),
    ]


def test_req_verify_2850_checksum_depends_on_inputs_state_seeds_and_scores(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2850: checksum changes when score rows change."""

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    inputs = mod.input_file_hashes(tmp_path)
    score_rows = [{"seed": 42, "condition_a_production_auroc": 0.9}]

    first = build_reproducibility_checksum(
        input_hashes=inputs,
        state_files=state_files,
        seeds=SEEDS,
        n_examples=4,
        score_rows=score_rows,
    )
    second = build_reproducibility_checksum(
        input_hashes=inputs,
        state_files=state_files,
        seeds=SEEDS,
        n_examples=4,
        score_rows=[{"seed": 42, "condition_a_production_auroc": 0.91}],
    )

    assert len(first) == 64
    assert first != second


def test_req_verify_2850_adversarial_flags_are_embedded(tmp_path: Path) -> None:
    """REQ-VERIFY-2850: adversarial verification summary is included in the artifact."""

    _minimal_repo(tmp_path)

    def flagged_report(_path: Path) -> dict[str, Any]:
        return {
            "loaded": True,
            "flag_count": 1,
            "max_severity": 1,
            "flags": [{"kind": "CHECK", "severity": "warn", "detail": "example"}],
        }

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=(42,),
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        condition_runner=lambda config, seed, condition, require_no_state: ConditionMeasurement(
            seed=seed,
            condition=condition,
            auroc=0.75 if condition == CONDITION_PRODUCTION else 0.70,
            per_verifier_auroc={"tier0r_curry_howard": 0.72},
            n_examples=4,
            state_visible_count=len(discover_fr11_state_files(config.repo_root)),
            fr11_state_loaded=condition == CONDITION_PRODUCTION,
            subset_sha256="subset",
            python_executable="in-process",
        ),
        adversarial_verify_runner=flagged_report,
        write=True,
    )

    assert artifact["adversarial_verify_passed"] is False
    assert artifact["adversarial_verify_flags"] == [
        {"kind": "CHECK", "severity": "warn", "detail": "example"}
    ]
    assert artifact["adversarial_verify_summary"]["flag_count"] == 1


def test_req_verify_2850_precondition_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-2850: preconditions distinguish data, metric, and state blockers."""

    fover = tmp_path / "data" / "fover_corpus.jsonl"
    fover.parent.mkdir(parents=True)
    fover.write_text(
        "\n"
        "{not json}\n"
        + json.dumps({"label": "correct"}) + "\n"
        + json.dumps({"label": "ignored"}) + "\n",
        encoding="utf-8",
    )
    assert mod._count_labeled_fover_rows(fover) == 1
    assert mod._count_labeled_fover_rows(tmp_path / "missing.jsonl") == 0

    checks = [
        PreconditionCheck("nexus_constraint_memory_v2", False, "optional"),
        PreconditionCheck("sklearn", False, "missing"),
    ]
    assert mod._blocked_verdict(checks) == "blocked_metrics_dependency"
    assert mod._blocked_verdict([PreconditionCheck("fr11_state_files", False, "count=0")]) == (
        "blocked_fr11_state_files"
    )
    assert mod._blocked_verdict([PreconditionCheck("fover_corpus", True, "ok")]) is None

    _write_fover_rows(fover, n_per_class=4)
    _write_state(tmp_path)

    def fake_dependency(name: str) -> tuple[bool, str]:
        if name == "sklearn":
            return False, "missing"
        return True, "ok"

    monkeypatch.setattr(mod, "_dependency_detail", fake_dependency)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_examples=4),
        adversarial_verify_runner=_clean_adversarial_report,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_metrics_dependency"


def test_req_verify_2850_score_condition_uses_dataset_scorer(tmp_path: Path) -> None:
    """REQ-VERIFY-2850: default condition scoring delegates to FoVer dataset rows."""

    _minimal_repo(tmp_path)
    measurement = mod.score_condition(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_examples=4),
        42,
        CONDITION_PRODUCTION,
        False,
    )

    assert measurement.condition == CONDITION_PRODUCTION
    assert measurement.n_examples == 4
    assert measurement.state_visible_count > 0
    assert measurement.fr11_state_loaded is True


def test_req_verify_2850_adversarial_runner_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2850: adversarial runner handles missing, invalid, and clean output."""

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    missing = mod.run_adversarial_verify(tmp_path / "artifact.json")
    assert missing["flags"] == [{"kind": "NOT_RUN", "severity": "info", "detail": "script missing"}]

    script = tmp_path / "scripts" / "adversarial_verify.py"
    script.parent.mkdir()
    script.write_text("# placeholder\n", encoding="utf-8")

    def invalid_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "not-json", "stderr detail")

    monkeypatch.setattr(mod.subprocess, "run", invalid_runner)
    invalid = mod.run_adversarial_verify(tmp_path / "artifact.json")
    assert invalid["flag_count"] == 1
    assert invalid["flags"][0]["kind"] == "ADVERSARIAL_VERIFY_ERROR"

    def report_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        payload = {"reports": [{"loaded": True, "flag_count": 0, "flags": []}]}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mod.subprocess, "run", report_runner)
    clean = mod.run_adversarial_verify(tmp_path / "artifact.json")
    assert clean["loaded"] is True
    assert clean["returncode"] == 0

    def empty_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps({"reports": []}), "")

    monkeypatch.setattr(mod.subprocess, "run", empty_runner)
    empty = mod.run_adversarial_verify(tmp_path / "artifact.json")
    assert empty == {"loaded": False, "flag_count": 0, "flags": [], "returncode": 0}


def test_req_verify_2850_restore_mismatch_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2850: a failed state-restore hash check aborts success."""

    _minimal_repo(tmp_path)

    def fake_condition_runner(
        config: ExperimentConfig,
        seed: int,
        condition: str,
        require_no_state: bool,
    ) -> ConditionMeasurement:
        return ConditionMeasurement(
            seed=seed,
            condition=condition,
            auroc=0.75,
            per_verifier_auroc={"tier0r_curry_howard": 0.75},
            n_examples=4,
            state_visible_count=len(discover_fr11_state_files(config.repo_root)),
            fr11_state_loaded=condition == CONDITION_PRODUCTION,
            subset_sha256="subset",
            python_executable="in-process",
        )

    monkeypatch.setattr(mod, "state_files_restored_sha_match", lambda _root, _files: False)
    with pytest.raises(mod.ConditionScoringError, match="restore SHA256 mismatch"):
        run_experiment(
            ExperimentConfig(
                repo_root=tmp_path,
                results_dir=tmp_path / "results",
                n_examples=4,
                random_seeds=(42,),
            ),
            condition_runner=fake_condition_runner,
            adversarial_verify_runner=_clean_adversarial_report,
            write=False,
        )
