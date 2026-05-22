"""Tests for the Exp 2846 milestone .269 capstone artifact.

Spec refs: REQ-REPORT-2846, SCENARIO-REPORT-2846.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import capstone_v269_2846 as exp2846


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_real_v269_inputs(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_2836_sota_runtime_preflight.json",
        {
            "honest_verdict": "success: .venv CUDA torch available",
            "sota_runtime_ready": True,
            "selected_python": "/repo/.venv/bin/python",
            "venv_torch_cuda_available": True,
            "system_python_torch_cuda_available": False,
            "sota_models_cached": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
            "smoke_load_results": [{"headline_usable": True, "n_gpu_layers": 0}],
            "models_missing_from_cache": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
        },
    )
    _write_json(
        root,
        "results/experiment_2837_fover_memory_leakage_v3.json",
        {
            "honest_verdict": "complete: FoVer measured",
            "condition_a_production_auroc_mean": 0.9131336,
            "condition_b_architecture_only_auroc_mean": 0.8946624,
            "learning_contribution": 0.0184712,
            "n_examples": 1000,
            "n_seeds": 5,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
        },
    )
    for exp_id, stem, corpus, blocker in [
        ("2838", "mbpp_dual_condition_v3", "MBPP-sanitized-test", "mbpp_dataset"),
        ("2839", "humaneval_dual_condition_v3", "HumanEval-full", "humaneval_dataset"),
        (
            "2840",
            "truthfulqa_dual_condition_v4",
            "TruthfulQA-generation",
            "truthfulqa_generation_split",
        ),
    ]:
        _write_json(
            root,
            f"results/experiment_{exp_id}_{stem}.json",
            {
                "honest_verdict": f"blocked_{blocker}",
                "corpus": corpus,
                "condition_a_production_auroc_mean": None,
                "condition_b_architecture_only_auroc_mean": None,
                "learning_contribution": None,
                "blocked_resources": [blocker],
                "n_seeds": 5,
            },
        )
    _write_json(
        root,
        "results/experiment_2841_halueval_fever_pilot.json",
        {
            "honest_verdict": "complete: HaluEval/FEVER readiness pilot",
            "pilot_only": True,
            "n_examples": 50,
            "pilot_auroc_by_dataset": {
                "FEVER": {"auroc": 0.4326923077, "ready_for_full_benchmark": True},
                "HaluEval": {"auroc": 0.5769230769, "ready_for_full_benchmark": True},
            },
            "recommendation": "Scale FEVER, HaluEval to N>=500.",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
        },
    )
    _write_json(
        root,
        "results/experiment_2843_beaver_epr_bounded_probe.json",
        {
            "honest_verdict": "complete: bounded-prefix/EPR proxy",
            "beaver_exact": False,
            "beaver_method_label": "bounded-prefix/EPR proxy, not exact BEAVER",
            "bounded_prefix_probe_auc": 0.7756,
            "n_examples": 100,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
        },
    )
    _write_json(
        root,
        "results/experiment_2844_loopus_fr11_self_learning_pilot.json",
        {
            "honest_verdict": "blocked_live_recurrence_backend",
            "continuous_self_learning_task": True,
            "requested_n_examples": 50,
            "n_examples": 0,
            "mean_energy_delta_loop0_to_final": 0.0,
            "correctness_delta": 0.0,
            "early_exit_rate": 0.0,
            "blocked_resources": ["live_recurrence_backend"],
        },
    )


def test_req_report_2846_writes_required_schema_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2846: the capstone emits every required top-level field."""

    _write_real_v269_inputs(tmp_path)

    artifact = exp2846.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    required = {
        "honest_verdict",
        "milestone",
        "sota_runtime_ready",
        "primary_corpus_results",
        "self_learning_result",
        "paper_ready",
        "top_3_next_actions",
        "docs_updated",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.269"
    assert artifact["sota_runtime_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["duration_s"] == 2.25


def test_scenario_report_2846_preserves_missing_blocked_and_flagged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2846: missing, blocked, and flagged inputs stay distinct."""

    _write_real_v269_inputs(tmp_path)

    artifact = exp2846.build_artifact(tmp_path)

    assert artifact["missing_artifacts"] == ["exp2842", "exp2845"]
    assert artifact["blocked_artifacts"] == ["exp2838", "exp2839", "exp2840", "exp2844"]
    assert artifact["flagged_artifacts"] == ["exp2836", "exp2837", "exp2841", "exp2843"]
    assert artifact["source_artifact_status"]["exp2842"]["status"] == "missing"
    assert artifact["source_artifact_status"]["exp2844"]["status"] == "blocked"
    assert artifact["source_artifact_status"]["exp2837"]["status"] == "flagged"
    assert "exp2842" in artifact["gate_blocked_or_not_run"]
    assert "exp2845" in artifact["gate_blocked_or_not_run"]


def test_req_report_2846_primary_corpus_results_use_only_source_metrics(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2846: corpus rows do not invent blocked or replacement metrics."""

    _write_real_v269_inputs(tmp_path)

    artifact = exp2846.build_artifact(tmp_path)
    corpora = artifact["primary_corpus_results"]

    assert corpora["FoVer"]["production_auroc_mean"] == 0.9131336
    assert corpora["FoVer"]["architecture_only_auroc_mean"] == 0.8946624
    assert corpora["FoVer"]["learning_contribution"] == 0.0184712
    assert corpora["FoVer"]["headline_eligible"] is False
    assert corpora["FoVer"]["excluded_from_headline_reason"] == "adversarially_flagged"
    for corpus in ("MBPP", "HumanEval", "TruthfulQA"):
        assert corpora[corpus]["production_auroc_mean"] is None
        assert corpora[corpus]["architecture_only_auroc_mean"] is None
        assert corpora[corpus]["status"] == "blocked"
        assert corpora[corpus]["headline_eligible"] is False


def test_req_report_2846_pilots_and_self_learning_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-2846: pilot/proxy/self-learning results retain their caveats."""

    _write_real_v269_inputs(tmp_path)

    artifact = exp2846.build_artifact(tmp_path)

    assert artifact["pilot_results"]["exp2841"]["pilot_only"] is True
    assert artifact["pilot_results"]["exp2841"]["headline_eligible"] is False
    assert artifact["pilot_results"]["exp2843"]["beaver_exact"] is False
    assert artifact["pilot_results"]["exp2843"]["headline_eligible"] is False
    assert artifact["self_learning_result"]["status"] == "blocked"
    assert artifact["self_learning_result"]["blocked_resources"] == ["live_recurrence_backend"]
    assert artifact["self_learning_result"]["correctness_delta"] == 0.0
    assert artifact["self_learning_result"]["measured_improvement"] is False


def test_req_report_2846_docs_updated_is_honest_about_stop_rule(tmp_path: Path) -> None:
    """REQ-REPORT-2846: docs reconciliation records actual edits, not imagined ops edits."""

    _write_real_v269_inputs(tmp_path)

    artifact = exp2846.build_artifact(tmp_path)

    assert artifact["docs_updated"] == ["openspec/capabilities/research-reporting/spec.md"]
    assert artifact["docs_reconciliation"]["ops/status.md"] == (
        "not_updated_per_stop_when_done_rule"
    )
    assert artifact["docs_reconciliation"]["ops/changelog.md"] == (
        "not_updated_per_stop_when_done_rule"
    )


def test_req_report_2846_write_artifact_creates_json(tmp_path: Path) -> None:
    """REQ-REPORT-2846: write_artifact persists the capstone deliverable."""

    _write_real_v269_inputs(tmp_path)

    out = exp2846.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2846_capstone_v269.json"
    assert payload["duration_s"] == 0.5
    assert payload["honest_verdict"].startswith("complete:")


def test_req_report_2846_helper_status_branches(tmp_path: Path) -> None:
    """REQ-REPORT-2846: helper branches classify malformed and nonterminal inputs."""

    assert exp2846.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2846.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2846.read_json(array) == {}

    assert exp2846.source_status({}) == "missing"
    assert exp2846.source_status({"flagged_adversarial": True}) == "flagged"
    assert exp2846.source_status({"honest_verdict": "blocked_cache"}) == "blocked"
    assert exp2846.source_status({"honest_verdict": "success: ok"}) == "complete"
    assert exp2846.source_status({"honest_verdict": "running"}) == "nonterminal"
