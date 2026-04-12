"""Spec: REQ-REPORT-001, REQ-REPORT-002, REQ-REPORT-003, REQ-REPORT-004,
SCENARIO-REPORT-001, SCENARIO-REPORT-002, SCENARIO-REPORT-003
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

README_TEMPLATE = """# Carnot

Carnot is an Energy-Based Model framework for **verifying and repairing LLM outputs**. Through 160+ experiments across 11 milestones, we proved that structural constraint verification via Ising models catches hallucinations that activation-based approaches miss — and that a verify-repair loop can fix them automatically.

**The breakthrough:** LLM proposes → Ising verifies → repair loop fixes. Full GSM8K (1,319 questions): +10-14% accuracy. Adversarial GSM8K (Apple methodology): +24-28% on number-swapped variants. HumanEval pass@1: 90%→96%. Self-learning pipeline: 67.6%→97.0% over 500 questions. 0.006ms per constraint check enables real-time guided decoding.

**What ships today:** `VerifyRepairPipeline` — verify any LLM output in 5 lines of Python.

**What we learned:** Activation-based EBMs detect confidence, not correctness (50% practical). The 14 principles from our systematic negative results save other researchers months of dead ends. Structural constraint verification is what actually works. See the [technical report](docs/technical-report.md) for full results.

## Key Results (160+ experiments, 16 models, 11 milestones)

### What actually works in practice

| Approach | Domain | Result | Practical? |
|----------|--------|--------|-----------|
| **Full GSM8K (1,319 questions)** | Math | 70-77% → 84-88% | **Yes** — publishable, +10-14% |
| **Adversarial GSM8K (Apple)** | Math | +24-28% on number-swapped | **Yes** — robust to adversarial |
| **Self-learning (Tier 1)** | All | 67.6% → 97.0% | **Yes** — gets smarter with use |
| **HumanEval + Ising fuzzing** | Code | pass@1: 90% → 96% | **Yes** — instrumentation + repair |
| **Factual coverage (Wikidata)** | Facts | 96% claim coverage | **Yes** — factual gap closed |

### What works on test sets but fails in practice
"""


REPORT_TEMPLATE = """# Carnot: Energy-Based Verification for LLM Output

## A Technical Report on 160+ Experiments Across Eleven Research Milestones

## Abstract

We present Carnot, an open-source framework that combines Energy-Based Models (EBMs) with Large Language Models (LLMs) to reduce hallucinations in generated output.

**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm shift from detection to verification yielded: (1) full GSM8K (1,319 questions) showing Qwen3.5 improving from 70.6% to 84.4% and Gemma4 from 77.1% to 87.8% with verify-repair, (2) adversarial GSM8K (Apple methodology) recovering +24-28% accuracy on number-swapped variants, (3) self-learning Tier 1 improving from 67.6% to 97.0% accuracy over 500 questions via online constraint generation, (4) 96% factual claim coverage via Wikidata knowledge base integration.

## 1. Introduction

### 1.4 The Paradigm Shift: From Detection to Verification

The resulting architecture — LLM proposes, Ising verifies, repair loop fixes — proved dramatically more effective than any activation-based approach. On live LLM output, it achieves 100% hallucination detection (vs 50% practical for activation EBMs), +27% accuracy improvement via repair (vs -3% to -6% for activation-based rejection), and 96% pass@1 on HumanEval (vs 90% baseline).

## 5. Phase 3: Live LLM End-to-End (Experiments 53-64)

Phase 2 validated individual components with simulated LLM outputs. Phase 3 connects a real LLM (Qwen3.5-0.8B, local) to the constraint pipeline and runs everything end-to-end. This is where the paradigm shift delivers its payoff.

## 12. Conclusion

### Part 2: Constraint-based verification works

- **Full GSM8K (1,319 questions)**: Qwen3.5 70.6% -> 84.4%, Gemma4 77.1% -> 87.8%
- **Adversarial GSM8K**: +24-28% on number-swapped variants (Apple methodology)
- **Self-learning Tier 1**: 67.6% -> 97.0% accuracy over 500 questions
- **Factual coverage**: 96% via Wikidata knowledge base integration
- **HumanEval pass@1: 90% -> 96%** with Ising-guided fuzzing and repair (Experiment 68)

## 15. Limitations

3. **Simulated fallbacks.** Some benchmark experiments (GSM8K, HumanEval) used simulated LLM outputs when model loading failed. Live results are available for Experiments 56-57 but not all benchmarks have been validated at full scale with live models.
"""


INDEX_TEMPLATE = """<section class="hero">
  <div class="hero-content">
    <div class="stats-bar">
      <div class="stat"><div class="stat-num">160+</div><div class="stat-label">Experiments</div></div>
      <div class="stat"><div class="stat-num">4</div><div class="stat-label">Energy Tiers</div></div>
      <div class="stat"><div class="stat-num">0.006ms</div><div class="stat-label">Verify Latency</div></div>
      <div class="stat"><div class="stat-num">2,251</div><div class="stat-label">Tests</div></div>
    </div>
  </div>
</section>

<section id="features">
  <div class="bento-grid">
    <div class="bento-card">
      <h3 class="bento-title">Verify-Repair Pipeline</h3>
      <p class="bento-text">VerifyRepairPipeline: extract constraints, verify via Ising, repair with LLM feedback. +15% on adversarial math, +6% on HumanEval. 5 lines of Python.</p>
    </div>
    <div class="bento-card">
      <h3 class="bento-title">Self-Learning</h3>
      <p class="bento-text">Gets smarter with use: 67.6% &rarr; 97.0% over 500 questions. Online constraint generation from memory patterns, persistent across sessions. 96% factual claim coverage via Wikidata.</p>
    </div>
  </div>
</section>

<section id="results">
  <div class="container">
    <span class="section-label">Evidence</span>
    <h2 class="section-title">Measured Performance</h2>
    <div class="results-grid">
      <div class="r-card">
        <span class="r-tag">GSM8K Full (1,319 Questions)</span>
        <h3 class="r-title">Math Reasoning at Scale</h3>
        <div class="r-stats"><span class="r-before">Baseline: 70-77%</span> <span class="r-after">+Repair: 84-88%</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Adversarial Math (Apple GSM8K)</span>
        <h3 class="r-title">+24-28% on Number-Swapped</h3>
        <div class="r-stats"><span class="r-before">LLM drops to: 46-53%</span> <span class="r-after">+Repair: 74-78%</span></div>
      </div>
    </div>
  </div>
</section>
"""


def load_cleanup_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_209_cleanup.py"
    spec = importlib.util.spec_from_file_location("experiment_209_cleanup", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "README.md").write_text(README_TEMPLATE)
    (repo / "docs" / "technical-report.md").write_text(REPORT_TEMPLATE)
    (repo / "docs" / "index.html").write_text(INDEX_TEMPLATE)
    return repo


def populate_core_results(repo: Path) -> None:
    write_json(
        repo / "results" / "experiment_134_results.json",
        {
            "experiment": 134,
            "summary": {
                "fixed_overall_accuracy": 0.676,
                "adaptive_overall_accuracy": 0.97,
            },
        },
    )
    write_json(
        repo / "results" / "experiment_158_results.json",
        {
            "experiment": 158,
            "coverage_pct": 96.0,
        },
    )
    write_json(
        repo / "results" / "experiment_161_results.json",
        {
            "experiment": 161,
            "metadata": {
                "inference_mode": "simulation",
            },
            "statistics": {
                "Qwen3.5-0.8B": {
                    "baseline": {"accuracy": 0.706},
                    "verify_repair": {"accuracy": 0.844},
                    "improvement_delta": 0.138,
                },
                "Gemma4-E4B-it": {
                    "baseline": {"accuracy": 0.771},
                    "verify_repair": {"accuracy": 0.878},
                    "improvement_delta": 0.107,
                },
            },
        },
    )
    write_json(
        repo / "results" / "experiment_178_results.json",
        {
            "experiment": 178,
            "inference_mode": "simulated",
            "per_variant": {
                "number_swapped": {
                    "improvement_delta_pp_per_model": {
                        "Qwen3.5-0.8B": 28.25,
                        "Gemma4-E4B-it": 24.0,
                    },
                },
            },
        },
    )
    write_json(
        repo / "results" / "experiment_184_results.json",
        {
            "experiment": 184,
            "metadata": {
                "inference_mode": "live_gpu",
            },
            "standard_statistics": {
                "baseline": {"accuracy": 0.63},
                "verify_repair": {"accuracy": 0.61},
                "improvement_delta": -0.02,
            },
        },
    )
    write_json(
        repo / "results" / "experiment_203_results.json",
        {
            "experiment": 203,
            "metadata": {"inference_mode": "live_gpu"},
        },
    )
    write_json(
        repo / "results" / "experiment_206_results.json",
        {
            "experiment": 206,
            "metadata": {"inference_mode": "live_gpu"},
            "statistics": {
                "baseline_accuracy": 0.91,
                "verify_repair_accuracy": 0.91,
                "improvement_delta": 0.0,
            },
        },
    )
    write_json(
        repo / "results" / "experiment_207_results.json",
        {
            "experiment": 207,
            "metadata": {"inference_mode": "live_gpu"},
            "statistics": {
                "baseline_accuracy": 0.91,
                "verify_repair_accuracy": 0.91,
                "improvement_delta": 0.0,
            },
        },
    )
    write_json(
        repo / "results" / "experiment_208_results.json",
        {
            "experiment": 208,
            "metadata": {"inference_mode": "live_gpu"},
            "statistics": {
                "baseline": {"pass_at_1": 0.1666666667},
                "verify_repair": {"pass_at_1": 0.2},
                "improvement": {"delta": 0.0333333333},
            },
        },
    )


def test_run_cleanup_marks_live_simulated_and_missing_results(tmp_path):
    """REQ-REPORT-001, REQ-REPORT-002, SCENARIO-REPORT-001/002/003: result artifacts are normalized and labeled honestly."""
    module = load_cleanup_module()
    repo = make_repo(tmp_path)

    write_json(
        repo / "results" / "experiment_001_results.json",
        {
            "experiment": 1,
            "metadata": {"inference_mode": "live_gpu"},
        },
    )
    write_json(
        repo / "results" / "experiment_002_results.json",
        {
            "experiment": 2,
            "inference_mode": "simulated",
        },
    )
    write_json(
        repo / "results" / "experiment_003_results.json",
        {
            "experiment": 3,
            "description": "missing provenance",
        },
    )

    summary = module.run_cleanup(repo)

    live = json.loads((repo / "results" / "experiment_001_results.json").read_text())
    simulated = json.loads((repo / "results" / "experiment_002_results.json").read_text())
    missing = json.loads((repo / "results" / "experiment_003_results.json").read_text())

    assert summary.total_results == 3
    assert summary.validated_live_gpu == 1
    assert summary.simulated_results == 1
    assert summary.unverified_results == 1

    assert live["inference_mode"] == "live_gpu"
    assert live["result_provenance"]["status"] == "validated_live_gpu"
    assert live["result_provenance"]["source"] == "metadata.inference_mode"
    assert "VALIDATED LIVE RESULT" in live["result_header"]

    assert simulated["result_provenance"]["status"] == "warning"
    assert simulated["result_provenance"]["normalized_mode"] == "simulated"
    assert "WARNING" in simulated["result_header"]

    assert "inference_mode" not in missing or missing["inference_mode"] is None
    assert missing["result_provenance"]["normalized_mode"] == "missing"
    assert missing["result_provenance"]["source"] == "missing"
    assert "WARNING" in missing["result_header"]


def test_run_cleanup_rewrites_public_docs_with_provenance_labels(tmp_path):
    """REQ-REPORT-003, REQ-REPORT-004: README, report, and landing page disclose live vs simulated evidence."""
    module = load_cleanup_module()
    repo = make_repo(tmp_path)
    populate_core_results(repo)

    exit_code = module.main(["--root", str(repo)])
    assert exit_code == 0

    readme = (repo / "README.md").read_text()
    report = (repo / "docs" / "technical-report.md").read_text()
    index = (repo / "docs" / "index.html").read_text()

    assert "validated live artifacts" in readme
    assert "Full GSM8K (Exp 161)" in readme
    assert "Simulated" in readme
    assert "Live HumanEval (Exp 208)" in readme
    assert "Missing explicit inference provenance" in readme

    assert "## Simulation vs Reality" in report
    assert "validated live_gpu artifacts" in report
    assert "Exp 208" in report
    assert "Exp 161" in report
    assert "Exp 178" in report
    assert "not yet validated as full live benchmarks" in report

    assert "Measured Performance With Provenance" in index
    assert "Validated live GPU" in index
    assert "Simulated benchmark" in index
    assert "Live HumanEval" in index
    assert "provenance audit" in index


def test_run_cleanup_is_idempotent(tmp_path):
    """REQ-REPORT-001 through REQ-REPORT-004: rerunning the cleanup is stable."""
    module = load_cleanup_module()
    repo = make_repo(tmp_path)
    populate_core_results(repo)

    first = module.run_cleanup(repo)
    snapshot = {
        "README.md": (repo / "README.md").read_text(),
        "docs/technical-report.md": (repo / "docs" / "technical-report.md").read_text(),
        "docs/index.html": (repo / "docs" / "index.html").read_text(),
        "exp161": (repo / "results" / "experiment_161_results.json").read_text(),
    }

    second = module.run_cleanup(repo)

    assert first == second
    assert snapshot["README.md"] == (repo / "README.md").read_text()
    assert (
        snapshot["docs/technical-report.md"] == (repo / "docs" / "technical-report.md").read_text()
    )
    assert snapshot["docs/index.html"] == (repo / "docs" / "index.html").read_text()
    assert snapshot["exp161"] == (repo / "results" / "experiment_161_results.json").read_text()


def test_cleanup_helper_error_paths_and_main_entrypoint(tmp_path, monkeypatch):
    """REQ-REPORT-001 through REQ-REPORT-004: helper branches and CLI entrypoint behave deterministically."""
    module = load_cleanup_module()

    assert module.normalize_mode("custom_mode") == "custom_mode"
    assert module.is_number(1.5) is True
    assert module.is_number("1.5") is False

    with pytest.raises(ValueError, match="start marker not found"):
        module.require_between("body", "missing", ("end",), "replacement")
    with pytest.raises(ValueError, match="end marker not found"):
        module.require_between("start only", "start", ("end",), "replacement")
    with pytest.raises(ValueError, match="start marker not found"):
        module.replace_block("body", "missing", ("end",), "replacement")
    with pytest.raises(ValueError, match="end marker not found"):
        module.replace_block("<section>", "<section>", ("</section>",), "replacement")

    repo = make_repo(tmp_path)
    populate_core_results(repo)
    (repo / "docs" / "technical-report.md").write_text("# broken report\n")
    summary = module.CleanupSummary(0, 0, 0, 0, (), (), ())
    with pytest.raises(ValueError, match="could not insert Simulation vs Reality section"):
        module.update_technical_report(repo, summary, {}, {})

    repo = make_repo(tmp_path / "cli")
    populate_core_results(repo)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_209_cleanup.py"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--root", str(repo)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")
    assert exc_info.value.code == 0
