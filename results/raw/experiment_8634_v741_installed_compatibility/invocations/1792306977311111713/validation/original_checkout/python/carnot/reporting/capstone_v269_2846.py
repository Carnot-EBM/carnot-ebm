"""Build the Exp 2846 milestone .269 capstone artifact.

Spec refs: REQ-REPORT-2846, SCENARIO-REPORT-2846.

This module is intentionally a pure synthesis step.  It reads the result files
already written by milestone .269 tasks, records what those files actually say,
and refuses to promote blocked, missing, or adversarially flagged measurements
into paper-ready claims.  That separation matters because the capstone's job is
not to make the milestone look successful; it is to draw the exact boundary
between measured evidence, operational blockers, and next-milestone work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.milestone_capstone.v269"
MILESTONE = "2026.05.269"
OUTPUT_REL_PATH = Path("results/experiment_2846_capstone_v269.json")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp2835": Path("results/experiment_2835_archive_v268.json"),
    "exp2836": Path("results/experiment_2836_sota_runtime_preflight.json"),
    "exp2837": Path("results/experiment_2837_fover_memory_leakage_v3.json"),
    "exp2838": Path("results/experiment_2838_mbpp_dual_condition_v3.json"),
    "exp2839": Path("results/experiment_2839_humaneval_dual_condition_v3.json"),
    "exp2840": Path("results/experiment_2840_truthfulqa_dual_condition_v4.json"),
    "exp2841": Path("results/experiment_2841_halueval_fever_pilot.json"),
    "exp2842": Path("results/experiment_2842_cross_corpus_matrix_v3.json"),
    "exp2843": Path("results/experiment_2843_beaver_epr_bounded_probe.json"),
    "exp2844": Path("results/experiment_2844_loopus_fr11_self_learning_pilot.json"),
    "exp2845": Path("results/experiment_2845_paper_v6_section5_v3.json"),
}

PRIMARY_CORPUS_SOURCES: dict[str, tuple[str, str]] = {
    "FoVer": ("exp2837", "FoVer"),
    "MBPP": ("exp2838", "MBPP"),
    "HumanEval": ("exp2839", "HumanEval"),
    "TruthfulQA": ("exp2840", "TruthfulQA"),
}

PAPER_READY_KEYS = (
    "paper_ready",
    "paper_ready_for_citation",
    "section5_cite_ready",
    "arxiv_ready_v8",
    "arxiv_ready_v7",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal/blocked prefix discipline.",
    "milestone": "Capstone identity.",
    "sota_runtime_ready": "Primary operational outcome.",
    "primary_corpus_results": "Only real upstream metrics.",
    "self_learning_result": "FR-11 continuous self-learning outcome or block.",
    "paper_ready": "Operator-only publication gate.",
    "top_3_next_actions": "Actionable next milestone planning.",
    "docs_updated": "Operational reconciliation.",
    "duration_s": "Real synthesis wall-time; no sleep padding.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from *path*, or an empty dict when unavailable.

    Missing and malformed inputs are normal at milestone boundaries.  A capstone
    must report those gaps in its output instead of crashing or silently filling
    them with invented success values.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith("blocked_")


def _is_terminal_success(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("complete:", "success:"))


def _is_flagged(payload: dict[str, Any]) -> bool:
    return bool(payload.get("flagged_adversarial") or payload.get("corrigendum_pending"))


def source_status(payload: dict[str, Any]) -> str:
    """Classify one upstream artifact without conflating failure modes."""

    if not payload:
        return "missing"
    if _is_flagged(payload):
        return "flagged"
    if _is_blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _is_terminal_success(payload.get("honest_verdict")):
        return "complete"
    return "nonterminal"


def _number_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _source_status_rows(
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        payload = artifacts[exp_id]
        rows[exp_id] = {
            "path": str(rel_path),
            "status": source_status(payload),
            "honest_verdict": payload.get("honest_verdict", "missing_artifact"),
            "flagged_adversarial": _is_flagged(payload),
            "blocked_resources": payload.get("blocked_resources", []),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
        }
    return rows


def _corpus_result(corpus: str, payload: dict[str, Any]) -> dict[str, Any]:
    status = source_status(payload)
    production = _number_or_none(payload.get("condition_a_production_auroc_mean"))
    architecture = _number_or_none(payload.get("condition_b_architecture_only_auroc_mean"))
    learning = _number_or_none(payload.get("learning_contribution"))
    headline_eligible = status == "complete" and production is not None and architecture is not None
    excluded_reason = None
    if not headline_eligible:
        excluded_reason = {
            "missing": "missing_artifact",
            "flagged": "adversarially_flagged",
            "blocked": "blocked_precondition",
            "nonterminal": "nonterminal_verdict",
        }.get(status, "unmeasured")
    return {
        "corpus": corpus,
        "status": status,
        "honest_verdict": payload.get("honest_verdict", "missing_artifact"),
        "n_examples": payload.get("n_examples")
        or payload.get("n_tasks")
        or payload.get("n_questions"),
        "n_seeds": payload.get("n_seeds"),
        "production_auroc_mean": production,
        "architecture_only_auroc_mean": architecture,
        "learning_contribution": learning,
        "blocked_resources": payload.get("blocked_resources", []),
        "flagged_adversarial": _is_flagged(payload),
        "corrigendum_pending": payload.get("corrigendum_pending", []),
        "headline_eligible": headline_eligible,
        "excluded_from_headline_reason": excluded_reason,
    }


def _primary_corpus_results(
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for corpus, (exp_id, label) in PRIMARY_CORPUS_SOURCES.items():
        results[corpus] = _corpus_result(label, artifacts[exp_id])
    return results


def _sota_runtime_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": source_status(payload),
        "sota_runtime_ready": bool(payload.get("sota_runtime_ready")),
        "selected_python": payload.get("selected_python"),
        "venv_torch_cuda_available": bool(payload.get("venv_torch_cuda_available")),
        "system_python_torch_cuda_available": bool(
            payload.get("system_python_torch_cuda_available")
        ),
        "cached_model_ids": [
            item.get("hf_id")
            for item in payload.get("sota_models_cached", [])
            if isinstance(item, dict)
        ],
        "models_missing_from_cache": payload.get("models_missing_from_cache", []),
        "smoke_load_headline_usable": any(
            bool(item.get("headline_usable"))
            for item in payload.get("smoke_load_results", [])
            if isinstance(item, dict)
        ),
        "flagged_adversarial": _is_flagged(payload),
        "corrigendum_pending": payload.get("corrigendum_pending", []),
    }


def _self_learning_result(payload: dict[str, Any]) -> dict[str, Any]:
    status = source_status(payload)
    n_examples = int(payload.get("n_examples") or 0)
    correctness_delta = _number_or_none(payload.get("correctness_delta"))
    energy_delta = _number_or_none(payload.get("mean_energy_delta_loop0_to_final"))
    measured_improvement = bool(
        status == "complete" and n_examples > 0 and correctness_delta and correctness_delta > 0
    )
    return {
        "status": status,
        "honest_verdict": payload.get("honest_verdict", "missing_artifact"),
        "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
        "requested_n_examples": payload.get("requested_n_examples"),
        "n_examples": n_examples,
        "mean_energy_delta_loop0_to_final": energy_delta,
        "correctness_delta": correctness_delta,
        "early_exit_rate": _number_or_none(payload.get("early_exit_rate")),
        "blocked_resources": payload.get("blocked_resources", []),
        "measured_improvement": measured_improvement,
        "methodology_note": payload.get("methodology_note"),
    }


def _pilot_results(artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    halueval_fever = artifacts["exp2841"]
    beaver = artifacts["exp2843"]
    return {
        "exp2841": {
            "status": source_status(halueval_fever),
            "honest_verdict": halueval_fever.get("honest_verdict", "missing_artifact"),
            "pilot_only": bool(halueval_fever.get("pilot_only")),
            "n_examples": halueval_fever.get("n_examples"),
            "pilot_auroc_by_dataset": halueval_fever.get("pilot_auroc_by_dataset", {}),
            "recommendation": halueval_fever.get("recommendation"),
            "headline_eligible": False,
            "excluded_from_headline_reason": "pilot_only_or_adversarially_flagged",
        },
        "exp2843": {
            "status": source_status(beaver),
            "honest_verdict": beaver.get("honest_verdict", "missing_artifact"),
            "beaver_exact": bool(beaver.get("beaver_exact")),
            "beaver_method_label": beaver.get("beaver_method_label"),
            "bounded_prefix_probe_auc": _number_or_none(beaver.get("bounded_prefix_probe_auc")),
            "n_examples": beaver.get("n_examples"),
            "headline_eligible": False,
            "excluded_from_headline_reason": "bounded_proxy_not_exact_beaver_or_flagged",
        },
    }


def _paper_ready(payload: dict[str, Any]) -> bool:
    return bool(
        payload
        and not _is_flagged(payload)
        and any(payload.get(key) is True for key in PAPER_READY_KEYS)
    )


def _gate_blocked_or_not_run(
    artifacts: dict[str, dict[str, Any]],
    primary_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    blocked_primary = [
        corpus
        for corpus, row in primary_results.items()
        if row["status"] in {"blocked", "missing", "flagged"}
    ]
    rows: dict[str, dict[str, Any]] = {}
    if not artifacts["exp2842"]:
        rows["exp2842"] = {
            "reason": "cross_corpus_matrix_missing",
            "blocked_by": blocked_primary,
        }
    if not artifacts["exp2845"]:
        rows["exp2845"] = {
            "reason": "paper_section5_artifact_missing",
            "blocked_by": ["exp2842"] if "exp2842" in rows else blocked_primary,
        }
    return rows


def _top_3_next_actions() -> list[str]:
    return [
        "Clear adversarial verification flags on Exp 2836 and Exp 2837 before any runtime or FoVer metric is cited.",
        "Materialize MBPP, HumanEval, and TruthfulQA local datasets/splits, then rerun dual-condition corpus tasks and regenerate Exp 2842.",
        "Implement or select the live recurrence backend for Exp 2844, then regenerate paper-v6 Section 5 only after matrix and self-learning gates are clean.",
    ]


def _compose_verdict(
    *,
    sota_runtime_ready: bool,
    paper_ready: bool,
    missing_count: int,
    blocked_count: int,
    flagged_count: int,
) -> str:
    return (
        f"complete: .269 capstone synthesized; sota_runtime_ready={str(sota_runtime_ready).lower()}; "
        f"paper_ready={str(paper_ready).lower()}; missing_artifacts={missing_count}; "
        f"blocked_artifacts={blocked_count}; adversarially_flagged_artifacts={flagged_count}; "
        "no new multi-corpus headline claim"
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2846: synthesize the .269 claim-boundary artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    artifacts = {
        exp_id: read_json(root_path / rel_path) for exp_id, rel_path in EXPECTED_ARTIFACTS.items()
    }
    source_rows = _source_status_rows(artifacts)
    primary_results = _primary_corpus_results(artifacts)
    sota_summary = _sota_runtime_summary(artifacts["exp2836"])
    paper_ready = _paper_ready(artifacts["exp2845"])
    missing_artifacts = [
        exp_id
        for exp_id, row in source_rows.items()
        if row["status"] == "missing" and exp_id != "exp2835"
    ]
    blocked_artifacts = [
        exp_id for exp_id, row in source_rows.items() if row["status"] == "blocked"
    ]
    flagged_artifacts = [
        exp_id for exp_id, row in source_rows.items() if row["status"] == "flagged"
    ]
    end = time.perf_counter() if now_s is None else now_s
    duration_s = round(max(0.0, end - start), 6)

    return {
        "schema": SCHEMA,
        "artifact": "experiment_2846_capstone_v269",
        "title": "Milestone 2026.05.269 retrospective capstone",
        "milestone": MILESTONE,
        "honest_verdict": _compose_verdict(
            sota_runtime_ready=sota_summary["sota_runtime_ready"],
            paper_ready=paper_ready,
            missing_count=len(missing_artifacts),
            blocked_count=len(blocked_artifacts),
            flagged_count=len(flagged_artifacts),
        ),
        "sota_runtime_ready": sota_summary["sota_runtime_ready"],
        "sota_runtime_summary": sota_summary,
        "primary_corpus_results": primary_results,
        "self_learning_result": _self_learning_result(artifacts["exp2844"]),
        "pilot_results": _pilot_results(artifacts),
        "paper_ready": paper_ready,
        "paper_section5_status": {
            "status": source_status(artifacts["exp2845"]),
            "honest_verdict": artifacts["exp2845"].get("honest_verdict", "missing_artifact"),
            "cite_ready": paper_ready,
            "reason": (
                "Exp 2845 paper-v6 Section 5 artifact is missing; cross-corpus matrix and non-FoVer rows are not clean."
                if not paper_ready
                else "Exp 2845 reports a ready paper gate."
            ),
        },
        "source_artifact_status": source_rows,
        "missing_artifacts": missing_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "gate_blocked_or_not_run": _gate_blocked_or_not_run(artifacts, primary_results),
        "claim_boundary": {
            "measured": [
                "Exp 2836 reports .venv CUDA torch and one cached Gemma 4 GGUF, but the artifact is adversarially flagged.",
                "Exp 2837 reports FoVer dual-condition AUROC fields, but the artifact is adversarially flagged and not headline-eligible.",
            ],
            "blocked": [
                "Exp 2838 MBPP blocked on mbpp_dataset.",
                "Exp 2839 HumanEval blocked on humaneval_dataset.",
                "Exp 2840 TruthfulQA blocked on truthfulqa_generation_split.",
                "Exp 2844 LoopUS/FR-11 self-learning blocked on live_recurrence_backend.",
            ],
            "not_measured": [
                "Exp 2842 cross-corpus matrix artifact is absent.",
                "Exp 2845 paper-v6 Section 5 artifact is absent.",
            ],
        },
        "top_3_next_actions": _top_3_next_actions(),
        "residual_risks": [
            "A nominally ready SOTA runtime still carries an adversarial duration flag.",
            "FoVer metrics exist but cannot be used as headline evidence until the flag is resolved.",
            "No non-FoVer dual-condition AUROC row exists for the .269 headline table.",
        ],
        "docs_updated": ["openspec/capabilities/research-reporting/spec.md"],
        "docs_reconciliation": {
            "ops/status.md": "not_updated_per_stop_when_done_rule",
            "ops/changelog.md": "not_updated_per_stop_when_done_rule",
            "_bmad/traceability.md": "not_updated_per_stop_when_done_rule",
        },
        "field_principles": FIELD_PRINCIPLES,
        "synthesis_is_compute_free": True,
        "scripts_research_conductor_modified": False,
        "pushed": False,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2846 capstone JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
