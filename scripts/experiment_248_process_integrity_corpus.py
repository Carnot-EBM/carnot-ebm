#!/usr/bin/env python3
"""Experiment 248: process integrity corpus from checked-in live traces.

Writes:
- ``data/research/process_integrity_corpus_248.jsonl``
- ``results/experiment_248_results.json``

Spec: REQ-VERIFY-060,
SCENARIO-VERIFY-070, SCENARIO-VERIFY-071, SCENARIO-VERIFY-072,
SCENARIO-VERIFY-073

Background
----------
Small models can produce the right answer via flawed reasoning ("right answer,
wrong process") or produce wrong answers despite partially sound intermediate
steps.  Neither pattern is visible from outcome accuracy alone.  This corpus
attaches an explicit process integrity label to every (case, iteration) trace
in the checked-in Exp 235 (GSM8K reasoning) and Exp 238 (HumanEval code)
artifacts so that downstream training and benchmarking can distinguish:

- ``right_answer_wrong_process``        — outcome correct, process unsound
- ``wrong_answer_partially_sound_process`` — outcome wrong, steps partially valid
- ``unsupported_step``                  — a specific step lacks premise support
- ``repair_fixed_outcome_only``         — repair improved the answer but not the process
- ``repair_fixed_process_and_outcome``  — repair improved both answer and process
- ``clean``                             — outcome correct, process sound

The schema is compact enough for both the reasoning verifier (TypedReasoningIR)
and the code verifier (SpecCodeVerifier) to consume directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

RUN_DATE = "20260413"
EXPERIMENT = 248
SCHEMA_VERSION = "carnot.process_integrity_corpus.v1"

SOURCE_235 = Path("results/experiment_235_results.json")
SOURCE_238 = Path("results/experiment_238_results.json")
SOURCE_243 = Path("results/experiment_243_results.json")

CORPUS_RELATIVE = Path("data/research/process_integrity_corpus_248.jsonl")
RESULTS_RELATIVE = Path("results/experiment_248_results.json")

# Threshold below which a reasoning claim is considered "unsupported"
_UNSUPPORTED_PREMISE_SUPPORT_THRESHOLD = 0.3
# Threshold above which a reasoning claim is considered "sound"
_SOUND_PREMISE_SUPPORT_THRESHOLD = 0.5


def get_repo_root() -> Path:
    """Return the repository root, respecting the CARNOT_REPO_ROOT override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _resolve(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Classification functions (pure, no I/O — unit-testable)
# ---------------------------------------------------------------------------


def classify_reasoning(
    is_correct: bool,
    verdict: str,
    claim_results: list[dict[str, Any]],
    is_repair: bool,
    prior_correct: bool,
) -> str:
    """Assign a process integrity label to one reasoning trace iteration.

    Parameters
    ----------
    is_correct:
        Whether the model's extracted answer matches the ground truth.
    verdict:
        The semantic verifier v2 verdict string ("verified", "violated", "abstain").
    claim_results:
        List of per-claim dicts from semantic_verifier_v2, each carrying
        ``claim_id``, ``is_final``, ``premise_support``, and
        ``missing_clause_ids``.
    is_repair:
        True when ``iteration > 0`` (i.e., this is a repair attempt).
    prior_correct:
        Whether the immediately preceding iteration produced a correct answer.
        Relevant only when ``is_repair=True``.

    Returns
    -------
    str
        One of the six process integrity label strings.
    """
    non_final = [cr for cr in claim_results if not cr.get("is_final", False)]
    n_total = len(non_final)
    n_unsupported = sum(
        1
        for cr in non_final
        if cr.get("premise_support", 1.0) < _UNSUPPORTED_PREMISE_SUPPORT_THRESHOLD
        and cr.get("missing_clause_ids")
    )
    n_sound = sum(
        1
        for cr in non_final
        if cr.get("premise_support", 0.0) >= _SOUND_PREMISE_SUPPORT_THRESHOLD
    )
    process_sound = (n_unsupported == 0) and (verdict != "violated")
    majority_sound = n_total > 0 and n_sound > n_unsupported

    # Repair-specific labels take highest priority
    if is_repair and is_correct and not prior_correct:
        if process_sound:
            return "repair_fixed_process_and_outcome"
        return "repair_fixed_outcome_only"

    # Outcome-correct with unsound process
    if is_correct and n_unsupported > 0:
        return "right_answer_wrong_process"

    # Outcome-incorrect, majority of steps are sound → partial soundness
    if not is_correct and majority_sound:
        return "wrong_answer_partially_sound_process"

    # Outcome-incorrect, at least one step explicitly unsupported
    if not is_correct and n_unsupported > 0:
        return "unsupported_step"

    return "clean"


def classify_code(
    is_correct: bool,
    n_pbt_failures: int,
    n_spec_violations: int,
    pbt_verified: bool,
    n_derived_props: int,
    is_repair: bool,
    prior_correct: bool,
) -> str:
    """Assign a process integrity label to one code verification trace iteration.

    Parameters
    ----------
    is_correct:
        Whether the candidate passes the official HumanEval test harness.
    n_pbt_failures:
        Number of Hypothesis-backed PBT property failures for this candidate.
    n_spec_violations:
        Number of explicit code-spec clause violations.
    pbt_verified:
        True when PBT found no failures (all derived properties held).
    n_derived_props:
        Total number of Hypothesis-derived properties checked.
    is_repair:
        True when ``iteration > 0``.
    prior_correct:
        Whether the immediately preceding iteration produced a correct answer.

    Returns
    -------
    str
        One of the six process integrity label strings.
    """
    # Repair-specific labels take highest priority
    if is_repair and is_correct and not prior_correct:
        if n_pbt_failures == 0 and n_spec_violations == 0:
            return "repair_fixed_process_and_outcome"
        return "repair_fixed_outcome_only"

    # Official tests pass but verifier layers found issues → right answer, wrong process
    if is_correct and (n_spec_violations > 0 or n_pbt_failures > 0):
        return "right_answer_wrong_process"

    # Official tests fail, PBT found no violations, but explicit spec did
    # → the model's code lacks support for specific spec clauses that random
    #   testing did not surface (analogous to "unsupported step")
    if not is_correct and pbt_verified and n_spec_violations > 0:
        return "unsupported_step"

    # Official tests fail, but some PBT properties hold (partial soundness)
    if not is_correct and n_pbt_failures > 0 and n_pbt_failures < n_derived_props:
        return "wrong_answer_partially_sound_process"

    return "clean"


# ---------------------------------------------------------------------------
# Evidence builders
# ---------------------------------------------------------------------------


def _reasoning_evidence(
    claim_results: list[dict[str, Any]],
    verdict: str,
    semantic_error_probability: float,
) -> dict[str, Any]:
    """Build a compact process_evidence dict for a reasoning trace."""
    non_final = [cr for cr in claim_results if not cr.get("is_final", False)]
    n_unsupported = sum(
        1
        for cr in non_final
        if cr.get("premise_support", 1.0) < _UNSUPPORTED_PREMISE_SUPPORT_THRESHOLD
        and cr.get("missing_clause_ids")
    )
    n_sound = sum(
        1 for cr in non_final if cr.get("premise_support", 0.0) >= _SOUND_PREMISE_SUPPORT_THRESHOLD
    )
    max_ps = max((cr.get("premise_support", 0.0) for cr in non_final), default=0.0)
    return {
        "n_total_non_final_claims": len(non_final),
        "n_unsupported_claims": n_unsupported,
        "n_sound_claims": n_sound,
        "max_premise_support": round(max_ps, 4),
        "semantic_error_probability": round(semantic_error_probability, 4),
        "verifier_verdict": verdict,
    }


def _code_evidence(
    is_correct: bool,
    n_pbt_failures: int,
    n_spec_violations: int,
    pbt_verified: bool,
    n_derived_props: int,
    stage_acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact process_evidence dict for a code trace."""
    return {
        "official_passed": is_correct,
        "pbt_verified": pbt_verified,
        "n_pbt_failures": n_pbt_failures,
        "n_derived_props": n_derived_props,
        "n_spec_violations": n_spec_violations,
        "pbt_verify_only_accepted": bool(stage_acceptance.get("pbt_verify_only")),
        "spec_verify_only_accepted": bool(stage_acceptance.get("spec_aware_verify_only")),
    }


# ---------------------------------------------------------------------------
# Reasoning entry extraction (Exp 235)
# ---------------------------------------------------------------------------


def _slugify_model(name: str) -> str:
    return name.lower().replace(".", "_").replace("-", "_").replace("/", "_")


def _extract_reasoning_steps(typed_reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a compact step list from a TypedReasoningIR dict."""
    steps = []
    for step in typed_reasoning.get("reasoning_steps", []):
        steps.append({
            "step_id": step.get("step_id", ""),
            "kind": step.get("kind", ""),
            "text": step.get("text", ""),
        })
    return steps


def _extract_final_answer(typed_reasoning: dict[str, Any]) -> dict[str, Any] | None:
    fa = typed_reasoning.get("final_answer")
    if not fa:
        return None
    return {
        "text": fa.get("text", ""),
        "normalized": fa.get("normalized"),
        "answer_type": fa.get("answer_type", "unknown"),
    }


def build_reasoning_entries(
    paired_runs: list[dict[str, Any]],
    source_artifact: str,
    source_experiment: int,
) -> list[dict[str, Any]]:
    """Extract process integrity entries from Exp 235 verify_repair paired runs.

    For each (benchmark, model) pair, iterate over every case's history to
    produce one corpus entry per (case_id, iteration).  Only verify_repair
    runs are used because they are the only runs that carry per-iteration
    semantic verifier data.
    """
    # Build a lookup for ground truth correctness per (model, benchmark, case_id)
    # from the baseline runs so we can cross-check typed_reasoning availability.
    entries: list[dict[str, Any]] = []

    for run in paired_runs:
        if run["mode"] != "verify_repair":
            continue
        benchmark = run["benchmark"]
        model_name = run["model_name"]
        model_slug = _slugify_model(model_name)

        for case in run["cases"]:
            case_id: str = case["case_id"]
            initial_correct: bool = bool(case.get("initial_correct", False))
            final_correct: bool = bool(case.get("correct", False))
            history: list[dict[str, Any]] = case.get("history", [])

            for hist_idx, h in enumerate(history):
                iteration: int = h["iteration"]

                # Determine outcome for this iteration
                if iteration == 0:
                    is_correct = initial_correct
                elif hist_idx == len(history) - 1:
                    is_correct = final_correct
                else:
                    # Intermediate iterations: extract final_answer from response
                    # and compare to the case's ground truth is not directly
                    # available here; use a heuristic based on verifier acceptance.
                    # When verified=True it's very likely correct.
                    is_correct = bool(h["verification"].get("verified", False))

                # Prior iteration correctness (for repair label assignment)
                prior_correct: bool
                if iteration == 0:
                    prior_correct = False
                else:
                    prior_h = history[hist_idx - 1]
                    if hist_idx - 1 == 0:
                        prior_correct = initial_correct
                    else:
                        prior_correct = bool(prior_h["verification"].get("verified", False))

                # Semantic verifier v2 data
                sv2: dict[str, Any] = h["verification"].get("semantic_verifier_v2", {})
                verdict: str = sv2.get("verdict", "abstain")
                semantic_error_probability: float = float(
                    sv2.get("semantic_error_probability", 0.0)
                )
                claim_results: list[dict[str, Any]] = sv2.get("claim_results", [])

                label = classify_reasoning(
                    is_correct=is_correct,
                    verdict=verdict,
                    claim_results=claim_results,
                    is_repair=(iteration > 0),
                    prior_correct=prior_correct,
                )

                evidence = _reasoning_evidence(
                    claim_results=claim_results,
                    verdict=verdict,
                    semantic_error_probability=semantic_error_probability,
                )

                # Typed reasoning steps (from history verification payload)
                tr = h["verification"].get("typed_reasoning") or {}
                steps = _extract_reasoning_steps(tr) if tr else []
                final_answer = _extract_final_answer(tr) if tr else None

                corpus_id = (
                    f"pi248-{source_experiment}-{benchmark}-{model_slug}-{case_id}-it{iteration}"
                )

                entry: dict[str, Any] = {
                    "corpus_id": corpus_id,
                    "run_date": RUN_DATE,
                    "experiment": EXPERIMENT,
                    "source_experiment": source_experiment,
                    "source_artifact": source_artifact,
                    "benchmark": benchmark,
                    "domain": "reasoning",
                    "model": model_name,
                    "case_id": case_id,
                    "iteration": iteration,
                    "outcome_label": "correct" if is_correct else "incorrect",
                    "process_label": label,
                    "process_evidence": evidence,
                    "steps": steps,
                    "final_answer": final_answer,
                    "repair_context": None
                    if iteration == 0
                    else {
                        "prior_outcome": "correct" if prior_correct else "incorrect",
                    },
                }
                entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Code entry extraction (Exp 238)
# ---------------------------------------------------------------------------


def build_code_entries(
    model_runs: dict[str, Any],
    source_artifact: str,
    source_experiment: int,
) -> list[dict[str, Any]]:
    """Extract process integrity entries from Exp 238 per_problem_results histories."""
    entries: list[dict[str, Any]] = []
    benchmark = "humaneval_dual_model_spec"

    for model_name, run in model_runs.items():
        model_slug = _slugify_model(model_name)
        ppr: list[dict[str, Any]] = run.get("per_problem_results", [])

        for prob in ppr:
            case_id: str = prob["case_id"]
            entry_point: str = prob.get("entry_point", "")
            history: list[dict[str, Any]] = prob.get("history", [])

            for hist_idx, h in enumerate(history):
                iteration: int = h["iteration"]
                ev: dict[str, Any] = h.get("evaluation", {})

                official = ev.get("official_tests", {})
                pbt = ev.get("pbt", {})
                specs = ev.get("explicit_specs", {})
                stage_acceptance = ev.get("stage_acceptance", {})

                is_correct: bool = bool(official.get("passed", False))
                n_pbt_failures: int = int(pbt.get("n_failures", 0))
                n_derived_props: int = len(pbt.get("derived_properties", []))
                n_spec_violations: int = int(specs.get("n_violations", 0))
                pbt_verified: bool = bool(pbt.get("verified", False))

                prior_correct: bool = False
                if hist_idx > 0:
                    prior_ev = history[hist_idx - 1].get("evaluation", {})
                    prior_correct = bool(prior_ev.get("official_tests", {}).get("passed", False))

                label = classify_code(
                    is_correct=is_correct,
                    n_pbt_failures=n_pbt_failures,
                    n_spec_violations=n_spec_violations,
                    pbt_verified=pbt_verified,
                    n_derived_props=n_derived_props,
                    is_repair=(iteration > 0),
                    prior_correct=prior_correct,
                )

                evidence = _code_evidence(
                    is_correct=is_correct,
                    n_pbt_failures=n_pbt_failures,
                    n_spec_violations=n_spec_violations,
                    pbt_verified=pbt_verified,
                    n_derived_props=n_derived_props,
                    stage_acceptance=stage_acceptance,
                )

                # Steps for code: represent the code body as a single step
                code_body: str = h.get("body", "")
                steps = [
                    {
                        "step_id": "code_body",
                        "kind": "implementation",
                        "text": code_body,
                    }
                ] if code_body else []

                corpus_id = (
                    f"pi248-{source_experiment}-{benchmark}-{model_slug}-{case_id}-it{iteration}"
                )

                entry: dict[str, Any] = {
                    "corpus_id": corpus_id,
                    "run_date": RUN_DATE,
                    "experiment": EXPERIMENT,
                    "source_experiment": source_experiment,
                    "source_artifact": source_artifact,
                    "benchmark": benchmark,
                    "domain": "code",
                    "model": model_name,
                    "case_id": case_id,
                    "entry_point": entry_point,
                    "iteration": iteration,
                    "outcome_label": "correct" if is_correct else "incorrect",
                    "process_label": label,
                    "process_evidence": evidence,
                    "steps": steps,
                    "final_answer": None,
                    "repair_context": None
                    if iteration == 0
                    else {
                        "prior_outcome": "correct" if prior_correct else "incorrect",
                    },
                }
                entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


def build_corpus(
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load source artifacts and produce the full corpus rows and summary.

    Returns
    -------
    tuple[list[dict], dict]
        ``(rows, summary)`` where ``rows`` is the ordered JSONL corpus and
        ``summary`` is the companion JSON summary payload.
    """
    all_entries: list[dict[str, Any]] = []
    source_artifacts_used: list[str] = []

    # --- Exp 235: GSM8K reasoning ---
    path_235 = _resolve(repo_root, SOURCE_235)
    if path_235.exists():
        d235 = _load_json(path_235)
        paired_runs = d235.get("paired_runs", [])
        entries_235 = build_reasoning_entries(
            paired_runs=paired_runs,
            source_artifact=str(SOURCE_235),
            source_experiment=235,
        )
        all_entries.extend(entries_235)
        source_artifacts_used.append(str(SOURCE_235))
    else:
        print(f"[warn] Exp 235 artifact not found at {path_235}, skipping reasoning entries")

    # --- Exp 238: HumanEval code ---
    path_238 = _resolve(repo_root, SOURCE_238)
    if path_238.exists():
        d238 = _load_json(path_238)
        model_runs = d238.get("model_runs", {})
        entries_238 = build_code_entries(
            model_runs=model_runs,
            source_artifact=str(SOURCE_238),
            source_experiment=238,
        )
        all_entries.extend(entries_238)
        source_artifacts_used.append(str(SOURCE_238))
    else:
        print(f"[warn] Exp 238 artifact not found at {path_238}, skipping code entries")

    # --- Exp 243: reference for provenance (no new entries, but noted in summary) ---
    path_243 = _resolve(repo_root, SOURCE_243)
    if path_243.exists():
        source_artifacts_used.append(str(SOURCE_243))

    # Sort for determinism: source_experiment, benchmark, model, case_id, iteration
    all_entries.sort(
        key=lambda r: (
            r["source_experiment"],
            r["benchmark"],
            r["model"],
            r["case_id"],
            r["iteration"],
        )
    )

    # Build summary
    from collections import Counter

    label_counts: dict[str, int] = Counter(r["process_label"] for r in all_entries)
    by_benchmark: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}

    for row in all_entries:
        bm = row["benchmark"]
        md = row["model"]
        lbl = row["process_label"]
        by_benchmark.setdefault(bm, Counter())[lbl] += 1  # type: ignore[arg-type]
        by_model.setdefault(md, Counter())[lbl] += 1  # type: ignore[arg-type]

    # Convert Counter objects to plain dicts for JSON serialisation
    by_benchmark_clean = {bm: dict(counts) for bm, counts in by_benchmark.items()}
    by_model_clean = {md: dict(counts) for md, counts in by_model.items()}

    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "Process Integrity Corpus — outcome-vs-process labels from checked-in traces",
        "run_date": RUN_DATE,
        "schema": SCHEMA_VERSION,
        "total_rows": len(all_entries),
        "label_counts": dict(label_counts),
        "by_source_benchmark": by_benchmark_clean,
        "by_model": by_model_clean,
        "corpus_path": str(CORPUS_RELATIVE),
        "source_artifacts": source_artifacts_used,
        "process_label_definitions": {
            "right_answer_wrong_process": (
                "Model produced the correct final answer but at least one reasoning step "
                "or code verification layer is unsound."
            ),
            "wrong_answer_partially_sound_process": (
                "Model produced the wrong final answer but the majority of intermediate "
                "steps or code properties are valid."
            ),
            "unsupported_step": (
                "A reasoning claim or code behaviour has no grounding in the provided "
                "premises or spec clauses."
            ),
            "repair_fixed_outcome_only": (
                "A repair attempt produced a correct answer but left the underlying "
                "process unsound (verifier still flags issues)."
            ),
            "repair_fixed_process_and_outcome": (
                "A repair attempt produced a correct answer AND the process is now sound "
                "(all verifier checks pass)."
            ),
            "clean": (
                "Outcome is correct and the process is sound (no verifier violations)."
            ),
        },
    }

    return all_entries, summary


# ---------------------------------------------------------------------------
# Public entry point (called by tests and by __main__)
# ---------------------------------------------------------------------------


def build_and_write(
    repo_root: Path,
    corpus_path: Path,
    summary_path: Path,
) -> None:
    """Build the corpus and write both output artifacts.

    This function is the single entry point used by tests for the
    deterministic-generation check.
    """
    rows, summary = build_corpus(repo_root=repo_root)
    _write_jsonl(corpus_path, rows)
    try:
        summary["corpus_path"] = str(corpus_path.relative_to(repo_root))
    except ValueError:
        summary["corpus_path"] = str(corpus_path)
    _write_json(summary_path, summary)
    print(f"Wrote {len(rows)} rows to {corpus_path}")
    print(f"Wrote summary to {summary_path}")
    _print_label_table(summary)


def _print_label_table(summary: dict[str, Any]) -> None:
    print("\n=== Process Integrity Label Counts ===")
    lc = summary.get("label_counts", {})
    total = summary.get("total_rows", 0)
    for label in [
        "right_answer_wrong_process",
        "wrong_answer_partially_sound_process",
        "unsupported_step",
        "repair_fixed_outcome_only",
        "repair_fixed_process_and_outcome",
        "clean",
    ]:
        count = lc.get(label, 0)
        pct = 100.0 * count / total if total else 0.0
        print(f"  {label:<45} {count:>5}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<45} {total:>5}")

    print("\n=== Label Counts by Source Benchmark ===")
    for bm, counts in sorted(summary.get("by_source_benchmark", {}).items()):
        print(f"  {bm}:")
        for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {cnt}")

    print("\n=== Label Counts by Model ===")
    for model, counts in sorted(summary.get("by_model", {}).items()):
        print(f"  {model}:")
        for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {cnt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Exp 248 process integrity corpus artifact.",
    )
    parser.add_argument(
        "--corpus-output",
        default=None,
        help="Override output path for the JSONL corpus (default: repo-relative standard path).",
    )
    parser.add_argument(
        "--results-output",
        default=None,
        help="Override output path for the JSON summary (default: repo-relative standard path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = get_repo_root()
    corpus_path = Path(args.corpus_output) if args.corpus_output else repo_root / CORPUS_RELATIVE
    summary_path = (
        Path(args.results_output) if args.results_output else repo_root / RESULTS_RELATIVE
    )
    build_and_write(repo_root=repo_root, corpus_path=corpus_path, summary_path=summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
