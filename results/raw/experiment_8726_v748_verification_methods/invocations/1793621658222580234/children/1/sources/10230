"""Deterministic meta-harness search for Carnot conductor policies.

Spec: REQ-HARNESS-010, REQ-HARNESS-011, REQ-HARNESS-012,
SCENARIO-HARNESS-005, SCENARIO-HARNESS-006, SCENARIO-HARNESS-007.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_FILES = (
    "traces/task_prompt.md",
    "traces/stdout.log",
    "traces/stderr.log",
    "traces/tool_calls.jsonl",
    "traces/gate_evaluation.json",
    "traces/artifact_timeline.jsonl",
    "traces/verifier_outputs.jsonl",
    "traces/diff.patch",
)


@dataclass(frozen=True)
class EvalCase:
    """A deterministic conductor-harness failure case."""

    case_id: str
    title: str
    required_capability: str
    desired_behavior: str
    penalty_type: str
    held_out: bool = False


@dataclass(frozen=True)
class CandidatePolicy:
    """A conductor-policy candidate evaluated by the meta-harness loop."""

    candidate_id: str
    title: str
    capabilities: tuple[str, ...]
    policy_text: str
    recommended_changes: tuple[str, ...]
    average_runtime_s: float


def default_eval_cases() -> list[EvalCase]:
    """Return the cheap deterministic eval suite.

    Spec: REQ-HARNESS-010, SCENARIO-HARNESS-005.
    """

    return [
        EvalCase(
            "bootstrap_only_artifact",
            "Bootstrap-only artifact is not complete",
            "bootstrap_detection",
            "Classify skeleton artifacts as non-terminal.",
            "false_complete",
        ),
        EvalCase(
            "stale_skeleton",
            "Stale skeleton is rejected",
            "stale_detection",
            "Classify stale artifacts as stale rather than complete.",
            "false_complete",
        ),
        EvalCase(
            "missing_sota_model",
            "Missing GPU or SOTA model writes blocked artifact",
            "blocked_prerequisite",
            "Record exact missing model or hardware prerequisite.",
            "false_block",
        ),
        EvalCase(
            "gated_downstream_task",
            "Gated task evaluates upstream path field operator value",
            "structured_gate",
            "Write terminal blocked artifact when gate is closed.",
            "gate_error",
        ),
        EvalCase(
            "local_verifier_mismatch",
            "Local verifier success cannot override final artifact gate",
            "acceptance_alignment",
            "Use final artifact or benchmark acceptance object.",
            "false_complete",
            held_out=True,
        ),
        EvalCase(
            "paper_unsupported_claim",
            "Unsupported paper claim is flagged",
            "paper_claim_audit",
            "Cite missing evidence instead of adding unsupported claims.",
            "false_complete",
        ),
        EvalCase(
            "timeout_with_progress",
            "Timeout with progress checkpoints honestly",
            "timeout_progress",
            "Classify partial evidence as timeout_with_progress.",
            "missing_trace",
        ),
        EvalCase(
            "timeout_without_progress",
            "Timeout without progress is separate",
            "timeout_no_progress",
            "Avoid pretending success when no evidence was produced.",
            "false_complete",
        ),
        EvalCase(
            "no_file_changes_produced",
            "No file changes are classified",
            "no_file_change_policy",
            "Retry, block, or retire with evidence.",
            "false_complete",
        ),
        EvalCase(
            "malformed_json_artifact",
            "Malformed JSON fails schema validation",
            "json_schema_validation",
            "Reject malformed JSON and request repair.",
            "false_complete",
        ),
        EvalCase(
            "missing_deliverable",
            "Missing deliverable is detected",
            "missing_deliverable_detection",
            "Do not count absent result paths as complete.",
            "false_complete",
            held_out=True,
        ),
        EvalCase(
            "blocked_missing_tool",
            "Missing local tool writes blocked artifact",
            "blocked_missing_tool",
            "Record exact missing command or package.",
            "false_block",
        ),
    ]


def default_candidate_policies() -> list[CandidatePolicy]:
    """Return baseline plus four deterministic candidate policies."""

    return [
        CandidatePolicy(
            "candidate_000",
            "Baseline current conductor policy",
            (
                "structured_gate",
                "json_schema_validation",
                "missing_deliverable_detection",
            ),
            "Baseline checks gates, JSON parseability, and missing deliverables.",
            ("Keep baseline as comparison only.",),
            1.0,
        ),
        CandidatePolicy(
            "candidate_001",
            "Terminal artifact guard policy",
            (
                "bootstrap_detection",
                "stale_detection",
                "json_schema_validation",
                "missing_deliverable_detection",
                "no_file_change_policy",
            ),
            "Adds strict terminal artifact checks for skeleton and stale outputs.",
            ("Promote terminal artifact audit before completion accounting.",),
            1.2,
        ),
        CandidatePolicy(
            "candidate_002",
            "Gate and prerequisite blocking policy",
            (
                "bootstrap_detection",
                "stale_detection",
                "structured_gate",
                "blocked_prerequisite",
                "blocked_missing_tool",
                "json_schema_validation",
                "missing_deliverable_detection",
            ),
            "Adds explicit blocked artifacts for gates, tools, models, and GPUs.",
            ("Write terminal blocked artifacts for every closed gate.",),
            1.4,
        ),
        CandidatePolicy(
            "candidate_003",
            "Trace-rich restart policy",
            (
                "bootstrap_detection",
                "stale_detection",
                "structured_gate",
                "blocked_prerequisite",
                "blocked_missing_tool",
                "json_schema_validation",
                "missing_deliverable_detection",
                "timeout_progress",
                "timeout_no_progress",
                "trace_store",
            ),
            "Adds full trace preservation and distinct timeout classifications.",
            ("Create run packets with stdout, stderr, gates, and artifacts.",),
            1.8,
        ),
        CandidatePolicy(
            "candidate_004",
            "Acceptance-aligned full charter policy",
            (
                "bootstrap_detection",
                "stale_detection",
                "structured_gate",
                "blocked_prerequisite",
                "blocked_missing_tool",
                "json_schema_validation",
                "missing_deliverable_detection",
                "timeout_progress",
                "timeout_no_progress",
                "trace_store",
                "acceptance_alignment",
                "paper_claim_audit",
                "no_file_change_policy",
            ),
            "Aligns local verifiers, gates, traces, and paper claims to the final artifact acceptance object.",
            (
                "Adopt file-backed run packets for conductor tasks.",
                "Require acceptance-object declarations for local verifiers.",
                "Run paper-claim audits before public claim upgrades.",
            ),
            2.1,
        ),
    ]


def candidate_has_hardcoded_leakage(policy: CandidatePolicy) -> bool:
    """Detect candidate leakage through hard-coded experiment identifiers.

    Spec: REQ-HARNESS-012, SCENARIO-HARNESS-007.
    """

    haystack = " ".join(
        (
            policy.candidate_id,
            policy.title,
            policy.policy_text,
            " ".join(policy.capabilities),
            " ".join(policy.recommended_changes),
        )
    )
    return bool(re.search(r"\bexp\d{3,5}\b", haystack, flags=re.IGNORECASE))


def evaluate_policy(policy: CandidatePolicy, eval_cases: list[EvalCase]) -> dict[str, Any]:
    """Evaluate one policy over deterministic conductor-harness cases.

    Spec: REQ-HARNESS-010, SCENARIO-HARNESS-006.
    """

    capabilities = set(policy.capabilities)
    case_results: list[dict[str, Any]] = []
    penalties = {
        "false_complete": 0,
        "false_block": 0,
        "gate_error": 0,
        "missing_trace": 0,
    }

    for case in eval_cases:
        passed = case.required_capability in capabilities
        if not passed:
            penalties[case.penalty_type] += 1
        case_results.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "required_capability": case.required_capability,
                "passed": passed,
                "held_out": case.held_out,
                "penalty_type": None if passed else case.penalty_type,
                "desired_behavior": case.desired_behavior,
            }
        )

    leakage_detected = candidate_has_hardcoded_leakage(policy)
    passed_count = sum(1 for row in case_results if row["passed"])
    score = (
        passed_count
        - 3 * penalties["false_complete"]
        - 2 * penalties["gate_error"]
        - penalties["false_block"]
        - penalties["missing_trace"]
        - (5 if leakage_detected else 0)
    )
    trace_completeness_score = 1.0 if "trace_store" in capabilities else 0.5

    return {
        "candidate_id": policy.candidate_id,
        "title": policy.title,
        "eval_cases_passed": passed_count,
        "eval_cases_total": len(eval_cases),
        "false_complete_count": penalties["false_complete"],
        "false_block_count": penalties["false_block"],
        "gate_error_count": penalties["gate_error"],
        "missing_trace_count": penalties["missing_trace"],
        "false_complete_rate": round(penalties["false_complete"] / len(eval_cases), 6),
        "false_block_rate": round(penalties["false_block"] / len(eval_cases), 6),
        "missing_artifact_detection_rate": 1.0
        if "missing_deliverable_detection" in capabilities
        else 0.0,
        "gate_correctness_rate": 1.0 if "structured_gate" in capabilities else 0.0,
        "trace_completeness_score": trace_completeness_score,
        "average_runtime_s": policy.average_runtime_s,
        "hardcoded_leakage_detected": leakage_detected,
        "score": float(score),
        "case_results": case_results,
        "recommended_policy_changes": list(policy.recommended_changes),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_candidate_trace(
    trace_store: Path,
    policy: CandidatePolicy,
    score: dict[str, Any],
) -> None:
    """Write a full candidate trace packet.

    Spec: REQ-HARNESS-011, SCENARIO-HARNESS-005.
    """

    candidate_dir = trace_store / policy.candidate_id
    traces_dir = candidate_dir / "traces"
    results_dir = candidate_dir / "results"
    traces_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    (candidate_dir / "policy.md").write_text(
        "\n".join(
            [
                f"# {policy.title}",
                "",
                policy.policy_text,
                "",
                "## Capabilities",
                *[f"- `{capability}`" for capability in policy.capabilities],
                "",
                "## Recommended Changes",
                *[f"- {change}" for change in policy.recommended_changes],
                "",
            ]
        )
    )
    (candidate_dir / "policy.py").write_text("CAPABILITIES = " + repr(policy.capabilities) + "\n")
    _write_json(
        candidate_dir / "roadmap_patch.yaml",
        {
            "candidate_id": policy.candidate_id,
            "policy_title": policy.title,
            "roadmap_patch_required": False,
        },
    )
    _write_json(candidate_dir / "score.json", score)

    (traces_dir / "task_prompt.md").write_text(
        "# Meta-Harness Candidate Evaluation\n\n"
        "Evaluate the candidate policy on deterministic conductor failure cases.\n"
    )
    (traces_dir / "stdout.log").write_text(
        f"{score['eval_cases_passed']}/{score['eval_cases_total']} cases passed\n"
    )
    (traces_dir / "stderr.log").write_text("")
    _write_jsonl(
        traces_dir / "tool_calls.jsonl",
        [
            {
                "tool": "evaluate_policy",
                "candidate_id": policy.candidate_id,
                "eval_cases_total": score["eval_cases_total"],
            }
        ],
    )
    _write_json(
        traces_dir / "gate_evaluation.json",
        {
            "candidate_id": policy.candidate_id,
            "structured_gate_supported": score["gate_correctness_rate"] == 1.0,
        },
    )
    _write_jsonl(
        traces_dir / "artifact_timeline.jsonl",
        [
            {
                "event": "candidate_evaluated",
                "candidate_id": policy.candidate_id,
                "score": score["score"],
            },
            {
                "event": "score_written",
                "path": str(candidate_dir / "score.json"),
            },
        ],
    )
    _write_jsonl(traces_dir / "verifier_outputs.jsonl", score["case_results"])
    (traces_dir / "diff.patch").write_text(
        "# Policy-only candidate; no repository patch generated.\n"
    )
    _write_json(
        results_dir / "final_artifact.json",
        {
            "candidate_id": policy.candidate_id,
            "status": "complete",
            "score": score["score"],
            "honest_verdict": "candidate_policy_evaluated",
        },
    )


def _pareto_frontier(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for score in scores:
        dominated = False
        for other in scores:
            if other is score:
                continue
            no_worse = (
                other["score"] >= score["score"]
                and other["trace_completeness_score"] >= score["trace_completeness_score"]
                and other["average_runtime_s"] <= score["average_runtime_s"]
            )
            strictly_better = (
                other["score"] > score["score"]
                or other["trace_completeness_score"] > score["trace_completeness_score"]
                or other["average_runtime_s"] < score["average_runtime_s"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "candidate_id": score["candidate_id"],
                    "score": score["score"],
                    "trace_completeness_score": score["trace_completeness_score"],
                    "average_runtime_s": score["average_runtime_s"],
                }
            )
    return sorted(
        frontier,
        key=lambda row: (-row["score"], -row["trace_completeness_score"], row["average_runtime_s"]),
    )


def run_search(trace_store: Path, result_path: Path) -> dict[str, Any]:
    """Run deterministic conductor meta-harness search and write artifacts.

    Spec: REQ-HARNESS-010, REQ-HARNESS-011, REQ-HARNESS-012,
    SCENARIO-HARNESS-005, SCENARIO-HARNESS-006.
    """

    eval_cases = default_eval_cases()
    policies = default_candidate_policies()
    scores = [evaluate_policy(policy, eval_cases) for policy in policies]

    trace_store.mkdir(parents=True, exist_ok=True)
    for policy, score in zip(policies, scores, strict=True):
        write_candidate_trace(trace_store, policy, score)

    frontier = _pareto_frontier(scores)
    _write_json(trace_store / "frontier.json", frontier)
    (trace_store / "search_log.md").write_text(
        "\n".join(
            [
                "# Meta-Harness Search Log",
                "",
                f"Candidates evaluated: {len(policies)}",
                f"Eval cases defined: {len(eval_cases)}",
                f"Best candidate: {max(scores, key=lambda row: row['score'])['candidate_id']}",
                "",
            ]
        )
    )

    baseline_score = scores[0]["score"]
    best = max(scores, key=lambda row: row["score"])
    improvement = best["score"] - baseline_score
    hardcoded_leakage_audit_passed = not any(
        score["hardcoded_leakage_detected"] for score in scores
    )

    result = {
        "experiment": "1281_meta_harness_conductor_search",
        "schema": "carnot.meta_harness_conductor_search.v1",
        "run_date": "20260505",
        "status": "complete",
        "candidate_harnesses_evaluated": len(policies),
        "eval_cases_defined": len(eval_cases),
        "held_out_cases": sum(1 for case in eval_cases if case.held_out),
        "baseline_score": baseline_score,
        "best_score": best["score"],
        "improvement_over_baseline": improvement,
        "best_candidate_id": best["candidate_id"],
        "pareto_frontier_written": (trace_store / "frontier.json").exists(),
        "trace_store_written": all(
            (trace_store / policy.candidate_id).exists() for policy in policies
        ),
        "trace_store_path": str(trace_store),
        "recommended_policy_changes": best["recommended_policy_changes"],
        "hardcoded_leakage_audit_passed": hardcoded_leakage_audit_passed,
        "candidate_scores": scores,
        "pareto_frontier": frontier,
        "honest_verdict": "meta_harness_conductor_search_complete"
        if improvement > 0
        else "no_candidate_improved_over_baseline",
    }
    _write_json(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-store",
        type=Path,
        default=Path("meta_harness_runs"),
        help="Directory for candidate trace packets.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/experiment_1281_meta_harness_conductor_search.json"),
        help="Terminal result artifact path.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the terminal artifact to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_search(args.trace_store, args.result)
    if args.print_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
