"""Exp 4022: decentralization gate driven by Exp 4012 local best-of-N.

Spec refs: REQ-PHASE4-031, SCENARIO-PHASE4-031.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4022_decentralization_gated.json"
CORPUS_NAME = "experiment_4022_distill_corpus.jsonl"
EXP4012_RESULT = REPO / "results" / "experiment_4012_gap4_local_best_of_n.json"
PROGRAM_ARTIFACTS = [
    REPO / "results" / "arc3_gap4_induced_programs.json",
    REPO / "results" / "arc3_gap4_arc2_induced_programs.json",
]
POOL_ARTIFACTS = [
    REPO / "results" / "arc3_gap3_stage2_eval_pool.json.gz",
    REPO / "results" / "arc3_gap4_arc2_eval_pool.json.gz",
]

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_decentralization_gated import (  # noqa: E402
    BRANCH_A,
    artifact_schema_errors,
    build_blocked_artifact,
    build_decentralization_artifact,
    choose_branch,
    harvest_distillation_traces,
    load_json,
    tiny_sanity_finetune,
    write_artifact,
    write_jsonl,
)


def _blocked_or_unavailable() -> bool:
    if not EXP4012_RESULT.exists():
        return True
    try:
        payload = load_json(EXP4012_RESULT)
    except Exception:
        return True
    return str(payload.get("honest_verdict", "")).startswith("blocked_")


def run(*, write: bool = True) -> dict[str, object]:
    started = time.time()
    if _blocked_or_unavailable():
        artifact = build_blocked_artifact(
            "blocked_exp4012_result_unavailable",
            duration_s=time.time() - started,
        )
        if write:
            write_artifact(artifact, REPO / "results" / RESULT_NAME)
        return artifact

    exp4012 = load_json(EXP4012_RESULT)
    branch_taken, _cited = choose_branch(exp4012)
    corpus_path = REPO / "results" / CORPUS_NAME
    traces: list[dict[str, object]] = []
    corpus_report: dict[str, object] = {
        "n_programs_scanned": 0,
        "n_execution_validated": 0,
        "n_clean_traces": 0,
        "rejection_reasons": {},
        "median_code_chars": 0,
        "hardcoded_grid_suspect_count": 0,
        "generic_trace_ratio": 0.0,
    }
    sanity = tiny_sanity_finetune([])

    if branch_taken == BRANCH_A:
        artifact = build_decentralization_artifact(
            exp4012,
            branch_taken=branch_taken,
            corpus_report=corpus_report,
            sanity_finetune=sanity,
            corpus_path=corpus_path,
            duration_s=time.time() - started,
        )
    else:
        traces, corpus_report = harvest_distillation_traces(PROGRAM_ARTIFACTS, POOL_ARTIFACTS)
        sanity = tiny_sanity_finetune(traces, subset_size=8)
        if write:
            write_jsonl(traces, corpus_path)
        artifact = build_decentralization_artifact(
            exp4012,
            branch_taken=branch_taken,
            corpus_report=corpus_report,
            sanity_finetune=sanity,
            corpus_path=corpus_path,
            duration_s=time.time() - started,
        )

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, REPO / "results" / RESULT_NAME)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - CLI exercised through direct run.
    main()
