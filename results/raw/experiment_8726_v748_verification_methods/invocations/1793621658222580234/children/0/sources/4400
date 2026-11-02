"""Build the Exp 2961 milestone .278 capstone artifact.

Spec refs: REQ-REPORT-2961, SCENARIO-REPORT-2961.

This module is an aggregation-only closeout layer. It reads the active
milestone roadmap, the .277 capstone, the available .278 result artifacts, and
the v12 matrix, then writes one compact closeout JSON. It does not rerun model
inference, verifier scoring, solver execution, synthesis, or hardware smoke
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
MILESTONE = "2026.05.278"
SCHEMA = "carnot.milestone_capstone.v278_aggregation.v1"
ARTIFACT = "experiment_2961_capstone_v278"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2961_capstone_v278.json")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")

CAPSTONE_V277_REL_PATH = Path("results/experiment_2948_capstone_v277.json")
EXP2950_REL_PATH = Path("results/experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json")
EXP2951_REL_PATH = Path("results/experiment_2951_structured_candidate_manifest_adapter_v1.json")
EXP2952_REL_PATH = Path("results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json")
EXP2953_REL_PATH = Path("results/experiment_2953_code_verifier_threshold_policy_v1.json")
EXP2954_REL_PATH = Path("results/experiment_2954_fr11_utility_gated_replay_curriculum_v2.json")
EXP2955_REL_PATH = Path("results/experiment_2955_gatemate_constraints_materialization_v4.json")
EXP2956_REL_PATH = Path("results/experiment_2956_gatemate_n16_bitstream_build_v4.json")
EXP2957_REL_PATH = Path("results/experiment_2957_gatemate_flash_timing_smoke_v2.json")
EXP2958_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_scorer_v2.json")
EXP2958_TRANSCRIPT_REL_PATH = Path(
    "results/experiment_2958_polarfire_1000_clause_transcript_v2.json"
)
EXP2959_REL_PATH = Path("results/experiment_2959_nl_to_z3_execution_repair_mini_v2.json")
EXP2960_REL_PATH = Path("results/experiment_2960_cross_corpus_matrix_v12.json")

CLASSIFICATIONS = (
    "clean",
    "flagged",
    "blocked",
    "gated-skipped",
    "missing",
    "pilot-only",
    "aggregation-only",
)

FORBIDDEN_CLAIMS_REAFFIRMED = [
    "KV260 speedup claims remain forbidden.",
    "KV260 Boltzmann, thermalization, or equilibrium-sampling claims remain forbidden.",
    "TSU or Kona performance claims remain forbidden.",
    "Broad hardware acceleration claims remain forbidden.",
    "Broad verifier-generalization claims beyond measured rows remain forbidden.",
]

TASK_PATH_OVERRIDES = {
    "exp2950": EXP2950_REL_PATH,
    "exp2951": EXP2951_REL_PATH,
    "exp2952": EXP2952_REL_PATH,
    "exp2953": EXP2953_REL_PATH,
    "exp2954": EXP2954_REL_PATH,
    "exp2955": EXP2955_REL_PATH,
    "exp2956": EXP2956_REL_PATH,
    "exp2957": EXP2957_REL_PATH,
    "exp2958": EXP2958_REL_PATH,
    "exp2959": EXP2959_REL_PATH,
    "exp2960": EXP2960_REL_PATH,
    "exp2961": OUTPUT_REL_PATH,
}


@dataclass(frozen=True)
class TaskSpec:
    """One planned roadmap task and the local artifact path it should emit."""

    task_id: str
    title: str
    deliverable: Path
    inference_substrate: str
    gated_on: tuple[Mapping[str, Any], ...]


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, returning an empty mapping for unusable input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA256 digest for a file, or None when it is absent."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2961: synthesize the terminal .278 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    tasks = _load_tasks(root_path)
    task_payloads = _load_task_payloads(root_path, tasks)
    capstone_v277 = read_json_object(root_path / CAPSTONE_V277_REL_PATH)
    matrix_v12 = task_payloads.get("exp2960") or read_json_object(root_path / EXP2960_REL_PATH)
    transcript = read_json_object(root_path / EXP2958_TRANSCRIPT_REL_PATH)

    classification_details = _classification_details(tasks, task_payloads, root_path)
    buckets = _bucket_task_ids(classification_details)
    outcome_summaries = _outcome_summaries(task_payloads, matrix_v12, transcript)
    forbidden_claims_absent = _forbidden_claims_absent(matrix_v12, outcome_summaries)
    gaps_closed = _gaps_closed(buckets, outcome_summaries)
    gaps_remaining = _gaps_remaining(
        buckets, outcome_summaries, matrix_v12, forbidden_claims_absent
    )
    paper_ready = _paper_ready(capstone_v277, matrix_v12, buckets, forbidden_claims_absent)
    source_paths = _source_paths(root_path, tasks)
    source_rows = _source_rows(root_path, source_paths)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(paper_ready, buckets),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "headline_outcome": _headline_outcome(paper_ready, gaps_closed, gaps_remaining),
        "clean_artifacts": buckets["clean"],
        "flagged_artifacts": buckets["flagged"],
        "blocked_artifacts": buckets["blocked"],
        "gated_skipped_artifacts": buckets["gated-skipped"],
        "missing_artifacts": buckets["missing"],
        "pilot_only_artifacts": buckets["pilot-only"],
        "aggregation_only_artifacts": buckets["aggregation-only"],
        "artifact_classification_counts": {name: len(buckets[name]) for name in CLASSIFICATIONS},
        "classification_details": classification_details,
        "gaps_closed": gaps_closed,
        "gaps_remaining": gaps_remaining,
        "forbidden_claims_absent": forbidden_claims_absent,
        "forbidden_claims_reaffirmed": FORBIDDEN_CLAIMS_REAFFIRMED,
        "paper_v6_safe_claims": _paper_v6_safe_claims(outcome_summaries, buckets),
        "next_milestone_recommendations": _next_milestone_recommendations(),
        "outcome_summaries": outcome_summaries,
        "source_artifacts_read": source_rows,
        "source_checksums": {row["path"]: row["sha256"] for row in source_rows},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2961 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_tasks(root: Path) -> list[TaskSpec]:
    roadmap = yaml.safe_load((root / ROADMAP_REL_PATH).read_text(encoding="utf-8"))
    raw_tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    tasks: list[TaskSpec] = []
    for item in raw_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "")
        if not task_id.startswith("exp"):
            continue
        deliverable = Path(str(item.get("deliverable") or TASK_PATH_OVERRIDES.get(task_id, "")))
        if task_id in TASK_PATH_OVERRIDES:
            deliverable = TASK_PATH_OVERRIDES[task_id]
        gated_on = item.get("gated_on")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                title=str(item.get("title") or ""),
                deliverable=deliverable,
                inference_substrate=str(item.get("inference_substrate") or ""),
                gated_on=tuple(gate for gate in gated_on if isinstance(gate, Mapping))
                if isinstance(gated_on, list)
                else (),
            )
        )
    return [task for task in tasks if task.task_id.startswith(("exp2949", "exp295", "exp296"))]


def _load_task_payloads(root: Path, tasks: list[TaskSpec]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for task in tasks:
        payloads[task.task_id] = read_json_object(root / task.deliverable)
    return payloads


def _classification_details(
    tasks: list[TaskSpec],
    payloads: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        payload = payloads.get(task.task_id, {})
        present = (root / task.deliverable).is_file()
        gate_failures = _gate_failures(task, payloads)
        classification = _classify_task(task, payload, present, gate_failures)
        rows.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "path": task.deliverable.as_posix(),
                "classification": classification,
                "present": present,
                "source_honest_verdict": str(payload.get("honest_verdict") or ""),
                "flag_kinds": _flag_kinds(payload),
                "gate_blocked_by": gate_failures,
            }
        )
    return rows


def _classify_task(
    task: TaskSpec,
    payload: Mapping[str, Any],
    present: bool,
    gate_failures: list[str],
) -> str:
    if task.task_id == "exp2961":
        return "aggregation-only"
    if not present or not payload:
        return "gated-skipped" if gate_failures else "missing"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    if task.task_id == "exp2960":
        return "aggregation-only"
    if payload.get("pilot_only") is True:
        return "pilot-only"
    return "clean"


def _gate_failures(task: TaskSpec, payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for gate in task.gated_on:
        upstream = str(gate.get("upstream") or "")
        field = str(gate.get("artifact_field") or "")
        expected = gate.get("value")
        actual = _get_path(payloads.get(upstream, {}), field)
        if gate.get("op", "==") == "==" and actual != expected:
            failures.append(f"{upstream}.{field}")
    return failures


def _bucket_task_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {name: [] for name in CLASSIFICATIONS}
    for row in rows:
        classification = str(row["classification"])
        if classification in buckets:
            buckets[classification].append(str(row["task_id"]))
    return buckets


def _source_paths(root: Path, tasks: list[TaskSpec]) -> list[Path]:
    paths = {CAPSTONE_V277_REL_PATH, EXP2958_TRANSCRIPT_REL_PATH}
    paths.update(task.deliverable for task in tasks)
    results_dir = root / "results"
    if results_dir.is_dir():
        paths.update(path.relative_to(root) for path in results_dir.glob("experiment_295*.json"))
        paths.update(path.relative_to(root) for path in results_dir.glob("experiment_2960*.json"))
    return sorted(paths, key=lambda path: path.as_posix())


def _source_rows(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "present": (root / path).is_file(),
            "sha256": sha256_file(root / path),
        }
        for path in paths
    ]


def _outcome_summaries(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v12: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "code_repair": _code_repair_summary(payloads, matrix_v12),
        "self_learning": _self_learning_summary(payloads, matrix_v12),
        "solver": _solver_summary(payloads, matrix_v12),
        "hardware": _hardware_summary(payloads, matrix_v12, transcript),
    }


def _code_repair_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v12: Mapping[str, Any],
) -> dict[str, Any]:
    repair = payloads.get("exp2952", {})
    threshold = payloads.get("exp2953", {})
    repair_flagged = _has_flags(repair) or "exp2952_structured_repair_delta" in matrix_v12.get(
        "flagged_rows", []
    )
    return {
        "manifest_ready": bool(payloads.get("exp2950", {}).get("repair_prompt_manifest_ready")),
        "structured_decode_manifest_ready": bool(
            payloads.get("exp2951", {}).get("structured_decode_manifest_ready")
        ),
        "n_tasks": _coerce_int(repair.get("n_tasks")),
        "baseline_pass_at_1": _coerce_float(repair.get("baseline_pass_at_1")),
        "repair_pass_at_1": _coerce_float(repair.get("repair_pass_at_1")),
        "pass_at_1_delta": _coerce_float(repair.get("pass_at_1_delta")),
        "pass_at_k_delta": _coerce_float(repair.get("pass_at_k_delta")),
        "syntax_failure_rate_delta": _coerce_float(repair.get("syntax_failure_rate_delta")),
        "false_accept_delta": _coerce_float(repair.get("false_accept_delta")),
        "taxonomy_repair_delta_pass": bool(repair.get("taxonomy_repair_delta_pass")),
        "repair_artifact_flagged": repair_flagged,
        "threshold_policy_ready": bool(threshold.get("threshold_policy_ready")),
        "selected_default_threshold": _coerce_float(threshold.get("selected_default_threshold")),
        "expected_ppv_at_default": _coerce_float(threshold.get("expected_ppv_at_default")),
        "safe_claim": (
            "pilot_delta_flagged_not_paper_ready"
            if repair.get("taxonomy_repair_delta_pass") and repair_flagged
            else "no_repair_delta_claim"
        ),
    }


def _self_learning_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v12: Mapping[str, Any],
) -> dict[str, Any]:
    payload = payloads.get("exp2954", {})
    flagged = _has_flags(payload) or "exp2954_self_learning_utility" in matrix_v12.get(
        "flagged_rows", []
    )
    return {
        "artifact_ready": bool(payload.get("self_learning_utility_artifact_ready")),
        "self_learning_utility_positive": bool(payload.get("self_learning_utility_positive")),
        "heldout_utility_baseline": _coerce_float(payload.get("heldout_utility_baseline")),
        "heldout_utility_after": _coerce_float(payload.get("heldout_utility_after")),
        "heldout_utility_delta": _coerce_float(payload.get("heldout_utility_delta")),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "model_weights_mutated": bool(payload.get("model_weights_mutated")),
        "rollback_triggered": bool(payload.get("rollback_triggered")),
        "artifact_flagged": flagged,
        "safe_claim": (
            "positive_utility_flagged_not_paper_ready"
            if payload.get("self_learning_utility_positive") and flagged
            else "no_self_learning_claim_upgrade"
        ),
    }


def _solver_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v12: Mapping[str, Any],
) -> dict[str, Any]:
    payload = payloads.get("exp2959", {})
    flagged = _has_flags(payload) or "exp2959_nl_to_z3_execution_repair" in matrix_v12.get(
        "flagged_rows", []
    )
    return {
        "z3_import_ok": bool(payload.get("z3_import_ok")),
        "z3_execution_repaired": bool(payload.get("z3_execution_repaired")),
        "z3_execution_rate": _coerce_float(payload.get("z3_execution_rate")),
        "solver_verified_accuracy": _coerce_float(payload.get("solver_verified_accuracy")),
        "answer_accuracy": _coerce_float(payload.get("answer_accuracy")),
        "parseability_rate": _coerce_float(payload.get("parseability_rate")),
        "n_items": _coerce_int(payload.get("n_items")),
        "artifact_flagged": flagged,
        "safe_claim": (
            "z3_execution_repaired_but_flagged_not_paper_ready"
            if payload.get("z3_execution_repaired") and flagged
            else "no_solver_claim_upgrade"
        ),
    }


def _hardware_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v12: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
    exp2955 = payloads.get("exp2955", {})
    exp2956 = payloads.get("exp2956", {})
    exp2957 = payloads.get("exp2957", {})
    exp2958 = payloads.get("exp2958", {})
    return {
        "gatemate": {
            "constraints_ready": bool(exp2955.get("gatemate_constraints_ready")),
            "dirtyjtag_detected": bool(exp2955.get("dirtyjtag_detected")),
            "bitstream_built": bool(exp2956.get("gatemate_bitstream_built")),
            "timing_met": _get_path(exp2956, "timing_summary.timing_met"),
            "flash_state": str(exp2957.get("honest_verdict") or "missing"),
            "board_detected": bool(exp2957.get("board_detected")),
            "flash_attempted": bool(exp2957.get("flash_attempted")),
            "flash_succeeded": bool(exp2957.get("flash_succeeded")),
            "smoke_vector_passed": bool(exp2957.get("smoke_vector_passed")),
        },
        "polarfire": {
            "board_reachable": bool(exp2958.get("board_reachable")),
            "clause_count": _coerce_int(exp2958.get("clause_count")),
            "hash_verified": bool(exp2958.get("polarfire_1000_clause_hash_verified")),
            "elapsed_ms": _coerce_float(exp2958.get("elapsed_ms")),
            "remote_arch": exp2958.get("remote_arch"),
            "transcript_total_wall_clock_s": _coerce_float(transcript.get("total_wall_clock_s")),
            "evaluation_cycles_per_clause": _coerce_int(
                transcript.get("evaluation_cycles_per_clause")
            ),
        },
        "matrix_hardware_blocked_rows": [
            row for row in matrix_v12.get("blocked_rows", []) if isinstance(row, str)
        ],
        "safe_claim": "materialization_and_hash_only_no_performance_claim",
    }


def _forbidden_claims_absent(
    matrix_v12: Mapping[str, Any],
    outcome_summaries: Mapping[str, Any],
) -> bool:
    safe_claim_text = json.dumps(_paper_v6_safe_claims(outcome_summaries, {}), sort_keys=True)
    forbidden_tokens = ("speedup", "thermalization", "boltzmann", "tsu", "kona")
    return bool(matrix_v12.get("forbidden_claims_absent")) and not any(
        token in safe_claim_text.lower() for token in forbidden_tokens
    )


def _gaps_closed(
    buckets: Mapping[str, list[str]],
    outcome_summaries: Mapping[str, Any],
) -> list[str]:
    closed: list[str] = []
    if "exp2953" in buckets["clean"]:
        closed.append("Code verifier threshold policy is clean and deployment-bounded.")
    hardware = outcome_summaries["hardware"]
    if "exp2955" in buckets["clean"] and "exp2956" in buckets["clean"]:
        closed.append("GateMate n=16 constraints and bitstream materialization landed.")
    if hardware["polarfire"]["hash_verified"] and "exp2958" in buckets["clean"]:
        closed.append("PolarFire 1000-clause scorer hash verification landed.")
    return closed


def _gaps_remaining(
    buckets: Mapping[str, list[str]],
    outcome_summaries: Mapping[str, Any],
    matrix_v12: Mapping[str, Any],
    forbidden_claims_absent: bool,
) -> list[str]:
    gaps: list[str] = []
    if "exp2952" in buckets["flagged"]:
        gaps.append(
            "Taxonomy-guided repair delta remains flagged; rerun or audit before paper use."
        )
    if "exp2954" in buckets["flagged"]:
        gaps.append("FR-11 utility-gated self-learning remains flagged despite positive utility.")
    if "exp2959" in buckets["flagged"]:
        gaps.append("NL-to-Z3 solver execution repair remains flagged with low verified accuracy.")
    if "exp2957" in buckets["blocked"]:
        gaps.append("GateMate flash/timing smoke remains blocked by board detection.")
    if buckets["missing"]:
        gaps.append(f"Missing planned .278 artifacts: {', '.join(buckets['missing'])}.")
    if not matrix_v12.get("matrix_v12_ready"):
        gaps.append("Cross-corpus matrix v12 is not ready.")
    if not forbidden_claims_absent:
        gaps.append("Forbidden claim scan failed in matrix v12.")
    if outcome_summaries["hardware"]["gatemate"]["flash_succeeded"] is False:
        gaps.append("No GateMate board flash/output-hash claim is available.")
    return _unique_strings(gaps)


def _paper_ready(
    capstone_v277: Mapping[str, Any],
    matrix_v12: Mapping[str, Any],
    buckets: Mapping[str, list[str]],
    forbidden_claims_absent: bool,
) -> bool:
    unresolved = (
        buckets["flagged"]
        + buckets["blocked"]
        + buckets["gated-skipped"]
        + buckets["missing"]
        + buckets["pilot-only"]
    )
    return (
        capstone_v277.get("paper_ready") is True
        and matrix_v12.get("matrix_v12_ready") is True
        and forbidden_claims_absent
        and not unresolved
    )


def _paper_v6_safe_claims(
    outcome_summaries: Mapping[str, Any],
    buckets: Mapping[str, list[str]],
) -> list[str]:
    claims = [
        "The verifier threshold policy is deployment-bounded when Exp 2953 is clean.",
        "GateMate evidence is limited to constraints and n=16 bitstream materialization.",
        "PolarFire evidence is limited to 1000-clause scorer hash verification.",
    ]
    if "exp2952" not in buckets.get("clean", []):
        claims.append("Taxonomy-guided repair remains a flagged pilot delta, not a paper upgrade.")
    return claims


def _headline_outcome(
    paper_ready: bool,
    gaps_closed: list[str],
    gaps_remaining: list[str],
) -> str:
    return (
        "paper_ready: .278 preserves the narrowed .277 claim set"
        if paper_ready
        else (
            "partial: materialization and threshold-policy evidence improved, "
            f"but {len(gaps_remaining)} unresolved gaps keep paper readiness unchanged"
            if gaps_closed
            else "blocked: no .278 gap closed cleanly"
        )
    )


def _honest_verdict(paper_ready: bool, buckets: Mapping[str, list[str]]) -> str:
    return (
        "complete: milestone_278_capstone; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"clean={len(buckets['clean'])}; "
        f"flagged={len(buckets['flagged'])}; "
        f"blocked={len(buckets['blocked'])}; "
        f"missing={len(buckets['missing'])}"
    )


def _next_milestone_recommendations() -> list[str]:
    return [
        "GateMate .279: resolve board detection and capture flash/timing/output-hash smoke.",
        "Code repair .279: repeat taxonomy-guided repair with adversarial flags eliminated.",
        "FR-11 .279: rerun utility-gated replay with non-tautological forgetting evidence.",
        "Solver .279: raise parseability and Z3 execution rate before any solver claim upgrade.",
    ]


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: Mapping[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    flags = payload.get("corrigendum_pending")
    return isinstance(flags, list) and bool(flags)


def _flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    kinds: list[str] = []
    if payload.get("flagged_adversarial") is True:
        kinds.append("flagged_adversarial=true")
    flags = payload.get("corrigendum_pending")
    if isinstance(flags, list):
        for item in flags:
            if isinstance(item, Mapping):
                kind = str(item.get("kind") or "unknown")
                severity = str(item.get("severity") or "unknown")
                kinds.append(f"{kind}:{severity}")
    return _unique_strings(kinds)


def _get_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
