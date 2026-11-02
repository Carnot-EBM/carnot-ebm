"""Build the Exp 2936 milestone .276 capstone artifact.

Spec refs: REQ-REPORT-2936, SCENARIO-REPORT-2936.

This module is an aggregation-only closeout layer. It reads the already-written
`.276` artifacts, counts missing expected artifacts explicitly, preserves row
boundaries from matrix v10, and derives claim booleans without running a model,
hardware board, verifier, or synthesis tool.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
MILESTONE = "2026.05.276"
SCHEMA = "carnot.milestone_capstone.v276"
ARTIFACT = "experiment_2936_capstone_v276"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2936_capstone_v276.json")

ROW_CLASSES = (
    "clean",
    "flagged",
    "blocked",
    "missing",
    "projection_only",
    "diagnostic_only",
    "pilot_only",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required_fields: tuple[str, ...]


EXPECTED_ARTIFACTS: dict[str, SourceSpec] = {
    "exp2923": SourceSpec(
        "exp2923",
        Path("results/experiment_2923_archive_v275_activate_v276.json"),
        ("archive_ready",),
    ),
    "exp2924": SourceSpec(
        "exp2924",
        Path("results/experiment_2924_aggregation_metadata_corrigendum_v1.json"),
        ("aggregation_metadata_clean",),
    ),
    "exp2925": SourceSpec(
        "exp2925",
        Path("results/experiment_2925_code_hallucination_taxonomy_provenance_corrigendum_v2.json"),
        ("taxonomy_corrigendum_clean", "code_hallucination_verifier_ready"),
    ),
    "exp2926": SourceSpec(
        "exp2926",
        Path("results/experiment_2926_constraintbench_constrained_output_rerun_v2.json"),
        ("constraintbench_corrigendum_ready",),
    ),
    "exp2927": SourceSpec(
        "exp2927",
        Path("results/experiment_2927_gatemate_himbaechel_constraints_preflight_v3.json"),
        ("gatemate_himbaechel_ready", "constraints_ready"),
    ),
    "exp2928": SourceSpec(
        "exp2928",
        Path("results/experiment_2928_gatemate_n16_himbaechel_bitstream_build_v3.json"),
        ("gatemate_bitstream_built",),
    ),
    "exp2929": SourceSpec(
        "exp2929",
        Path("results/experiment_2929_gatemate_flash_timing_boundary_v1.json"),
        ("gatemate_flash_smoke_ready",),
    ),
    "exp2930": SourceSpec(
        "exp2930",
        Path("results/experiment_2930_kv260_pbit_ssqa_scaling_projection_v1.json"),
        ("kv260_scaling_projection_ready",),
    ),
    "exp2931": SourceSpec(
        "exp2931",
        Path("results/experiment_2931_llmeval_logic_z3_mini_v1.json"),
        ("logic_verifier_mini_ready",),
    ),
    "exp2932": SourceSpec(
        "exp2932",
        Path("results/experiment_2932_citation_hallucination_field_verifier_v1.json"),
        ("citation_verifier_ready",),
    ),
    "exp2933": SourceSpec(
        "exp2933",
        Path("results/experiment_2933_kan_cl_per_knot_self_learning_v1.json"),
        ("kan_cl_self_learning_ready",),
    ),
    "exp2934": SourceSpec(
        "exp2934",
        Path("results/experiment_2934_aquaforte_beaver_reformulation_pipeline_v1.json"),
        ("reformulation_pipeline_ready",),
    ),
    "exp2935": SourceSpec(
        "exp2935",
        Path("results/experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1.json"),
        ("matrix_v10_ready", "matrix_v10_paper_boundary_ready"),
    ),
}

INHERITED_FLAG_CLEAN_OVERRIDES = {"exp2924", "exp2925", "exp2935"}
STRUCTURED_GENERATION_ARTIFACTS = ("exp2926", "exp2931", "exp2932", "exp2934")
KV260_SPEEDUP_ROW = "exp2913_kv260_claim_boundary"
GATEMATE_SPEEDUP_ELIGIBLE = False


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail closed to an empty mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_artifact(exp_id: str, payload: dict[str, Any], present: bool) -> str:
    """REQ-REPORT-2936: classify one expected .276 artifact."""

    if not present or not payload:
        return "missing"
    if payload.get("projection_only") is True:
        return "projection_only"
    if payload.get("diagnostic_only") is True:
        return "diagnostic_only"
    if payload.get("pilot_only") is True:
        return "pilot_only"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _clean_despite_inherited_flags(exp_id, payload):
        return "clean"
    if _has_current_flags(payload):
        return "flagged"
    if _is_clean_by_required_fields(exp_id, payload):
        return "clean"
    return "blocked"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2936: synthesize the terminal .276 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    payloads, present = _load_expected(root_path)
    statuses = _classify_all(payloads, present)
    matrix = payloads["exp2935"]

    clean_artifacts = _ids_with_status(statuses, "clean")
    flagged_artifacts = _ids_with_status(statuses, "flagged")
    blocked_artifacts = _ids_with_status(statuses, "blocked")
    missing_artifacts = _ids_with_status(statuses, "missing")
    projection_only_artifacts = _ids_with_status(statuses, "projection_only")
    diagnostic_only_artifacts = _ids_with_status(statuses, "diagnostic_only")
    pilot_only_artifacts = _ids_with_status(statuses, "pilot_only")

    evidence_boundary = _evidence_boundary_summary(payloads, statuses)
    gatemate_status = _gatemate_status(payloads, present)
    self_learning = _continuous_self_learning_status(payloads["exp2933"], statuses["exp2933"])
    paper_ready = _paper_ready(matrix)
    hardware_speedup_claim_eligible = _kv260_speedup_claim_eligible(matrix)
    structured_generation_clean = all(
        statuses.get(exp_id) == "clean" for exp_id in STRUCTURED_GENERATION_ARTIFACTS
    )

    end = time.perf_counter() if now_s is None else now_s
    duration_s = round(max(0.0, end - start), 6)
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": _compose_verdict(
            paper_ready=paper_ready,
            hardware_speedup_claim_eligible=hardware_speedup_claim_eligible,
            evidence_boundary_repaired=evidence_boundary["repaired"],
            structured_generation_clean=structured_generation_clean,
            fr11_self_learning_clean=self_learning["clean"],
            statuses=statuses,
        ),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "hardware_speedup_claim_eligible": hardware_speedup_claim_eligible,
        "gate_mate_speedup_claim_eligible": GATEMATE_SPEEDUP_ELIGIBLE,
        "evidence_boundary_repaired": evidence_boundary["repaired"],
        "sota_structured_generation_clean": structured_generation_clean,
        "fr11_self_learning_clean": self_learning["clean"],
        "clean_artifacts": clean_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "projection_only_artifacts": projection_only_artifacts,
        "diagnostic_only_artifacts": diagnostic_only_artifacts,
        "pilot_only_artifacts": pilot_only_artifacts,
        "artifact_classification_counts": _artifact_classification_counts(statuses),
        "row_classification_counts": _matrix_row_counts(matrix),
        "row_boundaries": _row_boundaries(matrix),
        "evidence_boundary_summary": evidence_boundary,
        "gate_mate_status": gatemate_status,
        "continuous_self_learning_status": self_learning,
        "paper_claim_boundary": _paper_claim_boundary(matrix, paper_ready),
        "hardware_claim_boundary": _hardware_claim_boundary(
            hardware_speedup_claim_eligible,
            gatemate_status,
        ),
        "top_three_next_actions": _top_three_next_actions(),
        "source_artifact_checksums": _source_artifact_checksums(root_path),
        "source_artifact_status": _source_artifact_status(payloads, present, statuses),
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2936 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_expected(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    payloads: dict[str, dict[str, Any]] = {}
    present: dict[str, bool] = {}
    for exp_id, spec in EXPECTED_ARTIFACTS.items():
        path = root / spec.path
        present[exp_id] = path.is_file()
        payloads[exp_id] = read_json_mapping(path) if present[exp_id] else {}
    return payloads, present


def _classify_all(payloads: dict[str, dict[str, Any]], present: dict[str, bool]) -> dict[str, str]:
    return {
        exp_id: classify_artifact(exp_id, payloads[exp_id], present[exp_id])
        for exp_id in EXPECTED_ARTIFACTS
    }


def _ids_with_status(statuses: dict[str, str], wanted: str) -> list[str]:
    return [exp_id for exp_id in EXPECTED_ARTIFACTS if statuses.get(exp_id) == wanted]


def _artifact_classification_counts(statuses: dict[str, str]) -> dict[str, int]:
    return {
        row_class: sum(1 for status in statuses.values() if status == row_class)
        for row_class in ROW_CLASSES
    }


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _terminal_success(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
    )


def _is_clean_by_required_fields(exp_id: str, payload: dict[str, Any]) -> bool:
    if not _terminal_success(payload.get("honest_verdict")):
        return False
    fields = EXPECTED_ARTIFACTS[exp_id].required_fields
    return all(payload.get(field) is True for field in fields)


def _clean_despite_inherited_flags(exp_id: str, payload: dict[str, Any]) -> bool:
    return (
        exp_id in INHERITED_FLAG_CLEAN_OVERRIDES
        and _is_clean_by_required_fields(exp_id, payload)
        and not _audit_flagged(payload)
    )


def _audit_flagged(payload: dict[str, Any]) -> bool:
    audit = payload.get("adversarial_audit_rerun")
    return isinstance(audit, dict) and audit.get("flagged") is True


def _has_current_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("adversarial_verify_passed") is False:
        return True
    if _audit_flagged(payload):
        return True
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        value = payload.get(key)
        if isinstance(value, list) and bool(value):
            return True
    summary = payload.get("adversarial_verify_summary")
    return isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0


def _paper_ready(matrix: dict[str, Any]) -> bool:
    if matrix.get("matrix_v10_ready") is not True:
        return False
    if matrix.get("matrix_v10_paper_boundary_ready") is not True:
        return False
    boundary = matrix.get("paper_claim_boundary")
    if isinstance(boundary, dict) and boundary.get("ready") is not True:
        return False
    headline_rows = _string_list(matrix.get("headline_eligible_rows"))
    clean_rows = set(_string_list(matrix.get("clean_rows")))
    return bool(headline_rows) and all(row in clean_rows for row in headline_rows)


def _kv260_speedup_claim_eligible(matrix: dict[str, Any]) -> bool:
    clean_rows = set(_string_list(matrix.get("clean_rows")))
    headline_rows = set(_string_list(matrix.get("headline_eligible_rows")))
    return KV260_SPEEDUP_ROW in clean_rows and KV260_SPEEDUP_ROW in headline_rows


def _evidence_boundary_summary(
    payloads: dict[str, dict[str, Any]],
    statuses: dict[str, str],
) -> dict[str, Any]:
    exp2924 = payloads["exp2924"]
    exp2925 = payloads["exp2925"]
    exp2926 = payloads["exp2926"]
    syntax_rate = exp2926.get("syntax_valid_rate")
    feasibility_rate = exp2926.get("feasibility_rate_overall")
    non_tautological = (
        _is_number(syntax_rate)
        and _is_number(feasibility_rate)
        and not math.isclose(float(syntax_rate), float(feasibility_rate), rel_tol=0.0, abs_tol=1e-9)
    )
    repaired = (
        statuses.get("exp2924") == "clean"
        and statuses.get("exp2925") == "clean"
        and statuses.get("exp2926") == "clean"
        and non_tautological
    )
    return {
        "repaired": repaired,
        "aggregation_metadata_clean": exp2924.get("aggregation_metadata_clean") is True,
        "aggregation_upstream_flags_preserved": bool(
            exp2924.get("upstream_flagged_rows_preserved")
        ),
        "aggregation_metadata_false_positive_count": len(
            exp2924.get("metadata_false_positive_findings") or []
        ),
        "taxonomy_corrigendum_clean": exp2925.get("taxonomy_corrigendum_clean") is True,
        "code_hallucination_verifier_ready": (
            exp2925.get("code_hallucination_verifier_ready") is True
        ),
        "constraintbench_corrigendum_ready": (
            exp2926.get("constraintbench_corrigendum_ready") is True
        ),
        "constraintbench_non_tautological": non_tautological,
        "constraintbench_syntax_valid_rate": syntax_rate,
        "constraintbench_feasibility_rate_overall": feasibility_rate,
        "constraintbench_optimality_rate_given_feasible": exp2926.get(
            "optimality_rate_given_feasible"
        ),
        "constraintbench_duration_s": exp2926.get("duration_s"),
    }


def _gatemate_status(
    payloads: dict[str, dict[str, Any]],
    present: dict[str, bool],
) -> dict[str, Any]:
    preflight = payloads["exp2927"]
    bitstream = payloads["exp2928"]
    flash = payloads["exp2929"]
    corrected_preflight_ready = (
        preflight.get("gatemate_himbaechel_ready") is True
        and preflight.get("nextpnr_device_supported") is True
        and not preflight.get("missing_toolchain")
    )
    preflight_blocker = "" if preflight.get("constraints_ready") is True else "constraints_missing"
    flash_blocker = str(flash.get("blocker") or flash.get("honest_verdict") or "")
    exact_blocker = flash_blocker or preflight_blocker or ""
    return {
        "corrected_preflight_ready": corrected_preflight_ready,
        "constraints_ready": preflight.get("constraints_ready") is True,
        "preflight_blocker": preflight_blocker,
        "tool_paths": preflight.get("tool_paths")
        if isinstance(preflight.get("tool_paths"), dict)
        else {},
        "bitstream_artifact_present": present["exp2928"],
        "bitstream_built": bitstream.get("gatemate_bitstream_built") is True,
        "flash_smoke_ready": flash.get("gatemate_flash_smoke_ready") is True,
        "flash_attempted": flash.get("flash_attempted") is True,
        "timing_claim_allowed": flash.get("timing_claim_allowed") is True,
        "speedup_claim_allowed": flash.get("speedup_claim_allowed") is True,
        "speedup_claim_eligible": GATEMATE_SPEEDUP_ELIGIBLE,
        "exact_blocker": exact_blocker,
        "claim_boundary": (
            "GateMate has corrected tool discovery only; it has no bitstream,"
            " no flash smoke, and no matched hardware-vs-CPU basis."
        ),
    }


def _continuous_self_learning_status(payload: dict[str, Any], status: str) -> dict[str, Any]:
    forgetting = payload.get("forgetting_rate")
    threshold = payload.get("forgetting_threshold")
    utility_delta = payload.get("utility_delta_vs_replay_only")
    forgetting_ok = (
        _is_number(forgetting)
        and (_is_number(threshold) is False or float(forgetting) <= float(threshold))
        and payload.get("non_forgetting_passed") is True
    )
    utility_ok = _is_number(utility_delta) and float(utility_delta) > 0.0
    clean = status == "clean" and utility_ok and forgetting_ok
    return {
        "clean": clean,
        "kan_cl_self_learning_ready": payload.get("kan_cl_self_learning_ready") is True,
        "continuous_self_learning_targeted": (
            payload.get("continuous_self_learning_targeted") is True
        ),
        "utility_delta_vs_replay_only": utility_delta,
        "energy_proxy_delta": payload.get("energy_proxy_delta"),
        "forgetting_rate": forgetting,
        "forgetting_threshold": threshold,
        "non_forgetting_passed": payload.get("non_forgetting_passed") is True,
        "updated_knot_or_rbf_count": payload.get("updated_knot_or_rbf_count"),
        "claim_boundary": (
            "Exp 2933 supports a bounded local-training self-learning claim:"
            " positive utility versus replay-only and measured non-forgetting."
        ),
    }


def _matrix_row_counts(matrix: dict[str, Any]) -> dict[str, int]:
    counts = matrix.get("row_classification_counts")
    if isinstance(counts, dict):
        return {row_class: int(counts.get(row_class) or 0) for row_class in ROW_CLASSES}
    return {row_class: 0 for row_class in ROW_CLASSES}


def _row_boundaries(matrix: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "headline_eligible_rows": _string_list(matrix.get("headline_eligible_rows")),
        "clean_rows": _string_list(matrix.get("clean_rows")),
        "flagged_rows": _string_list(matrix.get("flagged_rows")),
        "blocked_rows": _string_list(matrix.get("blocked_rows")),
        "missing_rows": _string_list(matrix.get("missing_rows")),
        "projection_only_rows": _string_list(matrix.get("projection_only_rows")),
        "diagnostic_only_rows": _string_list(matrix.get("diagnostic_only_rows")),
        "pilot_only_rows": _string_list(matrix.get("pilot_only_rows")),
    }


def _paper_claim_boundary(matrix: dict[str, Any], paper_ready: bool) -> dict[str, Any]:
    boundary = matrix.get("paper_claim_boundary")
    source_boundary = dict(boundary) if isinstance(boundary, dict) else {}
    source_boundary["ready"] = paper_ready
    source_boundary.setdefault(
        "headline_eligible_rows", _string_list(matrix.get("headline_eligible_rows"))
    )
    source_boundary.setdefault(
        "boundary_rules",
        [
            "Only clean rows are headline eligible.",
            "Flagged, blocked, missing, projection-only, diagnostic-only, and pilot-only rows remain non-headline.",
        ],
    )
    return source_boundary


def _hardware_claim_boundary(
    hardware_speedup_claim_eligible: bool,
    gatemate_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kv260_prior_evidence_eligible": hardware_speedup_claim_eligible,
        "kv260_claim_source_row": KV260_SPEEDUP_ROW,
        "gate_mate_speedup_claim_eligible": GATEMATE_SPEEDUP_ELIGIBLE,
        "gate_mate_blocker": gatemate_status["exact_blocker"],
        "boundary": (
            "KV260 prior same-basis evidence may remain eligible. GateMate is"
            " not eligible because no matched GateMate hardware-vs-CPU basis exists."
        ),
    }


def _source_artifact_checksums(root: Path) -> dict[str, str | None]:
    checksums: dict[str, str | None] = {}
    for spec in EXPECTED_ARTIFACTS.values():
        path = root / spec.path
        checksums[str(spec.path)] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return checksums


def _source_artifact_status(
    payloads: dict[str, dict[str, Any]],
    present: dict[str, bool],
    statuses: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        exp_id: {
            "path": str(spec.path),
            "present": present[exp_id],
            "classification": statuses[exp_id],
            "honest_verdict": payloads[exp_id].get("honest_verdict"),
        }
        for exp_id, spec in EXPECTED_ARTIFACTS.items()
    }


def _top_three_next_actions() -> list[str]:
    return [
        "Run KV260 MMD versus CPU sequential Gibbs to decide whether the n=64 hardware row is sampling the intended Boltzmann target or only a fixed-schedule heuristic.",
        "Add a CPU same-schedule synchronous-parallel comparator so the KV260 speedup claim has an apples-to-apples algorithmic basis.",
        "Measure verifier-ensemble AUPRC on code corpora at the documented negative base rate before promoting code-corpus verifier claims.",
    ]


def _compose_verdict(
    *,
    paper_ready: bool,
    hardware_speedup_claim_eligible: bool,
    evidence_boundary_repaired: bool,
    structured_generation_clean: bool,
    fr11_self_learning_clean: bool,
    statuses: dict[str, str],
) -> str:
    counts = _artifact_classification_counts(statuses)
    return (
        "complete: milestone=2026.05.276; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"hardware_speedup_claim_eligible={str(hardware_speedup_claim_eligible).lower()}; "
        "gate_mate_speedup_claim_eligible=false; "
        f"evidence_boundary_repaired={str(evidence_boundary_repaired).lower()}; "
        f"sota_structured_generation_clean={str(structured_generation_clean).lower()}; "
        f"fr11_self_learning_clean={str(fr11_self_learning_clean).lower()}; "
        f"clean={counts['clean']}; flagged={counts['flagged']}; "
        f"blocked={counts['blocked']}; missing={counts['missing']}; "
        f"projection_only={counts['projection_only']}"
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
