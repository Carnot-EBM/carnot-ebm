"""Build the Exp 2960 cross-corpus matrix v12 artifact.

Spec refs: REQ-REPORT-2960, SCENARIO-REPORT-2960.

This module is deliberately an aggregation layer. It reads the existing v11
matrix, the .277 capstone, and the completed .278 result artifacts, then emits
a compact v12 matrix. It does not rerun model inference, verifier scoring,
hardware builds, board flashing, or solver execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
SCHEMA = "carnot.cross_corpus_matrix.v12_278_aggregation.v1"
ARTIFACT = "experiment_2960_cross_corpus_matrix_v12"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2960_cross_corpus_matrix_v12.json")

MATRIX_V11_REL_PATH = Path("results/experiment_2943_cross_corpus_matrix_v11.json")
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

FORBIDDEN_PHRASES = (
    "KV260 hardware speedup",
    "FPGA acceleration over CPU",
    "runs faster on KV260",
    "thermalization",
    "Boltzmann-distributed energies",
    "equilibrium samples",
    "TSU performance",
    "Kona performance",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp2943", MATRIX_V11_REL_PATH),
    SourceSpec("exp2948", CAPSTONE_V277_REL_PATH),
    SourceSpec("exp2950", EXP2950_REL_PATH),
    SourceSpec("exp2951", EXP2951_REL_PATH),
    SourceSpec("exp2952", EXP2952_REL_PATH),
    SourceSpec("exp2953", EXP2953_REL_PATH),
    SourceSpec("exp2954", EXP2954_REL_PATH),
    SourceSpec("exp2955", EXP2955_REL_PATH),
    SourceSpec("exp2956", EXP2956_REL_PATH),
    SourceSpec("exp2957", EXP2957_REL_PATH),
    SourceSpec("exp2958", EXP2958_REL_PATH),
    SourceSpec("exp2958_transcript", EXP2958_TRANSCRIPT_REL_PATH),
    SourceSpec("exp2959", EXP2959_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one upstream JSON object, failing closed to an empty mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA256 digest for a source file, or None when absent."""

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
    """REQ-REPORT-2960: build matrix v12 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)
    payloads = _load_sources(root_path)
    source_rows = _upstream_artifacts_read(root_path, payloads)
    checksums = _upstream_checksums(source_rows)

    artifact = _base_artifact(payloads, source_rows, checksums, duration_s)
    required_errors = _required_source_errors(payloads)
    if required_errors:
        artifact.update(
            {
                "honest_verdict": "blocked_required_upstream_missing",
                "matrix_v12_ready": False,
                "required_upstream_errors": required_errors,
            }
        )
        artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
        return artifact

    self_learning_ready = bool(payloads["exp2954"].get("self_learning_utility_artifact_ready"))
    if not self_learning_ready:
        artifact["blocked_rows"] = _unique_strings(
            [*artifact["blocked_rows"], "exp2954_self_learning_utility"]
        )
        artifact["self_learning_delta_summary"] = _self_learning_delta_summary(payloads["exp2954"])
        artifact.update(
            {
                "honest_verdict": "blocked_self_learning_utility_artifact_not_ready",
                "matrix_v12_ready": False,
            }
        )
        artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
        return artifact

    artifact["matrix_v12_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2960 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json_object(root / spec.path) for spec in SOURCE_SPECS}


def _upstream_artifacts_read(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        present = path.is_file()
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": present,
                "required": spec.required,
                "readable_json_object": bool(payloads.get(spec.experiment_id)),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _upstream_checksums(source_rows: list[dict[str, Any]]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_rows}


def _required_source_errors(
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.required and not payloads.get(spec.experiment_id):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _base_artifact(
    payloads: Mapping[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    checksums: dict[str, str | None],
    duration_s: float,
) -> dict[str, Any]:
    v11 = payloads.get("exp2943", {})
    capstone = payloads.get("exp2948", {})
    v12_rows = _v12_rows(payloads)
    clean_rows = _unique_strings([*_v11_bucket(v11, "clean"), *_ids_by_class(v12_rows, "clean")])
    flagged_rows = _unique_strings(
        [*_v11_bucket(v11, "flagged"), *_ids_by_class(v12_rows, "flagged")]
    )
    blocked_rows = _unique_strings(
        [*_v11_bucket(v11, "blocked"), *_ids_by_class(v12_rows, "blocked")]
    )

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": "blocked_required_upstream_missing",
        "matrix_v12_ready": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "upstream_artifacts_read": source_rows,
        "upstream_checksums": checksums,
        "clean_rows": clean_rows,
        "flagged_rows": flagged_rows,
        "blocked_rows": blocked_rows,
        "gated_skipped_rows": _ids_by_class(v12_rows, "gated-skipped"),
        "pilot_only_rows": _v11_row_ids_by_class(v11, "pilot_only"),
        "projection_only_rows": _v11_row_ids_by_class(v11, "projection_only"),
        "aggregation_only_rows": [
            "exp2943_matrix_v11_carry_forward",
            "exp2948_capstone_v277_carry_forward",
            *_ids_by_class(v12_rows, "aggregation-only"),
        ],
        "forbidden_claims_absent": False,
        "code_repair_delta_summary": _code_repair_delta_summary(payloads),
        "self_learning_delta_summary": _self_learning_delta_summary(payloads.get("exp2954", {})),
        "hardware_state_summary": _hardware_state_summary(payloads),
        "solver_state_summary": _solver_state_summary(payloads.get("exp2959", {})),
        "matrix_rows": [
            {
                "row_id": "exp2943_matrix_v11_carry_forward",
                "row_class": "aggregation-only",
                "source_experiment_id": "exp2943",
                "claim_boundary": "v11 row buckets are copied without metric recomputation.",
            },
            {
                "row_id": "exp2948_capstone_v277_carry_forward",
                "row_class": "aggregation-only",
                "source_experiment_id": "exp2948",
                "claim_boundary": "The .277 narrowed paper boundary is carried forward.",
                "paper_ready": bool(capstone.get("paper_ready")),
                "headline_outcome": _get_path(
                    capstone, "deep_think_corrigenda_outcomes.headline_outcome"
                ),
            },
            *v12_rows,
        ],
        "delta_relative_to_277": _delta_relative_to_277(payloads),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }


def _v12_rows(payloads: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    exp2950 = payloads.get("exp2950", {})
    exp2951 = payloads.get("exp2951", {})
    exp2952 = payloads.get("exp2952", {})
    exp2953 = payloads.get("exp2953", {})
    exp2954 = payloads.get("exp2954", {})
    exp2955 = payloads.get("exp2955", {})
    exp2956 = payloads.get("exp2956", {})
    exp2957 = payloads.get("exp2957", {})
    exp2958 = payloads.get("exp2958", {})
    exp2959 = payloads.get("exp2959", {})
    return [
        _row(
            "exp2950_repair_prompt_manifest",
            _class_from_flags(exp2950, default="aggregation-only"),
            "exp2950",
            "Structured repair prompt manifest; no pass-rate claim is promoted.",
            {
                "manifest_ready": bool(exp2950.get("repair_prompt_manifest_ready")),
                "upstream_pass_at_1": _coerce_float(
                    _get_path(exp2950, "upstream_metrics.pass_at_1")
                ),
            },
            exp2950,
        ),
        _row(
            "exp2951_structured_candidate_manifest_adapter",
            _class_from_flags(exp2951, default="clean"),
            "exp2951",
            "Structured candidate schema adapter with deterministic fixture validation.",
            {
                "adapter_ready": bool(exp2951.get("structured_decode_manifest_ready")),
                "preferred_backend": exp2951.get("preferred_structured_output_backend"),
                "validation_fixture_passed": bool(exp2951.get("validation_fixture_passed")),
            },
            exp2951,
        ),
        _row(
            "exp2952_structured_repair_delta",
            _class_from_flags(exp2952, default="clean"),
            "exp2952",
            "Small code-repair pilot delta is recorded, with upstream flags preserved.",
            _code_repair_delta_summary({"exp2952": exp2952, "exp2950": exp2950}),
            exp2952,
        ),
        _row(
            "exp2953_threshold_policy",
            "clean" if exp2953.get("threshold_policy_ready") is True else "blocked",
            "exp2953",
            "Threshold policy is copied from the score-distribution aggregation.",
            _threshold_policy_summary(exp2953),
            exp2953,
        ),
        _row(
            "exp2954_self_learning_utility",
            _self_learning_row_class(exp2954),
            "exp2954",
            "Utility-gated replay result is recorded without model-weight mutation.",
            _self_learning_delta_summary(exp2954),
            exp2954,
        ),
        _row(
            "exp2955_gatemate_constraints_materialized",
            "clean" if exp2955.get("gatemate_constraints_ready") is True else "blocked",
            "exp2955",
            "GateMate constraints materialization landed; flashing was not part of this row.",
            {
                "constraints_ready": bool(exp2955.get("gatemate_constraints_ready")),
                "dirtyjtag_detected": bool(exp2955.get("dirtyjtag_detected")),
                "constraints_sha256": exp2955.get("constraints_sha256"),
            },
            exp2955,
        ),
        _row(
            "exp2956_gatemate_bitstream_built",
            "clean" if exp2956.get("gatemate_bitstream_built") is True else "blocked",
            "exp2956",
            "GateMate n=16 bitstream build landed; board execution is a separate row.",
            {
                "bitstream_built": bool(exp2956.get("gatemate_bitstream_built")),
                "bitstream_sha256": exp2956.get("bitstream_sha256"),
                "timing_met": _get_path(exp2956, "timing_summary.timing_met"),
            },
            exp2956,
        ),
        _row(
            "exp2957_gatemate_flash_smoke",
            _blocked_or_flagged_or_clean(exp2957),
            "exp2957",
            "GateMate board smoke remains separate from constraints and build rows.",
            {
                "board_detected": bool(exp2957.get("board_detected")),
                "flash_attempted": bool(exp2957.get("flash_attempted")),
                "flash_succeeded": bool(exp2957.get("flash_succeeded")),
                "smoke_vector_passed": bool(exp2957.get("smoke_vector_passed")),
            },
            exp2957,
        ),
        _row(
            "exp2958_polarfire_1000_clause_hash_verified",
            "clean" if exp2958.get("polarfire_1000_clause_hash_verified") is True else "blocked",
            "exp2958",
            "PolarFire 1000-clause scorer hash verification is copied as hardware smoke.",
            {
                "board_reachable": bool(exp2958.get("board_reachable")),
                "clause_count": _coerce_int(exp2958.get("clause_count")),
                "hash_verified": bool(exp2958.get("polarfire_1000_clause_hash_verified")),
                "elapsed_ms": _coerce_float(exp2958.get("elapsed_ms")),
            },
            exp2958,
        ),
        _row(
            "exp2959_nl_to_z3_execution_repair",
            _class_from_flags(exp2959, default="clean"),
            "exp2959",
            "NL-to-Z3 execution repair state is recorded without rerunning the solver.",
            _solver_state_summary(exp2959),
            exp2959,
        ),
    ]


def _row(
    row_id: str,
    row_class: str,
    source_experiment_id: str,
    claim_boundary: str,
    summary: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_class": row_class,
        "source_experiment_id": source_experiment_id,
        "headline_eligible": row_class == "clean",
        "paper_claim_eligible": row_class == "clean",
        "claim_boundary": claim_boundary,
        "source_honest_verdict": payload.get("honest_verdict", ""),
        "upstream_flags": _flag_kinds(payload),
        "summary": dict(summary),
    }


def _class_from_flags(payload: Mapping[str, Any], *, default: str) -> str:
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    return default


def _self_learning_row_class(payload: Mapping[str, Any]) -> str:
    if payload.get("self_learning_utility_artifact_ready") is not True:
        return "blocked"
    return _class_from_flags(payload, default="clean")


def _blocked_or_flagged_or_clean(payload: Mapping[str, Any]) -> str:
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    return "clean"


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


def _code_repair_delta_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp2950 = payloads.get("exp2950", {})
    exp2952 = payloads.get("exp2952", {})
    return {
        "source_experiment_id": "exp2952",
        "prior_277_pass_at_1_from_exp2950_upstream": _coerce_float(
            _get_path(exp2950, "upstream_metrics.pass_at_1")
        ),
        "prior_277_pass_at_k_from_exp2950_upstream": _coerce_float(
            _get_path(exp2950, "upstream_metrics.pass_at_k")
        ),
        "n_tasks": _coerce_int(exp2952.get("n_tasks")),
        "baseline_pass_at_1": _coerce_float(exp2952.get("baseline_pass_at_1")),
        "repair_pass_at_1": _coerce_float(exp2952.get("repair_pass_at_1")),
        "pass_at_1_delta": _coerce_float(exp2952.get("pass_at_1_delta")),
        "baseline_pass_at_k": _coerce_float(exp2952.get("baseline_pass_at_k")),
        "repair_pass_at_k": _coerce_float(exp2952.get("repair_pass_at_k")),
        "pass_at_k_delta": _coerce_float(exp2952.get("pass_at_k_delta")),
        "syntax_failure_rate_delta": _coerce_float(exp2952.get("syntax_failure_rate_delta")),
        "false_accept_delta": _coerce_float(exp2952.get("false_accept_delta")),
        "taxonomy_repair_delta_pass": bool(exp2952.get("taxonomy_repair_delta_pass")),
        "artifact_flagged": _has_flags(exp2952),
    }


def _threshold_policy_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_experiment_id": "exp2953",
        "threshold_policy_ready": bool(payload.get("threshold_policy_ready")),
        "selected_default_threshold": _coerce_float(payload.get("selected_default_threshold")),
        "expected_false_accept_rate_at_default": _coerce_float(
            payload.get("expected_false_accept_rate_at_default")
        ),
        "expected_recall_at_default": _coerce_float(payload.get("expected_recall_at_default")),
        "expected_ppv_at_default": _coerce_float(payload.get("expected_ppv_at_default")),
    }


def _self_learning_delta_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_experiment_id": "exp2954",
        "artifact_ready": bool(payload.get("self_learning_utility_artifact_ready")),
        "self_learning_utility_positive": bool(payload.get("self_learning_utility_positive")),
        "heldout_utility_baseline": _coerce_float(payload.get("heldout_utility_baseline")),
        "heldout_utility_after": _coerce_float(payload.get("heldout_utility_after")),
        "heldout_utility_delta": _coerce_float(payload.get("heldout_utility_delta")),
        "forgetting_guard_metric_before": _coerce_float(
            payload.get("forgetting_guard_metric_before")
        ),
        "forgetting_guard_metric_after": _coerce_float(
            payload.get("forgetting_guard_metric_after")
        ),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "live_model_invoked": bool(payload.get("live_model_invoked")),
        "model_weights_mutated": bool(payload.get("model_weights_mutated")),
        "rollback_triggered": bool(payload.get("rollback_triggered")),
        "artifact_flagged": _has_flags(payload),
    }


def _hardware_state_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp2955 = payloads.get("exp2955", {})
    exp2956 = payloads.get("exp2956", {})
    exp2957 = payloads.get("exp2957", {})
    exp2958 = payloads.get("exp2958", {})
    transcript = payloads.get("exp2958_transcript", {})
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
        },
        "polarfire": {
            "board_reachable": bool(exp2958.get("board_reachable")),
            "clause_count": _coerce_int(exp2958.get("clause_count")),
            "hash_verified": bool(exp2958.get("polarfire_1000_clause_hash_verified")),
            "elapsed_ms": _coerce_float(exp2958.get("elapsed_ms")),
            "remote_arch": exp2958.get("remote_arch"),
            "remote_python": exp2958.get("remote_python"),
            "transcript_total_wall_clock_s": _coerce_float(transcript.get("total_wall_clock_s")),
            "evaluation_cycles_per_clause": _coerce_int(
                transcript.get("evaluation_cycles_per_clause")
            ),
        },
        "claim_boundary": (
            "Hardware rows record materialization, reachability, and hash checks only; "
            "no comparative performance claim is asserted."
        ),
    }


def _solver_state_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_experiment_id": "exp2959",
        "z3_import_ok": bool(payload.get("z3_import_ok")),
        "z3_execution_repaired": bool(payload.get("z3_execution_repaired")),
        "z3_execution_rate": _coerce_float(payload.get("z3_execution_rate")),
        "solver_verified_accuracy": _coerce_float(payload.get("solver_verified_accuracy")),
        "answer_accuracy": _coerce_float(payload.get("answer_accuracy")),
        "parseability_rate": _coerce_float(payload.get("parseability_rate")),
        "n_items": _coerce_int(payload.get("n_items")),
        "formalization_manifest_sha256": payload.get("formalization_manifest_sha256"),
        "failure_categories": payload.get("failure_categories")
        if isinstance(payload.get("failure_categories"), Mapping)
        else {},
        "artifact_flagged": _has_flags(payload),
    }


def _delta_relative_to_277(payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    capstone = payloads.get("exp2948", {})
    return {
        "capstone_source_experiment_id": "exp2948",
        "capstone_paper_ready": bool(capstone.get("paper_ready")),
        "capstone_headline_outcome": _get_path(
            capstone, "deep_think_corrigenda_outcomes.headline_outcome"
        ),
        "repair_delta": _code_repair_delta_summary(payloads),
        "utility_delta": _self_learning_delta_summary(payloads.get("exp2954", {})),
        "hardware_materialization_state": _hardware_state_summary(payloads),
        "solver_execution_state": _solver_state_summary(payloads.get("exp2959", {})),
    }


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v12_ready=true; "
        f"clean={len(artifact['clean_rows'])}; "
        f"flagged={len(artifact['flagged_rows'])}; "
        f"blocked={len(artifact['blocked_rows'])}; "
        f"pilot_only={len(artifact['pilot_only_rows'])}"
    )


def _ids_by_class(rows: list[dict[str, Any]], row_class: str) -> list[str]:
    return [str(row["row_id"]) for row in rows if row.get("row_class") == row_class]


def _v11_bucket(v11: Mapping[str, Any], bucket: str) -> list[Any]:
    rows_current = v11.get(f"{bucket}_rows")
    if isinstance(rows_current, list):
        return rows_current
    rows_legacy = v11.get(f"rows_{bucket}")
    return rows_legacy if isinstance(rows_legacy, list) else []


def _v11_row_ids_by_class(v11: Mapping[str, Any], row_class: str) -> list[str]:
    rows = v11.get("matrix_rows")
    if not isinstance(rows, list):
        return []
    aliases = {row_class, row_class.replace("_", "-")}
    result: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("row_class") or "") in aliases and row.get("row_id"):
            result.append(str(row["row_id"]))
    return _unique_strings(result)


def _forbidden_claims_absent(payload: Mapping[str, Any]) -> bool:
    rendered = json.dumps(payload, sort_keys=True)
    rendered_lower = rendered.lower()
    return all(phrase.lower() not in rendered_lower for phrase in FORBIDDEN_PHRASES)


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _get_path(payload: Mapping[str, Any], dotted_field: str) -> Any:
    current: Any = payload
    for part in dotted_field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    return None
