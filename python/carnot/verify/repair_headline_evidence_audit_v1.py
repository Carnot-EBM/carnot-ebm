"""Build the Exp 3303 repair headline evidence audit artifact.

Spec refs: REQ-VERIFY-3303, SCENARIO-VERIFY-3303.

This audit is intentionally aggregation-only: it reads Exp 3302 and the fixed
Exp 3301 manifest, preserves the adversarial-verifier findings, and turns the
headline-promotion decision into a small machine-readable artifact. It does not
rerun repair generation or reinterpret correctness with an LLM judge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
AdversarialReporter = Callable[[Path], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.repair_headline_evidence_audit.v1"
EXPERIMENT_ID = "exp3303"
TASK_ID = "exp3303-repair-headline-evidence-audit-v1"
ARTIFACT = "experiment_3303_repair_headline_evidence_audit_v1"
MILESTONE = "2026.05.305"
RUN_DATE = "20260529"
RANDOM_SEED = 3303
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3303_repair_headline_evidence_audit_v1.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3301_REL_PATH = Path("results/experiment_3301_exact_repair_panel_manifest_v11.json")
MIN_PANEL_CASES = 30
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = {
    "repair_headline_evidence_audit_ready",
    "headline_claim_allowed_after_audit",
    "audited_artifact",
    "panel_case_count",
    "exact_successes_audited",
    "false_accept_count",
    "llm_judge_dependency_count",
    "adversarial_verify_flags",
    "substrate_consistency_passed",
    "confidence_interval_present",
    "claim_boundaries",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3303: audit Exp 3302 without rerunning generation."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    panel_path = resolve_path(root_path, EXP3302_REL_PATH)
    manifest_path = resolve_path(root_path, EXP3301_REL_PATH)
    panel = read_json_object(panel_path)
    manifest = read_json_object(manifest_path)
    adversarial_report = run_adversarial_report(
        panel_path,
        adversarial_reporter or default_adversarial_reporter,
    )
    flags = adversarial_flags(panel, adversarial_report)
    exact_provenance = exact_check_provenance(panel, manifest)
    manifest_status = manifest_consistency(panel, manifest)
    model_summary = model_invocation_summary(panel)
    ci_present = confidence_interval_present(panel)
    false_accept_count = count_value(panel.get("false_accept_count"))
    panel_case_count = count_value(panel.get("panel_case_count"))
    source_provenance_clean = panel.get("provenance_clean") is True
    critical_flags = [flag for flag in flags if flag.get("severity") == "critical"]
    substrate_ok = (
        model_summary["actual_model_declarations_present"]
        and source_provenance_clean
        and not critical_flags
    )
    audit_ready = bool(panel) and bool(manifest) and panel_case_count >= MIN_PANEL_CASES
    headline_allowed = (
        audit_ready
        and panel.get("headline_claim_allowed") is True
        and exact_provenance["all_claimed_successes_exact_checked"] is True
        and exact_provenance["reported_success_count_matches_rows"] is True
        and exact_provenance["llm_judge_dependency_count"] == 0
        and false_accept_count == 0
        and ci_present
        and manifest_status["hashes_match_exp3301"] is True
        and substrate_ok
    )
    boundaries = claim_boundaries(
        headline_allowed=headline_allowed,
        panel=panel,
        exact_provenance=exact_provenance,
        manifest_status=manifest_status,
        model_summary=model_summary,
        flags=flags,
        false_accept_count=false_accept_count,
        confidence_interval_present=ci_present,
        source_provenance_clean=source_provenance_clean,
    )
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3303", "SCENARIO-VERIFY-3303"],
        "repair_headline_evidence_audit_ready": audit_ready,
        "headline_claim_allowed_after_audit": headline_allowed,
        "audited_artifact": EXP3302_REL_PATH.as_posix(),
        "panel_case_count": panel_case_count,
        "exact_successes_audited": int(exact_provenance["claimed_success_count"]),
        "false_accept_count": false_accept_count,
        "llm_judge_dependency_count": int(exact_provenance["llm_judge_dependency_count"]),
        "adversarial_verify_flags": flags,
        "substrate_consistency_passed": substrate_ok,
        "confidence_interval_present": ci_present,
        "claim_boundaries": boundaries,
        "source_artifacts": source_artifacts(root_path),
        "adversarial_verify_report": {
            "loaded": adversarial_report.get("loaded") is True,
            "flag_count": count_value(adversarial_report.get("flag_count")),
            "max_severity": adversarial_report.get("max_severity"),
        },
        "exact_check_provenance": exact_provenance,
        "manifest_consistency": manifest_status,
        "model_invocation_summary": model_summary,
        "field_provenance": field_provenance(),
        "no_new_model_execution": True,
        "no_new_repair_generation": True,
        "no_llm_judge_used_by_audit": True,
        "source_headline_claim_allowed": panel.get("headline_claim_allowed") is True,
        "source_provenance_clean": source_provenance_clean,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3303 audit deliverable."""

    root_path = Path(root)
    output = resolve_path(root_path, output_path)
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        adversarial_reporter=adversarial_reporter,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def run_adversarial_report(path: Path, reporter: AdversarialReporter) -> JsonDict:
    """Invoke the available adversarial artifact checker and normalize output."""

    report = reporter(path)
    return dict(report) if isinstance(report, Mapping) else {"loaded": False, "flags": []}


def default_adversarial_reporter(path: Path) -> JsonDict:  # pragma: no cover - thin import wrapper.
    """Use the repository artifact verifier without shelling out to rerun models."""

    try:
        from scripts.adversarial_verify import verify_artifact

        return dict(verify_artifact(path))
    except Exception as exc:
        return {"artifact": str(path), "loaded": False, "error": str(exc), "flags": []}


def adversarial_flags(panel: Mapping[str, Any], report: Mapping[str, Any]) -> list[JsonDict]:
    """Return unique adversarial flags from the live checker plus source fields."""

    flags: list[JsonDict] = []
    for source in (report.get("flags"), panel.get("corrigendum_pending")):
        for row in mapping_list(source):
            flag = {
                "kind": str(row.get("kind") or "UNKNOWN"),
                "severity": str(row.get("severity") or "warn"),
                "detail": str(row.get("detail") or ""),
            }
            if flag not in flags:
                flags.append(flag)
    return flags


def exact_check_provenance(
    panel: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> JsonDict:
    """Summarize whether each claimed success has exact-check provenance."""

    rows = mapping_list(panel.get("candidate_results"))
    successes = [row for row in rows if row.get("verified_success") is True]
    missing_exact = [
        str(row.get("case_id") or row.get("case_hash") or "unknown")
        for row in successes
        if row.get("exact_check_passed") is not True
    ]
    success_types = sorted(
        {
            str(row.get("exact_checker_type") or "")
            for row in successes
            if row.get("exact_checker_type")
        }
    )
    manifest_llm_count = count_value(manifest.get("llm_judge_required_count"))
    manifest_llm_count += sum(
        1 for row in mapping_list(manifest.get("panel_cases")) if row.get("llm_judge_required") is True
    )
    candidate_llm_count = sum(1 for row in rows if row.get("llm_judge_required") is True)
    reported_success_count = count_value(panel.get("verified_success_count"))
    return {
        "candidate_result_count": len(rows),
        "claimed_success_count": len(successes),
        "reported_verified_success_count": reported_success_count,
        "reported_success_count_matches_rows": reported_success_count == len(successes),
        "all_claimed_successes_exact_checked": len(missing_exact) == 0,
        "claimed_successes_missing_exact_check": missing_exact,
        "exact_checker_types_for_successes": success_types,
        "llm_judge_dependency_count": manifest_llm_count + candidate_llm_count,
    }


def manifest_consistency(panel: Mapping[str, Any], manifest: Mapping[str, Any]) -> JsonDict:
    """Compare Exp 3302's frozen denominator with the Exp 3301 manifest."""

    panel_hashes = string_list(panel.get("manifest_case_hashes"))
    manifest_hashes = string_list(manifest.get("case_hashes"))
    return {
        "panel_manifest_hash_count": len(panel_hashes),
        "exp3301_manifest_hash_count": len(manifest_hashes),
        "hashes_match_exp3301": (
            bool(panel_hashes)
            and bool(manifest_hashes)
            and panel_hashes == manifest_hashes
            and panel.get("manifest_case_hashes_match") is not False
        ),
    }


def model_invocation_summary(panel: Mapping[str, Any]) -> JsonDict:
    """Check that model specs name the actually invoked model ids."""

    used_model_ids = unique_strings(
        [
            *[
                row.get("model_id") or row.get("hf_id")
                for row in mapping_list(panel.get("models_used"))
            ],
            *[
                row.get("model_id")
                for row in mapping_list(panel.get("candidate_results"))
                if row.get("model_id")
            ],
        ]
    )
    model_specs = mapping(panel.get("model_specs"))
    mandated_models = mapping(model_specs.get("mandated_models"))
    mandated_model_ids = unique_strings(
        [*string_list(model_specs.get("mandated_model_ids")), *list(mandated_models)]
    )
    missing_model_ids = unique_strings(
        row.get("model_id") or row.get("hf_id")
        for row in mapping_list(panel.get("missing_model_specs"))
    )
    return {
        "used_model_ids": used_model_ids,
        "mandated_model_ids": mandated_model_ids,
        "missing_model_ids": missing_model_ids,
        "used_model_count": len(used_model_ids),
        "actual_model_declarations_present": bool(used_model_ids)
        and all(model_id in mandated_model_ids for model_id in used_model_ids),
        "legacy_small_model_used": any(
            row.get("legacy_small_model") is True
            for row in mapping_list(panel.get("models_used"))
        ),
    }


def confidence_interval_present(panel: Mapping[str, Any]) -> bool:
    """Return true only when both headline-relevant CI fields are valid pairs."""

    return is_ci_pair(panel.get("repair_success_ci95")) and is_ci_pair(
        panel.get("false_accept_rate_ci95")
    )


def claim_boundaries(
    *,
    headline_allowed: bool,
    panel: Mapping[str, Any],
    exact_provenance: Mapping[str, Any],
    manifest_status: Mapping[str, Any],
    model_summary: Mapping[str, Any],
    flags: Sequence[Mapping[str, Any]],
    false_accept_count: int,
    confidence_interval_present: bool,
    source_provenance_clean: bool,
) -> list[str]:
    """List the exact claim restrictions implied by the audit gates."""

    if headline_allowed:
        return [
            "Headline repair claim allowed only for the audited Exp 3302 fixed 30-case exact panel and its recorded SOTA GGUF model set."
        ]
    boundaries: list[str] = []
    if any(flag.get("severity") == "critical" for flag in flags):
        boundaries.append(
            "Do not promote a headline repair claim until the critical duration/substrate adversarial verification flag is independently resolved."
        )
    if panel.get("headline_claim_allowed") is not True:
        boundaries.append(
            "Exp 3302 source artifact sets headline_claim_allowed=false, so downstream claims must remain audit/boundary evidence."
        )
    if not source_provenance_clean:
        boundaries.append(
            "Source provenance is not clean; cite only bounded exact-check and no-false-accept facts."
        )
    if exact_provenance.get("all_claimed_successes_exact_checked") is not True:
        boundaries.append(
            "At least one claimed success lacks exact checker provenance and cannot support a repair-success claim."
        )
    if exact_provenance.get("reported_success_count_matches_rows") is not True:
        boundaries.append(
            "Reported verified_success_count does not match audited candidate success rows."
        )
    if count_value(exact_provenance.get("llm_judge_dependency_count")) > 0:
        boundaries.append("LLM judge dependencies are present and cannot support headline repair evidence.")
    if false_accept_count != 0:
        boundaries.append("A nonzero false accept count blocks headline repair promotion.")
    if not confidence_interval_present:
        boundaries.append("Required repair-success and false-accept confidence interval fields are missing.")
    if manifest_status.get("hashes_match_exp3301") is not True:
        boundaries.append("Exp 3302 manifest hashes do not match the fixed Exp 3301 denominator.")
    if model_summary.get("actual_model_declarations_present") is not True:
        boundaries.append("The artifact does not name an actual invoked model in model_specs/models_used.")
    missing = string_list(model_summary.get("missing_model_ids"))
    if missing:
        boundaries.append(
            "Model claim is bounded to the used GGUF only; missing mandated models include "
            + ", ".join(missing)
            + "."
        )
    if model_summary.get("legacy_small_model_used") is True:
        boundaries.append("Legacy small-model use would block any SOTA repair claim.")
    return boundaries or ["Audit completed, but source evidence does not satisfy every headline gate."]


def source_artifacts(root: Path) -> JsonDict:
    """Record source artifact presence and checksums for reproducibility."""

    return {
        "exp3302": file_status(resolve_path(root, EXP3302_REL_PATH)),
        "exp3301": file_status(resolve_path(root, EXP3301_REL_PATH)),
    }


def file_status(path: Path) -> JsonDict:
    """Return readable/present/checksum metadata for one source file."""

    present = path.exists()
    status: JsonDict = {
        "path": path.relative_to(REPO_ROOT).as_posix() if path.is_absolute() and path.is_relative_to(REPO_ROOT) else path.as_posix(),
        "present": present,
        "readable": False,
        "sha256": None,
    }
    if not present:
        return status
    try:
        data = path.read_bytes()
    except OSError:
        return status
    status["readable"] = True
    status["sha256"] = hashlib.sha256(data).hexdigest()
    return status


def field_provenance() -> JsonDict:
    """Explain where the key audit fields come from."""

    return {
        "panel_case_count": "Exp 3302 top-level panel_case_count checked against Exp 3301 hashes.",
        "exact_successes_audited": "Candidate rows with verified_success=true and exact_check_passed=true.",
        "adversarial_verify_flags": "scripts.adversarial_verify.verify_artifact plus Exp 3302 corrigendum_pending.",
        "substrate_consistency_passed": "False when source provenance is dirty, a critical adversarial flag remains, or used models are not declared.",
        "claim_boundaries": "Derived from failed headline gates instead of discarding bounded evidence.",
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal audit artifact and block unsafe overclaiming."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("repair_headline_evidence_audit_ready"), bool):
        raise ValueError("repair_headline_evidence_audit_ready must be a bool")
    if not isinstance(artifact.get("headline_claim_allowed_after_audit"), bool):
        raise ValueError("headline_claim_allowed_after_audit must be a bool")
    if not str(artifact.get("audited_artifact") or ""):
        raise ValueError("audited_artifact must name the source result")
    if count_value(artifact.get("panel_case_count")) < MIN_PANEL_CASES:
        raise ValueError("panel_case_count must be >= 30")
    for field in (
        "exact_successes_audited",
        "false_accept_count",
        "llm_judge_dependency_count",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if not isinstance(artifact.get("adversarial_verify_flags"), list):
        raise ValueError("adversarial_verify_flags must be a list")
    if not isinstance(artifact.get("substrate_consistency_passed"), bool):
        raise ValueError("substrate_consistency_passed must be a bool")
    if not isinstance(artifact.get("confidence_interval_present"), bool):
        raise ValueError("confidence_interval_present must be a bool")
    if not isinstance(artifact.get("claim_boundaries"), list) or not artifact.get("claim_boundaries"):
        raise ValueError("claim_boundaries must be a non-empty list")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        raise ValueError("random_seed must be an integer")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or duration_s < 0:
        raise ValueError("duration_s must be a non-negative number")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a 64-character checksum")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a complete audit verdict that names the promotion decision."""

    return (
        "complete: "
        f"repair_headline_evidence_audit_ready={str(artifact['repair_headline_evidence_audit_ready']).lower()}; "
        f"headline_claim_allowed_after_audit={str(artifact['headline_claim_allowed_after_audit']).lower()}; "
        f"panel_case_count={artifact['panel_case_count']}; "
        f"exact_successes_audited={artifact['exact_successes_audited']}; "
        f"false_accept_count={artifact['false_accept_count']}; "
        f"adversarial_verify_flags={len(artifact['adversarial_verify_flags'])}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable audit content while excluding timing and self-hash fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    return stable_hash(stable)


def resolve_path(root: Path, value: Path | str) -> Path:
    """Resolve repository-relative paths."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def count_value(value: Any) -> int:
    """Parse non-bool integer-like values; invalid values fail closed to zero."""

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def is_ci_pair(value: Any) -> bool:
    """Return true for an ordered two-number confidence interval."""

    return (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
        and float(value[1]) >= float(value[0])
    )


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with explicit zero-denominator behavior."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started: float, finished: float) -> float:
    """Return non-negative elapsed seconds rounded for stable JSON."""

    return round(max(0.0, float(finished) - float(started)), 6)


def mapping(value: Any) -> JsonDict:
    """Return a dict for JSON-like mapping values."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from arbitrary JSON-like values."""

    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def string_list(value: Any) -> list[str]:
    """Return non-empty strings from list-like JSON values."""

    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item or "")]


def unique_strings(values: Sequence[Any]) -> list[str]:
    """Return sorted unique non-empty strings."""

    return sorted({str(value) for value in values if str(value or "")})


def stable_hash(payload: Any) -> str:
    """Return a SHA-256 checksum for JSON-compatible payloads."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:  # pragma: no cover - CLI wrapper.
    """Write the default Exp 3303 audit artifact."""

    output = write_artifact()
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
