"""Build and check the Exp 3309 live-runtime provenance contract.

Spec refs: REQ-REPORT-3309, SCENARIO-REPORT-3309.

This module is a contract and checker, not a live rerun. Its job is to make
the next expensive SOTA GGUF tasks fail closed unless they preserve enough
runtime, metric-lineage, and repair-substrate evidence to clear the `.305`
quality flags without relying on trust in a short wall-clock artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.live_runtime_provenance_contract.v1"
CONTRACT_VERSION = SCHEMA_VERSION
EXPERIMENT_ID = "exp3309"
TASK_ID = "exp3309-live-runtime-provenance-contract-v1"
ARTIFACT = "experiment_3309_live_runtime_provenance_contract_v1"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3309

SPEC_REL_PATH = Path("openspec/capabilities/research-reporting/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3309_live_runtime_provenance_contract_v1.json")
EXP3308_REL_PATH = Path("results/experiment_3308_quality_flag_root_cause_autopsy_v1.json")
EXECUTABLE_CHECKER_PATH = "python/carnot/reporting/live_runtime_provenance_contract_3309.py"
MINIMUM_LIVE_DURATION_S = 60.0
CPU_SMOKE_MINIMUM_DURATION_S = 0.0
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "runtime_contract_ready",
    "contract_version",
    "minimum_live_duration_s",
    "required_provenance_fields",
    "tautology_guard_rules",
    "duration_guard_rules",
    "repair_substrate_rules",
    "executable_checker_path",
    "honest_verdict",
)

CHECKER_VERSION_KEYS: tuple[str, ...] = (
    "live_runtime_provenance_contract",
    "executable_checker_path",
    "checker_file_sha256",
    "adversarial_verify",
    "spec_coverage",
    "llama_cpp_python",
    "selected_python_cuda_probe",
)

HEADLINE_RUNTIME_FIELDS: tuple[str, ...] = (
    "runtime_provenance.command",
    "runtime_provenance.cwd",
    "runtime_provenance.pid",
    "runtime_provenance.cuda_visible_devices",
    "runtime_provenance.wall_clock_duration_s",
    "runtime_provenance.model_load_started_at",
    "runtime_provenance.model_load_finished_at",
    "runtime_provenance.model_load_duration_s",
    "runtime_provenance.gpu_memory_samples",
    "runtime_provenance.per_case_generation",
)

HEADLINE_MODEL_FIELDS: tuple[str, ...] = (
    "models_used[].model_id",
    "models_used[].hf_id",
    "models_used[].model_path",
    "models_used[].size_bytes",
    "models_used[].quantization",
)

METRIC_LINEAGE_KEYS: tuple[str, ...] = (
    "numerator",
    "denominator",
    "source_filter",
    "source_row_count",
    "calculation_function",
    "source_artifact_sha256",
)

TAUTOLOGY_METRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("refusal_rate", "aligned_instruction_false_positive_rate"),
    ("refusal_rate", "aligned_benign_false_positive_rate"),
    ("repair_success_rate", "false_accept_rate"),
    ("dataflip_detection_rate", "attack_success_rate"),
)

REQUIRED_PROVENANCE_FIELDS: tuple[JsonDict, ...] = (
    {
        "category": "model_identity",
        "fields": ["models_used[].model_id", "models_used[].hf_id", "models_used[].name", "models_used[].role"],
        "required_for": ["headline_live", "cpu_smoke"],
    },
    {
        "category": "cache_path",
        "fields": ["models_used[].model_path", "models_used[].cache_root", "models_used[].snapshot_revision"],
        "required_for": ["headline_live"],
    },
    {
        "category": "model_size",
        "fields": ["models_used[].size_bytes", "models_used[].quantization"],
        "required_for": ["headline_live"],
    },
    {
        "category": "load_timing",
        "fields": [
            "runtime_provenance.model_load_started_at",
            "runtime_provenance.model_load_finished_at",
            "runtime_provenance.model_load_duration_s",
        ],
        "required_for": ["headline_live"],
    },
    {
        "category": "wall_clock_duration",
        "fields": ["duration_s", "runtime_provenance.wall_clock_duration_s"],
        "required_for": ["headline_live", "cpu_smoke", "aggregation_audit"],
    },
    {
        "category": "generated_token_count",
        "fields": ["tokens_generated", "runtime_provenance.per_case_generation[].generated_tokens"],
        "required_for": ["headline_live"],
    },
    {
        "category": "command",
        "fields": ["runtime_provenance.command", "runtime_provenance.argv", "runtime_provenance.cwd", "runtime_provenance.pid"],
        "required_for": ["headline_live", "cpu_smoke"],
    },
    {
        "category": "cuda_visibility",
        "fields": ["runtime_provenance.cuda_visible_devices", "runtime_provenance.selected_python_cuda"],
        "required_for": ["headline_live"],
    },
    {
        "category": "gpu_memory",
        "fields": ["runtime_provenance.gpu_memory_samples"],
        "required_for": ["headline_live"],
    },
    {
        "category": "checker_versions",
        "fields": [f"checker_versions.{key}" for key in CHECKER_VERSION_KEYS],
        "required_for": ["headline_live", "cpu_smoke", "aggregation_audit"],
    },
)

TAUTOLOGY_GUARD_RULES: JsonDict = {
    "required_lineage_object": "metric_lineage",
    "lineage_required_fields": list(METRIC_LINEAGE_KEYS),
    "distinct_metrics_must_not_share": [
        "numerator",
        "denominator",
        "source_filter",
        "calculation_function",
    ],
    "equal_rate_policy": (
        "Equal distinct metric values are allowed only when both metrics can be "
        "recomputed from independent lineage or declare an explicit alias."
    ),
    "guarded_metric_pairs": [list(pair) for pair in TAUTOLOGY_METRIC_PAIRS],
}

DURATION_GUARD_RULES: JsonDict = {
    "headline_live_evidence": {
        "minimum_duration_s": MINIMUM_LIVE_DURATION_S,
        "applies_when": "GGUF/CUDA/live-model markers appear and headline promotion is possible",
        "must_fail_when_short": True,
    },
    "cpu_smoke_exception": {
        "minimum_duration_s": CPU_SMOKE_MINIMUM_DURATION_S,
        "required_labels": ["evidence_tier=cpu_smoke", "cpu_smoke_only=true", "headline_result=false"],
        "headline_promotion_allowed": False,
    },
    "aggregation_audit_exception": {
        "minimum_duration_s": CPU_SMOKE_MINIMUM_DURATION_S,
        "required_labels": ["evidence_tier=aggregation_audit", "no_new_model_execution=true"],
        "requires_source_runtime_contract": True,
        "headline_promotion_allowed": False,
    },
}

REPAIR_SUBSTRATE_RULES: JsonDict = {
    "shared_fields": [
        "source_artifact_hashes",
        "panel_case_count",
        "manifest_case_hashes",
        "exact_checker_types",
        "used_model_ids",
        "missing_model_ids",
        "source_panel_runtime_contract",
    ],
    "audit_substrate_exception": {
        "allowed_audit_substrate": "aggregation_from_upstream_artifacts",
        "requires_no_new_model_execution": True,
    },
    "promotion_gate": (
        "substrate_consistency_passed is true only when the source panel's "
        "runtime contract is clean, critical adversarial flags are absent, "
        "exact checkers are preserved, and no legacy small model is substituted."
    ),
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3309: build the executable contract artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-REPORT-3309", "SCENARIO-REPORT-3309"],
        "runtime_contract_ready": True,
        "contract_version": CONTRACT_VERSION,
        "minimum_live_duration_s": MINIMUM_LIVE_DURATION_S,
        "required_provenance_fields": [dict(row) for row in REQUIRED_PROVENANCE_FIELDS],
        "tautology_guard_rules": dict(TAUTOLOGY_GUARD_RULES),
        "duration_guard_rules": dict(DURATION_GUARD_RULES),
        "repair_substrate_rules": dict(REPAIR_SUBSTRATE_RULES),
        "executable_checker_path": EXECUTABLE_CHECKER_PATH,
        "checker_entrypoint": "carnot.reporting.live_runtime_provenance_contract_3309.check_runtime_evidence_artifact",
        "checker_versions_required": list(CHECKER_VERSION_KEYS),
        "checker_file_sha256": sha256_file_or_empty(root_path / EXECUTABLE_CHECKER_PATH),
        "source_artifacts": [source_artifact_summary(root_path, EXP3308_REL_PATH)],
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_garak_run": True,
        "no_new_dataflip_run": True,
        "no_new_repair_generation": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
        "downstream_implementation_steps": downstream_implementation_steps(),
        "duration_s": duration(started, finished),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_contract_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Persist the Exp 3309 contract artifact after validation."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_runtime_evidence_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Check one downstream artifact against the Exp 3309 contract."""

    tier = evidence_tier(artifact)
    violations: list[JsonDict] = []
    warnings: list[str] = []
    headline_live = tier == "headline_live"
    cpu_smoke = tier == "cpu_smoke"
    aggregation_audit = tier == "aggregation_audit"
    if headline_live:
        violations.extend(headline_missing_provenance(artifact))
    if headline_live and live_markers_present(artifact) and numeric(artifact.get("duration_s")) < MINIMUM_LIVE_DURATION_S:
        violations.append(
            violation(
                "DURATION_TOO_SHORT",
                f"duration_s is below {MINIMUM_LIVE_DURATION_S}s for headline live GGUF evidence",
            )
        )
    if cpu_smoke:
        smoke_ok = (
            artifact.get("cpu_smoke_only") is True
            and artifact.get("headline_result") is False
            and artifact.get("headline_claim_allowed") is False
            and bool(mapping(artifact.get("runtime_provenance")).get("command"))
        )
        warnings.append("non_headline_cpu_smoke_exception") if smoke_ok else violations.append(
            violation("CPU_SMOKE_EXCEPTION_INVALID", "short smoke evidence must disable headline promotion")
        )
    tautology_violations = tautology_guard_violations(artifact)
    repair_violations = repair_substrate_violations(artifact) if aggregation_audit else []
    violations.extend(tautology_violations)
    violations.extend(repair_violations)
    duration_passed = not any(row["kind"] in {"DURATION_TOO_SHORT", "CPU_SMOKE_EXCEPTION_INVALID"} for row in violations)
    tautology_passed = not tautology_violations
    repair_passed = not repair_violations
    promotion_allowed = headline_live and duration_passed and tautology_passed and repair_passed
    runtime_passed = duration_passed and tautology_passed and repair_passed and not any(
        row["kind"] == "MISSING_PROVENANCE" for row in violations
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "evidence_tier": tier,
        "runtime_contract_passed": runtime_passed,
        "duration_contract_passed": duration_passed,
        "tautology_guard_passed": tautology_passed,
        "repair_substrate_passed": repair_passed,
        "headline_promotion_allowed": promotion_allowed,
        "violations": violations,
        "warnings": warnings,
    }


def validate_contract_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal contract artifact and block incomplete contracts."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("runtime_contract_ready"), bool):
        raise ValueError("runtime_contract_ready must be a bool")
    if artifact.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("contract_version must match the executable checker")
    if numeric(artifact.get("minimum_live_duration_s")) < MINIMUM_LIVE_DURATION_S:
        raise ValueError("minimum_live_duration_s must preserve the headline floor")
    if not artifact.get("required_provenance_fields"):
        raise ValueError("required_provenance_fields must be non-empty")
    if not artifact.get("tautology_guard_rules"):
        raise ValueError("tautology_guard_rules must be non-empty")
    if not artifact.get("duration_guard_rules"):
        raise ValueError("duration_guard_rules must be non-empty")
    if not artifact.get("repair_substrate_rules"):
        raise ValueError("repair_substrate_rules must be non-empty")
    if artifact.get("executable_checker_path") != EXECUTABLE_CHECKER_PATH:
        raise ValueError("executable_checker_path must name the checker module")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def headline_missing_provenance(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return missing headline-live provenance fields as checker violations."""

    missing: list[str] = []
    models = mapping_list(artifact.get("models_used"))
    if not models:
        missing.append("models_used[]")
    for field in HEADLINE_MODEL_FIELDS:
        key = field.split(".", 1)[1]
        if any(not model.get(key) for model in models):
            missing.append(field)
    runtime = mapping(artifact.get("runtime_provenance"))
    for field in HEADLINE_RUNTIME_FIELDS:
        key = field.split(".", 1)[1]
        if not runtime.get(key):
            missing.append(field)
    versions = mapping(artifact.get("checker_versions"))
    for key in CHECKER_VERSION_KEYS:
        if not versions.get(key):
            missing.append(f"checker_versions.{key}")
    if numeric(artifact.get("tokens_generated")) <= 0:
        missing.append("tokens_generated")
    return [violation("MISSING_PROVENANCE", f"missing required provenance: {', '.join(missing)}")] if missing else []


def tautology_guard_violations(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Detect equal distinct metrics that are not independently sourced."""

    lineage = mapping(artifact.get("metric_lineage"))
    violations: list[JsonDict] = []
    for left, right in TAUTOLOGY_METRIC_PAIRS:
        left_value = artifact.get(left)
        right_value = artifact.get(right)
        if left_value is not None and right_value is not None and abs(numeric(left_value) - numeric(right_value)) <= 0.000001:
            left_lineage = mapping(lineage.get(left))
            right_lineage = mapping(lineage.get(right))
            if not independent_lineage(left_lineage, right_lineage):
                violations.append(
                    violation("TAUTOLOGY", f"{left} and {right} are equal without independent metric lineage")
                )
    return violations


def repair_substrate_violations(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Detect repair-panel/audit substrate inconsistencies."""

    source_contract = mapping(artifact.get("source_panel_runtime_contract"))
    audit_models = mapping(artifact.get("model_invocation_summary"))
    source_models = mapping(artifact.get("source_model_invocation_summary"))
    checks = [
        artifact.get("inference_substrate") == "aggregation_from_upstream_artifacts",
        artifact.get("no_new_model_execution") is True,
        source_contract.get("runtime_contract_passed") is True,
        source_contract.get("duration_contract_passed") is True,
        not source_contract.get("critical_adversarial_flags"),
        int(artifact.get("panel_case_count") or -1) == int(artifact.get("source_panel_case_count") or -2),
        set(string_list(artifact.get("manifest_case_hashes"))) == set(string_list(artifact.get("source_manifest_case_hashes"))),
        set(string_list(artifact.get("exact_checker_types"))) == set(string_list(artifact.get("source_exact_checker_types"))),
        set(string_list(audit_models.get("used_model_ids"))) == set(string_list(source_models.get("used_model_ids"))),
        set(string_list(audit_models.get("missing_model_ids"))) == set(string_list(source_models.get("missing_model_ids"))),
        audit_models.get("legacy_small_model_used") is False,
    ]
    return [] if all(checks) else [violation("REPAIR_SUBSTRATE_INCONSISTENCY", "repair audit and source panel substrate facts do not match")]


def independent_lineage(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return true when two metric lineages are complete and non-identical."""

    required_present = all(key in left and key in right for key in METRIC_LINEAGE_KEYS)
    identity_fields = ("numerator", "denominator", "source_filter", "calculation_function")
    return required_present and any(left.get(key) != right.get(key) for key in identity_fields)


def evidence_tier(artifact: Mapping[str, Any]) -> str:
    """Classify the artifact tier used for duration and substrate rules."""

    explicit = str(artifact.get("evidence_tier") or "")
    if explicit:
        return explicit
    if artifact.get("cpu_smoke_only") is True:
        return "cpu_smoke"
    if artifact.get("inference_substrate") == "aggregation_from_upstream_artifacts":
        return "aggregation_audit"
    return "headline_live" if artifact.get("headline_result") is True or artifact.get("headline_claim_allowed") is True else "non_headline"


def live_markers_present(artifact: Mapping[str, Any]) -> bool:
    """Return true when the artifact carries GGUF/CUDA/live-model markers."""

    rendered = json.dumps(artifact, sort_keys=True, default=str).casefold()
    return any(marker in rendered for marker in ("gguf", "cuda", "llama_cpp", "live"))


def downstream_implementation_steps() -> list[str]:
    """Record the bounded steps downstream live reruns must implement."""

    return [
        "import check_runtime_evidence_artifact before claiming runtime_provenance_clean",
        "record every required_provenance_fields entry before writing the terminal artifact",
        "store metric_lineage for guarded metrics and recompute rates from raw rows",
        "copy the checker result into duration_contract_passed and runtime_contract_passed",
        "have repair audits preserve the source panel runtime-contract result instead of rerunning live inference",
    ]


def source_artifact_summary(root: Path, rel_path: Path) -> JsonDict:
    """Return source presence, readiness, and checksum for the Exp 3308 autopsy."""

    path = root / rel_path
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    return {
        "experiment_id": "exp3308",
        "path": rel_path.as_posix(),
        "present": path.exists(),
        "readable_json_object": bool(payload),
        "ready": payload.get("quality_flag_autopsy_ready") is True,
        "reported_experiment_id": str(payload.get("experiment_id") or ""),
        "artifact": str(payload.get("artifact") or ""),
        "sha256": sha256_file_or_empty(path),
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a compact terminal verdict for the contract artifact."""

    return (
        "complete: "
        f"runtime_contract_ready={str(artifact['runtime_contract_ready']).lower()}; "
        f"contract_version={artifact['contract_version']}; "
        f"minimum_live_duration_s={artifact['minimum_live_duration_s']}; "
        f"executable_checker_path={artifact['executable_checker_path']}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable contract content while excluding self-referential fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    return stable_hash(stable)


def violation(kind: str, detail: str) -> JsonDict:
    """Build one machine-readable checker violation."""

    return {"kind": kind, "severity": "critical", "detail": detail}


def mapping(value: Any) -> JsonDict:
    """Return a plain dict for JSON-like mappings."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from a JSON-like list."""

    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list | tuple) else []


def string_list(value: Any) -> list[str]:
    """Return stable non-empty strings from an iterable JSON value."""

    if isinstance(value, str) or value is None:
        return []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return []


def numeric(value: Any) -> float:
    """Return a float with explicit bad-value fallback for artifact checks."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def duration(started: float, finished: float) -> float:
    """Return non-negative elapsed seconds rounded for stable JSON."""

    return round(max(0.0, float(finished) - float(started)), 6)


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible content."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file_or_empty(path: Path) -> str:
    """Return a file digest, or an empty string when the source is absent."""

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
