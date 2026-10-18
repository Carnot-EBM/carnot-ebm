"""Build the Exp 3283 prompt-injection corrigendum audit.

Spec refs: REQ-REPORT-3283, SCENARIO-REPORT-3283.

This module is a boundary ledger, not a relabeling job. It reads the already
materialized prompt-injection artifacts and records what later Garak, KAN,
clean-verifier, repair, and paper-claim tasks may cite. The important point is
to keep tiny cached model evidence panels, deterministic template expansion,
blocked benchmarks, sidecar detector scores, and aggregation-only matrices in
separate buckets so downstream work cannot accidentally turn mixed provenance
into a clean live-LLM headline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.prompt_injection_corrigendum_duration_audit.v1"
EXPERIMENT_ID = "exp3283"
TASK_ID = "exp3283-prompt-injection-corrigendum-duration-audit-v1"
ARTIFACT = "experiment_3283_prompt_injection_corrigendum_duration_audit_v1"
MILESTONE = "2026.05.304"
RANDOM_SEED = 3283
OUTPUT_REL_PATH = Path(
    "results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json"
)

EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")
EXP3269_REL_PATH = Path(
    "results/experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.json"
)
EXP3270_REL_PATH = Path("results/experiment_3270_prompt_injection_teacher_label_shards_2_4_v1.json")
EXP3271_REL_PATH = Path(
    "results/experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1.json"
)
EXP3272_REL_PATH = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
EXP3273_REL_PATH = Path(
    "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
)
EXP3274_REL_PATH = Path(
    "results/experiment_3274_prompt_injection_v4_garak_dataflip_redteam_eval_v1.json"
)
EXP3275_REL_PATH = Path("results/experiment_3275_clean_local_sota_verifier_rerun_v14.json")
EXP3276_REL_PATH = Path(
    "results/experiment_3276_repair_gate_decision_v8_after_v4_garak_clean_verifier.json"
)
EXP3277_REL_PATH = Path("results/experiment_3277_sota_repair_micro_panel_v9.json")
EXP3279_REL_PATH = Path("results/experiment_3279_evidence_matrix_v35.json")

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
ARTIFACT_CLASSES = {"live-LLM", "cached", "template-backed", "aggregation-only", "blocked"}
REQUIRED_PRESENT_IDS = {
    "exp3264",
    "exp3269",
    "exp3270",
    "exp3271",
    "exp3272",
    "exp3273",
    "exp3274",
    "exp3275",
    "exp3276",
    "exp3279",
}
REQUIRED_ARTIFACT_FIELDS = {
    "corrigendum_ready",
    "audited_artifacts",
    "provenance_by_artifact",
    "duration_flags",
    "tautology_flags",
    "headline_eligible_metrics",
    "provisional_or_sidecar_metrics",
    "downstream_usage_rules",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
FORBIDDEN_HEADLINE_METRICS = {
    "full_corpus_auroc",
    "full_corpus_auprc",
    "new_label_count",
    "cumulative_label_count",
    "assembled_example_count",
    "raw_example_count",
    "garak_gate_passed",
    "dataflip_gate_passed",
    "false_accept_rate",
    "false_reject_rate",
}


@dataclass(frozen=True)
class SourceSpec:
    """One source artifact whose provenance must be explicit in the ledger."""

    experiment_id: str
    role: str
    path: Path


EXPECTED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("exp3264", "prior_live_seed_teacher_labels", EXP3264_REL_PATH),
    SourceSpec("exp3269", "v4_full_corpus_split_manifest", EXP3269_REL_PATH),
    SourceSpec("exp3270", "teacher_label_shards_2_4", EXP3270_REL_PATH),
    SourceSpec("exp3271", "teacher_label_shards_5_7_garak_seed", EXP3271_REL_PATH),
    SourceSpec("exp3272", "full_corpus_assembly_leakage_audit", EXP3272_REL_PATH),
    SourceSpec("exp3273", "kan_full_corpus_delong_sidecar", EXP3273_REL_PATH),
    SourceSpec("exp3274", "garak_dataflip_redteam_blocked", EXP3274_REL_PATH),
    SourceSpec("exp3275", "clean_local_sota_verifier_blocked", EXP3275_REL_PATH),
    SourceSpec("exp3276", "repair_gate_blocked", EXP3276_REL_PATH),
    SourceSpec("exp3277", "repair_micro_panel_missing", EXP3277_REL_PATH),
    SourceSpec("exp3279", "evidence_matrix_v35", EXP3279_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact source bytes so the corrigendum can be tied to its inputs."""

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
) -> JsonDict:
    """REQ-REPORT-3283: build the corrigendum from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    records = [_source_record(root_path, spec) for spec in EXPECTED_SOURCES]
    payloads = {str(record["experiment_id"]): _as_mapping(record.get("payload")) for record in records}
    audited = [_public_audit_record(record) for record in records]
    provenance = {
        str(record["experiment_id"]): _provenance_record(record) for record in records
    }
    missing_required = [
        row["experiment_id"]
        for row in audited
        if row["experiment_id"] in REQUIRED_PRESENT_IDS and row["present"] is False
    ]
    duration_flags = _flag_records(records, {"DURATION_TOO_SHORT"})
    tautology_flags = _flag_records(records, {"TAUTOLOGY", "IMPLAUSIBLE_PERFECT"})
    leakage_flags = _leakage_flags(payloads.get("exp3272", {}))
    label_totals = _label_provenance_totals(provenance)
    usage_rules = _downstream_usage_rules()

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "corrigendum_ready": not missing_required,
        "audited_artifacts": audited,
        "provenance_by_artifact": provenance,
        "label_provenance_totals": label_totals,
        "duration_flags": duration_flags,
        "tautology_flags": tautology_flags,
        "leakage_flags": leakage_flags,
        "headline_eligible_metrics": _headline_eligible_metrics(payloads, leakage_flags),
        "provisional_or_sidecar_metrics": _provisional_or_sidecar_metrics(payloads),
        "downstream_usage_rules": usage_rules,
        "missing_required_artifacts": missing_required,
        "protected_files_untouched": {
            "scripts/research_conductor.py": True,
            "ops/status.md": True,
            "ops/changelog.md": True,
            "_bmad/traceability.md": True,
        },
        "no_relabeling_performed": True,
        "no_new_model_execution": True,
        "no_new_garak_run": True,
        "no_new_kan_training_or_scoring": True,
        "no_new_clean_verifier_run": True,
        "no_new_repair_run": True,
        "no_push": True,
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    if artifact["corrigendum_ready"]:
        validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3283 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the corrigendum omits fields or promotes bounded evidence."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3283")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3283-prompt-injection-corrigendum-duration-audit-v1")
    if artifact.get("corrigendum_ready") is not True:
        raise ValueError("corrigendum_ready must be true for a completed corrigendum")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    promoted = {
        str(metric.get("metric"))
        for metric in _as_list(artifact.get("headline_eligible_metrics"))
        if isinstance(metric, Mapping)
    } & FORBIDDEN_HEADLINE_METRICS
    if promoted:
        raise ValueError(f"forbidden headline metrics promoted: {sorted(promoted)}")


def _source_record(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    present = path.is_file() and bool(payload)
    artifact_class = _classify_artifact(spec, payload, present)
    return {
        "experiment_id": spec.experiment_id,
        "role": spec.role,
        "path": spec.path.as_posix(),
        "present": present,
        "payload": payload,
        "artifact_class": artifact_class,
        "sha256": sha256_file(path),
    }


def _public_audit_record(record: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(record.get("payload"))
    return {
        "experiment_id": str(record.get("experiment_id") or ""),
        "reported_experiment_id": _artifact_id(payload, SourceSpec("", "", Path(""))),
        "path": str(record.get("path") or ""),
        "role": str(record.get("role") or ""),
        "present": record.get("present") is True,
        "artifact_class": str(record.get("artifact_class") or "blocked"),
        "duration_s": _float_value(payload.get("duration_s")),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": record.get("sha256"),
    }


def _provenance_record(record: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(record.get("payload"))
    artifact_class = str(record.get("artifact_class") or "blocked")
    return {
        "artifact_class": artifact_class,
        "source_path": str(record.get("path") or ""),
        "claim_boundary": _claim_boundary(str(record.get("experiment_id") or ""), artifact_class),
        "row_provenance_counts": _row_provenance_counts(str(record.get("experiment_id") or ""), payload),
        "duration_s": _float_value(payload.get("duration_s")),
        "flags": [
            _flag_payload(str(flag.get("kind") or ""), flag)
            for flag in _as_list(payload.get("corrigendum_pending"))
            if isinstance(flag, Mapping)
        ],
    }


def _classify_artifact(spec: SourceSpec, payload: Mapping[str, Any], present: bool) -> str:
    if not present or _is_blocked(payload):
        return "blocked"
    if spec.experiment_id in {"exp3269", "exp3272", "exp3273", "exp3279"}:
        return "aggregation-only"
    counts = _row_provenance_counts(spec.experiment_id, payload)
    if counts.get("live_llm_seed", 0) > 0:
        return "live-LLM"
    if counts.get("cached_llm_panel", 0) > 0:
        return "cached"
    if counts.get("template_backed", 0) > 0 or counts.get("garak_deterministic_seed", 0) > 0:
        return "template-backed"
    return "aggregation-only"


def _is_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or bool(_as_list(payload.get("blocked_reasons")))
        or bool(_as_list(payload.get("gate_reasons")))
        or payload.get("garak_redteam_eval_ready") is False
        or payload.get("clean_verifier_rerun_ready") is False
    )


def _row_provenance_counts(experiment_id: str, payload: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if experiment_id == "exp3264":
        seed_count = len(_as_list(payload.get("per_example_labels"))) or _int_value(
            _as_mapping(payload.get("model_specs")).get("target_shard_size")
        )
        if seed_count:
            counts["live_llm_seed"] = seed_count
    for used in _as_list(payload.get("models_used")):
        row = _as_mapping(used)
        count = _int_value(row.get("examples_labeled"))
        role = str(row.get("label_source_role") or "")
        runtime = str(row.get("runtime") or "")
        if count <= 0:
            continue
        if runtime == "llama_cpp" or role == "headline_label_evidence_panel":
            counts["cached_llm_panel"] = counts.get("cached_llm_panel", 0) + count
        elif "garak" in role or runtime == "deterministic_garak_adaptive_seed":
            counts["garak_deterministic_seed"] = counts.get("garak_deterministic_seed", 0) + count
        elif "taxonomy" in role or "deterministic" in runtime:
            counts["template_backed"] = counts.get("template_backed", 0) + count
    return counts


def _label_provenance_totals(provenance: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in provenance.values():
        for key, value in _as_mapping(row.get("row_provenance_counts")).items():
            count = _int_value(value)
            if count:
                totals[key] = totals.get(key, 0) + count
    return dict(sorted(totals.items()))


def _flag_records(records: list[Mapping[str, Any]], kinds: set[str]) -> list[JsonDict]:
    flags: list[JsonDict] = []
    for record in records:
        payload = _as_mapping(record.get("payload"))
        for flag in _as_list(payload.get("corrigendum_pending")):
            flag_map = _as_mapping(flag)
            kind = str(flag_map.get("kind") or "")
            if kind in kinds:
                flags.append(
                    {
                        "experiment_id": str(record.get("experiment_id") or ""),
                        "source_path": str(record.get("path") or ""),
                        "kind": kind,
                        "severity": str(flag_map.get("severity") or ""),
                        "detail": str(flag_map.get("detail") or ""),
                        "duration_s": _float_value(payload.get("duration_s")),
                        "headline_impact": _flag_headline_impact(kind),
                    }
                )
    return flags


def _flag_payload(kind: str, flag: Mapping[str, Any]) -> JsonDict:
    return {
        "kind": kind,
        "severity": str(flag.get("severity") or ""),
        "detail": str(flag.get("detail") or ""),
        "headline_impact": _flag_headline_impact(kind),
    }


def _flag_headline_impact(kind: str) -> str:
    impacts = {
        "DURATION_TOO_SHORT": "blocks live-LLM or compute-bound headline claims from this artifact",
        "TAUTOLOGY": "count equality must be cited as inventory or provisional evidence only",
        "IMPLAUSIBLE_PERFECT": "exact-zero or perfect-equality movement cannot support an improvement headline",
    }
    return impacts.get(kind, "carry forward as bounded methodology evidence")


def _leakage_flags(exp3272: Mapping[str, Any]) -> list[JsonDict]:
    audit = _as_mapping(exp3272.get("leakage_audit"))
    flags: list[JsonDict] = []
    garak_overlap = _int_value(audit.get("garak_template_family_overlap_count"))
    if garak_overlap:
        flags.append(
            {
                "experiment_id": "exp3272",
                "kind": "GARAK_TEMPLATE_FAMILY_OVERLAP_BOUNDED",
                "detail": (
                    f"garak_template_family_overlap_count={garak_overlap}; "
                    f"garak_training_eligible_false={audit.get('garak_training_eligible_false') is True}"
                ),
                "headline_impact": "usable as leakage-boundary evidence only, not detector-performance evidence",
            }
        )
    within_source = _int_value(exp3272.get("within_source_duplicate_count"))
    if within_source:
        flags.append(
            {
                "experiment_id": "exp3272",
                "kind": "WITHIN_SOURCE_DUPLICATES_PRESENT",
                "detail": f"within_source_duplicate_count={within_source}",
                "headline_impact": "carry as corpus-composition sidecar; split leakage audit still passed",
            }
        )
    return flags


def _headline_eligible_metrics(
    payloads: Mapping[str, Mapping[str, Any]], leakage_flags: list[Mapping[str, Any]]
) -> list[JsonDict]:
    exp3272 = _as_mapping(payloads.get("exp3272"))
    exp3279 = _as_mapping(payloads.get("exp3279"))
    leakage = _as_mapping(exp3272.get("leakage_audit"))
    return [
        {
            "metric": "artifact_checksums_available",
            "source_experiment_id": "exp3272",
            "value": bool(_as_mapping(_as_mapping(exp3272.get("checksums")).get("output_files"))),
            "boundary": "integrity claim only; not label-quality or detector-performance evidence",
        },
        {
            "metric": "split_leakage_boundary",
            "source_experiment_id": "exp3272",
            "value": {
                "leakage_audit_passed": exp3272.get("leakage_audit_passed") is True,
                "exact_duplicate_overlap_rows": _overlap_count(leakage, "exact_duplicate_overlap"),
                "near_duplicate_overlap_rows": _overlap_count(leakage, "near_duplicate_overlap"),
                "normal_template_family_overlap_rows": _overlap_count(
                    leakage, "normal_template_family_overlap"
                ),
                "bounded_leakage_sidecars": [flag["kind"] for flag in leakage_flags],
            },
            "boundary": "split leakage boundary only; template-backed label provenance still applies",
        },
        {
            "metric": "paper_ready_false_blocker_state",
            "source_experiment_id": "exp3279",
            "value": {
                "paper_ready": exp3279.get("paper_ready") is True,
                "publication_blocker_count_estimate": _int_value(
                    exp3279.get("publication_blocker_count_estimate")
                ),
                "blocking_rows": _as_list(_as_mapping(exp3279.get("publication_readiness")).get("blocking_rows")),
                "flagged_rows": _as_list(_as_mapping(exp3279.get("publication_readiness")).get("flagged_rows")),
            },
            "boundary": "negative readiness and blocker-state claim only",
        },
    ]


def _provisional_or_sidecar_metrics(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    exp3270 = _as_mapping(payloads.get("exp3270"))
    exp3271 = _as_mapping(payloads.get("exp3271"))
    exp3272 = _as_mapping(payloads.get("exp3272"))
    exp3273 = _as_mapping(payloads.get("exp3273"))
    exp3274 = _as_mapping(payloads.get("exp3274"))
    exp3275 = _as_mapping(payloads.get("exp3275"))
    exp3279 = _as_mapping(payloads.get("exp3279"))
    return [
        {
            "metric": "new_label_count",
            "source_experiment_id": "exp3270+exp3271",
            "value": _int_value(exp3270.get("new_label_count")) + _int_value(exp3271.get("new_label_count")),
            "boundary": "mixed cached-panel plus template-backed labels; not fully live-LLM labeled",
        },
        {
            "metric": "assembled_example_count",
            "source_experiment_id": "exp3272",
            "value": _int_value(exp3272.get("assembled_example_count")),
            "boundary": "operational inventory only because count-equality tautology flags remain open",
        },
        {
            "metric": "full_corpus_auroc",
            "source_experiment_id": "exp3273",
            "value": exp3273.get("full_corpus_auroc"),
            "boundary": "KAN sidecar only; DeLong non-inferiority failed",
        },
        {
            "metric": "full_corpus_auprc",
            "source_experiment_id": "exp3273",
            "value": exp3273.get("full_corpus_auprc"),
            "boundary": "KAN sidecar only; DeLong non-inferiority failed",
        },
        {
            "metric": "garak_gate_passed",
            "source_experiment_id": "exp3274",
            "value": exp3274.get("garak_gate_passed") is True,
            "boundary": "blocked because Garak was unavailable",
        },
        {
            "metric": "clean_verifier_abstention_rate",
            "source_experiment_id": "exp3275",
            "value": exp3275.get("abstention_rate"),
            "boundary": "blocked gate evidence only; n_eval remains a tiny diagnostic panel",
        },
        {
            "metric": "publication_blocker_delta_from_v302",
            "source_experiment_id": "exp3279",
            "value": exp3279.get("publication_blocker_delta_from_v302"),
            "boundary": "exact-zero movement flag means no improvement headline",
        },
    ]


def _downstream_usage_rules() -> JsonDict:
    return {
        "garak": {
            "allowed": True,
            "requires_citation": OUTPUT_REL_PATH.as_posix(),
            "rule": "May use frozen splits as input, but must not call the corpus fully live-LLM labeled and must rerun real Garak after toolchain availability.",
        },
        "kan": {
            "allowed": True,
            "headline_allowed": False,
            "rule": "May use corpus for sidecar/autopsy work only; AUROC/AUPRC remain provisional until leakage-aware non-inferiority and Garak pressure pass.",
        },
        "clean_verifier": {
            "allowed": True,
            "headline_allowed": False,
            "rule": "Use Exp 3275 as blocked abstention evidence only until a calibrated rerun accepts/rejects exact rows without abstain-all behavior.",
        },
        "repair": {
            "allowed": False,
            "headline_allowed": False,
            "rule": "Repair claims remain gated until Garak and clean-verifier gates reopen and the missing repair micro-panel is produced or explicitly skipped.",
        },
        "paper_claims": {
            "headline_performance_metrics_allowed": False,
            "rule": "Only artifact-integrity, leakage-boundary, and blocker-state claims are clean headline material from this corrigendum.",
        },
    }


def _overlap_count(leakage: Mapping[str, Any], key: str) -> int:
    return _int_value(_as_mapping(leakage.get(key)).get("overlap_row_count"))


def _claim_boundary(experiment_id: str, artifact_class: str) -> str:
    if artifact_class == "blocked":
        return "blocked or missing evidence; may only support blocker-state claims"
    boundaries = {
        "exp3264": "prior seed rows have live llama.cpp provenance but are not new .303 labels",
        "exp3270": "tiny cached GGUF panel plus deterministic taxonomy expansion; duration flag blocks fully-live label claims",
        "exp3271": "tiny cached GGUF panel plus deterministic taxonomy and Garak-seed expansion; duration flag blocks fully-live label claims",
        "exp3272": "assembly/leakage ledger only; count tautology flags bound corpus-size claims",
        "exp3273": "KAN result is sidecar-only because non-inferiority failed",
        "exp3279": "evidence matrix is aggregation-only and preserves paper_ready=false",
    }
    return boundaries.get(experiment_id, f"{artifact_class} evidence boundary")


def _artifact_id(payload: Mapping[str, Any], spec: SourceSpec) -> str:
    raw = payload.get("experiment_id", payload.get("experiment", spec.experiment_id))
    return f"exp{raw}" if isinstance(raw, int) else str(raw or spec.experiment_id)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    stable["honest_verdict"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("corrigendum_ready") is not True:
        missing = ",".join(_as_list(artifact.get("missing_required_artifacts")))
        return f"complete: corrigendum_ready=false; missing required source artifacts={missing}"
    return (
        "complete: corrigendum_ready=true; "
        f"duration_flags={len(_as_list(artifact.get('duration_flags')))}; "
        f"tautology_flags={len(_as_list(artifact.get('tautology_flags')))}; "
        "headline_performance_metrics_allowed=false"
    )


def _duration(started_s: float, now_s: float | None = None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - float(started_s)), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
