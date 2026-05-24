"""Exp 3015 offline repair-candidate acceptance controller.

Spec: REQ-CODE-3015, SCENARIO-CODE-3015.

This module turns cached Exp 3003 repair evidence into an inspectable
accept/reject policy. It deliberately uses only deterministic local artifacts:
original and metamorphic verifier results, Exp 3014 taxonomy rows, verifier
logs, and optional telemetry records. No live model call and no LLM judge can
be part of the acceptance decision because the controller is meant to gate a
future live rerun, not become another unverifiable source of authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_NAME = "experiment_3015_cactus_style_repair_acceptance_controller_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
SCHEMA = "carnot.repair_acceptance_controller.v1"
EXP3002_FILENAME = "experiment_3002_metamorphic_repair_oracle_audit_v1.json"
EXP3003_FILENAME = "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json"
EXP3014_FILENAME = "experiment_3014_repair_syntax_schema_failure_taxonomy_v1.json"
EXP3013_FILENAME = "experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json"
RAW_REL_DIR = Path("results/raw") / ARTIFACT_NAME
CONFIG_REL_PATH = RAW_REL_DIR / "controller_config.json"
REJECTED_TABLE_REL_PATH = RAW_REL_DIR / "rejected_candidates.jsonl"
TAXONOMY_TABLE_REL_PATH = (
    Path("results/raw")
    / "experiment_3014_repair_syntax_schema_failure_taxonomy_v1"
    / "taxonomy_table.jsonl"
)
METAMORPHIC_MANIFEST_REL_PATH = (
    Path("results/raw")
    / "experiment_3002_metamorphic_repair_oracle_audit_v1"
    / "metamorphic_manifest.jsonl"
)
INFERENCE_SUBSTRATE = "deterministic_cached_replay_no_live_llm"
FEATURE_NAMES: tuple[str, ...] = (
    "schema_valid",
    "syntax_success",
    "entry_point_present",
    "false_accept_probe_clean",
    "tautology_probe_clean",
    "intent_drift_class",
    "original_passed",
    "metamorphic_passed_all",
    "optional_telemetry_available",
)
GRID_FEATURE_FLAGS: tuple[str, ...] = (
    "require_schema_valid",
    "require_syntax_success",
    "require_entry_point_present",
    "require_false_accept_probe_clean",
    "require_no_intent_drift",
    "require_original_passed",
    "require_metamorphic_passed_all",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "acceptance_controller_ready",
    "controller_config_path",
    "n_candidates_evaluated",
    "false_accept_delta_offline",
    "syntax_failure_delta_offline",
    "schema_failure_delta_offline",
    "pass_at_1_delta_offline",
    "rejected_candidate_table_path",
    "llm_judge_used",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths for deterministic Exp 3015 controller construction."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    controller_config_path: Path | None = None
    rejected_table_path: Path | None = None
    exp3002_artifact_path: Path | None = None
    exp3003_artifact_path: Path | None = None
    exp3014_artifact_path: Path | None = None
    taxonomy_table_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    telemetry_artifact_path: Path | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFunc = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_controller_config_path(self) -> Path:
        return self.controller_config_path or self.repo_root / CONFIG_REL_PATH

    def resolved_rejected_table_path(self) -> Path:
        return self.rejected_table_path or self.repo_root / REJECTED_TABLE_REL_PATH

    def resolved_exp3002_artifact_path(self) -> Path:
        return self.exp3002_artifact_path or self.repo_root / "results" / EXP3002_FILENAME

    def resolved_exp3003_artifact_path(self) -> Path:
        return self.exp3003_artifact_path or self.repo_root / "results" / EXP3003_FILENAME

    def resolved_exp3014_artifact_path(self) -> Path:
        return self.exp3014_artifact_path or self.repo_root / "results" / EXP3014_FILENAME

    def resolved_taxonomy_table_path(self, exp3014: Mapping[str, Any]) -> Path:
        if self.taxonomy_table_path is not None:
            return self.taxonomy_table_path
        return _resolve_repo_path(
            self.repo_root,
            exp3014.get("taxonomy_table_path") or TAXONOMY_TABLE_REL_PATH,
        )

    def resolved_metamorphic_manifest_path(self, exp3002: Mapping[str, Any]) -> Path:
        if self.metamorphic_manifest_path is not None:
            return self.metamorphic_manifest_path
        return _resolve_repo_path(
            self.repo_root,
            exp3002.get("metamorphic_manifest_path") or METAMORPHIC_MANIFEST_REL_PATH,
        )

    def resolved_telemetry_artifact_path(self) -> Path:
        return self.telemetry_artifact_path or self.repo_root / "results" / EXP3013_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 3015 terminal artifact and sidecar audit tables."""

    config = config or ExperimentConfig()
    started = config.start_time()
    exp3002 = _read_json_if_present(config.resolved_exp3002_artifact_path())
    exp3003 = _read_json_if_present(config.resolved_exp3003_artifact_path())
    exp3014 = _read_json_if_present(config.resolved_exp3014_artifact_path())
    taxonomy_path = config.resolved_taxonomy_table_path(exp3014)
    taxonomy_rows = _read_jsonl(taxonomy_path) if taxonomy_path.is_file() else []
    candidate_rows = [row for row in taxonomy_rows if row.get("row_type") == "candidate"]
    if not candidate_rows:
        return _blocked_artifact(config, started, exp3002, exp3003, exp3014, taxonomy_path)

    telemetry = _read_json_if_present(config.resolved_telemetry_artifact_path())
    features = _feature_rows(config, candidate_rows, exp3002, exp3003, telemetry)
    accept_all = _policy_metrics(features, _accept_all_rule())
    conservative = _policy_metrics(features, _conservative_clean_rule())
    search = _search_rules(features, accept_all)
    selected_rule = search["selected_rule"]
    selected_metrics = search["selected_metrics"]
    rejected_rows = _rejected_rows(features, selected_rule)
    config_path = config.resolved_controller_config_path()
    rejected_path = config.resolved_rejected_table_path()
    controller_config = _controller_config(
        selected_rule=selected_rule,
        selected_metrics=selected_metrics,
        accept_all=accept_all,
        conservative=conservative,
        search_evaluated_rule_count=search["evaluated_rule_count"],
        telemetry_available=bool(telemetry),
    )
    _write_json(config_path, controller_config)
    _write_jsonl(rejected_path, rejected_rows)

    deltas = _offline_deltas(selected_metrics, accept_all)
    ready = bool(
        selected_metrics["accepted_count"] > 0
        and _nonpositive(deltas["false_accept_delta_offline"])
        and _nonpositive(deltas["syntax_failure_delta_offline"])
        and _nonpositive(deltas["schema_failure_delta_offline"])
        and _tautology_probe_clean(exp3002, taxonomy_rows)
        and config_path.is_file()
        and rejected_path.is_file()
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "acceptance_controller_ready": ready,
        "controller_config_path": str(_relative_or_absolute(config.repo_root, config_path)),
        "n_candidates_evaluated": len(features),
        "false_accept_delta_offline": deltas["false_accept_delta_offline"],
        "syntax_failure_delta_offline": deltas["syntax_failure_delta_offline"],
        "schema_failure_delta_offline": deltas["schema_failure_delta_offline"],
        "pass_at_1_delta_offline": deltas["pass_at_1_delta_offline"],
        "rejected_candidate_table_path": str(
            _relative_or_absolute(config.repo_root, rejected_path)
        ),
        "llm_judge_used": False,
        "honest_verdict": (
            "complete: offline repair acceptance controller ready"
            if ready
            else "blocked: exp3015 selected controller is not usable"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_inference_run": False,
        "black_box_judge_used": False,
        "selected_rule": selected_rule,
        "selected_policy_metrics": selected_metrics,
        "baseline_policy_metrics": {
            "accept_all": accept_all,
            "conservative_accept_only_clean": conservative,
        },
        "search_evaluated_rule_count": search["evaluated_rule_count"],
        "candidate_feature_rows": features,
        "rejected_candidate_count": len(rejected_rows),
        "tautology_probe_clean": _tautology_probe_clean(exp3002, taxonomy_rows),
        "false_accept_probe_ready": bool(exp3002.get("false_accept_probe_ready")),
        "optional_telemetry_available": bool(telemetry),
        "source_artifacts": _source_artifacts(config, exp3002, exp3003, exp3014, taxonomy_path),
        "controller_config_sha256": _sha256_file(config_path),
        "rejected_candidate_table_sha256": _sha256_file(rejected_path),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 3015 terminal JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    exp3002: Mapping[str, Any],
    exp3003: Mapping[str, Any],
    exp3014: Mapping[str, Any],
    taxonomy_path: Path,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "acceptance_controller_ready": False,
        "controller_config_path": "",
        "n_candidates_evaluated": 0,
        "false_accept_delta_offline": 0.0,
        "syntax_failure_delta_offline": 0.0,
        "schema_failure_delta_offline": 0.0,
        "pass_at_1_delta_offline": 0.0,
        "rejected_candidate_table_path": "",
        "llm_judge_used": False,
        "honest_verdict": "blocked: exp3015 cached evidence unavailable",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_inference_run": False,
        "black_box_judge_used": False,
        "selected_rule": {},
        "selected_policy_metrics": _empty_metrics(),
        "baseline_policy_metrics": {
            "accept_all": _empty_metrics(),
            "conservative_accept_only_clean": _empty_metrics(),
        },
        "search_evaluated_rule_count": 0,
        "candidate_feature_rows": [],
        "rejected_candidate_count": 0,
        "tautology_probe_clean": _tautology_probe_clean(exp3002, []),
        "false_accept_probe_ready": bool(exp3002.get("false_accept_probe_ready")),
        "optional_telemetry_available": config.resolved_telemetry_artifact_path().is_file(),
        "source_artifacts": _source_artifacts(config, exp3002, exp3003, exp3014, taxonomy_path),
        "controller_config_sha256": None,
        "rejected_candidate_table_sha256": None,
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _feature_rows(
    config: ExperimentConfig,
    taxonomy_candidates: Sequence[Mapping[str, Any]],
    exp3002: Mapping[str, Any],
    exp3003: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> list[JsonDict]:
    exp3003_index = _candidate_index(exp3003.get("candidate_evaluations") or [])
    telemetry_models = {
        str(row.get("hf_id") or "")
        for row in telemetry.get("headline_models_attempted") or []
        if row.get("telemetry_observation")
    }
    out: list[JsonDict] = []
    for row in taxonomy_candidates:
        key = _candidate_key(row)
        exp3003_row = exp3003_index.get(key, {})
        verifier = _read_json_if_present(
            _resolve_repo_path(config.repo_root, row.get("verifier_log_path"))
        )
        original_passed = bool(row.get("original_passed"))
        metamorphic_passed = bool(row.get("metamorphic_passed_all"))
        false_accept = bool(row.get("false_accept"))
        intent_drift = bool(row.get("intent_drift"))
        model_id = str(row.get("model_hf_id") or exp3003_row.get("model_hf_id") or "")
        out.append(
            {
                "item_id": str(row.get("item_id") or ""),
                "candidate_sha256": str(row.get("candidate_sha256") or ""),
                "schema_valid": bool(row.get("schema_valid")),
                "syntax_success": bool(row.get("syntax_success")),
                "entry_point_present": bool(row.get("entry_point_present")),
                "false_accept_probe_clean": bool(
                    exp3002.get("false_accept_probe_ready") is True and not false_accept
                ),
                "tautology_probe_clean": _tautology_probe_clean(exp3002, []),
                "intent_drift_class": "intent_drift" if intent_drift else "clean",
                "original_passed": original_passed,
                "metamorphic_passed_all": metamorphic_passed,
                "passed": bool(original_passed and metamorphic_passed),
                "false_accept": false_accept,
                "intent_drift": intent_drift,
                "primary_root_cause": str(row.get("primary_root_cause") or ""),
                "failure_modes": list(row.get("failure_modes") or []),
                "metamorphic_variant_count": int(row.get("metamorphic_variant_count") or 0),
                "verifier_log_path": str(row.get("verifier_log_path") or ""),
                "verifier_log_loaded": bool(verifier),
                "verifier_passed": bool(verifier.get("passed", exp3003_row.get("passed", False))),
                "optional_telemetry_available": bool(model_id and model_id in telemetry_models),
                "model_hf_id": model_id,
            }
        )
    return out


def _candidate_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        indexed[_candidate_key(row)] = row
    return indexed


def _candidate_key(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_sha256") or row.get("item_id") or "")


def _search_rules(features: Sequence[Mapping[str, Any]], accept_all: Mapping[str, Any]) -> JsonDict:
    evaluated: list[JsonDict] = []
    for rule in _grid_rules():
        metrics = _policy_metrics(features, rule)
        evaluated.append({"rule": rule, "metrics": metrics})
    selected = max(evaluated, key=lambda item: _rule_score(item, accept_all))
    return {
        "selected_rule": selected["rule"],
        "selected_metrics": selected["metrics"],
        "evaluated_rule_count": len(evaluated),
    }


def _grid_rules() -> list[JsonDict]:
    rules: list[JsonDict] = []
    for bits in itertools.product((False, True), repeat=len(GRID_FEATURE_FLAGS)):
        rule = dict(zip(GRID_FEATURE_FLAGS, bits, strict=True))
        rule["require_tautology_probe_clean"] = True
        rules.append(rule)
    return rules


def _rule_score(entry: Mapping[str, Any], accept_all: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = entry["metrics"]
    rule = entry["rule"]
    safety_ok = (
        metrics["false_accept_rate"] <= accept_all["false_accept_rate"]
        and metrics["syntax_failure_rate"] <= accept_all["syntax_failure_rate"]
        and metrics["schema_failure_rate"] <= accept_all["schema_failure_rate"]
    )
    safety_requirements = sum(1 for value in rule.values() if value)
    return (
        int(safety_ok),
        int(metrics["accepted_count"] > 0),
        round(metrics["pass_at_1"] - accept_all["pass_at_1"], 12),
        -metrics["false_accept_rate"],
        -metrics["syntax_failure_rate"],
        -metrics["schema_failure_rate"],
        safety_requirements,
        metrics["accepted_count"],
    )


def _accept_all_rule() -> JsonDict:
    return {flag: False for flag in (*GRID_FEATURE_FLAGS, "require_tautology_probe_clean")}


def _conservative_clean_rule() -> JsonDict:
    return {flag: True for flag in (*GRID_FEATURE_FLAGS, "require_tautology_probe_clean")}


def _policy_metrics(features: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> JsonDict:
    accepted = [row for row in features if _accepts(row, rule)]
    return {
        "candidate_count": len(features),
        "accepted_count": len(accepted),
        "rejected_count": len(features) - len(accepted),
        "accepted_item_ids": [str(row.get("item_id") or "") for row in accepted],
        "pass_at_1": _rate(accepted, lambda row: row.get("passed") is True),
        "false_accept_rate": _rate(accepted, lambda row: row.get("false_accept") is True),
        "syntax_failure_rate": _rate(accepted, lambda row: row.get("syntax_success") is False),
        "schema_failure_rate": _rate(accepted, lambda row: row.get("schema_valid") is False),
        "tautology_exposure_rate": _rate(
            accepted, lambda row: row.get("tautology_probe_clean") is False
        ),
    }


def _accepts(row: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    return not _rejection_reasons(row, rule)


def _rejected_rows(
    features: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in features:
        reasons = _rejection_reasons(row, rule)
        if reasons:
            rows.append(
                {
                    "item_id": str(row.get("item_id") or ""),
                    "candidate_sha256": str(row.get("candidate_sha256") or ""),
                    "rejection_reasons": reasons,
                    "primary_root_cause": str(row.get("primary_root_cause") or ""),
                    "original_passed": bool(row.get("original_passed")),
                    "metamorphic_passed_all": bool(row.get("metamorphic_passed_all")),
                    "false_accept": bool(row.get("false_accept")),
                }
            )
    return rows


def _rejection_reasons(row: Mapping[str, Any], rule: Mapping[str, Any]) -> list[str]:
    checks = [
        ("require_schema_valid", "schema_valid", row.get("schema_valid") is True),
        ("require_syntax_success", "syntax_success", row.get("syntax_success") is True),
        (
            "require_entry_point_present",
            "entry_point_present",
            row.get("entry_point_present") is True,
        ),
        (
            "require_false_accept_probe_clean",
            "false_accept",
            row.get("false_accept_probe_clean") is True,
        ),
        (
            "require_no_intent_drift",
            "intent_drift",
            row.get("intent_drift_class") == "clean",
        ),
        ("require_original_passed", "original_passed", row.get("original_passed") is True),
        (
            "require_metamorphic_passed_all",
            "metamorphic_passed_all",
            row.get("metamorphic_passed_all") is True,
        ),
        (
            "require_tautology_probe_clean",
            "tautology_probe_clean",
            row.get("tautology_probe_clean") is True,
        ),
    ]
    return [reason for flag, reason, passed in checks if rule.get(flag) and not passed]


def _controller_config(
    *,
    selected_rule: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    accept_all: Mapping[str, Any],
    conservative: Mapping[str, Any],
    search_evaluated_rule_count: int,
    telemetry_available: bool,
) -> JsonDict:
    return {
        "policy_type": "transparent_grid_rule",
        "selected_rule": dict(selected_rule),
        "feature_names": list(FEATURE_NAMES),
        "selection_objective": [
            "do_not_increase_false_accept_rate",
            "do_not_increase_syntax_failure_rate",
            "do_not_increase_schema_failure_rate",
            "maximize_accepted_pass_rate",
            "prefer_more_explicit_safety_requirements_on_ties",
        ],
        "baseline_policy_metrics": {
            "accept_all": dict(accept_all),
            "conservative_accept_only_clean": dict(conservative),
        },
        "selected_policy_metrics": dict(selected_metrics),
        "search_evaluated_rule_count": search_evaluated_rule_count,
        "optional_telemetry_available": telemetry_available,
        "llm_judge_used": False,
    }


def _offline_deltas(selected: Mapping[str, Any], accept_all: Mapping[str, Any]) -> JsonDict:
    return {
        "false_accept_delta_offline": _delta(
            selected["false_accept_rate"], accept_all["false_accept_rate"]
        ),
        "syntax_failure_delta_offline": _delta(
            selected["syntax_failure_rate"], accept_all["syntax_failure_rate"]
        ),
        "schema_failure_delta_offline": _delta(
            selected["schema_failure_rate"], accept_all["schema_failure_rate"]
        ),
        "pass_at_1_delta_offline": _delta(selected["pass_at_1"], accept_all["pass_at_1"]),
    }


def _tautology_probe_clean(
    exp3002: Mapping[str, Any], taxonomy_rows: Sequence[Mapping[str, Any]]
) -> bool:
    rejected = exp3002.get("rejected_variants") or []
    taxonomy_has_tautology = any(row.get("failure_mode") == "tautology" for row in taxonomy_rows)
    return bool(
        exp3002.get("tautology_probe_ready") is True
        and (
            taxonomy_has_tautology
            or any(row.get("reason") == "tautological_oracle_rejected" for row in rejected)
        )
    )


def _source_artifacts(
    config: ExperimentConfig,
    exp3002: Mapping[str, Any],
    exp3003: Mapping[str, Any],
    exp3014: Mapping[str, Any],
    taxonomy_path: Path,
) -> list[JsonDict]:
    paths = [
        config.resolved_exp3002_artifact_path(),
        config.resolved_metamorphic_manifest_path(exp3002),
        config.resolved_exp3003_artifact_path(),
        config.resolved_exp3014_artifact_path(),
        taxonomy_path,
        config.resolved_telemetry_artifact_path(),
    ]
    loaded_by_name = {
        EXP3002_FILENAME: bool(exp3002),
        EXP3003_FILENAME: bool(exp3003),
        EXP3014_FILENAME: bool(exp3014),
    }
    return [
        {
            "path": str(_relative_or_absolute(config.repo_root, path)),
            "present": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "artifact_loaded": loaded_by_name.get(path.name),
        }
        for path in paths
    ]


def _empty_metrics() -> JsonDict:
    return {
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted_item_ids": [],
        "pass_at_1": 0.0,
        "false_accept_rate": 0.0,
        "syntax_failure_rate": 0.0,
        "schema_failure_rate": 0.0,
        "tautology_exposure_rate": 0.0,
    }


def _rate(
    rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 12)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 12)


def _nonpositive(value: float) -> bool:
    return float(value) <= 0.0


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


__all__ = [
    "ARTIFACT_FILENAME",
    "EXP3002_FILENAME",
    "EXP3003_FILENAME",
    "EXP3014_FILENAME",
    "METAMORPHIC_MANIFEST_REL_PATH",
    "TAXONOMY_TABLE_REL_PATH",
    "ExperimentConfig",
    "build_artifact",
    "write_artifact",
]
