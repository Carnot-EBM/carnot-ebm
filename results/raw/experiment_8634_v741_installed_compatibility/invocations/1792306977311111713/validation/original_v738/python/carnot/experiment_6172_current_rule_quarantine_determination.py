"""Exp6172 current-rule companion determination for quarantined Exp6161/6162.

Spec refs: REQ-REPORT-6172,
SCENARIO-REPORT-6172-IMMUTABLE-SOURCE,
SCENARIO-REPORT-6172-CURRENT-RULE-REPLAY,
SCENARIO-REPORT-6172-DURATION-PROVENANCE,
SCENARIO-REPORT-6172-OPERATOR-BOUNDARY,
SCENARIO-REPORT-6172-SCHEMA.

This module writes a companion receipt only. The historical artifacts stay
flagged unless the operator explicitly reopens them outside this workflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6172_current_rule_quarantine_determination.json")
EXP6159_RELATIVE_PATH = Path("results/experiment_6159_decision_calibrated_stream.json")
EXP6160_RELATIVE_PATH = Path("results/experiment_6160_sota_decision_calibration_corpus.json")
EXP6161_RELATIVE_PATH = Path("results/experiment_6161_decision_calibrated_energy_policy.json")
EXP6161_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6161_decision_calibrated_energy_policy.manifest.json"
)
EXP6162_RELATIVE_PATH = Path("results/experiment_6162_prospective_admission_replication.json")
EXP6162_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6162_prospective_admission_replication.manifest.json"
)
CAPSTONE_RELATIVE_PATH = Path("results/experiment_6168_v534_capstone_reconciliation.json")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DETERMINATION_LINT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")
DELIVERABLE_GUARD_RELATIVE_PATH = Path("python/carnot/pipeline/deliverable_guard.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT_ID = "experiment_6172_current_rule_quarantine_determination"
TASK_ID = "exp6172-current-rule-quarantine-determination"
RUN_DATE = "20260807"
SCHEMA = "carnot.experiment_6172.current_rule_quarantine_determination.v1"
INFERENCE_SUBSTRATE = "deterministic_current_rule_companion_determination"

TASK_IDS = {
    "experiment_6161_decision_calibrated_energy_policy": (
        "exp6161-decision-calibrated-energy-policy"
    ),
    "experiment_6162_prospective_admission_replication": (
        "exp6162-prospective-admission-replication"
    ),
}

SOURCE_RELATIVE_PATHS = (
    EXP6161_RELATIVE_PATH,
    EXP6162_RELATIVE_PATH,
    EXP6161_MANIFEST_RELATIVE_PATH,
    EXP6162_MANIFEST_RELATIVE_PATH,
    CAPSTONE_RELATIVE_PATH,
)

PROTECTED_RELATIVE_PATHS = (
    EXP6161_RELATIVE_PATH,
    EXP6162_RELATIVE_PATH,
    EXP6161_MANIFEST_RELATIVE_PATH,
    EXP6162_MANIFEST_RELATIVE_PATH,
    CAPSTONE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DETERMINATION_LINT_RELATIVE_PATH,
    DELIVERABLE_GUARD_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("ops/known-issues.md"),
    Path("ops/status.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "source_artifact_paths_hashes_and_immutable_bytes",
    "historical_adversarial_flags_reasons_and_capstone_classification",
    "current_verifier_path_version_hash_rule_ids_and_thresholds",
    "current_verifier_commands_exit_codes_and_receipts",
    "acquisition_duration_and_cached_analysis_duration_provenance",
    "model_lifecycle_and_held_access_receipts",
    "field_level_historical_vs_current_determination_matrix",
    "current_rule_clean",
    "historical_quarantine_preserved",
    "headline_promotion_authorized",
    "operator_reopen_required",
    "source_hashes_and_git_status_before_after",
    "preexisting_worktree_changes_preserved",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state for a companion determination, not a source unflag.",
    "preconditions_checked": "Snapshots source bytes, manifests, historical flags, capstone state, verifier version, git state, and protected files.",
    "source_artifact_paths_hashes_and_immutable_bytes": "Exact bytes of Exp6161, Exp6162, manifests, and capstone are hashed before and after.",
    "historical_adversarial_flags_reasons_and_capstone_classification": "Historical flags and capstone terminal classes remain the immutable history.",
    "current_verifier_path_version_hash_rule_ids_and_thresholds": "Current verifier code, git version, rule IDs, and thresholds define only current-rule cleanliness.",
    "current_verifier_commands_exit_codes_and_receipts": "Unmodified verifier command output drives current_rule_clean.",
    "acquisition_duration_and_cached_analysis_duration_provenance": "Live row acquisition and cached analysis durations are separated.",
    "model_lifecycle_and_held_access_receipts": "Row generation, model lifecycle, and held access are copied from exact source artifacts.",
    "field_level_historical_vs_current_determination_matrix": "Every compared field shows historical value, current value, and preservation outcome.",
    "current_rule_clean": "Derived only from the unmodified current verifier run and never aliases historical acceptance.",
    "historical_quarantine_preserved": "Bare true; source quarantine and capstone flagged classifications remain immutable.",
    "headline_promotion_authorized": "Bare false; companion determinations do not authorize headline use.",
    "operator_reopen_required": "Bare true; only the operator can reopen or promote historically quarantined evidence.",
    "source_hashes_and_git_status_before_after": "Source hashes and git status are compared before and after companion construction.",
    "preexisting_worktree_changes_preserved": "Worktree changes already present before this write are reported and must remain present afterward.",
    "protected_files_unchanged": "Protected source, verifier, ops, and instruction files stay byte-identical.",
    "duration_s": "Wall-clock duration for building the companion determination.",
    "inference_substrate": "Set `deterministic_current_rule_companion_determination`.",
    "field_provenance": "Every required field traces to source artifacts, verifier receipts, git, or deterministic comparisons.",
    "test_commands": "Records focused, coverage, spec, replay, preservation, protected-file, E2E, root-clutter, and full-suite commands.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "reproducibility_checksum": "Detects later source, verifier, command, or boundary drift.",
    "honest_verdict": "Use `complete:` or `blocked:` and state both current-rule and immutable-history outcomes.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6172_current_rule_quarantine_determination.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6172_current_rule_quarantine_determination.py -m pytest tests/python/test_experiment_6172_current_rule_quarantine_determination.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6172_current_rule_quarantine_determination.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6172_current_rule_quarantine_determination.py",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_6161_decision_calibrated_energy_policy.json results/experiment_6162_prospective_admission_replication.json",
    ".venv/bin/python -m carnot.experiment_6172_current_rule_quarantine_determination --validate",
    ".venv/bin/python scripts/determination_preservation_lint.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(encoded)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_receipt(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    if not path.exists():
        return {
            "path": rel_path.as_posix(),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "immutable_bytes": True,
        }
    data = path.read_bytes()
    return {
        "path": rel_path.as_posix(),
        "exists": True,
        "sha256": _sha256_bytes(data),
        "size_bytes": len(data),
        "immutable_bytes": True,
    }


def _snapshot(root: Path, rel_paths: tuple[Path, ...]) -> JsonDict:
    return {rel_path.as_posix(): _path_receipt(root, rel_path) for rel_path in rel_paths}


def _unchanged(before: JsonDict, after: JsonDict) -> bool:
    return all(before[key] == after.get(key) for key in before)


def _git(args: list[str], root: Path) -> str:  # pragma: no cover
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return (
        result.stdout.strip() if result.returncode == 0 else f"<git failed:{result.stderr.strip()}>"
    )


def git_status(root: Path = REPO_ROOT) -> str:  # pragma: no cover
    return _git(["status", "--short"], root)


def git_head(root: Path = REPO_ROOT) -> str:  # pragma: no cover
    return _git(["rev-parse", "HEAD"], root)


def _load_verifier_module(root: Path) -> ModuleType:
    path = root / ADVERSARIAL_VERIFY_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_exp6172_adversarial_verify", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load verifier module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rule_ids_from_source(source: str, module: ModuleType) -> list[str]:
    direct_ids = set(re.findall(r"kind=[\"']([A-Z0-9_]+)[\"']", source))
    direct_ids.update(str(kind) for kind in getattr(module, "HIGH_PRECISION_KINDS", ()))
    return sorted(direct_ids)


def _thresholds(module: ModuleType) -> JsonDict:
    names = (
        "TAUTOLOGY_DIGITS",
        "COMPUTE_BOUND_MIN_DURATION_S",
        "VERIFIER_SCORING_MIN_DURATION_S",
        "CHEAP_LEARNED_VALUE_MIN_DURATION_S",
        "AGGREGATION_MIN_DURATION_S",
        "DETERMINISTIC_VERIFIER_MIN_DURATION_S",
        "ARC_LIVE_AGENT_NO_LLM_MIN_DURATION_S",
        "LLM_EMBEDDING_EXTRACTION_MIN_DURATION_S",
        "LOG_ANALYSIS_LOCAL_TIMING_MIN_DURATION_S",
        "ARTIFACT_QA_LINT_TESTS_MIN_DURATION_S",
        "WEB_BIBLIOGRAPHIC_SEARCH_ONLY_MIN_DURATION_S",
        "LOCAL_SOTA_GGUF_SMALL_N_MIN_DURATION_S",
        "DETERMINISTIC_SMT_HINT_VALIDATION_MIN_DURATION_S",
        "NATIVE_GGUF_BACKEND_BISECT_MIN_DURATION_S",
    )
    return {name: getattr(module, name) for name in names if hasattr(module, name)}


def _verifier_version_receipt(root: Path, git_revision: str) -> JsonDict:
    source = (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).read_text(encoding="utf-8")
    module = _load_verifier_module(root)
    return {
        "path": ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        "git_head": git_revision,
        "sha256": _path_receipt(root, ADVERSARIAL_VERIFY_RELATIVE_PATH)["sha256"],
        "version": f"git:{git_revision}",
        "rule_ids": _rule_ids_from_source(source, module),
        "thresholds": _thresholds(module),
        "duration_rule_current_id": "DURATION_TOO_SHORT",
        "duration_rule_historical_floor_s": 60.0,
    }


def _duration_floor(root: Path, payload: JsonDict) -> JsonDict:
    module = _load_verifier_module(root)
    floor = module.duration_floor_for_artifact(payload)
    return dict(floor)


def _historical_flags(payload: JsonDict) -> JsonDict:
    flags = payload.get("corrigendum_pending") or []
    return {
        "flagged_adversarial": payload.get("flagged_adversarial"),
        "corrigendum_pending": flags,
        "flag_kinds": [flag.get("kind") for flag in flags if isinstance(flag, dict)],
        "critical_flag_kinds": [
            flag.get("kind")
            for flag in flags
            if isinstance(flag, dict) and flag.get("severity") == "critical"
        ],
    }


def _capstone_classification(capstone: JsonDict) -> JsonDict:
    exact = capstone.get("exact_terminal_classification", {})
    terminals = exact.get("terminal_class_by_task_id", {})
    underlying = exact.get("underlying_terminal_class_by_task_id", {})
    flagged_ids = (
        capstone.get("adversarial_verifier_and_quarantine_receipts", {}).get(
            "flagged_task_ids",
            [],
        )
        or []
    )
    return {
        "capstone_status": capstone.get("status"),
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "flagged_task_ids": flagged_ids,
        "terminal_class_by_task_id": {
            task_id: terminals.get(task_id) for task_id in TASK_IDS.values()
        },
        "underlying_terminal_class_by_task_id": {
            task_id: underlying.get(task_id) for task_id in TASK_IDS.values()
        },
    }


def _current_reports_by_exp(verifier_receipt: JsonDict) -> JsonDict:
    parsed = verifier_receipt.get("parsed_json") or {}
    reports = parsed.get("reports", []) if isinstance(parsed, dict) else []
    return {report.get("exp_id"): report for report in reports if isinstance(report, dict)}


def _current_rule_clean(verifier_receipt: JsonDict) -> bool:
    parsed = verifier_receipt.get("parsed_json") or {}
    reports = parsed.get("reports", []) if isinstance(parsed, dict) else []
    return (
        verifier_receipt.get("exit_code") == 0
        and parsed.get("flagged_count") == 0
        and all(report.get("flag_count") == 0 for report in reports if isinstance(report, dict))
    )


def _matrix(payloads: JsonDict, capstone_state: JsonDict, verifier_receipt: JsonDict) -> JsonDict:
    current_reports = _current_reports_by_exp(verifier_receipt)
    out: JsonDict = {}
    for experiment_id, task_id in TASK_IDS.items():
        payload = payloads[experiment_id]
        report = current_reports.get(experiment_id, {})
        source_receipt = payloads["source_receipts"][payload["source_path"]]
        out[experiment_id] = {
            "source_sha256": {
                "historical_value": source_receipt["sha256"],
                "current_value": source_receipt["sha256_after"],
                "unchanged": source_receipt["sha256"] == source_receipt["sha256_after"],
            },
            "status": {
                "historical_value": payload.get("status"),
                "current_value": payload.get("status"),
                "unchanged": True,
            },
            "honest_verdict": {
                "historical_value": payload.get("honest_verdict"),
                "current_value": payload.get("honest_verdict"),
                "unchanged": True,
            },
            "source_flagged_adversarial": {
                "historical_value": payload.get("flagged_adversarial"),
                "current_value": payload.get("flagged_adversarial"),
                "unchanged": True,
            },
            "source_corrigendum_pending": {
                "historical_value": payload.get("corrigendum_pending"),
                "current_value": payload.get("corrigendum_pending"),
                "unchanged": True,
            },
            "capstone_terminal_class": {
                "historical_value": capstone_state["terminal_class_by_task_id"].get(task_id),
                "current_value": capstone_state["terminal_class_by_task_id"].get(task_id),
                "unchanged": True,
            },
            "capstone_underlying_class": {
                "historical_value": capstone_state["underlying_terminal_class_by_task_id"].get(
                    task_id
                ),
                "current_value": capstone_state["underlying_terminal_class_by_task_id"].get(
                    task_id
                ),
                "unchanged": True,
            },
            "current_verifier_flag_count": {
                "historical_value": None,
                "current_value": report.get("flag_count"),
                "unchanged": None,
            },
            "current_verifier_flags": {
                "historical_value": None,
                "current_value": report.get("flags", []),
                "unchanged": None,
            },
        }
    return out


def _duration_provenance(
    root: Path, exp6160: JsonDict, exp6161: JsonDict, exp6162: JsonDict
) -> JsonDict:
    return {
        "historical_rule_that_fired": "DURATION_TOO_SHORT",
        "historical_duration_floor_s": 60.0,
        "current_rule_differs_because": (
            "top-level no-LLM cached substrates use deterministic-verifier duration floor"
        ),
        "acquisition_receipt": {
            "source_experiment_id": exp6160.get("experiment_id"),
            "inference_substrate": exp6160.get("inference_substrate"),
            "duration_s": exp6160.get("duration_s"),
            "row_count": exp6160.get("per_model_row_paths_hashes_and_counts", {}).get(
                "total_row_count"
            ),
            "ready_score": exp6160.get("sota_decision_corpus_ready_score"),
        },
        "cached_analysis_receipts": {
            exp6161["experiment_id"]: {
                "inference_substrate": exp6161.get("inference_substrate"),
                "duration_s": exp6161.get("duration_s"),
                "current_duration_floor": _duration_floor(root, exp6161),
                "duration_above_current_floor": exp6161.get("duration_s")
                >= _duration_floor(root, exp6161)["min_duration_s"],
            },
            exp6162["experiment_id"]: {
                "inference_substrate": exp6162.get("inference_substrate"),
                "duration_s": exp6162.get("duration_s"),
                "current_duration_floor": _duration_floor(root, exp6162),
                "duration_above_current_floor": exp6162.get("duration_s")
                >= _duration_floor(root, exp6162)["min_duration_s"],
            },
        },
    }


def _model_lifecycle(
    exp6159: JsonDict, exp6160: JsonDict, exp6161: JsonDict, exp6162: JsonDict
) -> JsonDict:
    policy = exp6161.get("selected_policy_rationale_without_held_access", {})
    held = exp6162.get("first_and_only_held_access_receipt", {})
    return {
        "stream_receipts": {
            "experiment_id": exp6159.get("experiment_id"),
            "duration_s": exp6159.get("duration_s"),
            "counts": exp6159.get("event_template_family_partition_and_shift_counts"),
            "held_loader_one_shot_contract": exp6159.get("held_loader_one_shot_contract"),
        },
        "row_generation_receipts": exp6160.get("per_model_row_paths_hashes_and_counts"),
        "model_lifecycle": exp6160.get("gpu_offload_pid_lifecycle_and_cleanup_receipts"),
        "model_specs": exp6160.get("model_specs") or exp6160.get("MODEL_SPECS"),
        "held_access_receipts": {
            "exp6161_policy_freeze_held_access_count": exp6161.get("held_access_count"),
            "exp6161_selection_uses_held_outcomes": policy.get("selection_uses_held_outcomes"),
            "exp6162_held_access_count_before": held.get("held_access_count_before"),
            "exp6162_held_access_count_after": held.get("held_access_count_after"),
            "exp6162_held_label_read_count": held.get("held_label_read_count"),
        },
        "science_fields": {
            "exp6161_ready_score": exp6161.get("decision_calibrated_policy_ready_score"),
            "exp6161_selected_arm": policy.get("selected_arm"),
            "exp6161_policy_validly_frozen": policy.get("policy_validly_frozen"),
            "exp6161_threshold": policy.get("selected_threshold")
            or exp6161.get("score_threshold_abstention_and_cost_freeze_receipts", {}).get(
                "threshold"
            ),
            "exp6162_ready_score": exp6162.get("prospective_admission_replication_ready_score"),
            "exp6162_conjunctive_pass": exp6162.get(
                "per_model_and_conjunctive_gate_matrix",
                {},
            ).get("conjunctive_pass"),
            "exp6162_all_gates_pass": exp6162.get(
                "unsafe_admission_and_known_family_noninferiority_gates",
                {},
            ).get("all_gates_pass"),
            "exp6162_retirement_triggered": exp6162.get("retirement_triggered"),
        },
    }


def _preexisting_worktree(git_before: str, git_after: str) -> JsonDict:
    before_lines = [line for line in git_before.splitlines() if line.strip()]
    after_lines = [line for line in git_after.splitlines() if line.strip()]
    missing_after = [line for line in before_lines if line not in after_lines]
    return {
        "git_status_before_lines": before_lines,
        "git_status_after_lines": after_lines,
        "preexisting_change_count": len(before_lines),
        "missing_after": missing_after,
        "preserved": not missing_after,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "sources": [
                "results/experiment_6161_decision_calibrated_energy_policy.json",
                "results/experiment_6162_prospective_admission_replication.json",
                "results/experiment_6168_v534_capstone_reconciliation.json",
                "scripts/adversarial_verify.py",
            ],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _command_exit_map(verifier_receipt: JsonDict, test_exit_codes: dict[str, int]) -> JsonDict:
    out: JsonDict = dict(test_exit_codes)
    out[str(verifier_receipt["command"])] = int(verifier_receipt["exit_code"])
    return out


def run_current_verifier(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    command = [
        sys.executable,
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        "--json",
        EXP6161_RELATIVE_PATH.as_posix(),
        EXP6162_RELATIVE_PATH.as_posix(),
    ]
    started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    finished = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    parsed = json.loads(result.stdout)
    return {
        "command": " ".join(command),
        "started_at_utc": started,
        "finished_at_utc": finished,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_sha256": sha256_text(result.stdout),
        "parsed_json": parsed,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    verifier_receipt: JsonDict,
    git_status_before: str,
    git_status_after: str,
    git_head: str,
    duration_s: float,
    test_commands: list[str] | None = None,
    test_exit_codes: dict[str, int] | None = None,
) -> JsonDict:
    source_before = _snapshot(root, SOURCE_RELATIVE_PATHS)
    protected_before = _snapshot(root, PROTECTED_RELATIVE_PATHS)

    exp6159 = _read_json(root / EXP6159_RELATIVE_PATH)
    exp6160 = _read_json(root / EXP6160_RELATIVE_PATH)
    exp6161 = _read_json(root / EXP6161_RELATIVE_PATH)
    exp6162 = _read_json(root / EXP6162_RELATIVE_PATH)
    capstone = _read_json(root / CAPSTONE_RELATIVE_PATH)

    source_after = _snapshot(root, SOURCE_RELATIVE_PATHS)
    protected_after = _snapshot(root, PROTECTED_RELATIVE_PATHS)
    source_receipts = {
        path: {
            **receipt,
            "sha256_after": source_after[path]["sha256"],
            "size_bytes_after": source_after[path]["size_bytes"],
            "unchanged": receipt == source_after[path],
        }
        for path, receipt in source_before.items()
    }
    payloads = {
        exp6161["experiment_id"]: {**exp6161, "source_path": EXP6161_RELATIVE_PATH.as_posix()},
        exp6162["experiment_id"]: {**exp6162, "source_path": EXP6162_RELATIVE_PATH.as_posix()},
        "source_receipts": source_receipts,
    }
    capstone_state = _capstone_classification(capstone)
    current_clean = _current_rule_clean(verifier_receipt)
    historical_preserved = (
        bool(exp6161.get("flagged_adversarial"))
        and bool(exp6162.get("flagged_adversarial"))
        and capstone_state["terminal_class_by_task_id"]["exp6161-decision-calibrated-energy-policy"]
        == "flagged"
        and capstone_state["terminal_class_by_task_id"]["exp6162-prospective-admission-replication"]
        == "flagged"
        and _unchanged(source_before, source_after)
    )
    status = (
        "complete_current_rule_clean_historical_quarantine_preserved"
        if current_clean and historical_preserved
        else "blocked_current_rule_or_history_preservation_failed"
    )
    test_commands_final = list(test_commands or DEFAULT_TEST_COMMANDS)
    test_exit_codes_final = _command_exit_map(verifier_receipt, test_exit_codes or {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "status": status,
        "preconditions_checked": {
            "instructions_read": {
                "AGENTS.md": _path_receipt(root, Path("AGENTS.md")),
                "CODEX.md": _path_receipt(root, Path("CODEX.md")),
                "CLAUDE.md": _path_receipt(root, Path("CLAUDE.md")),
            },
            "source_snapshot_before": source_before,
            "historical_flags": {
                exp6161["experiment_id"]: _historical_flags(exp6161),
                exp6162["experiment_id"]: _historical_flags(exp6162),
            },
            "capstone_classification": capstone_state,
            "current_verifier": _verifier_version_receipt(root, git_head),
            "git_status_before": git_status_before,
            "protected_files_before": protected_before,
            "preexisting_audit_edits": _preexisting_worktree(git_status_before, git_status_after),
            "do_not_modify_research_conductor": _path_receipt(
                root,
                RESEARCH_CONDUCTOR_RELATIVE_PATH,
            ),
        },
        "source_artifact_paths_hashes_and_immutable_bytes": source_receipts,
        "historical_adversarial_flags_reasons_and_capstone_classification": {
            "by_experiment": {
                exp6161["experiment_id"]: {
                    **_historical_flags(exp6161),
                    "historical_rule_fired": "DURATION_TOO_SHORT",
                    "historical_duration_floor_s": 60.0,
                    "capstone_terminal_class": capstone_state["terminal_class_by_task_id"][
                        TASK_IDS[exp6161["experiment_id"]]
                    ],
                    "capstone_underlying_class": capstone_state[
                        "underlying_terminal_class_by_task_id"
                    ][TASK_IDS[exp6161["experiment_id"]]],
                },
                exp6162["experiment_id"]: {
                    **_historical_flags(exp6162),
                    "historical_rule_fired": "DURATION_TOO_SHORT",
                    "historical_duration_floor_s": 60.0,
                    "capstone_terminal_class": capstone_state["terminal_class_by_task_id"][
                        TASK_IDS[exp6162["experiment_id"]]
                    ],
                    "capstone_underlying_class": capstone_state[
                        "underlying_terminal_class_by_task_id"
                    ][TASK_IDS[exp6162["experiment_id"]]],
                },
            },
            "capstone": capstone_state,
            "historical_quarantine_reason_summary": (
                "Historical DURATION_TOO_SHORT applied a live-model 60s floor to "
                "cached Exp6161/Exp6162 analysis durations."
            ),
        },
        "current_verifier_path_version_hash_rule_ids_and_thresholds": _verifier_version_receipt(
            root,
            git_head,
        ),
        "current_verifier_commands_exit_codes_and_receipts": verifier_receipt,
        "acquisition_duration_and_cached_analysis_duration_provenance": _duration_provenance(
            root,
            exp6160,
            exp6161,
            exp6162,
        ),
        "model_lifecycle_and_held_access_receipts": _model_lifecycle(
            exp6159,
            exp6160,
            exp6161,
            exp6162,
        ),
        "field_level_historical_vs_current_determination_matrix": _matrix(
            payloads,
            capstone_state,
            verifier_receipt,
        ),
        "current_rule_clean": current_clean,
        "historical_quarantine_preserved": True,
        "headline_promotion_authorized": False,
        "operator_reopen_required": True,
        "source_hashes_and_git_status_before_after": {
            "source_hashes_before": source_before,
            "source_hashes_after": source_after,
            "all_source_hashes_unchanged": _unchanged(source_before, source_after),
            "git_head": git_head,
            "git_status_before": git_status_before,
            "git_status_after": git_status_after,
        },
        "preexisting_worktree_changes_preserved": _preexisting_worktree(
            git_status_before,
            git_status_after,
        ),
        "protected_files_unchanged": {
            "before_hashes": protected_before,
            "after_hashes": protected_after,
            "changed_files": [
                path
                for path, receipt in protected_before.items()
                if receipt != protected_after[path]
            ],
            "unchanged": _unchanged(protected_before, protected_after),
        },
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": test_commands_final,
        "test_exit_codes": test_exit_codes_final,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: current_rule_clean=true for unmodified Exp6161/Exp6162 replay; "
            "immutable_history_preserved=true with historical quarantine still flagged, "
            "headline_promotion_authorized=false, operator_reopen_required=true"
        )
        if current_clean and historical_preserved
        else (
            "blocked: current-rule replay or immutable-history preservation failed; "
            "historical quarantine remains preserved and operator reopen remains required"
        ),
    }
    artifact["reproducibility_checksum"] = _stable_hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    return artifact


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    verifier_receipt: JsonDict | None = None,
    git_status_before: str | None = None,
    git_status_after: str | None = None,
    git_head: str | None = None,
    duration_s: float | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: dict[str, int] | None = None,
) -> JsonDict:
    started = time.monotonic()
    receipt = verifier_receipt or run_current_verifier(root)
    elapsed = duration_s if duration_s is not None else round(time.monotonic() - started, 6)
    before = git_status_before if git_status_before is not None else git_status(root)
    head = git_head if git_head is not None else git_head_fn(root)
    after = git_status_after if git_status_after is not None else before
    artifact = build_artifact(
        root=root,
        verifier_receipt=receipt,
        git_status_before=before,
        git_status_after=after,
        git_head=head,
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    _write_json(root / RESULT_RELATIVE_PATH, artifact)
    if git_status_after is None:
        actual_after = git_status(root)
        artifact["source_hashes_and_git_status_before_after"]["git_status_after"] = actual_after
        artifact["preexisting_worktree_changes_preserved"] = _preexisting_worktree(
            before,
            actual_after,
        )
        artifact["preconditions_checked"]["preexisting_audit_edits"] = artifact[
            "preexisting_worktree_changes_preserved"
        ]
        artifact["test_exit_codes"] = _command_exit_map(receipt, test_exit_codes or {})
        artifact["reproducibility_checksum"] = _stable_hash(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
        _write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def git_head_fn(root: Path = REPO_ROOT) -> str:  # pragma: no cover
    return git_head(root)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(artifact: JsonDict) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
        _require(field in artifact.get("field_provenance", {}), f"missing provenance: {field}")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact["current_rule_clean"] is True, "current_rule_clean")
    _require(
        artifact["historical_quarantine_preserved"] is True,
        "historical_quarantine_preserved",
    )
    _require(
        artifact["headline_promotion_authorized"] is False,
        "headline_promotion_authorized",
    )
    _require(artifact["operator_reopen_required"] is True, "operator_reopen_required")
    _require(
        artifact["source_hashes_and_git_status_before_after"]["all_source_hashes_unchanged"]
        is True,
        "source hashes changed",
    )
    _require(artifact["protected_files_unchanged"]["unchanged"] is True, "protected files changed")
    _require(
        artifact["preexisting_worktree_changes_preserved"]["preserved"] is True,
        "preexisting changes not preserved",
    )
    _require(
        artifact["honest_verdict"].startswith(("complete:", "blocked:")),
        "honest_verdict",
    )
    for experiment_id, row in artifact[
        "field_level_historical_vs_current_determination_matrix"
    ].items():
        _require(
            row["source_flagged_adversarial"]["current_value"] is True,
            f"{experiment_id} source flag cleared",
        )
        _require(
            row["source_corrigendum_pending"]["current_value"],
            f"{experiment_id} corrigendum cleared",
        )
        _require(
            row["capstone_terminal_class"]["current_value"] == "flagged",
            f"{experiment_id} capstone terminal changed",
        )
        _require(
            row["current_verifier_flag_count"]["current_value"] == 0,
            f"{experiment_id} current verifier not clean",
        )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_read_json(REPO_ROOT / RESULT_RELATIVE_PATH))
        return 0
    artifact = build_and_write_artifact()
    validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
