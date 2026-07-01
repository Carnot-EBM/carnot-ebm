#!/usr/bin/env python3
"""Experiment 5056: archive .464, activate .465, and record the close-state.

Spec refs: REQ-CAPSTONE-5056, SCENARIO-CAPSTONE-5056,
SCENARIO-CAPSTONE-5056-BLOCKED-YAML.

This is a record-only transition. It reads roadmap YAML plus the Exp5055 and
Exp5050 authority artifacts, runs only its own ``--no-cov`` pre-test gate, and
records the .464 close-state without promoting any blocked or flagged evidence
into a moat claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(PYTHON_ROOT))


@dataclass(frozen=True)
class CommandResult:
    """Completed command summary for the local pre-test gate."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


CommandRunner = Callable[[list[str], Path], CommandResult]

EXPERIMENT = "experiment_5056_archive_464_activate_465"
EXPERIMENT_ID = 5056
SCHEMA = "carnot.exp5056.archive_464_activate_465.v1"
RANDOM_SEED = 20260701
PRIOR_MILESTONE = "2026.06.464"
NEXT_MILESTONE = "2026.06.465"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

OUTPUT_REL_PATH = Path("results/experiment_5056_archive_464_activate_465.json")
ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5055_capstone_v464.json")
MOAT_GATE_REL_PATH = Path("results/experiment_5050_moat_gate_resolution_v464.json")

PRETEST_COMMAND = [
    ".venv/bin/pytest",
    "tests/python/test_experiment_5056_archive_464_activate_465.py",
    "-q",
    "--no-cov",
]

SPEC_REFS = [
    "REQ-CAPSTONE-5056",
    "SCENARIO-CAPSTONE-5056",
    "SCENARIO-CAPSTONE-5056-BLOCKED-YAML",
]

TERMINAL_PREFIXES = (
    "complete_",
    "success_",
    "passed_",
    "shipped_",
    "blocked_",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; clean transition is "
            "complete_464_archived_465_activated_execution_incomplete_not_moat_claim."
        )
    },
    "prior_capstone_verdict": {
        "principle": "the Exp5055 blunt .464 capstone verdict, copied without reinterpretation.",
    },
    "moat_state": {
        "principle": (
            "the Exp5055/Exp5050 moat state; execution_incomplete is not a moat claim "
            "and not a bounded retirement."
        )
    },
    "d1_state": {
        "principle": (
            "D1 remained blocked by SOTA candidate refresh despite a +0.080 MuSR "
            "signal, so it is not headline-countable."
        )
    },
    "d4_state": {
        "principle": (
            "D4 was promising on the second corpus but flagged/not counted as a clean "
            "confirmation."
        )
    },
    "d6_state": {
        "principle": "D6 was gate-blocked and therefore cannot establish a cascade efficiency win.",
    },
    "fr11_state": {
        "principle": "FR-11 replay memory regressed on held-out accuracy and is guarded-negative.",
    },
    "kv260_state": {
        "principle": "KV260 succeeded only as a local timing-ratio packet, not a general speedup claim.",
    },
    "arc_state": {
        "principle": "ARC banked no new level; the reproducible total remains unchanged.",
    },
    "activation_ready": {
        "principle": (
            "true only when roadmaps parse, required source fields are present, the own "
            "pre-test gate is green, and the record remains non-submitting/non-moat."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "prior_milestone",
    "next_milestone",
    "prior_capstone_verdict",
    "moat_state",
    "d1_state",
    "d4_state",
    "d6_state",
    "fr11_state",
    "kv260_state",
    "arc_state",
    "activation_ready",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "prior_milestone",
    "next_milestone",
    "prior_capstone_verdict",
    "moat_state",
    "d1_state",
    "d4_state",
    "d6_state",
    "fr11_state",
    "kv260_state",
    "arc_state",
    "activation_ready",
    "inference_substrate",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "leaderboard_submission",
    "moat_claim",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

SOURCE_PATHS: dict[str, Path] = {
    "CAPSTONE": CAPSTONE_REL_PATH,
    "MOAT_GATE": MOAT_GATE_REL_PATH,
}

REQUIRED_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "CAPSTONE": (
        "honest_verdict",
        "capstone_ready",
        "moat_state",
        "best_arm_and_delta",
        "best_verifier_evidence",
        "second_corpus_state",
        "cascade_state",
        "fr11_state",
        "fr11_self_learning_result",
        "hardware_state",
        "hardware_result",
        "arc_state",
        "arc_result",
    ),
    "MOAT_GATE": (
        "honest_verdict",
        "moat_state",
        "best_arm",
        "best_arm_delta",
        "best_arm_ci",
        "second_corpus_confirmed",
        "cascade_efficiency_win",
        "blocked_upstream_artifacts",
        "flagged_upstream_artifacts",
        "missing_upstream_artifacts",
        "execution_incomplete_reasons",
        "per_arm_table",
        "cascade_artifact",
        "second_corpus_artifact",
    ),
}

REQUIRED_NESTED_SOURCE_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("CAPSTONE", "best_arm_and_delta"): (
        "arm_id",
        "delta",
        "ci95",
        "evidence_status",
        "headline_countable",
    ),
    ("CAPSTONE", "best_verifier_evidence"): (
        "arm_id",
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "execution_status",
        "proper_musr_win",
    ),
    ("CAPSTONE", "second_corpus_state"): (
        "state",
        "execution_status",
        "honest_verdict",
        "reported_confirmed",
        "headline_counted",
        "delta_vs_tuned_sc_second",
        "paired_ci95_second",
    ),
    ("CAPSTONE", "cascade_state"): (
        "state",
        "execution_status",
        "honest_verdict",
        "efficiency_win",
        "judge_call_fraction",
        "paired_ci95",
    ),
    ("CAPSTONE", "fr11_self_learning_result"): (
        "state",
        "honest_verdict",
        "self_learning_loop_executed",
        "credible_evidence",
        "heldout_delta",
    ),
    ("CAPSTONE", "hardware_result"): (
        "state",
        "honest_verdict",
        "kv260_ssh_reachable",
        "overlay_loaded",
        "timing_ratio_packet_built",
        "cpu_reference_ok",
        "kv260_result_ok",
    ),
    ("CAPSTONE", "arc_result"): (
        "state",
        "honest_verdict",
        "new_levels_banked",
        "reproducible_total_levels_after",
    ),
    ("MOAT_GATE", "cascade_artifact"): (
        "execution_status",
        "honest_verdict",
        "efficiency_win",
    ),
    ("MOAT_GATE", "second_corpus_artifact"): (
        "execution_status",
        "honest_verdict",
        "second_corpus_confirmed",
        "delta_vs_tuned_sc_second",
        "paired_ci95_second",
    ),
}


def run_command(command: list[str], cwd: Path) -> CommandResult:
    """Run a subprocess for the pre-test gate."""

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        command=list(command),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=round(max(time.perf_counter() - started, 0.0001), 6),
    )


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration_from(started_s: float, now_s: float | None = None) -> float:
    end = time.perf_counter() if now_s is None else now_s
    return max(0.0001, round(float(end - started_s), 6))


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text.removeprefix("sha256:"))
    )


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _parse_yaml_status(root: Path, rel_path: Path, *, absent_status: str) -> tuple[JsonDict, JsonDict]:
    path = root / rel_path
    status: JsonDict = {"path": str(rel_path), "exists": path.exists()}
    if not path.exists():
        status.update({"parse_ok": None, "status": absent_status})
        return {}, status
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        status.update({"parse_ok": False, "error": str(exc)})
        return {}, status
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        status.update({"parse_ok": False, "error": "yaml_not_mapping"})
        return {}, status
    status.update(
        {
            "parse_ok": True,
            "status": "parsed",
            "milestone": str(payload.get("milestone", "")),
            "sha256": file_sha256(path),
        }
    )
    return dict(payload), status


def check_roadmaps(root: Path) -> tuple[JsonDict, JsonDict]:
    """Parse active and pre-staged roadmaps with explicit YAML failure status."""

    active, active_status = _parse_yaml_status(
        root,
        ROADMAP_ACTIVE_REL_PATH,
        absent_status="missing_active_roadmap",
    )
    staged, staged_status = _parse_yaml_status(
        root,
        ROADMAP_NEXT_REL_PATH,
        absent_status="absent_already_promoted",
    )
    return (
        {"active": active, "pre_staged": staged},
        {"active": active_status, "pre_staged": staged_status},
    )


def roadmap_blocker(roadmaps_checked: Mapping[str, Any]) -> str:
    active = _mapping(roadmaps_checked.get("active"))
    staged = _mapping(roadmaps_checked.get("pre_staged"))
    if active.get("parse_ok") is False or staged.get("parse_ok") is False:
        return "blocked_yaml_parse"
    if active.get("parse_ok") is not True:
        return "blocked_missing_active_roadmap"
    return ""


def load_sources(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    statuses: dict[str, JsonDict] = {}
    for source, rel_path in SOURCE_PATHS.items():
        artifact, status = _read_json_mapping(root / rel_path)
        artifacts[source] = artifact
        statuses[source] = {"path": str(rel_path), **status}
    return artifacts, statuses


def source_blocker(source_statuses: Mapping[str, Any]) -> str:
    for status in source_statuses.values():
        mapping = _mapping(status)
        if mapping.get("exists") is not True:
            return "blocked_missing_required_artifact"
        if mapping.get("loadable") is not True:
            return "blocked_unloadable_required_artifact"
    return ""


def missing_required_fields(artifacts: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    missing: list[JsonDict] = []
    for source, fields in REQUIRED_SOURCE_FIELDS.items():
        artifact = artifacts.get(source, {})
        path = SOURCE_PATHS[source]
        for field in fields:
            if field not in artifact:
                missing.append({"path": str(path), "field": field})
                continue
            nested_fields = REQUIRED_NESTED_SOURCE_FIELDS.get((source, field), ())
            if not nested_fields:
                continue
            nested = _mapping(artifact.get(field))
            if not nested:
                missing.append({"path": str(path), "field": field})
                continue
            for nested_field in nested_fields:
                if nested_field not in nested:
                    missing.append({"path": str(path), "field": f"{field}.{nested_field}"})
    return missing


def _find_arm_row(rows: Any, arm_id: str) -> JsonDict:
    for row in _list(rows):
        mapping = _mapping(row)
        if mapping.get("arm_id") == arm_id:
            return mapping
    return {}


def _transition_state(roadmaps: Mapping[str, Any], roadmaps_checked: Mapping[str, Any]) -> JsonDict:
    active = _mapping(roadmaps.get("active"))
    staged = _mapping(roadmaps.get("pre_staged"))
    active_status = _mapping(roadmaps_checked.get("active"))
    staged_status = _mapping(roadmaps_checked.get("pre_staged"))
    active_milestone = str(active.get("milestone", active_status.get("milestone", "")))
    staged_milestone = str(staged.get("milestone", staged_status.get("milestone", "")))
    return {
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "active_milestone_confirmed": active_milestone,
        "pre_staged_milestone_confirmed": staged_milestone,
        "active_roadmap_path": str(ROADMAP_ACTIVE_REL_PATH),
        "pre_staged_roadmap_path": str(ROADMAP_NEXT_REL_PATH)
        if staged_status.get("exists") is True
        else "",
        "pre_staged_roadmap_status": str(staged_status.get("status", "")),
        "activation_state": "already_active_or_activated_465"
        if active_milestone == NEXT_MILESTONE
        else "pre_staged_465_not_promoted_by_script",
        "active_conductor_changed": False,
    }


def build_close_state(artifacts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the true .464 close-state from Exp5055 and Exp5050 fields."""

    capstone = _mapping(artifacts.get("CAPSTONE"))
    gate = _mapping(artifacts.get("MOAT_GATE"))
    best = _mapping(capstone.get("best_verifier_evidence"))
    best_delta = _mapping(capstone.get("best_arm_and_delta"))
    second = _mapping(capstone.get("second_corpus_state"))
    cascade = _mapping(capstone.get("cascade_state"))
    fr11 = _mapping(capstone.get("fr11_self_learning_result"))
    hardware = _mapping(capstone.get("hardware_result"))
    arc = _mapping(capstone.get("arc_result"))
    d1_blocked = _find_arm_row(gate.get("blocked_upstream_artifacts"), "D1")
    d4_flagged = _find_arm_row(gate.get("flagged_upstream_artifacts"), "D4")
    d6_blocked = _find_arm_row(gate.get("blocked_upstream_artifacts"), "D6")
    d6_missing = _find_arm_row(gate.get("missing_upstream_artifacts"), "D6")

    d1_verdict = str(d1_blocked.get("honest_verdict") or best.get("honest_verdict"))
    d1_status = str(best.get("execution_status"))
    d1_state = "blocked_candidate_refresh" if d1_verdict == "blocked_sota_candidate_refresh_unavailable" else d1_status
    fr11_delta = _number(fr11.get("heldout_delta"))
    arc_banked = _number(arc.get("new_levels_banked"))

    return {
        "prior_capstone_verdict": str(capstone.get("honest_verdict")),
        "moat_state": str(capstone.get("moat_state")),
        "moat_gate_verdict": str(gate.get("honest_verdict")),
        "d1_state": {
            "arm_id": "D1",
            "state": d1_state,
            "execution_status": d1_status,
            "honest_verdict": d1_verdict,
            "delta_vs_tuned_sc": _number(best.get("delta_vs_tuned_sc")),
            "ci95": best.get("paired_ci95"),
            "best_arm_delta": _number(best_delta.get("delta")),
            "best_arm_ci95": best_delta.get("ci95"),
            "evidence_status": str(best_delta.get("evidence_status")),
            "headline_countable": best_delta.get("headline_countable") is True,
            "proper_musr_win": best.get("proper_musr_win") is True,
            "moat_claim": False,
        },
        "d4_state": {
            "arm_id": "D4",
            "state": str(second.get("state")),
            "execution_status": str(second.get("execution_status")),
            "honest_verdict": str(d4_flagged.get("honest_verdict") or second.get("honest_verdict")),
            "reported_confirmed": second.get("reported_confirmed") is True,
            "headline_counted": second.get("headline_counted") is True,
            "delta_vs_tuned_sc_second": _number(second.get("delta_vs_tuned_sc_second")),
            "paired_ci95_second": second.get("paired_ci95_second"),
            "moat_claim": False,
        },
        "d6_state": {
            "arm_id": "D6",
            "state": str(cascade.get("state")),
            "execution_status": str(cascade.get("execution_status")),
            "honest_verdict": str(d6_blocked.get("honest_verdict") or cascade.get("honest_verdict")),
            "efficiency_win": cascade.get("efficiency_win") is True,
            "judge_call_fraction": _number(cascade.get("judge_call_fraction")),
            "paired_ci95": cascade.get("paired_ci95"),
            "missing_artifact_path": str(d6_missing.get("path", "")),
            "moat_claim": False,
        },
        "fr11_state": {
            "state": str(fr11.get("state") or capstone.get("fr11_state")),
            "honest_verdict": str(fr11.get("honest_verdict")),
            "self_learning_loop_executed": fr11.get("self_learning_loop_executed") is True,
            "credible_evidence": fr11.get("credible_evidence") is True,
            "heldout_delta": fr11_delta,
            "regressed": fr11_delta is not None and fr11_delta < 0.0,
        },
        "kv260_state": {
            "state": str(hardware.get("state") or capstone.get("hardware_state")),
            "honest_verdict": str(hardware.get("honest_verdict")),
            "kv260_ssh_reachable": hardware.get("kv260_ssh_reachable") is True,
            "overlay_loaded": hardware.get("overlay_loaded") is True,
            "timing_ratio_packet_built": hardware.get("timing_ratio_packet_built") is True,
            "cpu_reference_ok": hardware.get("cpu_reference_ok") is True,
            "kv260_result_ok": hardware.get("kv260_result_ok") is True,
            "claim_scope": str(hardware.get("claim_scope", "")),
        },
        "arc_state": {
            "state": str(arc.get("state") or capstone.get("arc_state")),
            "honest_verdict": str(arc.get("honest_verdict")),
            "new_levels_banked": int(arc_banked) if arc_banked is not None else None,
            "reproducible_total_levels_after": int(_number(arc.get("reproducible_total_levels_after")) or 0),
            "no_new_level": arc_banked == 0,
        },
        "execution_incomplete_reasons": [
            str(reason) for reason in _list(gate.get("execution_incomplete_reasons"))
        ],
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            rows.append({"path": str(rel_path), "sha256": file_sha256(path)})
    for source, rel_path in SOURCE_PATHS.items():
        path = root / rel_path
        if path.exists():
            rows.append({"source": source, "path": str(rel_path), "sha256": file_sha256(path)})
    return rows


def _command_summary(result: CommandResult) -> JsonDict:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "duration_s": result.duration_s,
    }


def _pretest_gate_from_result(result: CommandResult) -> JsonDict:
    return {"ran": True, "green": result.exit_code == 0, **_command_summary(result)}


def _empty_pretest(reason: str) -> JsonDict:
    return {"ran": False, "green": False, "reason": reason}


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    roadmaps: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    close_state: Mapping[str, Any],
    activation_ready: bool,
    missing_fields: list[JsonDict] | None = None,
    duration_s: float,
) -> JsonDict:
    close = _mapping(close_state)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "honest_verdict": honest_verdict,
        "prior_capstone_verdict": str(close.get("prior_capstone_verdict", "")),
        "moat_state": str(close.get("moat_state", "")),
        "d1_state": _mapping(close.get("d1_state")),
        "d4_state": _mapping(close.get("d4_state")),
        "d6_state": _mapping(close.get("d6_state")),
        "fr11_state": _mapping(close.get("fr11_state")),
        "kv260_state": _mapping(close.get("kv260_state")),
        "arc_state": _mapping(close.get("arc_state")),
        "activation_ready": activation_ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "transition": {
            **_transition_state(roadmaps, _mapping(preconditions_checked.get("roadmaps"))),
            "transition_performed": transition_performed,
        },
        "transition_performed": transition_performed,
        "leaderboard_submission": False,
        "moat_claim": False,
        "close_state_464": dict(close_state),
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": max(0.0001, round(float(duration_s), 6)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    if missing_fields:
        payload["missing_required_fields"] = missing_fields
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run the record-only .464/.465 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    roadmaps, roadmaps_checked = check_roadmaps(root)
    preconditions: JsonDict = {"roadmaps": roadmaps_checked}

    blocker = roadmap_blocker(roadmaps_checked)
    if blocker:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker,
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=_empty_pretest("skipped_after_yaml_precondition_failure"),
            transition_performed=False,
            close_state={},
            activation_ready=False,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifacts, source_statuses = load_sources(root)
    preconditions["source_artifacts"] = source_statuses
    blocker = source_blocker(source_statuses)
    missing_fields = missing_required_fields(artifacts)
    if blocker or missing_fields:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker or "blocked_missing_required_field",
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=_empty_pretest("skipped_after_required_source_failure"),
            transition_performed=False,
            close_state={},
            activation_ready=False,
            missing_fields=missing_fields,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    pretest_result = command_runner(PRETEST_COMMAND, root)
    pretest_gate = _pretest_gate_from_result(pretest_result)
    if pretest_result.exit_code != 0:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_pretest_gate_failed",
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            close_state={},
            activation_ready=False,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    close_state = build_close_state(artifacts)
    artifact = build_artifact(
        root=root,
        honest_verdict="complete_464_archived_465_activated_execution_incomplete_not_moat_claim",
        roadmaps=roadmaps,
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        close_state=close_state,
        activation_ready=True,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 5056 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("prior_milestone") != PRIOR_MILESTONE:
        errors.append("invalid_prior_milestone")
    if payload.get("next_milestone") != NEXT_MILESTONE:
        errors.append("invalid_next_milestone")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")
    if payload.get("moat_claim") is not False:
        errors.append("invalid_moat_claim")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not blocked:
        if payload.get("honest_verdict") != (
            "complete_464_archived_465_activated_execution_incomplete_not_moat_claim"
        ):
            errors.append("invalid_complete_verdict")
        if payload.get("prior_capstone_verdict") != (
            "complete_capstone_v464_execution_incomplete_fr11_no_credible_positive_evidence"
        ):
            errors.append("invalid_prior_capstone_verdict")
        if payload.get("moat_state") != "execution_incomplete":
            errors.append("invalid_moat_state")
        if payload.get("activation_ready") is not True:
            errors.append("invalid_activation_ready")
        d1 = _mapping(payload.get("d1_state"))
        d4 = _mapping(payload.get("d4_state"))
        d6 = _mapping(payload.get("d6_state"))
        fr11 = _mapping(payload.get("fr11_state"))
        kv260 = _mapping(payload.get("kv260_state"))
        arc = _mapping(payload.get("arc_state"))
        if d1.get("state") != "blocked_candidate_refresh" or _number(d1.get("delta_vs_tuned_sc")) != 0.08:
            errors.append("invalid_d1_state")
        if d4.get("state") != "flagged_not_counted" or d4.get("headline_counted") is not False:
            errors.append("invalid_d4_state")
        if d6.get("state") != "blocked" or d6.get("efficiency_win") is not False:
            errors.append("invalid_d6_state")
        if fr11.get("state") != "guarded_negative" or fr11.get("regressed") is not True:
            errors.append("invalid_fr11_state")
        if kv260.get("state") != "packet_built" or kv260.get("timing_ratio_packet_built") is not True:
            errors.append("invalid_kv260_state")
        if arc.get("state") != "no_bank" or arc.get("new_levels_banked") != 0:
            errors.append("invalid_arc_state")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 5056 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 5056 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
