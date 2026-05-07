"""Build the Exp 1479 `.114` activation manifest.

Spec: REQ-REPORT-050, SCENARIO-REPORT-050.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
PREDECESSOR_MILESTONE = "2026.04.113"
TARGET_MILESTONE = "2026.04.114"
EXPERIMENT = "1479_113_completion_archive_114_activation"
SCHEMA = "milestone_114_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1479_113_completion_archive_114_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_114_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1478_milestone_113_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_criteria_met",
    "predecessor_criteria_total",
    "activation_manifest_complete",
    "telemetry_headline_block_preserved",
    "self_learning_followup_allowed",
    "hardware_claim_boundaries",
    "allowed_114_tracks",
    "forbidden_reopen_tracks",
    "research_complete_has_113_entry",
    "honest_verdict",
}

ALLOWED_114_TRACKS = [
    {
        "track": "adversarial_balanced_telemetry",
        "name": "Adversarial Balanced Telemetry",
        "guardrail": "Use live local SOTA telemetry only with balanced labels and superficial baselines.",
    },
    {
        "track": "beaver_lite_calibration",
        "name": "BEAVER-lite Calibration",
        "guardrail": "Expand only the sound bounded-prefix calibration lane; any violation blocks claims.",
    },
    {
        "track": "halluguard_risk_bound_fit",
        "name": "HalluGuard-style Risk-Bound Fit",
        "guardrail": "Fit risk-bound language only for implemented assumptions and label missing assumptions.",
    },
    {
        "track": "fr11_query_time_self_learning",
        "name": "FR-11 Query-Time Self-Learning",
        "guardrail": "Promote only opt-in query-time utility with zero soundness mistakes.",
    },
    {
        "track": "cctu_executable_constraints",
        "name": "CCTU-style Executable Constraints",
        "guardrail": "Use deterministic local validators; no closed model dependency or broad benchmark claim.",
    },
    {
        "track": "v1_pairwise_verification",
        "name": "V_1 Pairwise Verification",
        "guardrail": "Compare pairwise self-verification against Carnot energy on bounded candidate sets.",
    },
    {
        "track": "thrml_preflight_parity",
        "name": "THRML Preflight/Parity",
        "guardrail": "Run install/import preflight and simulator parity only; no TSU hardware claim.",
    },
    {
        "track": "partial_trace_localization",
        "name": "Partial-Trace Localization",
        "guardrail": "Audit injected-failure localization without claiming decoded quality or Kona internals.",
    },
]

FORBIDDEN_REOPEN_TRACKS = [
    {
        "track": "telemetry_headline_claims",
        "name": "Telemetry Headline Claims",
        "source": "Exp 1473 adversarial validity audit",
        "rule": "Do not make a headline telemetry claim unless a future adversarial audit beats superficial baselines.",
    },
    {
        "track": "repair_executor_reruns",
        "name": "Repair-Executor Reruns",
        "source": "Exp 1464 and .113 carry-forward guardrails",
        "rule": "Do not rerun repair-executor or validation-error-context work without a new root cause and falsifiable gate.",
    },
    {
        "track": "grpo_vprm",
        "name": "GRPO/VPRM",
        "source": "Exp 1456 retirement",
        "rule": "Do not reopen GRPO/VPRM variants unless an operator reopens the line with changed evidence.",
    },
    {
        "track": "wopr_puzzle_cartridges",
        "name": "WOPR Puzzle Cartridges",
        "source": "Exp 1457 retirement",
        "rule": "Do not add puzzle cartridges or gallery work unless an operator reopens the thesis link.",
    },
    {
        "track": "hardnet_dsp",
        "name": "HardNet++/DSP",
        "source": "Exp 1458 retirement",
        "rule": "Do not reopen HardNet++/DSP or FSNet-as-DSP work without non-replay evidence.",
    },
    {
        "track": "broad_vnn_comp_runners",
        "name": "Broad VNN-COMP Runners",
        "source": "Exp 1465 and Exp 1470 BEAVER-lite narrowing",
        "rule": "Do not build broad VNNLIB/VNN-COMP runners before bounded BEAVER calibration earns expansion.",
    },
    {
        "track": "kv260_board_claims",
        "name": "KV260 Board Claims",
        "source": "Exp 1476 source-level RTL regression",
        "rule": "Do not claim board, bitfile, or latency evidence without live board execution evidence.",
    },
    {
        "track": "thrml_tsu_hardware_claims",
        "name": "THRML/TSU Hardware Claims",
        "source": "Exp 1477 THRML unavailable simulator probe",
        "rule": "Do not claim THRML, TSU, XTR-0, Z1, or Extropic hardware execution from simulator preflight.",
    },
]

HARDWARE_CLAIM_BOUNDARIES = {
    "dual_rtx_3090_runtime": {
        "allowed_evidence": ["local_sota_gguf_runtime", "live_logprob_telemetry"],
        "hardware_claim_allowed": True,
        "boundary": "Runtime evidence only for local open GGUF inference; no accelerator-substrate claim.",
    },
    "kv260": {
        "allowed_evidence": ["rtl_source", "rtl_simulation"],
        "hardware_claim_allowed": False,
        "boundary": "Source-level RTL lint/simulation only; no board, bitfile, or latency claim.",
    },
    "thrml_tsu": {
        "allowed_evidence": ["install_import_preflight", "simulator_parity"],
        "hardware_claim_allowed": False,
        "boundary": "THRML software preflight and simulator parity only; no TSU or Extropic hardware execution claim.",
    },
}

_EXP_LOG_NEEDLES = {
    "exp1467": ".112 Completion Archive + .113 Activation Manifest",
    "exp1468": "Live SOTA GGUF Logprob Telemetry Preflight",
    "exp1469": "HALT + Spilled Energy Diagnostic Micro-Benchmark",
    "exp1470": "BEAVER-Lite Deterministic Bound Smoke",
    "exp1471": "FR-11 v8 Verified-Memory-Growth Pivot",
    "exp1472": "Online Verifier Asymmetric Mistake-Budget Audit",
    "exp1473": "Live Telemetry Adversarial Validity Audit",
    "exp1474": "T-SKM Linear Constraint Projection Smoke",
    "exp1475": "STATIC CSR Certificate Automaton Smoke",
    "exp1476": "KV260 Discrete SB RTL Regression Pack",
    "exp1477": "THRML + NPIM Simulator Parity Micro-Probe",
    "exp1478": "Milestone .113 Retrospective",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-050: record that Exp 1479 started before evidence loading.

    The conductor treats this bootstrap file as proof that activation work
    began.  The terminal run replaces it after reading the .113 retro, ops
    evidence, and archive state.
    """

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "predecessor_milestone": PREDECESSOR_MILESTONE,
            "activation_manifest_complete": False,
            "telemetry_headline_block_preserved": False,
            "self_learning_followup_allowed": False,
            "hardware_claim_boundaries": dict(HARDWARE_CLAIM_BOUNDARIES),
            "allowed_114_tracks": [],
            "forbidden_reopen_tracks": [],
            "research_complete_has_113_entry": None,
            "honest_verdict": "in_progress_113_completion_archive_114_activation_seeded",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        if "ops" in parts:
            return str(Path(*parts[parts.index("ops") :]))
        if "results" in parts:
            return str(Path(*parts[parts.index("results") :]))
        return path.name


def _research_complete_has_113_entry(research_complete_text: str) -> bool:
    return (
        "- id: 2026.04.113" in research_complete_text
        or "id: 2026.04.113" in research_complete_text
        or 'id: "2026.04.113"' in research_complete_text
        or "id: '2026.04.113'" in research_complete_text
    )


def _track_by_name(retro: Mapping[str, Any], track_name: str) -> Mapping[str, Any]:
    for track in retro.get("carry_forward_tracks", []):
        if isinstance(track, Mapping) and track.get("track") == track_name:
            return track
    return {}


def _text_has_all(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def _telemetry_headline_block_preserved(
    retro: Mapping[str, Any],
    ops_status_text: str,
    ops_changelog_text: str,
) -> bool:
    headline_track = _track_by_name(retro, "telemetry_headline_claim")
    retro_blocks_claim = (
        headline_track.get("status") == "blocked"
        or "telemetry_headline_blocked" in str(retro.get("honest_verdict", ""))
    )
    ops_text = f"{ops_status_text}\n{ops_changelog_text}"
    ops_preserves_guardrail = (
        _text_has_all(ops_text, "headline", "blocked")
        or _text_has_all(ops_text, "adversarial", "telemetry")
    )
    return bool(retro_blocks_claim and ops_preserves_guardrail)


def _self_learning_followup_allowed(retro: Mapping[str, Any]) -> bool:
    self_learning = _track_by_name(retro, "self_learning")
    return bool(
        self_learning.get("status") == "preserved"
        and int(self_learning.get("self_learning_delta_overall") or 0) > 0
        and int(self_learning.get("soundness_mistakes") or 0) == 0
    )


def _hardware_boundaries_preserved(retro: Mapping[str, Any]) -> bool:
    hardware_track = _track_by_name(retro, "hardware_simulation")
    preserved_text = json.dumps(retro.get("preserved_lineages", []), sort_keys=True).lower()
    return bool(
        hardware_track.get("hardware_claim_allowed") is False
        and "kv260" in preserved_text
        and "thrml" in preserved_text
    )


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for exp_id, title in _EXP_LOG_NEEDLES.items():
        matches = [
            row
            for row in rows
            if exp_id in row or title in row or title[:38] in row
        ]
        entries[exp_id] = {
            "found": bool(matches),
            "ok": any("| OK |" in row for row in matches),
            "line": matches[-1] if matches else None,
        }
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "expected_count": len(_EXP_LOG_NEEDLES),
        "missing_experiments": [
            exp_id for exp_id, entry in entries.items() if not entry["found"]
        ],
    }


def _source_inputs_read(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(retro)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_track_table(rows: list[dict[str, str]], *, forbidden: bool) -> list[str]:
    if forbidden:
        lines = ["| track | source | rule |", "|---|---|---|"]
        for row in rows:
            lines.append(
                f"| {_md_cell(row['name'])} | {_md_cell(row['source'])} | {_md_cell(row['rule'])} |"
            )
        return lines
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def render_manifest(
    *,
    artifact: Mapping[str, Any],
    blocked_reasons: list[str],
) -> str:
    """REQ-REPORT-050: render the operator-facing `.114` activation manifest.

    The markdown is the terminal operator version of the JSON: what may run,
    what remains closed, and which hardware evidence can and cannot be claimed.
    """

    lines = [
        "# Milestone .114 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".113 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])
    if not artifact["research_complete_has_113_entry"]:
        lines.extend(["Archive gap: `research-complete.yaml` lacks `2026.04.113`.", ""])

    lines.extend(["## Allowed .114 Tracks", ""])
    lines.extend(_render_track_table(list(ALLOWED_114_TRACKS), forbidden=False))
    lines.extend(["", "## Forbidden Reopen Tracks", ""])
    lines.extend(_render_track_table(list(FORBIDDEN_REOPEN_TRACKS), forbidden=True))
    lines.extend(["", "## Hardware Claim Boundaries", ""])
    for boundary_id, boundary in artifact["hardware_claim_boundaries"].items():
        evidence = ", ".join(boundary["allowed_evidence"])
        lines.append(
            f"- {boundary_id}: allowed_evidence={evidence}; "
            f"hardware_claim_allowed={boundary['hardware_claim_allowed']}; "
            f"{boundary['boundary']}"
        )
    lines.extend(
        [
            "",
            "## Carry-Forward Guardrails",
            "",
            f"- telemetry_headline_block_preserved: {artifact['telemetry_headline_block_preserved']}",
            f"- self_learning_followup_allowed: {artifact['self_learning_followup_allowed']}",
            "- live telemetry remains non-headline until adversarial baselines are beaten.",
            "- FR-11 follow-up must prove query-time utility without soundness mistakes.",
            "- hardware evidence remains bounded to dual RTX 3090 runtime, KV260 RTL source/sim, and THRML simulator preflight.",
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1479_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1479_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    manifest_path: str,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-050: summarize `.113` closure and activate bounded `.114` work."""

    criteria_met = int(retro.get("criteria_met") or 0)
    criteria_total = int(retro.get("criteria_total") or 0)
    predecessor_complete = (
        str(retro.get("milestone")) == PREDECESSOR_MILESTONE
        and criteria_met == 12
        and criteria_total == 12
    )
    has_113_archive = _research_complete_has_113_entry(research_complete_text)
    telemetry_block = _telemetry_headline_block_preserved(
        retro,
        ops_status_text,
        ops_changelog_text,
    )
    self_learning_allowed = _self_learning_followup_allowed(retro)
    hardware_preserved = _hardware_boundaries_preserved(retro)

    blocked_reasons: list[str] = []
    if not predecessor_complete:
        blocked_reasons.append("predecessor retro criteria not complete")
    if not telemetry_block:
        blocked_reasons.append("telemetry headline block not preserved")
    if not self_learning_allowed:
        blocked_reasons.append("self-learning follow-up guardrail missing")
    if not hardware_preserved:
        blocked_reasons.append("hardware claim boundaries not preserved")

    activation_manifest_complete = not blocked_reasons
    archive_gap = None
    if not has_113_archive:
        archive_gap = {
            "missing_milestone": PREDECESSOR_MILESTONE,
            "recommended_action": (
                "append .113 archive row to research-complete.yaml without modifying research-roadmap.yaml"
            ),
        }

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if activation_manifest_complete else "blocked",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_honest_verdict": retro.get("honest_verdict"),
        "predecessor_criteria_met": criteria_met,
        "predecessor_criteria_total": criteria_total,
        "activation_manifest_complete": activation_manifest_complete,
        "telemetry_headline_block_preserved": telemetry_block,
        "self_learning_followup_allowed": self_learning_allowed,
        "self_learning_followup_guardrail": (
            "allowed only for opt-in query-time utility with zero soundness mistakes"
        ),
        "hardware_claim_boundaries": dict(HARDWARE_CLAIM_BOUNDARIES),
        "hardware_boundaries_preserved": hardware_preserved,
        "allowed_114_tracks": list(ALLOWED_114_TRACKS),
        "forbidden_reopen_tracks": list(FORBIDDEN_REOPEN_TRACKS),
        "research_complete_has_113_entry": has_113_archive,
        "research_complete_archive_update_needed": not has_113_archive,
        "archive_gap": archive_gap,
        "manifest_path": manifest_path,
        "blocked_reasons": blocked_reasons,
        "conductor_log_exp1467_to_exp1478": _conductor_log_summary(conductor_log_text),
        "predecessor_carry_forward_tracks": list(retro.get("carry_forward_tracks", [])),
        "source_inputs_read": _source_inputs_read(
            retro=retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
        ),
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "no_change_confirmations": {
            "research-roadmap.yaml": "unchanged_by_exp1479_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1479_activation_workflow",
        },
        "honest_verdict": (
            "milestone_114_activation_complete_113_archived_guardrails_preserved"
            if activation_manifest_complete and has_113_archive
            else "milestone_114_activation_complete_research_complete_113_archive_gap_recorded"
            if activation_manifest_complete
            else "milestone_114_activation_blocked_missing_predecessor_or_guardrail_evidence"
        ),
    }
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-050: write bootstrap, markdown manifest, and terminal JSON.

    The activation step deliberately writes only the Exp 1479 artifact and the
    operator manifest.  It does not mutate `research-roadmap.yaml`,
    `research-roadmap-next.yaml`, `ops/status.md`, `ops/changelog.md`, or the
    conductor.
    """

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    retro = _read_json(root_path / "results" / PREDECESSOR_RETRO_FILE) or {}
    artifact, manifest = build_artifact(
        retro=retro,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        manifest_path=_relative_path(manifest_out),
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
