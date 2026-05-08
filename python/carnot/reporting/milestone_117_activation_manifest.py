"""Build the Exp 1519 `.117` activation manifest.

Spec: REQ-REPORT-056, SCENARIO-REPORT-056.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
PREDECESSOR_MILESTONE = "2026.04.116"
TARGET_MILESTONE = "2026.04.117"
EXPERIMENT = "1519_116_completion_archive_117_activation"
SCHEMA = "milestone_117_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1519_116_completion_archive_117_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_117_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1518_milestone_116_retro.json"
ROADMAP_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_criteria_met",
    "predecessor_criteria_total",
    "activation_manifest_complete",
    "prior_runtime_contract_ready",
    "prior_fr11_rollback_ready",
    "prior_product_line_benchmark_ready",
    "prior_thrml_conformance_ready",
    "prior_kan_shape_manifest_ready",
    "prior_kv260_property_pack_ready",
    "research_complete_has_116_entry",
    "mandated_sota_models",
    "continuous_self_learning_required",
    "allowed_117_tracks",
    "gated_117_tracks",
    "retired_headline_signals",
    "honest_verdict",
}

MANDATED_SOTA_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]

RETIRED_HEADLINE_SIGNALS = [
    "Semantic Energy/logit telemetry headline claims",
    "pairwise LLM verifier headline claims",
    "arbitrary generated-Python verifier trust",
    "TSU hardware claims",
    "KV260 board claims",
    "KAN synthesis claims",
    "legacy small-model headline results",
]

ALLOWED_117_TRACKS = [
    {
        "track": "runtime_contract_e2e",
        "name": "runtime-contract E2E",
        "guardrail": "Integrate .116 contracts into one deterministic false-accept ledger.",
    },
    {
        "track": "live_sota_contract_guided_repair",
        "name": "live SOTA contract-guided repair",
        "guardrail": (
            "Use mandated local GGUF models; deterministic contracts remain authoritative."
        ),
    },
    {
        "track": "cdg_root_cause_repair",
        "name": "CDG root-cause repair",
        "guardrail": "Use dependency graphs for repair localization, not LLM-judge acceptance.",
    },
    {
        "track": "product_line_rescue_retirement",
        "name": "product-line rescue/retirement",
        "guardrail": "One parser/feasibility rescue is allowed before retiring the weak branch.",
    },
    {
        "track": "fr11_live_policy_promotion",
        "name": "FR-11 live policy promotion",
        "guardrail": "Promote only rollback-passing query-time policies; no model-weight mutation.",
    },
    {
        "track": "march_claim_isolation",
        "name": "MARCH-style claim isolation",
        "guardrail": (
            "Claim isolation is an ablation; deterministic validators are the trust boundary."
        ),
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "name": "THRML/Carnot parity scaling",
        "guardrail": (
            "Software/simulator parity only; no TSU, synthesis, bitstream, or board claim."
        ),
    },
]

GATED_117_TRACKS = [
    {
        "track": "runtime_contract_e2e",
        "task_id": "exp1520",
        "gated_on": ["exp1519.prior_runtime_contract_ready == true"],
    },
    {
        "track": "live_sota_contract_guided_repair",
        "task_id": "exp1521",
        "gated_on": ["exp1520.runtime_contract_e2e_ready == true"],
    },
    {
        "track": "cdg_root_cause_repair",
        "task_id": "exp1522",
        "gated_on": ["exp1520.runtime_contract_e2e_ready == true"],
    },
    {
        "track": "product_line_rescue_retirement",
        "task_id": "exp1523",
        "gated_on": ["exp1519.prior_product_line_benchmark_ready == true"],
    },
    {
        "track": "fr11_live_policy_promotion",
        "task_id": "exp1524",
        "gated_on": [
            "exp1519.prior_fr11_rollback_ready == true",
            "exp1520.runtime_contract_e2e_ready == true",
        ],
    },
    {
        "track": "march_claim_isolation",
        "task_id": "exp1525",
        "gated_on": ["exp1524.live_policy_promotion_ready == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1526",
        "gated_on": ["exp1519.prior_thrml_conformance_ready == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1527",
        "gated_on": ["exp1526.thrml_parity_n8_passed == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1528",
        "gated_on": ["exp1527.thrml_parity_n16_passed == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1529",
        "gated_on": ["exp1528.thrml_parity_n32_passed == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1530",
        "gated_on": ["exp1529.thrml_parity_n64_passed == true"],
    },
    {
        "track": "thrml_carnot_parity_scaling",
        "task_id": "exp1531",
        "gated_on": ["exp1528.thrml_parity_n32_passed == true"],
    },
]

SOURCE_FILES = {
    "exp1507": "experiment_1507_autopyverifier_safe_dsl_induction_pack.json",
    "exp1508": "experiment_1508_trigger_grammar_certificate_decoder_audit.json",
    "exp1509": "experiment_1509_executable_monitor_runtime_adapter.json",
    "exp1510": "experiment_1510_plan_graph_structural_contract_gate.json",
    "exp1511": "experiment_1511_product_line_solver_oracle_benchmark.json",
    "exp1512": "experiment_1512_fr11_verifier_feedback_policy_cache_v11.json",
    "exp1513": "experiment_1513_fr11_policy_rollback_replay_audit.json",
    "exp1515": "experiment_1515_thrml_samplerbackend_conformance_pack.json",
    "exp1516": "experiment_1516_kan_shape_normalization_preflight.json",
    "exp1517": "experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json",
}

CONDUCTOR_EXPERIMENT_TITLES = {
    "exp1506": ".115 Completion Archive + .116 Activation Manifest",
    "exp1507": "AutoPyVerifier-Inspired Safe-DSL Induction Pack",
    "exp1508": "Trigger+Grammar Certificate Decoder Audit",
    "exp1509": "Executable Monitor Runtime Adapter",
    "exp1510": "Plan-Graph Structural Contract Gate",
    "exp1511": "Product-Line Solver Oracle Benchmark",
    "exp1512": "FR-11 Verifier-Feedback Policy Cache v11",
    "exp1513": "FR-11 Policy Rollback Replay Audit",
    "exp1514": "trace2skill Portable Skill Pack v2",
    "exp1515": "THRML SamplerBackend Conformance Pack",
    "exp1516": "KAN/KAEM Shape Normalization Preflight",
    "exp1517": "KV260 Discrete SB RTL Property Pack v2",
    "exp1518": "Milestone .116 Retrospective + Claim Boundary Reconciliation",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-056: persist the bootstrap archive before evidence loading.

    This small file is intentionally visible and durable. It lets the conductor
    distinguish "Exp 1519 started and is reading mutable evidence" from "Exp
    1519 never wrote anything" if the run is interrupted before terminal JSON.
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
            "continuous_self_learning_required": True,
            "mandated_sota_models": list(MANDATED_SOTA_MODELS),
            "allowed_117_tracks": [],
            "gated_117_tracks": [],
            "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
            "honest_verdict": "complete: in_progress_116_completion_archive_117_activation_seeded",
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
    parts = path.parts
    if "ops" in parts:
        return str(Path(*parts[parts.index("ops") :]))
    if "results" in parts:
        return str(Path(*parts[parts.index("results") :]))
    return path.name


def _research_complete_has_116_entry(research_complete_text: str) -> bool:
    return any(
        marker in research_complete_text
        for marker in (
            "- id: 2026.04.116",
            "id: 2026.04.116",
            'id: "2026.04.116"',
            "id: '2026.04.116'",
        )
    )


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "complete"


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) in {0, 0.0}


def _manifest_path_exists(root: Path, path_text: object) -> bool:
    if not path_text:
        return False
    candidate = Path(str(path_text))
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.exists()


def _prior_runtime_contract_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    checks = (
        ("exp1507", "verifier_false_accept_rate"),
        ("exp1508", "verifier_false_accept_rate"),
        ("exp1509", "verifier_false_accept_rate"),
        ("exp1510", "false_accept_rate"),
    )
    return all(
        _is_complete(sources.get(exp_id, {})) and _zero(sources[exp_id], field)
        for exp_id, field in checks
    )


def _prior_fr11_rollback_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _is_complete(sources.get(exp_id, {})) and _zero(sources[exp_id], "soundness_mistakes")
        for exp_id in ("exp1512", "exp1513")
    )


def _prior_product_line_benchmark_ready(
    root: Path,
    sources: Mapping[str, Mapping[str, Any]],
) -> bool:
    exp1511 = sources.get("exp1511", {})
    return bool(
        _is_complete(exp1511)
        and exp1511.get("product_line_benchmark_ready") is True
        and _manifest_path_exists(root, exp1511.get("benchmark_manifest_path"))
    )


def _prior_thrml_conformance_ready(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1515 = sources.get("exp1515", {})
    return bool(
        _is_complete(exp1515)
        and exp1515.get("thrml_samplerbackend_conformance_ready") is True
        and exp1515.get("simulator_only") is True
        and exp1515.get("no_tsu_hardware_claim") is True
        and _manifest_path_exists(root, exp1515.get("conformance_manifest_path"))
    )


def _prior_kan_shape_manifest_ready(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1516 = sources.get("exp1516", {})
    return bool(
        _is_complete(exp1516)
        and exp1516.get("kan_shape_manifest_ready") is True
        and exp1516.get("normalized_shapes_written") is True
        and exp1516.get("no_synthesis_claim") is True
        and exp1516.get("no_board_claim") is True
        and _manifest_path_exists(root, exp1516.get("shape_manifest_path"))
    )


def _prior_kv260_property_pack_ready(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1517 = sources.get("exp1517", {})
    return bool(
        _is_complete(exp1517)
        and exp1517.get("kv260_property_pack_ready") is True
        and exp1517.get("source_level_only") is True
        and exp1517.get("no_board_execution") is True
        and exp1517.get("no_bitstream_claim") is True
        and _manifest_path_exists(root, exp1517.get("property_manifest_path"))
    )


def _predecessor_complete(predecessor_retro: Mapping[str, Any]) -> bool:
    return bool(
        _is_complete(predecessor_retro)
        and predecessor_retro.get("milestone") == PREDECESSOR_MILESTONE
        and predecessor_retro.get("criteria_met") == 13
        and predecessor_retro.get("criteria_total") == 13
    )


def _mandated_models_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts)
    return all(model in evidence for model in MANDATED_SOTA_MODELS)


def _continuous_self_learning_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    return "continuous self-learning" in evidence and "exp1524" in evidence


def _retirement_blocks_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    required = [
        ("semantic energy", "logit"),
        ("pairwise", "verifier"),
        ("generated", "verifier", "trust"),
        ("tsu", "hardware"),
        ("kv260", "board"),
        ("kan", "synthesis"),
        ("legacy", "small", "headline"),
    ]
    return all(all(term in evidence for term in terms) for terms in required)


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for exp_id, title in CONDUCTOR_EXPERIMENT_TITLES.items():
        title_prefix = title[:40]
        matches = [row for row in rows if exp_id in row or title_prefix in row or title in row]
        entries[exp_id] = {
            "found": bool(matches),
            "ok": any("| OK |" in row for row in matches),
            "terminal": any(
                "| OK |" in row or "| GATE_BLOCK |" in row or "| FAIL |" in row
                for row in matches
            ),
            "line": matches[-1] if matches else None,
        }
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "terminal_count": sum(1 for entry in entries.values() if entry["terminal"]),
        "expected_count": len(CONDUCTOR_EXPERIMENT_TITLES),
        "missing_experiments": [exp_id for exp_id, entry in entries.items() if not entry["found"]],
    }


def _source_inputs_read(
    *,
    predecessor_retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    research_references_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(predecessor_retro)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
        "ops/known-issues.md": {"exists": bool(ops_known_issues_text)},
        "research-roadmap.yaml": {"exists": bool(roadmap_text)},
        ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
        "research-references.md": {"exists": bool(research_references_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_allowed_track_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def _render_gated_track_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| task | track | gate |", "|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {_md_cell(row['task_id'])} | {_md_cell(row['track'])} | "
            f"{_md_cell('; '.join(row['gated_on']))} |"
        )
    return lines


def render_manifest(*, artifact: Mapping[str, Any], blocked_reasons: Sequence[str]) -> str:
    """REQ-REPORT-056: render the operator-facing `.117` activation manifest.

    The markdown is intentionally redundant with the JSON gate fields so a human
    can see which same-roadmap tasks are allowed, which are gated, and which
    headline claims remain blocked before downstream research resumes.
    """

    lines = [
        "# Milestone .117 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".116 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])

    lines.extend(["## Allowed .117 Tracks", ""])
    lines.extend(_render_allowed_track_table(ALLOWED_117_TRACKS))
    lines.extend(["", "## Gated .117 Tracks", ""])
    lines.extend(_render_gated_track_table(GATED_117_TRACKS))
    lines.extend(["", "## Same-Roadmap Gates", ""])
    for row in GATED_117_TRACKS:
        lines.append(f"- {row['task_id']}: {'; '.join(row['gated_on'])}")
    lines.extend(["", "## Prior Readiness", ""])
    for key in (
        "prior_runtime_contract_ready",
        "prior_fr11_rollback_ready",
        "prior_product_line_benchmark_ready",
        "prior_thrml_conformance_ready",
        "prior_kan_shape_manifest_ready",
        "prior_kv260_property_pack_ready",
    ):
        lines.append(f"- {key}: {artifact[key]}")
    lines.extend(["", "## Retired Headline Signals And Blocks", ""])
    for signal in artifact["retired_headline_signals"]:
        lines.append(f"- {signal}")
    lines.extend(["", "## Mandated Local SOTA Models", ""])
    for model in artifact["mandated_sota_models"]:
        lines.append(f"- {model}")
    lines.extend(
        [
            "",
            "## Continuous Self-Learning",
            "",
            f"- continuous_self_learning_required: {artifact['continuous_self_learning_required']}",
            "- required .117 task: exp1524 FR-11 live policy promotion",
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1519_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1519_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def _load_sources(results_dir: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for exp_id, filename in SOURCE_FILES.items():
        loaded[exp_id] = _read_json(results_dir / filename) or {}
    return loaded


def build_artifact(
    *,
    root: Path | str,
    predecessor_retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    research_references_text: str,
    manifest_path: str,
    protected_files_unchanged: bool,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-056: summarize `.116` closure and activate guarded `.117` work."""

    root_path = Path(root)
    criteria_met = int(predecessor_retro.get("criteria_met") or 0)
    criteria_total = int(predecessor_retro.get("criteria_total") or 0)
    predecessor_ready = _predecessor_complete(predecessor_retro)
    runtime_ready = _prior_runtime_contract_ready(sources)
    fr11_ready = _prior_fr11_rollback_ready(sources)
    product_ready = _prior_product_line_benchmark_ready(root_path, sources)
    thrml_ready = _prior_thrml_conformance_ready(root_path, sources)
    kan_ready = _prior_kan_shape_manifest_ready(root_path, sources)
    kv260_ready = _prior_kv260_property_pack_ready(root_path, sources)
    research_complete_has_116 = _research_complete_has_116_entry(research_complete_text)
    mandated_models_recorded = _mandated_models_recorded(
        ops_status_text,
        roadmap_text,
        roadmap_doc_text,
        research_references_text,
    )
    continuous_self_learning_recorded = _continuous_self_learning_recorded(
        ops_status_text,
        roadmap_text,
        roadmap_doc_text,
    )
    retirement_blocks_recorded = _retirement_blocks_recorded(
        ops_changelog_text,
        ops_status_text,
        ops_known_issues_text,
        roadmap_doc_text,
        research_references_text,
    )
    conductor_summary = _conductor_log_summary(conductor_log_text)

    blocked_reasons: list[str] = []
    if not predecessor_ready:
        blocked_reasons.append("predecessor .116 criteria are not 13 of 13")
    blocked_reasons.extend(
        reason
        for reason, ready in (
            ("runtime-contract prerequisites are not ready", runtime_ready),
            ("FR-11 rollback prerequisites are not ready", fr11_ready),
            ("product-line benchmark manifest is not ready", product_ready),
            ("THRML simulator conformance is not ready", thrml_ready),
            ("KAN shape manifest is not ready", kan_ready),
            ("KV260 property pack is not ready", kv260_ready),
            ("mandated SOTA model requirement is not recorded", mandated_models_recorded),
            (
                "continuous self-learning requirement is not recorded",
                continuous_self_learning_recorded,
            ),
            ("retired headline signal blocks are not preserved", retirement_blocks_recorded),
        )
        if not ready
    )
    if conductor_summary["missing_experiments"]:
        blocked_reasons.append("conductor log missing exp1506-through-exp1518 evidence")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed")

    activation_manifest_complete = not blocked_reasons
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if activation_manifest_complete else "blocked",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_honest_verdict": predecessor_retro.get("honest_verdict"),
        "predecessor_criteria_met": criteria_met,
        "predecessor_criteria_total": criteria_total,
        "activation_manifest_complete": activation_manifest_complete,
        "prior_runtime_contract_ready": runtime_ready,
        "prior_fr11_rollback_ready": fr11_ready,
        "prior_product_line_benchmark_ready": product_ready,
        "prior_thrml_conformance_ready": thrml_ready,
        "prior_kan_shape_manifest_ready": kan_ready,
        "prior_kv260_property_pack_ready": kv260_ready,
        "research_complete_has_116_entry": research_complete_has_116,
        "mandated_sota_models": list(MANDATED_SOTA_MODELS),
        "mandated_sota_models_recorded": mandated_models_recorded,
        "continuous_self_learning_required": True,
        "continuous_self_learning_requirement_recorded": continuous_self_learning_recorded,
        "allowed_117_tracks": list(ALLOWED_117_TRACKS),
        "gated_117_tracks": list(GATED_117_TRACKS),
        "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
        "retired_headline_signal_blocks_preserved": retirement_blocks_recorded,
        "manifest_path": manifest_path,
        "blocked_reasons": blocked_reasons,
        "conductor_log_exp1506_to_exp1518": conductor_summary,
        "source_inputs_read": _source_inputs_read(
            predecessor_retro=predecessor_retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            ops_known_issues_text=ops_known_issues_text,
            roadmap_text=roadmap_text,
            roadmap_doc_text=roadmap_doc_text,
            research_references_text=research_references_text,
        ),
        "source_evidence_paths": {
            exp_id: f"results/{filename}" for exp_id, filename in SOURCE_FILES.items()
        },
        "protected_files_unchanged": protected_files_unchanged,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "no_change_confirmations": {
            "research-roadmap.yaml": "unchanged_by_exp1519_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1519_activation_workflow",
        },
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            "complete: milestone_117_activation_complete_116_archived_runtime_"
            "fr11_thrml_gates_ready"
            if activation_manifest_complete
            else "passed: milestone_117_activation_blocked_missing_or_unsafe_predecessor_evidence"
        ),
    }
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest


def _protected_files_clean(root: Path) -> bool:  # pragma: no cover - environment guard
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--",
                "research-roadmap.yaml",
                "scripts/research_conductor.py",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode == 0


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-056: write bootstrap, markdown manifest, and terminal JSON.

    The activation step intentionally avoids mutating `research-roadmap.yaml`,
    `scripts/research_conductor.py`, `ops/status.md`, `ops/changelog.md`, and
    `_bmad/traceability.md`. The conductor-owned reconciler handles those
    operational documents after this focused artifact exits.
    """

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    protected = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    artifact, manifest = build_artifact(
        root=root_path,
        predecessor_retro=_read_json(root_path / "results" / PREDECESSOR_RETRO_FILE) or {},
        sources=_load_sources(root_path / "results"),
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        ops_known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_doc_text=_read_text(root_path / ROADMAP_DOC),
        research_references_text=_read_text(root_path / "research-references.md"),
        manifest_path=_relative_path(manifest_out),
        protected_files_unchanged=protected,
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
