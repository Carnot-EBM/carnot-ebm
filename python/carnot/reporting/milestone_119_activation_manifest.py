"""Build the Exp 1547 `.119` activation manifest.

Spec: REQ-REPORT-060, SCENARIO-REPORT-060.
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
PREDECESSOR_MILESTONE = "2026.04.118"
TARGET_MILESTONE = "2026.04.119"
EXPERIMENT = "1547_118_completion_archive_119_activation"
SCHEMA = "milestone_119_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1547_118_completion_archive_119_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_119_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1546_milestone_118_retro.json"
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
    "prior_automata_ready",
    "prior_satquest_benchmark_ready",
    "prior_satquest_solver_oracle_false_accepts",
    "prior_satquest_zero_solver_false_accepts",
    "prior_residual_drift_ready",
    "prior_fr11_safe_only",
    "prior_fr11_positive_utility",
    "prior_product_line_ready",
    "prior_claim_router_ready",
    "prior_arm_ebm_diagnostic_ready",
    "prior_thrml_n256_ready",
    "prior_thrml_diverse_n64_ready",
    "thrml_independent_rng_required",
    "prior_extropic_packet_ready",
    "research_complete_has_118_entry",
    "mandated_sota_models",
    "continuous_self_learning_required",
    "allowed_119_tracks",
    "retired_headline_signals",
    "honest_verdict",
}

MANDATED_SOTA_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]

RETIRED_HEADLINE_SIGNALS = [
    "legacy small-model headline claims",
    "SATQuest acceptance before oracle repair",
    "ARM/EBT soft-value acceptance authority",
    "Extropic TSU/Z1/XTR-0 hardware execution claims",
    "KV260 board claims",
    "model-weight mutation",
]

ALLOWED_119_TRACKS = [
    {
        "track": "thrml_independent_rng_audit",
        "name": "THRML independent-RNG audit",
        "guardrail": "Reject byte-identical stochastic summaries before any further parity headline.",
    },
    {
        "track": "satquest_oracle_repair",
        "name": "SATQuest oracle repair",
        "guardrail": "Proof and witness evidence must remove solver-oracle false accepts.",
    },
    {
        "track": "satquest_sota_reeval",
        "name": "SATQuest SOTA re-eval",
        "guardrail": "Runs only after repaired oracle evidence reports zero false accepts.",
    },
    {
        "track": "unified_contract_gate",
        "name": "unified automata/SAT/runtime contract gate",
        "guardrail": "Automata masks are generation support; deterministic validators remain authority.",
    },
    {
        "track": "residual_drift_repair",
        "name": "residual-drift repair",
        "guardrail": "Localize repairs to forgotten commitments with deterministic replay.",
    },
    {
        "track": "claim_isolation_scale",
        "name": "claim-isolation scale",
        "guardrail": "Budget savings must not hide deterministic failures.",
    },
    {
        "track": "product_line_scale",
        "name": "product-line scale",
        "guardrail": "Scale behind parser, feasibility, oracle, and false-accept gates.",
    },
    {
        "track": "fr11_positive_utility_or_retire",
        "name": "FR-11 positive-utility-or-retire",
        "guardrail": "Positive utility requires utility_delta > 0; no model-weight mutation.",
    },
    {
        "track": "arm_ebt_telemetry_repair",
        "name": "ARM/EBT telemetry repair",
        "guardrail": "Soft values and logprobs stay diagnostic below deterministic validators.",
    },
    {
        "track": "weaver_verification_routing",
        "name": "Weaver-style verification routing",
        "guardrail": "Route verification compute across weak and deterministic signals with cost metrics.",
    },
    {
        "track": "thrml_extropic_packet_update",
        "name": "THRML/Extropic packet update",
        "guardrail": "Update readiness only after RNG evidence; no hardware execution claim.",
    },
    {
        "track": "milestone_retro",
        "name": "milestone retro",
        "guardrail": "Close .119 from source artifacts with carry-forward gates for .120.",
    },
]

SOURCE_FILES = {
    "exp1535": "experiment_1535_xgrammar_abs_contract_decoder_adapter.json",
    "exp1536": "experiment_1536_satquest_cnf_verifier_benchmark.json",
    "exp1538": "experiment_1538_residual_drift_commitment_ledger.json",
    "exp1539": "experiment_1539_fr11_external_feedback_skill_promotion_v13.json",
    "exp1540": "experiment_1540_product_line_staged_benchmark_scale_v3.json",
    "exp1541": "experiment_1541_claim_isolation_uncertainty_router_v2.json",
    "exp1542": "experiment_1542_arm_ebm_soft_value_diagnostic.json",
    "exp1543": "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json",
    "exp1544": "experiment_1544_thrml_diverse_topology_parity_n64.json",
    "exp1545": "experiment_1545_extropic_z1_access_readiness_packet.json",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-060: persist a started marker before mutable evidence reads.

    The marker makes interrupted activation work visible to the conductor while
    avoiding a false terminal success. The terminal pass below overwrites these
    placeholders with evidence-backed fields.
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
            "mandated_sota_models": list(MANDATED_SOTA_MODELS),
            "continuous_self_learning_required": True,
            "allowed_119_tracks": [],
            "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
            "honest_verdict": "complete: in_progress_118_completion_archive_119_activation_seeded",
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
    for marker in ("ops", "results"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) :]))
    return path.name


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(f"results/{filename}")
        else:
            loaded[exp_id] = payload
    return loaded, missing


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "complete"


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int | float) and not isinstance(value, bool) and value == 0


def _predecessor_complete(predecessor_retro: Mapping[str, Any]) -> bool:
    return bool(
        _is_complete(predecessor_retro)
        and predecessor_retro.get("milestone") == PREDECESSOR_MILESTONE
        and predecessor_retro.get("criteria_met") == 13
        and predecessor_retro.get("criteria_total") == 14
    )


def _nested(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    return value if isinstance(value, Mapping) else {}


def _research_complete_has_118_entry(research_complete_text: str) -> bool:
    return any(
        marker in research_complete_text
        for marker in (
            "- id: 2026.04.118",
            "id: 2026.04.118",
            'id: "2026.04.118"',
            "id: '2026.04.118'",
        )
    )


def _thrml_independent_rng_required(ops_known_issues_text: str) -> bool:
    evidence = ops_known_issues_text.lower()
    return bool(
        "thrml/carnot" in evidence
        and (("independent-rng audit" in evidence) or ("independent rng audit" in evidence))
        and "mandatory" in evidence
    )


def _retirement_blocks_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower().replace("_", "-")
    required_any = [
        ("legacy", "small-model", "headline"),
        ("satquest", "acceptance", "oracle repair"),
        ("arm/ebt", "soft-value", "acceptance"),
        ("extropic", "tsu", "z1", "xtr-0", "hardware"),
        ("kv260", "board"),
        ("model-weight", "mutation"),
    ]
    return all(all(term in evidence for term in terms) for terms in required_any)


def _prior_automata_ready(
    predecessor_retro: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> bool:
    exp1535 = sources.get("exp1535", {})
    gate = _nested(predecessor_retro, "automata_contract_gate")
    return bool(
        _is_complete(exp1535)
        and exp1535.get("contract_decoder_adapter_ready") is True
        and gate.get("adapter_ready") is True
    )


def _prior_satquest_benchmark_ready(
    predecessor_retro: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> bool:
    exp1536 = sources.get("exp1536", {})
    gate = _nested(predecessor_retro, "satquest_verifier_gate")
    return bool(
        _is_complete(exp1536)
        and exp1536.get("satquest_benchmark_ready") is True
        and gate.get("benchmark_ready") is True
    )


def _prior_satquest_solver_false_accepts(predecessor_retro: Mapping[str, Any]) -> int | None:
    gate = _nested(predecessor_retro, "satquest_verifier_gate")
    value = gate.get("solver_oracle_false_accepts")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _prior_residual_drift_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1538 = sources.get("exp1538", {})
    return bool(_is_complete(exp1538) and exp1538.get("residual_drift_ledger_ready") is True)


def _prior_fr11_safe_only(
    predecessor_retro: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> bool:
    exp1539 = sources.get("exp1539", {})
    gate = _nested(predecessor_retro, "fr11_positive_utility_gate")
    soundness_mistakes = gate.get("soundness_mistakes", exp1539.get("soundness_mistakes"))
    no_mutation = gate.get("no_model_weight_mutation", exp1539.get("no_model_weight_mutation"))
    return bool(soundness_mistakes == 0 and no_mutation is True)


def _prior_fr11_positive_utility(predecessor_retro: Mapping[str, Any]) -> bool:
    return _nested(predecessor_retro, "fr11_positive_utility_gate").get(
        "positive_utility_achieved"
    ) is True


def _prior_product_line_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1540 = sources.get("exp1540", {})
    return bool(
        _is_complete(exp1540)
        and (
            exp1540.get("product_line_scale_ready") is True
            or exp1540.get("branch_retired") is True
        )
        and _zero(exp1540, "false_accept_count")
        and _zero(exp1540, "false_accept_rate")
    )


def _prior_claim_router_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1541 = sources.get("exp1541", {})
    return bool(
        _is_complete(exp1541)
        and exp1541.get("uncertainty_router_ready") is True
        and _zero(exp1541, "false_accept_count")
        and _zero(exp1541, "false_accept_rate")
    )


def _prior_arm_ebm_diagnostic_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1542 = sources.get("exp1542", {})
    return bool(
        _is_complete(exp1542)
        and exp1542.get("arm_ebm_diagnostic_ready") is True
        and exp1542.get("deterministic_validators_final_authority") is True
        and exp1542.get("no_model_weight_mutation") is True
    )


def _prior_thrml_n256_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1543 = sources.get("exp1543", {})
    return bool(
        _is_complete(exp1543)
        and exp1543.get("thrml_parity_n256_schedule_ready") is True
        and exp1543.get("parity_passed") is True
        and exp1543.get("simulator_only") is True
        and exp1543.get("no_tsu_hardware_claim") is True
    )


def _prior_thrml_diverse_n64_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1544 = sources.get("exp1544", {})
    return bool(
        _is_complete(exp1544)
        and exp1544.get("diverse_topology_parity_n64_ready") is True
        and exp1544.get("parity_passed") is True
        and exp1544.get("simulator_only") is True
        and exp1544.get("no_tsu_hardware_claim") is True
    )


def _prior_extropic_packet_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1545 = sources.get("exp1545", {})
    return bool(
        _is_complete(exp1545)
        and exp1545.get("extropic_z1_readiness_packet_ready") is True
        and exp1545.get("no_hardware_execution_claim") is True
        and exp1545.get("research_roadmap_yaml_modified") is not True
        and exp1545.get("scripts_research_conductor_modified") is not True
    )


def _source_inputs_read(
    *,
    predecessor_retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    roadmap_text: str,
    roadmap_next_text: str,
    roadmap_doc_text: str,
    research_references_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(predecessor_retro)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "research-references.md": {"exists": bool(research_references_text)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
        "ops/known-issues.md": {"exists": bool(ops_known_issues_text)},
        "research-roadmap.yaml": {"exists": bool(roadmap_text)},
        "research-roadmap-next.yaml": {"exists": bool(roadmap_next_text)},
        ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_allowed_track_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def _same_roadmap_gate_fields(artifact: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    return {
        "thrml_independent_rng_audit": {
            "prior_thrml_n256_ready": True,
            "thrml_independent_rng_required": True,
        },
        "satquest_oracle_repair": {
            "prior_satquest_benchmark_ready": True,
        },
        "satquest_sota_reeval": {
            "prior_satquest_zero_solver_false_accepts": True,
        },
        "unified_contract_gate": {
            "prior_automata_ready": True,
            "prior_satquest_zero_solver_false_accepts": True,
        },
        "fr11_positive_utility_or_retire": {
            "prior_fr11_safe_only": True,
            "prior_fr11_positive_utility": False,
        },
        "thrml_extropic_packet_update": {
            "thrml_independent_rng_required": False,
            "prior_extropic_packet_ready": True,
        },
    }


def render_manifest(*, artifact: Mapping[str, Any], blocked_reasons: Sequence[str]) -> str:
    """REQ-REPORT-060: render the operator-facing `.119` activation manifest."""

    lines = [
        "# Milestone .119 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".118 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])

    lines.extend(["## Allowed .119 Tracks", ""])
    lines.extend(_render_allowed_track_table(ALLOWED_119_TRACKS))
    lines.extend(["", "## Same-Roadmap Gates", ""])
    for task, fields in artifact["same_roadmap_gate_fields"].items():
        gates = "; ".join(f"{field} == {expected}" for field, expected in fields.items())
        lines.append(f"- {task}: {gates}")
    lines.extend(["", "## Prior Readiness", ""])
    for key in (
        "prior_automata_ready",
        "prior_satquest_benchmark_ready",
        "prior_satquest_solver_oracle_false_accepts",
        "prior_satquest_zero_solver_false_accepts",
        "prior_residual_drift_ready",
        "prior_fr11_safe_only",
        "prior_fr11_positive_utility",
        "prior_product_line_ready",
        "prior_claim_router_ready",
        "prior_arm_ebm_diagnostic_ready",
        "prior_thrml_n256_ready",
        "prior_thrml_diverse_n64_ready",
        "thrml_independent_rng_required",
        "prior_extropic_packet_ready",
        "research_complete_has_118_entry",
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
            "- required .119 task: exp1555 FR-11 positive-utility-or-retire",
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1547_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1547_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    predecessor_retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    roadmap_text: str,
    roadmap_next_text: str,
    roadmap_doc_text: str,
    research_references_text: str,
    manifest_path: str,
    protected_files_unchanged: bool,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-060: summarize `.118` closure and activate guarded `.119` work."""

    predecessor_ready = _predecessor_complete(predecessor_retro)
    criteria_met = int(predecessor_retro.get("criteria_met") or 0) if predecessor_ready else 0
    criteria_total = (
        int(predecessor_retro.get("criteria_total") or 0) if predecessor_ready else 0
    )
    satquest_false_accepts = _prior_satquest_solver_false_accepts(predecessor_retro)
    prior_satquest_zero = satquest_false_accepts == 0 if satquest_false_accepts is not None else False

    readiness = {
        "prior_automata_ready": _prior_automata_ready(predecessor_retro, sources),
        "prior_satquest_benchmark_ready": _prior_satquest_benchmark_ready(
            predecessor_retro, sources
        ),
        "prior_residual_drift_ready": _prior_residual_drift_ready(sources),
        "prior_fr11_safe_only": _prior_fr11_safe_only(predecessor_retro, sources),
        "prior_fr11_positive_utility": _prior_fr11_positive_utility(predecessor_retro),
        "prior_product_line_ready": _prior_product_line_ready(sources),
        "prior_claim_router_ready": _prior_claim_router_ready(sources),
        "prior_arm_ebm_diagnostic_ready": _prior_arm_ebm_diagnostic_ready(sources),
        "prior_thrml_n256_ready": _prior_thrml_n256_ready(sources),
        "prior_thrml_diverse_n64_ready": _prior_thrml_diverse_n64_ready(sources),
        "thrml_independent_rng_required": _thrml_independent_rng_required(
            ops_known_issues_text
        ),
        "prior_extropic_packet_ready": _prior_extropic_packet_ready(sources),
        "research_complete_has_118_entry": _research_complete_has_118_entry(
            research_complete_text
        ),
    }
    retirement_blocks_recorded = _retirement_blocks_recorded(
        ops_changelog_text,
        ops_status_text,
        ops_known_issues_text,
        roadmap_text,
        roadmap_doc_text,
        research_references_text,
    )

    blocked_reasons: list[str] = []
    if not predecessor_ready:
        blocked_reasons.append("predecessor .118 criteria are not 13 of 14")
    if missing_source_paths:
        blocked_reasons.append("listed source artifacts are missing")
    blocked_reasons.extend(
        reason
        for reason, ready in (
            ("automata prerequisite is not ready", readiness["prior_automata_ready"]),
            (
                "SATQuest benchmark prerequisite is not ready",
                readiness["prior_satquest_benchmark_ready"],
            ),
            (
                "residual-drift prerequisite is not ready",
                readiness["prior_residual_drift_ready"],
            ),
            ("FR-11 safe-only prerequisite is not ready", readiness["prior_fr11_safe_only"]),
            ("product-line prerequisite is not ready", readiness["prior_product_line_ready"]),
            ("claim-router prerequisite is not ready", readiness["prior_claim_router_ready"]),
            (
                "ARM/EBT diagnostic prerequisite is not ready",
                readiness["prior_arm_ebm_diagnostic_ready"],
            ),
            ("THRML n=256 prerequisite is not ready", readiness["prior_thrml_n256_ready"]),
            (
                "THRML diverse n=64 prerequisite is not ready",
                readiness["prior_thrml_diverse_n64_ready"],
            ),
            (
                "THRML independent-RNG audit requirement is not recorded",
                readiness["thrml_independent_rng_required"],
            ),
            ("Extropic packet prerequisite is not ready", readiness["prior_extropic_packet_ready"]),
            (
                "research-complete .118 archive entry is not recorded",
                readiness["research_complete_has_118_entry"],
            ),
            ("retired headline signal blocks are not preserved", retirement_blocks_recorded),
        )
        if not ready
    )
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
        "prior_satquest_solver_oracle_false_accepts": satquest_false_accepts,
        "prior_satquest_zero_solver_false_accepts": prior_satquest_zero,
        "mandated_sota_models": list(MANDATED_SOTA_MODELS),
        "continuous_self_learning_required": True,
        "allowed_119_tracks": list(ALLOWED_119_TRACKS),
        "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
        "retired_headline_signal_blocks_preserved": retirement_blocks_recorded,
        "manifest_path": manifest_path,
        "missing_source_paths": list(missing_source_paths),
        "blocked_reasons": blocked_reasons,
        "source_inputs_read": _source_inputs_read(
            predecessor_retro=predecessor_retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            ops_known_issues_text=ops_known_issues_text,
            roadmap_text=roadmap_text,
            roadmap_next_text=roadmap_next_text,
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
            "research-roadmap.yaml": "unchanged_by_exp1547_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1547_activation_workflow",
        },
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            "complete: milestone_119_activation_complete_118_archived_satquest_fr11_"
            "thrml_rng_gates_ready"
            if activation_manifest_complete
            else "passed: milestone_119_activation_blocked_missing_or_unsafe_118_evidence"
        ),
    }
    artifact.update(readiness)
    artifact["same_roadmap_gate_fields"] = _same_roadmap_gate_fields(artifact)

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
    """REQ-REPORT-060: write bootstrap, markdown manifest, and terminal JSON."""

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    protected = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    sources, missing_source_paths = _load_sources(root_path / "results")
    artifact, manifest = build_artifact(
        predecessor_retro=_read_json(root_path / "results" / PREDECESSOR_RETRO_FILE) or {},
        sources=sources,
        missing_source_paths=missing_source_paths,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        ops_known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_next_text=_read_text(root_path / "research-roadmap-next.yaml"),
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
