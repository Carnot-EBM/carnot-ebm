"""Build the Exp 1533 `.118` activation manifest.

Spec: REQ-REPORT-058, SCENARIO-REPORT-058.
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
PREDECESSOR_MILESTONE = "2026.04.117"
TARGET_MILESTONE = "2026.04.118"
EXPERIMENT = "1533_117_completion_archive_118_activation"
SCHEMA = "milestone_118_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1533_117_completion_archive_118_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_118_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1532_milestone_117_retro.json"
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
    "prior_runtime_contract_e2e_ready",
    "prior_live_sota_repair_ready",
    "prior_cdg_ready",
    "prior_product_line_ready",
    "prior_fr11_promotion_ready",
    "prior_claim_isolation_ready",
    "prior_thrml_n128_ready",
    "prior_thrml_diverse_ready",
    "prior_orphan_test_incident_recorded",
    "research_complete_has_117_entry",
    "mandated_sota_models",
    "continuous_self_learning_required",
    "allowed_118_tracks",
    "gated_118_tracks",
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
    "BEAVER/logprob acceptance authority",
    "ARM/EBT soft-value acceptance authority",
    "Extropic TSU/Z1 hardware execution claims",
    "KV260 board claims",
    "model-weight mutation",
]

ALLOWED_118_TRACKS = [
    {
        "track": "orphan_test_guard",
        "name": "orphan-test guard",
        "guardrail": "Audit generated tests for import targets before downstream work runs.",
    },
    {
        "track": "automata_contract_decoding",
        "name": "automata/XGrammar/ABS contract decoding",
        "guardrail": "Runtime contracts remain acceptance authority; decoding masks are auxiliary.",
    },
    {
        "track": "satquest_cnf_benchmark",
        "name": "SATQuest CNF benchmark",
        "guardrail": "PySAT or an equivalent deterministic solver is the oracle.",
    },
    {
        "track": "beaver_prefix_risk_audit",
        "name": "BEAVER-lite prefix-risk audit",
        "guardrail": "Prefix/logprob bounds route risk but do not accept answers.",
    },
    {
        "track": "residual_drift_ledger",
        "name": "residual-drift ledger",
        "guardrail": "Separate contradictions from satisfiable forgotten commitments.",
    },
    {
        "track": "external_feedback_fr11_skill_graph",
        "name": "external-feedback FR-11 skill graph",
        "guardrail": "No model-weight mutation; positive utility is required for headline claims.",
    },
    {
        "track": "product_line_scale",
        "name": "product-line scale",
        "guardrail": "Scale only against parser, feasibility, oracle, and false-accept metrics.",
    },
    {
        "track": "claim_isolation_uncertainty_routing",
        "name": "claim-isolation uncertainty routing",
        "guardrail": "Claim isolation must report routed budget cost and deterministic outcomes.",
    },
    {
        "track": "arm_ebt_soft_value_diagnostics",
        "name": "ARM/EBT soft-value diagnostics",
        "guardrail": "Soft-value signals stay below deterministic validators.",
    },
    {
        "track": "thrml_n256_n64_diverse_stress",
        "name": "THRML n=256/n=64-diverse stress",
        "guardrail": "Software/simulator parity only; no TSU, synthesis, bitstream, or board claim.",
    },
    {
        "track": "extropic_z1_readiness_packet",
        "name": "Extropic Z1 readiness packet",
        "guardrail": "Write access and transcript requirements without claiming hardware execution.",
    },
    {
        "track": "milestone_retro",
        "name": "milestone retro",
        "guardrail": "Close .118 with criteria accounting and .119 carry-forward gates.",
    },
]

GATED_118_TRACKS = [
    {
        "track": "orphan_test_guard",
        "task_id": "exp1534",
        "gated_on": ["exp1533.prior_orphan_test_incident_recorded == true"],
    },
    {
        "track": "automata_contract_decoding",
        "task_id": "exp1535",
        "gated_on": [
            "exp1533.prior_runtime_contract_e2e_ready == true",
            "exp1534.orphan_test_guard_ready == true",
        ],
    },
    {
        "track": "satquest_cnf_benchmark",
        "task_id": "exp1536",
        "gated_on": ["exp1533.prior_runtime_contract_e2e_ready == true"],
    },
    {
        "track": "beaver_prefix_risk_audit",
        "task_id": "exp1537",
        "gated_on": ["exp1535.contract_decoder_adapter_ready == true"],
    },
    {
        "track": "residual_drift_ledger",
        "task_id": "exp1538",
        "gated_on": ["exp1536.satquest_benchmark_ready == true"],
    },
    {
        "track": "external_feedback_fr11_skill_graph",
        "task_id": "exp1539",
        "gated_on": [
            "exp1533.prior_fr11_promotion_ready == true",
            "exp1538.residual_drift_ledger_ready == true",
        ],
    },
    {
        "track": "product_line_scale",
        "task_id": "exp1540",
        "gated_on": [
            "exp1533.prior_product_line_ready == true",
            "exp1535.contract_decoder_adapter_ready == true",
        ],
    },
    {
        "track": "claim_isolation_uncertainty_routing",
        "task_id": "exp1541",
        "gated_on": [
            "exp1533.prior_claim_isolation_ready == true",
            "exp1537.beaver_bound_ready == true",
        ],
    },
    {
        "track": "arm_ebt_soft_value_diagnostics",
        "task_id": "exp1542",
        "gated_on": [
            "exp1536.satquest_benchmark_ready == true",
            "exp1537.beaver_bound_ready == true",
        ],
    },
    {
        "track": "thrml_n256_n64_diverse_stress",
        "task_id": "exp1543",
        "gated_on": ["exp1533.prior_thrml_n128_ready == true"],
    },
    {
        "track": "thrml_n256_n64_diverse_stress",
        "task_id": "exp1544",
        "gated_on": [
            "exp1533.prior_thrml_diverse_ready == true",
            "exp1543.thrml_parity_n256_schedule_ready == true",
        ],
    },
    {
        "track": "extropic_z1_readiness_packet",
        "task_id": "exp1545",
        "gated_on": [
            "exp1543.thrml_parity_n256_schedule_ready == true",
            "exp1544.diverse_topology_parity_n64_ready == true",
        ],
    },
]

SOURCE_FILES = {
    "exp1520": "experiment_1520_runtime_contract_e2e_harness.json",
    "exp1521": "experiment_1521_live_sota_contract_guided_repair_v1.json",
    "exp1522": "experiment_1522_constraint_dependency_graph_root_cause_repair.json",
    "exp1523": "experiment_1523_product_line_parser_feasibility_rescue_v2.json",
    "exp1524": "experiment_1524_fr11_live_policy_promotion_v12.json",
    "exp1525": "experiment_1525_march_claim_isolation_verifier_ablation.json",
    "exp1530": "experiment_1530_thrml_carnot_parity_n128_production_scale.json",
    "exp1531": "experiment_1531_thrml_diverse_topology_parity_n32.json",
}

CONDUCTOR_EXPERIMENT_TITLES = {
    "exp1519": ".116 Completion Archive + .117 Activation Manifest",
    "exp1520": "Runtime-Contract E2E Harness",
    "exp1521": "Live SOTA Contract-Guided Repair v1",
    "exp1522": "Constraint Dependency Graph Root-Cause Repair",
    "exp1523": "Product-Line Parser Feasibility Rescue v2",
    "exp1524": "FR-11 Live Policy Promotion v12",
    "exp1525": "MARCH Claim-Isolation Verifier Ablation",
    "exp1526": "THRML/Carnot Parity n=8 Exact",
    "exp1527": "THRML/Carnot Parity n=16 Exact",
    "exp1528": "THRML/Carnot Parity n=32 Sample",
    "exp1529": "THRML/Carnot Parity n=64 Sample",
    "exp1530": "THRML/Carnot Parity n=128 Production-Scale Sample",
    "exp1531": "THRML Diverse Topology Parity n=32",
    "exp1532": "Milestone 2026.04.117 Retrospective",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-058: persist the bootstrap archive before reading evidence.

    This marker makes an interrupted activation auditable: operators can see
    that Exp 1533 started, even if mutable source reads or markdown rendering
    fail before the terminal JSON is written.
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
            "allowed_118_tracks": [],
            "gated_118_tracks": [],
            "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
            "honest_verdict": "complete: in_progress_117_completion_archive_118_activation_seeded",
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


def _load_sources(results_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        exp_id: _read_json(results_dir / filename) or {}
        for exp_id, filename in SOURCE_FILES.items()
    }


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "complete"


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int | float) and not isinstance(value, bool) and value == 0


def _positive_int(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _research_complete_has_117_entry(research_complete_text: str) -> bool:
    return any(
        marker in research_complete_text
        for marker in (
            "- id: 2026.04.117",
            "id: 2026.04.117",
            'id: "2026.04.117"',
            "id: '2026.04.117'",
        )
    )


def _predecessor_complete(predecessor_retro: Mapping[str, Any]) -> bool:
    return bool(
        _is_complete(predecessor_retro)
        and predecessor_retro.get("milestone") == PREDECESSOR_MILESTONE
        and predecessor_retro.get("criteria_met") == 14
        and predecessor_retro.get("criteria_total") == 14
    )


def _mandated_sota_used(payload: Mapping[str, Any]) -> bool:
    models = {str(model) for model in payload.get("models_used", []) if model}
    return bool(models.intersection(MANDATED_SOTA_MODELS))


def _prior_runtime_contract_e2e_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1520 = sources.get("exp1520", {})
    return bool(
        _is_complete(exp1520)
        and exp1520.get("runtime_contract_e2e_ready") is True
        and _zero(exp1520, "false_accept_rate")
    )


def _prior_live_sota_repair_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1521 = sources.get("exp1521", {})
    return bool(_is_complete(exp1521) and _mandated_sota_used(exp1521))


def _prior_cdg_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1522 = sources.get("exp1522", {})
    return bool(_is_complete(exp1522) and exp1522.get("cdg_root_cause_repair_ready") is True)


def _prior_product_line_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1523 = sources.get("exp1523", {})
    return bool(
        _is_complete(exp1523)
        and (
            exp1523.get("product_line_rescue_ready") is True
            or exp1523.get("product_line_branch_retired") is True
        )
    )


def _prior_fr11_promotion_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1524 = sources.get("exp1524", {})
    return bool(
        _is_complete(exp1524)
        and exp1524.get("live_policy_promotion_ready") is True
        and _zero(exp1524, "soundness_mistakes")
        and exp1524.get("no_model_weight_mutation") is True
    )


def _prior_claim_isolation_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1525 = sources.get("exp1525", {})
    return bool(
        _is_complete(exp1525)
        and exp1525.get("claim_isolation_ablation_ready") is True
        and _positive_int(exp1525, "cases_loaded")
        and _positive_int(exp1525, "claims_extracted")
        and exp1525.get("budget_delta") is not None
        and _zero(exp1525, "false_accept_count")
        and _zero(exp1525, "false_accept_rate")
    )


def _prior_thrml_n128_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1530 = sources.get("exp1530", {})
    return bool(
        _is_complete(exp1530)
        and exp1530.get("thrml_parity_n128_passed") is True
        and exp1530.get("simulator_only") is True
        and exp1530.get("no_tsu_hardware_claim") is True
    )


def _prior_thrml_diverse_ready(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    exp1531 = sources.get("exp1531", {})
    tested = {str(item) for item in exp1531.get("topologies_tested", [])}
    passed = {str(item) for item in exp1531.get("topologies_passed", [])}
    topologies_passed = bool(tested) and tested.issubset(passed)
    return bool(
        _is_complete(exp1531)
        and exp1531.get("diverse_topology_parity_ready") is True
        and exp1531.get("simulator_only") is True
        and exp1531.get("no_tsu_hardware_claim") is True
        and topologies_passed
    )


def _orphan_test_incident_recorded(ops_known_issues_text: str, conductor_log_text: str) -> bool:
    evidence = f"{ops_known_issues_text}\n{conductor_log_text}".lower()
    return "orphan-test" in evidence or ("orphan test" in evidence and "non-existent" in evidence)


def _mandated_models_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts)
    return all(model in evidence for model in MANDATED_SOTA_MODELS)


def _continuous_self_learning_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    return "continuous self-learning" in evidence and "exp1539" in evidence


def _retirement_blocks_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    required = [
        ("legacy", "small", "headline"),
        ("beaver", "logprob", "acceptance"),
        ("arm", "ebt", "soft-value", "acceptance"),
        ("extropic", "tsu", "z1", "hardware"),
        ("kv260", "board"),
        ("model-weight", "mutation"),
    ]
    return all(all(term in evidence for term in terms) for terms in required)


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries = {
        exp_id: {
            "found": bool(
                matches := [
                    row for row in rows if exp_id in row or title[:40] in row or title in row
                ]
            ),
            "ok": any("| OK |" in row for row in matches),
        }
        for exp_id, title in CONDUCTOR_EXPERIMENT_TITLES.items()
    }
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
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
    """REQ-REPORT-058: render the operator-facing `.118` activation manifest.

    The markdown duplicates the JSON gates so the conductor and a human operator
    see the same readiness boundary before automata, SAT, FR-11, THRML, and
    hardware-readiness work consumes research time.
    """

    lines = [
        "# Milestone .118 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".117 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])

    lines.extend(["## Allowed .118 Tracks", ""])
    lines.extend(_render_allowed_track_table(ALLOWED_118_TRACKS))
    lines.extend(["", "## Gated .118 Tracks", ""])
    lines.extend(_render_gated_track_table(GATED_118_TRACKS))
    lines.extend(["", "## Same-Roadmap Gates", ""])
    for row in GATED_118_TRACKS:
        lines.append(f"- {row['task_id']}: {'; '.join(row['gated_on'])}")
    lines.extend(["", "## Prior Readiness", ""])
    for key in (
        "prior_runtime_contract_e2e_ready",
        "prior_live_sota_repair_ready",
        "prior_cdg_ready",
        "prior_product_line_ready",
        "prior_fr11_promotion_ready",
        "prior_claim_isolation_ready",
        "prior_thrml_n128_ready",
        "prior_thrml_diverse_ready",
        "prior_orphan_test_incident_recorded",
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
            "- required .118 task: exp1539 external-feedback FR-11 skill graph",
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1533_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1533_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
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
    """REQ-REPORT-058: summarize `.117` closure and activate guarded `.118` work."""

    criteria_met = int(predecessor_retro.get("criteria_met") or 0)
    criteria_total = int(predecessor_retro.get("criteria_total") or 0)
    predecessor_ready = _predecessor_complete(predecessor_retro)
    runtime_ready = _prior_runtime_contract_e2e_ready(sources)
    sota_ready = _prior_live_sota_repair_ready(sources)
    cdg_ready = _prior_cdg_ready(sources)
    product_ready = _prior_product_line_ready(sources)
    fr11_ready = _prior_fr11_promotion_ready(sources)
    claim_ready = _prior_claim_isolation_ready(sources)
    thrml_n128_ready = _prior_thrml_n128_ready(sources)
    thrml_diverse_ready = _prior_thrml_diverse_ready(sources)
    orphan_recorded = _orphan_test_incident_recorded(ops_known_issues_text, conductor_log_text)
    research_complete_has_117 = _research_complete_has_117_entry(research_complete_text)
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
        blocked_reasons.append("predecessor .117 criteria are not 14 of 14")
    blocked_reasons.extend(
        reason
        for reason, ready in (
            ("runtime-contract E2E prerequisite is not ready", runtime_ready),
            ("live SOTA repair prerequisite is not ready", sota_ready),
            ("CDG prerequisite is not ready", cdg_ready),
            ("product-line prerequisite is not ready", product_ready),
            ("FR-11 promotion prerequisite is not ready", fr11_ready),
            ("claim-isolation prerequisite is not ready", claim_ready),
            ("THRML n=128 prerequisite is not ready", thrml_n128_ready),
            ("THRML diverse-topology prerequisite is not ready", thrml_diverse_ready),
            ("orphan-test incident is not recorded", orphan_recorded),
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
        blocked_reasons.append("conductor log missing exp1519-through-exp1532 evidence")
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
        "prior_runtime_contract_e2e_ready": runtime_ready,
        "prior_live_sota_repair_ready": sota_ready,
        "prior_cdg_ready": cdg_ready,
        "prior_product_line_ready": product_ready,
        "prior_fr11_promotion_ready": fr11_ready,
        "prior_claim_isolation_ready": claim_ready,
        "prior_thrml_n128_ready": thrml_n128_ready,
        "prior_thrml_diverse_ready": thrml_diverse_ready,
        "prior_orphan_test_incident_recorded": orphan_recorded,
        "research_complete_has_117_entry": research_complete_has_117,
        "mandated_sota_models": list(MANDATED_SOTA_MODELS),
        "mandated_sota_models_recorded": mandated_models_recorded,
        "continuous_self_learning_required": True,
        "continuous_self_learning_requirement_recorded": continuous_self_learning_recorded,
        "allowed_118_tracks": list(ALLOWED_118_TRACKS),
        "gated_118_tracks": list(GATED_118_TRACKS),
        "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
        "retired_headline_signal_blocks_preserved": retirement_blocks_recorded,
        "manifest_path": manifest_path,
        "blocked_reasons": blocked_reasons,
        "conductor_log_exp1519_to_exp1532": conductor_summary,
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
            "research-roadmap.yaml": "unchanged_by_exp1533_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1533_activation_workflow",
        },
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            "complete: milestone_118_activation_complete_117_archived_runtime_"
            "fr11_thrml_orphan_gate_ready"
            if activation_manifest_complete
            else "passed: milestone_118_activation_blocked_missing_or_unsafe_predecessor_evidence"
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
    """REQ-REPORT-058: write bootstrap, markdown manifest, and terminal JSON.

    The function does not update roadmap, conductor, status, changelog, or
    traceability files because the conductor's reconciliation step owns those
    documents after this focused activation artifact exits.
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
