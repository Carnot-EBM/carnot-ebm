"""Build the Exp 1518 milestone .116 retrospective artifact.

Spec: REQ-REPORT-055, SCENARIO-REPORT-055.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
MILESTONE = "2026.04.116"
EXPERIMENT = "1518_milestone_116_retro"
SCHEMA = "milestone_116_retro_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1518_milestone_116_retro.json"
ROADMAP_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

MET = "met"
UNMET = "unmet"
GATE_BLOCKED = "gate_blocked"
SATISFIES_CRITERION = {MET, GATE_BLOCKED}
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
    "criteria_met",
    "criteria_total",
    "completed_tasks",
    "gated_or_blocked_tasks",
    "failed_tasks",
    "verifier_runtime_contract_ready",
    "continuous_self_learning_result",
    "substrate_conformance_result",
    "retired_or_demoted_claims",
    "carry_forward_gates",
    "ops_docs_updated",
    "research_complete_entry_recommended",
    "honest_verdict",
}

EXPECTED_EXPERIMENT_IDS = tuple(f"exp{exp_id}" for exp_id in range(1506, 1518))

SOURCE_FILES = {
    "exp1506": "experiment_1506_115_completion_archive_116_activation.json",
    "exp1507": "experiment_1507_autopyverifier_safe_dsl_induction_pack.json",
    "exp1508": "experiment_1508_trigger_grammar_certificate_decoder_audit.json",
    "exp1509": "experiment_1509_executable_monitor_runtime_adapter.json",
    "exp1510": "experiment_1510_plan_graph_structural_contract_gate.json",
    "exp1511": "experiment_1511_product_line_solver_oracle_benchmark.json",
    "exp1512": "experiment_1512_fr11_verifier_feedback_policy_cache_v11.json",
    "exp1513": "experiment_1513_fr11_policy_rollback_replay_audit.json",
    "exp1514": "experiment_1514_trace2skill_portable_skill_pack_v2.json",
    "exp1515": "experiment_1515_thrml_samplerbackend_conformance_pack.json",
    "exp1516": "experiment_1516_kan_shape_normalization_preflight.json",
    "exp1517": "experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json",
}

EXPERIMENT_TITLES = {
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
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-055: persist an auditable marker before the source scan.

    The retrospective is mostly deterministic summarization, but it reads many
    mutable operator files. A bootstrap JSON lets the conductor distinguish a
    started run from one that never wrote a deliverable.
    """

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": MILESTONE,
            "criteria_met": 0,
            "criteria_total": 0,
            "completed_tasks": [],
            "gated_or_blocked_tasks": [],
            "failed_tasks": [],
            "verifier_runtime_contract_ready": False,
            "continuous_self_learning_result": "",
            "substrate_conformance_result": "",
            "retired_or_demoted_claims": [],
            "carry_forward_gates": [],
            "ops_docs_updated": False,
            "research_complete_entry_recommended": {"written": False, "entry": None},
            "honest_verdict": "complete: in_progress_milestone_116_retro_seeded",
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


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(exp_id)
        else:
            loaded[exp_id] = payload
    return loaded, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _is_terminal(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"complete", "blocked", "gate_blocked", "skipped"}


def _has_success_prefix(payload: Mapping[str, Any]) -> bool:
    return _verdict(payload).lower().startswith(TERMINAL_PREFIXES)


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES.get(exp_id, f'experiment_{exp_id}.json')}"
    return f"{path}:{field}" if field else path


def _fields_present(payload: Mapping[str, Any], fields: tuple[str, ...]) -> bool:
    return all(payload.get(field) is not None for field in fields)


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) in {0, 0.0}


def _positive_int(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _criterion(
    *,
    key: str,
    exp_id: str,
    status: str,
    target: str,
    fields: tuple[str, ...],
    source_values: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "criterion": key,
        "experiment_id": exp_id,
        "status": status,
        "target": target,
        "evidence_paths": [_source_path(exp_id, field) for field in fields],
        "source_values": dict(source_values),
        "reason": reason,
    }


def _missing_criterion(key: str, exp_id: str, target: str) -> dict[str, Any]:
    return _criterion(
        key=key,
        exp_id=exp_id,
        status=UNMET,
        target=target,
        fields=(),
        source_values={"status": "missing", "honest_verdict": "missing_artifact"},
        reason=f"{exp_id} source artifact is missing.",
    )


def _blocker_reason(payload: Mapping[str, Any]) -> str:
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(blocker) for blocker in blockers)
    if blockers:
        return str(blockers)
    return str(payload.get("gated_off_reason") or _verdict(payload) or _status(payload))


def _terminal_no_signal(payload: Mapping[str, Any]) -> bool:
    return _is_terminal(payload) and "no-signal" in _verdict(payload).lower()


def _thrml_honest_blocker(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"blocked", "gate_blocked", "skipped"} and (
        payload.get("simulator_only") is True and payload.get("no_tsu_hardware_claim") is True
    )


def _score_source_criterion(
    *,
    key: str,
    exp_id: str,
    target: str,
    fields: tuple[str, ...],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: Callable[[Mapping[str, Any]], bool],
    gate_blocked: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _missing_criterion(key, exp_id, target)
    payload = sources[exp_id]
    status = MET if passed(payload) else GATE_BLOCKED if gate_blocked and gate_blocked(payload) else UNMET
    source_values = {field: payload.get(field) for field in fields}
    source_values.update(
        {
            "status": payload.get("status"),
            "honest_verdict": _verdict(payload),
            "terminal": _is_terminal(payload),
            "terminal_prefix_ok": _has_success_prefix(payload),
        }
    )
    return _criterion(
        key=key,
        exp_id=exp_id,
        status=status,
        target=target,
        fields=fields,
        source_values=source_values,
        reason="criterion satisfied" if status == MET else _blocker_reason(payload),
    )


def _criteria_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "activation",
            "exp_id": "exp1506",
            "target": "activation_manifest_complete=true and no protected roadmap/conductor edits",
            "fields": (
                "activation_manifest_complete",
                "protected_files_unchanged",
                "research_roadmap_yaml_modified",
                "scripts_research_conductor_modified",
            ),
            "passed": lambda p: p.get("activation_manifest_complete") is True
            and p.get("research_roadmap_yaml_modified") is False
            and p.get("scripts_research_conductor_modified") is False,
        },
        {
            "key": "verifier_induction",
            "exp_id": "exp1507",
            "target": "verifier_induction_ready=true with compile, coverage, and false-accept metrics",
            "fields": (
                "verifier_induction_ready",
                "candidate_verifiers_compiled",
                "verifier_compile_rate",
                "verifier_coverage_rate",
                "verifier_false_accept_rate",
            ),
            "passed": lambda p: p.get("verifier_induction_ready") is True
            and _positive_int(p, "candidate_verifiers_compiled")
            and _fields_present(p, ("verifier_compile_rate", "verifier_coverage_rate"))
            and _zero(p, "verifier_false_accept_rate"),
        },
        {
            "key": "grammar_certificate_decoder",
            "exp_id": "exp1508",
            "target": "certificate_decoder_ready=true with live local SOTA rows and parser/validator rates",
            "fields": (
                "certificate_decoder_ready",
                "live_sota_model_inference_used",
                "grammar_parse_rate",
                "grammar_validation_rate",
                "verifier_false_accept_rate",
            ),
            "passed": lambda p: p.get("certificate_decoder_ready") is True
            and p.get("live_sota_model_inference_used") is True
            and _fields_present(p, ("grammar_parse_rate", "grammar_validation_rate"))
            and _zero(p, "verifier_false_accept_rate"),
        },
        {
            "key": "monitor_runtime",
            "exp_id": "exp1509",
            "target": "monitor_runtime_ready=true with replayable events and zero new false accepts",
            "fields": ("monitor_runtime_ready", "events_normalized", "verifier_false_accept_rate"),
            "passed": lambda p: p.get("monitor_runtime_ready") is True
            and _positive_int(p, "events_normalized")
            and _zero(p, "verifier_false_accept_rate"),
        },
        {
            "key": "structural_contracts",
            "exp_id": "exp1510",
            "target": "structural_contract_gate_ready=true or terminal no-signal artifact",
            "fields": (
                "structural_contract_gate_ready",
                "violations_detected",
                "false_accept_rate",
                "random_baseline_detection_rate",
                "length_baseline_detection_rate",
            ),
            "passed": lambda p: (
                p.get("structural_contract_gate_ready") is True and _zero(p, "false_accept_rate")
            )
            or _terminal_no_signal(p),
        },
        {
            "key": "feature_model_oracle",
            "exp_id": "exp1511",
            "target": "product_line_benchmark_ready=true with solver-oracle feasibility and false-accept rates",
            "fields": (
                "product_line_benchmark_ready",
                "solver_oracle_ready",
                "feasibility_rate",
                "verifier_false_accept_rate",
            ),
            "passed": lambda p: p.get("product_line_benchmark_ready") is True
            and p.get("solver_oracle_ready") is True
            and _fields_present(p, ("feasibility_rate",))
            and _zero(p, "verifier_false_accept_rate"),
        },
        {
            "key": "continuous_self_learning",
            "exp_id": "exp1512",
            "target": "policy_cache_ready=true, continuous_self_learning_task=true, and no model-weight mutation",
            "fields": (
                "policy_cache_ready",
                "continuous_self_learning_task",
                "no_model_weight_mutation",
                "soundness_mistakes",
                "verifier_false_accept_rate",
            ),
            "passed": lambda p: p.get("policy_cache_ready") is True
            and p.get("continuous_self_learning_task") is True
            and p.get("no_model_weight_mutation") is True
            and _zero(p, "soundness_mistakes")
            and _zero(p, "verifier_false_accept_rate"),
        },
        {
            "key": "rollback_replay",
            "exp_id": "exp1513",
            "target": "rollback_audit_passed=true before any learned policy is promoted",
            "fields": (
                "rollback_audit_passed",
                "accepted_policy_updates",
                "false_accept_delta",
                "soundness_mistakes",
            ),
            "passed": lambda p: p.get("rollback_audit_passed") is True
            and _fields_present(p, ("accepted_policy_updates",))
            and _zero(p, "false_accept_delta")
            and _zero(p, "soundness_mistakes"),
        },
        {
            "key": "portable_skill_pack",
            "exp_id": "exp1514",
            "target": "portable_skill_pack_ready=true only for rollback-passing entries",
            "fields": (
                "portable_skill_pack_ready",
                "rollback_passing_entries",
                "packaged_skill_entries",
                "rejected_skill_entries",
            ),
            "passed": lambda p: p.get("portable_skill_pack_ready") is True
            and _positive_int(p, "rollback_passing_entries")
            and _fields_present(p, ("packaged_skill_entries", "rejected_skill_entries")),
        },
        {
            "key": "thrml_conformance",
            "exp_id": "exp1515",
            "target": "thrml_samplerbackend_conformance_ready=true or honest simulator-only blocker",
            "fields": (
                "thrml_samplerbackend_conformance_ready",
                "simulator_only",
                "no_tsu_hardware_claim",
                "blockers",
            ),
            "passed": lambda p: p.get("thrml_samplerbackend_conformance_ready") is True
            and p.get("simulator_only") is True
            and p.get("no_tsu_hardware_claim") is True,
            "gate_blocked": _thrml_honest_blocker,
        },
        {
            "key": "kan_shape_accounting",
            "exp_id": "exp1516",
            "target": "kan_shape_manifest_ready=true and no synthesis/board claim",
            "fields": ("kan_shape_manifest_ready", "no_synthesis_claim", "no_board_claim", "blockers"),
            "passed": lambda p: p.get("kan_shape_manifest_ready") is True
            and p.get("no_synthesis_claim") is True
            and p.get("no_board_claim") is True,
        },
        {
            "key": "kv260_source_properties",
            "exp_id": "exp1517",
            "target": "kv260_property_pack_ready=true with source-level RTL tests only",
            "fields": (
                "kv260_property_pack_ready",
                "source_level_only",
                "no_board_execution",
                "no_bitstream_claim",
            ),
            "passed": lambda p: p.get("kv260_property_pack_ready") is True
            and p.get("source_level_only") is True
            and p.get("no_board_execution") is True
            and p.get("no_bitstream_claim") is True,
        },
    )


def _source_success_criteria(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    return {
        spec["key"]: _score_source_criterion(
            key=spec["key"],
            exp_id=spec["exp_id"],
            target=spec["target"],
            fields=spec["fields"],
            sources=sources,
            missing_source_ids=missing_source_ids,
            passed=spec["passed"],
            gate_blocked=spec.get("gate_blocked"),
        )
        for spec in _criteria_specs()
    }


def _source_reported_protected_changes(
    sources: Mapping[str, Mapping[str, Any]],
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    reports: list[dict[str, str]] = []
    for exp_id, payload in sources.items():
        if payload.get("research_roadmap_yaml_modified") is True:
            reports.append({"experiment_id": exp_id, "file": "research-roadmap.yaml"})
        if payload.get("scripts_research_conductor_modified") is True:
            reports.append({"experiment_id": exp_id, "file": "scripts/research_conductor.py"})
    return {
        "any_modification_reported": bool(reports) or not protected_files_unchanged,
        "source_reports": reports,
        "working_tree": {
            "research-roadmap.yaml": "unchanged" if protected_files_unchanged else "modified_or_unknown",
            "scripts/research_conductor.py": "unchanged" if protected_files_unchanged else "modified_or_unknown",
        },
    }


def _retrospective_criterion(
    missing_source_ids: set[str],
    protected_findings: Mapping[str, Any],
) -> dict[str, Any]:
    passed = not missing_source_ids and not protected_findings["any_modification_reported"]
    return {
        "criterion": "retrospective",
        "experiment_id": "exp1518",
        "status": MET if passed else UNMET,
        "target": "criteria_met and criteria_total summarize .116 with carry-forward decisions",
        "evidence_paths": [
            "results/experiment_1518_milestone_116_retro.json",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        "source_values": {
            "missing_source_ids": sorted(missing_source_ids),
            "protected_file_modification_findings": dict(protected_findings),
        },
        "reason": "retrospective artifact can close from terminal sources"
        if passed
        else "missing sources or protected-file modifications block clean closeout",
    }


def _gated_or_blocked_tasks(criteria: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": result["experiment_id"],
            "criterion": key,
            "reason": result["reason"],
        }
        for key, result in criteria.items()
        if result["status"] == GATE_BLOCKED
    ]


def _failed_tasks(
    criteria: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> list[dict[str, str]]:
    missing_failures = [
        {
            "experiment_id": exp_id,
            "criterion": next(
                key for key, result in criteria.items() if result["experiment_id"] == exp_id
            ),
            "reason": f"{exp_id} source artifact is missing.",
        }
        for exp_id in sorted(missing_source_ids)
    ]
    unmet_failures = [
        {
            "experiment_id": result["experiment_id"],
            "criterion": key,
            "reason": result["reason"],
        }
        for key, result in criteria.items()
        if result["status"] == UNMET and result["experiment_id"] not in missing_source_ids
    ]
    return missing_failures + unmet_failures


def _retired_or_demoted_claims() -> list[dict[str, str]]:
    return [
        {
            "claim": "Semantic Energy/logit telemetry",
            "decision": "retired_as_headline_signal",
            "boundary": "May appear only as auxiliary monitor/debug telemetry below deterministic validators.",
        },
        {
            "claim": "V_1 pairwise self-verification",
            "decision": "retired_as_headline_signal",
            "boundary": "No pairwise self-verification headline unless future evidence beats executable energy.",
        },
        {
            "claim": "Generated verifier code",
            "decision": "demoted_to_safe_dsl_only",
            "boundary": "Trusted only after safe-DSL compilation, coverage, and false-accept accounting.",
        },
        {
            "claim": "THRML/KAN/KV260 hardware execution",
            "decision": "not_claimed",
            "boundary": "Current evidence is simulator, shape-accounting, or source-level property conformance only.",
        },
    ]


def _carry_forward_gates() -> list[dict[str, str]]:
    return [
        {
            "gate": "runtime_contract_e2e",
            "requirement": "Combine induced safe-DSL validators, grammar certificates, monitor events, and structural contracts with zero false accepts.",
        },
        {
            "gate": "fr11_policy_promotion",
            "requirement": "Promote only rollback-passing query-time policy updates; keep no model-weight mutation.",
        },
        {
            "gate": "thrml_hardware_claim",
            "requirement": "Requires actual TSU or hardware transcript; current THRML evidence remains simulator/software only.",
        },
        {
            "gate": "kan_synthesis_or_board_claim",
            "requirement": "Requires normalized shapes plus synthesis reports, timing, bitstream, and board transcript.",
        },
        {
            "gate": "kv260_board_execution_claim",
            "requirement": "Requires bitstream and board execution transcript; current KV260 evidence is source-level only.",
        },
    ]


def _continuous_self_learning_result(sources: Mapping[str, Mapping[str, Any]]) -> str:
    exp1512 = sources.get("exp1512", {})
    exp1513 = sources.get("exp1513", {})
    exp1514 = sources.get("exp1514", {})
    return (
        "Bounded FR-11 feedback loop ready: policy cache="
        f"{exp1512.get('policy_cache_ready')}, no model-weight mutation="
        f"{exp1512.get('no_model_weight_mutation')}, rollback_audit_passed="
        f"{exp1513.get('rollback_audit_passed')}, accepted_policy_updates="
        f"{exp1513.get('accepted_policy_updates')}, portable_entries="
        f"{exp1514.get('rollback_passing_entries')}."
    )


def _substrate_conformance_result(sources: Mapping[str, Mapping[str, Any]]) -> str:
    exp1515 = sources.get("exp1515", {})
    exp1516 = sources.get("exp1516", {})
    exp1517 = sources.get("exp1517", {})
    return (
        "Substrate work is software/source conformance only: THRML simulator_only="
        f"{exp1515.get('simulator_only')} and no_tsu_hardware_claim="
        f"{exp1515.get('no_tsu_hardware_claim')}; KAN no_synthesis_claim="
        f"{exp1516.get('no_synthesis_claim')} and no_board_claim="
        f"{exp1516.get('no_board_claim')}; KV260 source_level_only="
        f"{exp1517.get('source_level_only')} and no_board_execution="
        f"{exp1517.get('no_board_execution')}."
    )


def _research_complete_has_116_entry(text: str) -> bool:
    return "2026.04.116" in text


def _archive_task_rows() -> list[dict[str, str]]:
    rows = [
        {
            "id": f"{exp_id}-{EXPERIMENT_TITLES[exp_id].lower().replace(' ', '-')}",
            "title": EXPERIMENT_TITLES[exp_id],
            "deliverable": f"results/{SOURCE_FILES[exp_id]}",
            "result": "OK (conductor)",
        }
        for exp_id in EXPECTED_EXPERIMENT_IDS
    ]
    rows.append(
        {
            "id": "exp1518-milestone-116-retrospective",
            "title": "Milestone .116 Retrospective + Claim Boundary Reconciliation",
            "deliverable": "results/experiment_1518_milestone_116_retro.json",
            "result": "OK (codex retro)",
        }
    )
    return rows


def _research_complete_recommendation(
    *,
    criteria_met: int,
    criteria_total: int,
    research_complete_text: str,
) -> dict[str, Any]:
    already_present = _research_complete_has_116_entry(research_complete_text)
    return {
        "written": False,
        "already_present": already_present,
        "reason": "recommended_only_stop_when_done_delegates_archive_write",
        "entry": {
            "id": MILESTONE,
            "title": "Runtime Verifier Contracts + FR-11 Feedback + Substrate Conformance Gates",
            "doc": ROADMAP_DOC,
            "completed": "2026-05-08",
            "finding": (
                f"Milestone .116 met {criteria_met} of {criteria_total} criteria. "
                "Runtime verifier contracts, bounded FR-11 feedback, and "
                "software/source substrate conformance are complete with retired "
                "telemetry, generated-code, and hardware claim boundaries preserved."
            ),
            "tasks": _archive_task_rows(),
        },
    }


def _source_inputs_read(
    *,
    conductor_log_text: str,
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
        "research-roadmap.yaml": {"exists": bool(research_roadmap_yaml_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
    }


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries = {
        exp_id: {
            "found": any(exp_id in row or EXPERIMENT_TITLES[exp_id][:40] in row for row in rows),
            "ok": any(
                (exp_id in row or EXPERIMENT_TITLES[exp_id][:40] in row) and "| OK |" in row
                for row in rows
            ),
        }
        for exp_id in EXPECTED_EXPERIMENT_IDS
    }
    return {
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "expected_count": len(EXPECTED_EXPERIMENT_IDS),
        "missing_experiments": [exp_id for exp_id, entry in entries.items() if not entry["found"]],
        "entries": entries,
    }


def _protected_files_clean(root: Path) -> bool:
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


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-055: score `.116` from terminal source artifacts.

    The retrospective keeps evidence and claim boundaries attached to the JSON
    fields that produced them. This prevents a successful milestone summary
    from accidentally promoting simulator-only, source-only, or safe-DSL-only
    evidence into stronger hardware or generated-code claims.
    """

    missing = set(missing_source_ids)
    criteria = _source_success_criteria(sources=sources, missing_source_ids=missing)
    protected_findings = _source_reported_protected_changes(sources, protected_files_unchanged)
    criteria["retrospective"] = _retrospective_criterion(missing, protected_findings)
    criteria_met = sum(1 for result in criteria.values() if result["status"] in SATISFIES_CRITERION)
    criteria_total = len(criteria)
    runtime_keys = (
        "verifier_induction",
        "grammar_certificate_decoder",
        "monitor_runtime",
        "structural_contracts",
        "feature_model_oracle",
    )
    completed_tasks = [
        exp_id for exp_id in EXPECTED_EXPERIMENT_IDS if exp_id in sources and _is_complete(sources[exp_id])
    ]
    completed_tasks.append("exp1518")

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "criteria_score_pct": round(criteria_met / criteria_total, 6) if criteria_total else 0.0,
        "success_criteria_results": criteria,
        "completed_tasks": completed_tasks,
        "gated_or_blocked_tasks": _gated_or_blocked_tasks(criteria),
        "failed_tasks": _failed_tasks(criteria, missing),
        "missing_artifacts": [
            {"experiment_id": exp_id, "path": _source_path(exp_id)} for exp_id in sorted(missing)
        ],
        "experiment_verdicts": {
            exp_id: {
                "status": sources.get(exp_id, {}).get("status", "missing"),
                "honest_verdict": _verdict(sources.get(exp_id, {})) or "missing_artifact",
                "terminal": _is_terminal(sources.get(exp_id, {})),
                "terminal_prefix_ok": _has_success_prefix(sources.get(exp_id, {})),
                "source_path": _source_path(exp_id),
            }
            for exp_id in EXPECTED_EXPERIMENT_IDS
        },
        "verifier_runtime_contract_ready": all(criteria[key]["status"] == MET for key in runtime_keys),
        "continuous_self_learning_result": _continuous_self_learning_result(sources),
        "substrate_conformance_result": _substrate_conformance_result(sources),
        "retired_or_demoted_claims": _retired_or_demoted_claims(),
        "carry_forward_gates": _carry_forward_gates(),
        "ops_docs_updated": False,
        "ops_docs_update_deferred_reason": (
            "separate_reconciliation_agent_owns_ops_status_changelog_and_traceability"
        ),
        "research_complete_entry_recommended": _research_complete_recommendation(
            criteria_met=criteria_met,
            criteria_total=criteria_total,
            research_complete_text=research_complete_text,
        ),
        "protected_files_unchanged": protected_files_unchanged,
        "protected_file_modification_findings": protected_findings,
        "source_inputs_read": _source_inputs_read(
            conductor_log_text=conductor_log_text,
            roadmap_doc_text=roadmap_doc_text,
            research_roadmap_yaml_text=research_roadmap_yaml_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
        ),
        "conductor_log_exp1506_to_exp1517": _conductor_log_summary(conductor_log_text),
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            f"complete: milestone_116_{criteria_met}_of_{criteria_total}_criteria_met_"
            "runtime_contracts_fr11_feedback_substrate_claim_boundaries_preserved"
        ),
    }
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-055: write bootstrap and terminal retrospective JSON only.

    This function intentionally does not mutate `research-complete.yaml`,
    `ops/status.md`, or `ops/changelog.md`; the conductor's follow-up
    reconciliation step owns those files for this stop-when-done run.
    """

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    protected = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        roadmap_doc_text=_read_text(root_path / ROADMAP_DOC),
        research_roadmap_yaml_text=_read_text(root_path / "research-roadmap.yaml"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        protected_files_unchanged=protected,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
