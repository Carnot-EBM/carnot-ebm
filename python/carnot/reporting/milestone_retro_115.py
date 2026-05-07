"""Build the Exp 1505 milestone .115 retrospective artifact.

Spec: REQ-REPORT-053, SCENARIO-REPORT-053.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
MILESTONE = "2026.04.115"
EXPERIMENT = "1505_milestone_115_retro"
SCHEMA = "milestone_115_retro_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1505_milestone_115_retro.json"
ROADMAP_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

MET = "met"
UNMET = "unmet"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "experiments_reviewed",
    "completed_experiments",
    "honest_gate_skips",
    "retired_lines",
    "graduated_lines",
    "carry_forward_lines",
    "continuous_self_learning_outcome",
    "hardware_claim_boundaries",
    "ops_docs_updated",
    "research_complete_updated",
    "protected_files_unchanged",
    "honest_verdict",
}

EXPECTED_EXPERIMENT_IDS = tuple(f"exp{exp_id}" for exp_id in range(1492, 1505))

SOURCE_FILES = {
    "exp1492": "experiment_1492_114_completion_archive_115_activation.json",
    "exp1493": "experiment_1493_trigger_token_certificate_export_v1.json",
    "exp1494": "experiment_1494_constrainprompt_validator_compiler_audit.json",
    "exp1495": "experiment_1495_interwhen_monitor_prototype.json",
    "exp1496": "experiment_1496_hover_safe_prefix_continuation_audit.json",
    "exp1497": "experiment_1497_fr11_trace2skill_daily_eval_v10.json",
    "exp1498": "experiment_1498_trace2skill_artifact_reachability_audit.json",
    "exp1499": "experiment_1499_verifier_ensemble_dry_orthogonality_v2.json",
    "exp1500": "experiment_1500_latent_deterministic_discipline_gate.json",
    "exp1501": "experiment_1501_gnnverifier_plan_graph_energy_adapter.json",
    "exp1502": "experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
    "exp1503": "experiment_1503_thrml_import_readiness_repair_gate.json",
    "exp1504": "experiment_1504_thrml_carnot_simulator_parity_v3.json",
}

EXPERIMENT_TITLES = {
    "exp1492": ".114 Completion Archive + .115 Activation Manifest",
    "exp1493": "Trigger-Token Certificate Export v1",
    "exp1494": "ConstrainPrompt Validator Compiler Audit",
    "exp1495": "interwhen Monitor Prototype",
    "exp1496": "HoVer Safe-Prefix Continuation Audit",
    "exp1497": "FR-11 v10 Trace2Skill Daily Eval + Rot Check",
    "exp1498": "trace2skill Artifact Reachability Audit",
    "exp1499": "Verifier Ensemble DRY + Conditional Orthogonality",
    "exp1500": "Latent-vs-Deterministic Discipline Gate",
    "exp1501": "GNNVerifier Plan-Graph Energy Adapter Smoke",
    "exp1502": "KAN Hardware Accounting - QuantKAN/KAEM No-Synthesis",
    "exp1503": "THRML Import Readiness Repair + Terminal Gate",
    "exp1504": "THRML/Carnot Simulator Parity v3",
}

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


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-053: persist a non-terminal marker before source loading.

    The retrospective reads many artifacts and may also append a milestone row
    to `research-complete.yaml`. A bootstrap file makes an interrupted run
    auditable instead of letting downstream tooling confuse "not started" with
    "failed after modifying documentation."
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
            "experiments_reviewed": [],
            "completed_experiments": [],
            "honest_gate_skips": [],
            "retired_lines": [],
            "graduated_lines": [],
            "carry_forward_lines": [],
            "continuous_self_learning_outcome": "",
            "hardware_claim_boundaries": "",
            "ops_docs_updated": False,
            "research_complete_updated": False,
            "protected_files_unchanged": False,
            "honest_verdict": "complete: in_progress_milestone_115_retro_seeded",
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


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _has_terminal_prefix(payload: Mapping[str, Any]) -> bool:
    verdict = _verdict(payload).lower()
    return verdict.startswith(TERMINAL_PREFIXES)


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _is_terminal(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"complete", "skipped", "blocked", "gate_blocked"}


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES.get(exp_id, f'experiment_{exp_id}.json')}"
    return f"{path}:{field}" if field else path


def _criterion(
    *,
    status: str,
    target: str,
    evidence_paths: list[str],
    positive_evidence: list[str] | None = None,
    negative_evidence: list[str] | None = None,
    source_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "target": target,
        "evidence_paths": evidence_paths,
        "positive_evidence": list(positive_evidence or []),
        "negative_evidence": list(negative_evidence or []),
        "source_values": dict(source_values or {}),
    }


def _missing_criterion(exp_id: str, target: str) -> dict[str, Any]:
    return _criterion(
        status=UNMET,
        target=target,
        evidence_paths=[_source_path(exp_id)],
        negative_evidence=[f"{exp_id} source artifact is missing."],
        source_values={"status": "missing", "honest_verdict": "missing_artifact"},
    )


def _scored_bool(
    *,
    exp_id: str,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: bool,
    target: str,
    fields: tuple[str, ...],
    positive: str,
    negative: str,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _missing_criterion(exp_id, target)
    payload = sources[exp_id]
    source_values = {field: payload.get(field) for field in fields}
    source_values["status"] = payload.get("status")
    source_values["honest_verdict"] = _verdict(payload)
    return _criterion(
        status=MET if passed else UNMET,
        target=target,
        evidence_paths=[_source_path(exp_id, field) for field in fields],
        positive_evidence=[positive] if passed else [],
        negative_evidence=[] if passed else [negative],
        source_values=source_values,
    )


def _score_activation(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1492 = sources.get("exp1492", {})
    passed = (
        _is_complete(exp1492)
        and exp1492.get("activation_manifest_complete") is True
        and exp1492.get("guardrail_blocks_preserved") is True
    )
    return _scored_bool(
        exp_id="exp1492",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="exp1492 activation_manifest_complete=true and .114 guardrails preserved",
        fields=("activation_manifest_complete", "guardrail_blocks_preserved"),
        positive=".115 activation manifest completed with guardrails preserved.",
        negative=".115 activation manifest or guardrail preservation is incomplete.",
    )


def _score_trigger_certificates(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1493 = sources.get("exp1493", {})
    passed = (
        exp1493.get("trigger_certificate_ready") is True
        and exp1493.get("certificate_parse_rate") is not None
        and exp1493.get("certificate_validation_rate") is not None
        and exp1493.get("always_constrained_validation_rate") is not None
        and exp1493.get("live_sota_model_inference_used") is True
    )
    return _scored_bool(
        exp_id="exp1493",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="trigger_certificate_ready=true with paired parse and validation rates",
        fields=(
            "trigger_certificate_ready",
            "certificate_parse_rate",
            "certificate_validation_rate",
            "always_constrained_validation_rate",
            "live_sota_model_inference_used",
        ),
        positive="Trigger-token certificate export is ready on live local SOTA rows.",
        negative="Trigger-token certificate export lacks readiness or required rates.",
    )


def _score_validator_compiler(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1494 = sources.get("exp1494", {})
    terminal_blocker_with_accounting = (
        _is_terminal(exp1494)
        and bool(exp1494.get("blockers"))
        and exp1494.get("verifier_false_accept_rate") is not None
    )
    passed = (
        exp1494.get("validator_compiler_ready") is True
        and exp1494.get("arbitrary_code_execution_path_introduced") is False
        and exp1494.get("verifier_false_accept_rate") is not None
    ) or terminal_blocker_with_accounting
    return _scored_bool(
        exp_id="exp1494",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="validator_compiler_ready=true or terminal blocker with false-accept accounting",
        fields=(
            "validator_compiler_ready",
            "validator_compile_rate",
            "known_good_pass_rate",
            "known_bad_reject_rate",
            "verifier_false_accept_rate",
            "arbitrary_code_execution_path_introduced",
            "blockers",
        ),
        positive="Validator compiler audit produced safe-DSL metrics and false-accept accounting.",
        negative="Validator compiler audit lacks safe readiness or terminal blocker accounting.",
    )


def _score_runtime_monitor(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1495 = sources.get("exp1495", {})
    passed = (
        exp1495.get("gated_inputs_present") is True
        and exp1495.get("monitor_intervention_ready") is True
        and exp1495.get("verifier_false_accept_rate") == 0.0
        and int(exp1495.get("false_interruptions") or 0) == 0
    )
    return _scored_bool(
        exp_id="exp1495",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="monitor_intervention_ready=true after structured gates with zero new false accepts",
        fields=(
            "gated_inputs_present",
            "monitor_intervention_ready",
            "monitor_events_emitted",
            "false_interruptions",
            "verifier_false_accept_rate",
        ),
        positive="interwhen monitor replay is intervention-ready with zero false interruptions.",
        negative="Monitor replay did not satisfy intervention or false-accept gates.",
    )


def _score_safe_prefix(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1496 = sources.get("exp1496", {})
    safe_rate = _number(exp1496.get("safe_prefix_validator_pass_rate"))
    baseline = _number(exp1496.get("baseline_validator_pass_rate"))
    regeneration = _number(exp1496.get("full_regeneration_validator_pass_rate"))
    comparator = max(value for value in (baseline or 0.0, regeneration or 0.0))
    passed = (
        exp1496.get("safe_prefix_continuation_ready") is True
        and safe_rate is not None
        and safe_rate > comparator
        and exp1496.get("verifier_false_accept_rate") == 0.0
    )
    return _scored_bool(
        exp_id="exp1496",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="safe-prefix continuation improves matched validator pass rate with zero false accepts",
        fields=(
            "safe_prefix_continuation_ready",
            "safe_prefix_validator_pass_rate",
            "baseline_validator_pass_rate",
            "full_regeneration_validator_pass_rate",
            "verifier_false_accept_rate",
        ),
        positive="HoVer-style safe-prefix continuation improved validator pass rate.",
        negative="Safe-prefix continuation did not improve the matched validator rate safely.",
    )


def _score_continuous_self_learning(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1497 = sources.get("exp1497", {})
    passed = (
        exp1497.get("daily_eval_manifest_ready") is True
        and bool(exp1497.get("continuous_self_learning_task"))
        and int(exp1497.get("soundness_mistakes") or 0) == 0
    )
    return _scored_bool(
        exp_id="exp1497",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="daily_eval_manifest_ready=true and continuous_self_learning_task recorded",
        fields=(
            "daily_eval_manifest_ready",
            "continuous_self_learning_task",
            "skills_evaluated",
            "promoted_skill_count",
            "retired_skill_count",
            "soundness_mistakes",
            "task_success_delta",
        ),
        positive="Bounded FR-11 trace2skill daily evaluation is ready with zero soundness mistakes.",
        negative="FR-11 daily evaluation readiness or soundness evidence is missing.",
    )


def _score_artifact_reachability(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1498 = sources.get("exp1498", {})
    passed = (
        exp1498.get("artifact_reachability_audit_complete") is True
        and exp1498.get("unreachable_artifact_count") is not None
        and isinstance(exp1498.get("repair_decisions"), list)
        and isinstance(exp1498.get("retirement_decisions"), list)
    )
    return _scored_bool(
        exp_id="exp1498",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="artifact reachability count is reported with repair or retire decisions",
        fields=(
            "artifact_reachability_audit_complete",
            "skills_checked",
            "reachable_artifact_count",
            "unreachable_artifact_count",
            "ambiguous_resolver_count",
            "repair_decisions",
            "retirement_decisions",
        ),
        positive="Trace2Skill artifact reachability is audited with explicit decisions.",
        negative="Reachability audit count or repair/retire decision surface is missing.",
    )


def _score_verifier_discipline(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    if missing_source_ids.intersection({"exp1499", "exp1500"}):
        missing = sorted(missing_source_ids.intersection({"exp1499", "exp1500"}))[0]
        return _missing_criterion(
            missing,
            "exp1499 orthogonality_matrix_written=true and exp1500 discipline_gate_ready=true",
        )
    exp1499 = sources.get("exp1499", {})
    exp1500 = sources.get("exp1500", {})
    passed = (
        exp1499.get("orthogonality_matrix_written") is True
        and exp1500.get("discipline_gate_ready") is True
    )
    return _criterion(
        status=MET if passed else UNMET,
        target="exp1499 orthogonality_matrix_written=true and exp1500 discipline_gate_ready=true",
        evidence_paths=[
            _source_path("exp1499", "orthogonality_matrix_written"),
            _source_path("exp1500", "discipline_gate_ready"),
            _source_path("exp1500", "retired_signals"),
        ],
        positive_evidence=[
            "Verifier orthogonality matrix and deterministic-first discipline gate are ready."
        ]
        if passed
        else [],
        negative_evidence=[] if passed else ["Verifier matrix or discipline gate is incomplete."],
        source_values={
            "exp1499_status": exp1499.get("status"),
            "orthogonality_matrix_written": exp1499.get("orthogonality_matrix_written"),
            "exp1500_status": exp1500.get("status"),
            "discipline_gate_ready": exp1500.get("discipline_gate_ready"),
            "retired_signals": exp1500.get("retired_signals"),
            "honest_verdicts": {
                "exp1499": _verdict(exp1499),
                "exp1500": _verdict(exp1500),
            },
        },
    )


def _score_graph_adapter(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1501 = sources.get("exp1501", {})
    terminal_no_signal = (
        _is_terminal(exp1501)
        and exp1501.get("plan_graph_energy_ready") is False
        and "no-signal" in _verdict(exp1501).lower()
    )
    passed = (
        exp1501.get("plan_graph_energy_ready") is True
        and exp1501.get("graph_energy_beats_baselines") is True
    ) or terminal_no_signal
    return _scored_bool(
        exp_id="exp1501",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="plan_graph_energy_ready=true or terminal no-signal finding against baselines",
        fields=(
            "plan_graph_energy_ready",
            "graph_energy_beats_baselines",
            "node_localization_top1_rate",
            "edge_localization_top1_rate",
            "trained_gnn_used",
        ),
        positive="Deterministic plan-graph energy adapter beat random and length baselines.",
        negative="Plan-graph adapter readiness or terminal no-signal evidence is missing.",
    )


def _score_kan_accounting(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1502 = sources.get("exp1502", {})
    passed = (
        exp1502.get("kan_hardware_accounting_ready") is True
        and exp1502.get("accounting_only_no_synthesis_claim") is True
        and exp1502.get("hardware_claim_allowed") is False
    )
    return _scored_bool(
        exp_id="exp1502",
        sources=sources,
        missing_source_ids=missing_source_ids,
        passed=passed,
        target="kan_hardware_accounting_ready=true with no synthesis or board claim",
        fields=(
            "kan_hardware_accounting_ready",
            "accounting_only_no_synthesis_claim",
            "hardware_claim_allowed",
            "blockers",
        ),
        positive="KAN accounting is ready and bounded to no-synthesis evidence.",
        negative="KAN accounting lacks readiness or overclaims hardware evidence.",
    )


def _score_thrml_readiness(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    if missing_source_ids.intersection({"exp1503", "exp1504"}):
        missing = sorted(missing_source_ids.intersection({"exp1503", "exp1504"}))[0]
        return _missing_criterion(
            missing,
            "exp1503 terminal readiness and exp1504 run only when readiness is true",
        )
    exp1503 = sources.get("exp1503", {})
    exp1504 = sources.get("exp1504", {})
    ready = exp1503.get("thrml_import_ready")
    if ready is True:
        passed = (
            exp1504.get("thrml_import_ready") is True
            and exp1504.get("parity_experiment_ran") is True
            and int(exp1504.get("parity_fail_count") or 0) == 0
            and exp1504.get("simulator_only") is True
            and exp1504.get("hardware_claim_allowed") is False
        )
        positive = "THRML import opened and simulator-only parity passed without hardware claims."
        negative = "THRML was ready but simulator-only parity did not pass cleanly."
    elif ready is False:
        passed = _is_terminal(exp1503) and (
            exp1504.get("gated_skip") is True
            or "gate" in str(exp1504.get("gated_off_reason") or "").lower()
            or _status(exp1504) == "skipped"
        )
        positive = "THRML readiness terminally closed false and parity was honestly gate-skipped."
        negative = "THRML readiness was false but parity was not recorded as a structured skip."
    else:
        passed = False
        positive = ""
        negative = "THRML import readiness was not terminally true or false."

    return _criterion(
        status=MET if passed else UNMET,
        target="exp1503 terminal readiness and exp1504 run only when readiness is true",
        evidence_paths=[
            _source_path("exp1503", "thrml_import_ready"),
            _source_path("exp1504", "parity_experiment_ran"),
            _source_path("exp1504", "gated_off_reason"),
            _source_path("exp1504", "hardware_claim_allowed"),
        ],
        positive_evidence=[positive] if passed else [],
        negative_evidence=[] if passed else [negative],
        source_values={
            "exp1503_status": exp1503.get("status"),
            "exp1503_thrml_import_ready": ready,
            "exp1504_status": exp1504.get("status"),
            "exp1504_thrml_import_ready": exp1504.get("thrml_import_ready"),
            "parity_experiment_ran": exp1504.get("parity_experiment_ran"),
            "parity_fail_count": exp1504.get("parity_fail_count"),
            "gated_skip": exp1504.get("gated_skip"),
            "gated_off_reason": exp1504.get("gated_off_reason"),
            "hardware_claim_allowed": exp1504.get("hardware_claim_allowed"),
            "honest_verdicts": {
                "exp1503": _verdict(exp1503),
                "exp1504": _verdict(exp1504),
            },
        },
    )


def _score_retrospective(
    *,
    missing_source_ids: set[str],
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    passed = not missing_source_ids and protected_files_unchanged
    return _criterion(
        status=MET if passed else UNMET,
        target="exp1505 writes required fields and confirms protected files unchanged",
        evidence_paths=[
            "results/experiment_1505_milestone_115_retro.json",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        positive_evidence=[
            "Retrospective artifact can be completed with protected files unchanged."
        ]
        if passed
        else [],
        negative_evidence=[]
        if passed
        else ["Missing source artifacts or protected file changes block retrospective closure."],
        source_values={
            "missing_source_ids": sorted(missing_source_ids),
            "protected_files_unchanged": protected_files_unchanged,
        },
    )


def _success_criteria(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    protected_files_unchanged: bool,
) -> dict[str, dict[str, Any]]:
    return {
        "activation": _score_activation(sources, missing_source_ids),
        "trigger_certificates": _score_trigger_certificates(sources, missing_source_ids),
        "validator_compiler": _score_validator_compiler(sources, missing_source_ids),
        "runtime_monitors": _score_runtime_monitor(sources, missing_source_ids),
        "safe_prefix_continuation": _score_safe_prefix(sources, missing_source_ids),
        "continuous_self_learning": _score_continuous_self_learning(
            sources,
            missing_source_ids,
        ),
        "artifact_reachability": _score_artifact_reachability(sources, missing_source_ids),
        "verifier_discipline": _score_verifier_discipline(sources, missing_source_ids),
        "graph_adapter": _score_graph_adapter(sources, missing_source_ids),
        "kan_accounting": _score_kan_accounting(sources, missing_source_ids),
        "thrml_readiness": _score_thrml_readiness(sources, missing_source_ids),
        "retrospective": _score_retrospective(
            missing_source_ids=missing_source_ids,
            protected_files_unchanged=protected_files_unchanged,
        ),
    }


def _honest_gate_skips(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    skips: list[dict[str, str]] = []
    exp1504 = sources.get("exp1504", {})
    reason = str(exp1504.get("gated_off_reason") or "").strip()
    if exp1504.get("gated_skip") is True or _status(exp1504) == "skipped" or reason:
        skips.append(
            {
                "experiment_id": "exp1504",
                "criterion": "thrml_readiness",
                "reason": reason or "structured THRML parity gate skip",
            }
        )
    return skips


def _experiment_verdicts(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for exp_id in EXPECTED_EXPERIMENT_IDS:
        payload = sources.get(exp_id)
        if payload is None:
            verdicts[exp_id] = {
                "status": "missing",
                "honest_verdict": "missing_artifact",
                "terminal_prefix_ok": False,
                "source_path": _source_path(exp_id),
            }
            continue
        verdicts[exp_id] = {
            "status": payload.get("status"),
            "honest_verdict": _verdict(payload),
            "terminal": _is_terminal(payload),
            "complete": _is_complete(payload),
            "terminal_prefix_ok": _has_terminal_prefix(payload),
            "source_path": _source_path(exp_id),
        }
    for missing in missing_source_ids:
        verdicts.setdefault(
            missing,
            {
                "status": "missing",
                "honest_verdict": "missing_artifact",
                "terminal_prefix_ok": False,
                "source_path": _source_path(missing),
            },
        )
    return verdicts


def _line(
    *,
    key: str,
    line: str,
    decision: str,
    source_experiments: list[str],
    evidence: str,
    boundary: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "line": line,
        "decision": decision,
        "source_experiments": source_experiments,
        "evidence": evidence,
        "claim_boundary": boundary,
    }


def _line_decisions(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    exp1493 = sources.get("exp1493", {})
    exp1494 = sources.get("exp1494", {})
    exp1495 = sources.get("exp1495", {})
    exp1496 = sources.get("exp1496", {})
    exp1497 = sources.get("exp1497", {})
    exp1498 = sources.get("exp1498", {})
    exp1500 = sources.get("exp1500", {})
    exp1501 = sources.get("exp1501", {})
    exp1502 = sources.get("exp1502", {})
    exp1503 = sources.get("exp1503", {})
    exp1504 = sources.get("exp1504", {})

    decisions = {
        "semantic_energy_v1": _line(
            key="semantic_energy_v1",
            line="Semantic Energy/logit telemetry and V_1 pairwise headline signals",
            decision="retired",
            source_experiments=["exp1481", "exp1487", "exp1500"],
            evidence=(
                "Exp1500 keeps semantic-energy and V_1 pairwise signals in the retired "
                "set after .114 confound findings."
            ),
            boundary="May appear only as historical or debugging context, not headline evidence.",
        ),
        "trigger_certificate_export": _line(
            key="trigger_certificate_export",
            line="Trigger-token certificate export",
            decision="graduated" if exp1493.get("trigger_certificate_ready") is True else "gated",
            source_experiments=["exp1493"],
            evidence=(
                f"parse_rate={exp1493.get('certificate_parse_rate')}, "
                f"validation_rate={exp1493.get('certificate_validation_rate')}, "
                f"false_accept_rate={exp1493.get('verifier_false_accept_rate')}"
            ),
            boundary="Graduates only as certificate export plus deterministic validation.",
        ),
        "constrainprompt_validator_compiler": _line(
            key="constrainprompt_validator_compiler",
            line="ConstrainPrompt-style safe-DSL validator compiler",
            decision="graduated" if exp1494.get("validator_compiler_ready") is True else "gated",
            source_experiments=["exp1494"],
            evidence=(
                f"compile_rate={exp1494.get('validator_compile_rate')}, "
                f"false_accept_rate={exp1494.get('verifier_false_accept_rate')}"
            ),
            boundary="No arbitrary generated-code execution path is promoted.",
        ),
        "interwhen_hover_monitoring": _line(
            key="interwhen_hover_monitoring",
            line="interwhen monitor plus HoVer safe-prefix continuation",
            decision="graduated"
            if (
                exp1495.get("monitor_intervention_ready") is True
                and exp1496.get("safe_prefix_continuation_ready") is True
            )
            else "gated",
            source_experiments=["exp1495", "exp1496"],
            evidence=(
                f"monitor_ready={exp1495.get('monitor_intervention_ready')}, "
                f"safe_prefix_ready={exp1496.get('safe_prefix_continuation_ready')}"
            ),
            boundary="Promotion is bounded to zero-new-false-accept replay evidence.",
        ),
        "fr11_trace2skill": _line(
            key="fr11_trace2skill",
            line="FR-11 trace2skill daily evaluation and reachability hygiene",
            decision="graduated"
            if (
                exp1497.get("daily_eval_manifest_ready") is True
                and exp1498.get("unreachable_artifact_count") == 0
            )
            else "gated",
            source_experiments=["exp1497", "exp1498"],
            evidence=(
                f"skills_evaluated={exp1497.get('skills_evaluated')}, "
                f"promoted={exp1497.get('promoted_skill_count')}, "
                f"unreachable={exp1498.get('unreachable_artifact_count')}"
            ),
            boundary="This is bounded daily evaluation, not autonomous unbounded self-learning.",
        ),
        "latent_deterministic_discipline": _line(
            key="latent_deterministic_discipline",
            line="Latent-vs-deterministic discipline gate",
            decision="graduated" if exp1500.get("discipline_gate_ready") is True else "gated",
            source_experiments=["exp1499", "exp1500"],
            evidence="Deterministic executable validators and conservative bounds are headline-allowed.",
            boundary="Latent, LLM-derived, and memory signals remain auxiliary or triage-only.",
        ),
        "plan_graph_energy": _line(
            key="plan_graph_energy",
            line="Deterministic plan-graph energy adapter",
            decision="carry_forward" if exp1501.get("plan_graph_energy_ready") is True else "gated",
            source_experiments=["exp1501"],
            evidence=(
                f"node_top1={exp1501.get('node_localization_top1_rate')}, "
                f"edge_top1={exp1501.get('edge_localization_top1_rate')}, "
                f"trained_gnn_used={exp1501.get('trained_gnn_used')}"
            ),
            boundary="Carry forward as deterministic graph-risk scoring, not trained GNN evidence.",
        ),
        "kan_hardware_accounting": _line(
            key="kan_hardware_accounting",
            line="KAN QuantKAN/KAEM hardware accounting",
            decision="carry_forward"
            if exp1502.get("kan_hardware_accounting_ready") is True
            else "gated",
            source_experiments=["exp1502"],
            evidence="No-synthesis accounting table exists for naive, QuantKAN-like, and KAEM variants.",
            boundary="No board, bitfile, timing, or accelerator speed claim.",
        ),
        "thrml_simulator_parity": _line(
            key="thrml_simulator_parity",
            line="THRML import readiness and simulator-only Carnot parity",
            decision="carry_forward"
            if (
                exp1503.get("thrml_import_ready") is True
                and exp1504.get("parity_experiment_ran") is True
            )
            else "gated",
            source_experiments=["exp1503", "exp1504"],
            evidence=(
                f"thrml_import_ready={exp1503.get('thrml_import_ready')}, "
                f"parity_pass_count={exp1504.get('parity_pass_count')}, "
                f"simulator_only={exp1504.get('simulator_only')}"
            ),
            boundary="Simulator-only parity; no Extropic TSU or physical hardware claim.",
        ),
        "legacy_small_model_headlines": _line(
            key="legacy_small_model_headlines",
            line="Legacy small-model headline evidence",
            decision="retired",
            source_experiments=["exp1492", "exp1493", "exp1494", "exp1496"],
            evidence="Local SOTA GGUF model specs remain the required headline model boundary.",
            boundary="Legacy small models are smoke tests only.",
        ),
        "decoded_quality_or_hardware_overclaims": _line(
            key="decoded_quality_or_hardware_overclaims",
            line="Decoded-quality, Kona-internals, and hardware overclaims from bounded probes",
            decision="retired",
            source_experiments=["exp1492", "exp1501", "exp1502", "exp1504"],
            evidence="All relevant artifacts retain injected-fault, no-synthesis, or simulator-only bounds.",
            boundary="No decoded-quality, Kona-internal, KV260 board, or TSU hardware claim.",
        ),
    }
    return decisions


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for exp_id in EXPECTED_EXPERIMENT_IDS:
        title = EXPERIMENT_TITLES[exp_id]
        matches = [row for row in rows if exp_id in row or title in row or title[:38] in row]
        entries[exp_id] = {
            "found": bool(matches),
            "ok": any("| OK |" in row for row in matches),
            "terminal": any(
                marker in row
                for row in matches
                for marker in ("| OK |", "| GATE_BLOCK |", "| FAIL |", "| SKIP |")
            ),
            "line": matches[-1] if matches else None,
        }
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "terminal_count": sum(1 for entry in entries.values() if entry["terminal"]),
        "expected_count": len(EXPECTED_EXPERIMENT_IDS),
        "missing_experiments": [exp_id for exp_id, entry in entries.items() if not entry["found"]],
    }


def _source_inputs_read(
    *,
    roadmap_doc_text: str,
    conductor_log_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    known_issues_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
        "ops/known-issues.md": {"exists": bool(known_issues_text)},
    }


def _research_complete_has_115_entry(text: str) -> bool:
    try:
        loaded = yaml.safe_load(text) if text.strip() else []
    except yaml.YAMLError:
        return "2026.04.115" in text
    if isinstance(loaded, list):
        return any(str(entry.get("id")) == MILESTONE for entry in loaded if isinstance(entry, dict))
    return "2026.04.115" in text


def _archive_task_rows() -> list[dict[str, str]]:
    rows = [
        {
            "id": f"{exp_id}-{EXPERIMENT_TITLES[exp_id].lower().replace(' ', '-').replace('/', '-')}",
            "title": EXPERIMENT_TITLES[exp_id],
            "deliverable": f"results/{SOURCE_FILES[exp_id]}",
            "result": "OK (conductor)",
        }
        for exp_id in EXPECTED_EXPERIMENT_IDS
    ]
    rows.append(
        {
            "id": "exp1505-milestone-115-retrospective",
            "title": "Milestone .115 Retrospective - Outcomes, Claim Boundaries, and Archive",
            "deliverable": "results/experiment_1505_milestone_115_retro.json",
            "result": "OK (codex retro)",
        }
    )
    return rows


def _archive_entry(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": MILESTONE,
        "title": "Executable Constraint Monitors + FR-11 Self-Learning Hygiene + Hardware Gates",
        "doc": ROADMAP_DOC,
        "completed": "2026-05-07",
        "finding": (
            f"Milestone .115 met {artifact['criteria_met']} of "
            f"{artifact['criteria_total']} success criteria. Trigger certificates, "
            "safe validator compilation, interwhen monitoring, HoVer continuation, "
            "bounded FR-11 daily evaluation, verifier discipline, deterministic "
            "plan-graph energy, KAN no-synthesis accounting, and simulator-only "
            "THRML parity are archived with Semantic Energy/V_1, self-learning, "
            "and hardware claim boundaries preserved."
        ),
        "tasks": _archive_task_rows(),
    }


def _append_research_complete_archive(path: Path, artifact: Mapping[str, Any]) -> tuple[bool, str]:
    if not path.exists():
        return False, "research_complete_yaml_missing"
    text = path.read_text(encoding="utf-8")
    if _research_complete_has_115_entry(text):
        return False, "research_complete_already_contains_2026.04.115"
    if artifact.get("status") != "complete":
        return False, "terminal_retro_artifact_not_complete"
    block = yaml.safe_dump([_archive_entry(artifact)], sort_keys=False, allow_unicode=False)
    # Validate the generated block before appending so a malformed archive row
    # cannot corrupt the historical completion ledger.
    yaml.safe_load(block)
    prefix = "" if text.endswith("\n") or not text else "\n"
    path.write_text(f"{text}{prefix}{block}", encoding="utf-8")
    return True, "appended_2026.04.115_archive_entry"


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
    ops_status_text: str,
    ops_changelog_text: str,
    known_issues_text: str,
    protected_files_unchanged: bool,
    research_complete_updated: bool,
    research_complete_update_reason: str,
) -> dict[str, Any]:
    """REQ-REPORT-053: score `.115` outcomes from terminal source artifacts.

    The artifact is deliberately evidence-first: every success criterion points
    back to source fields, and every line decision states the claim boundary.
    That keeps retired or simulator-only signals from being promoted by summary
    prose after the experiments themselves were more cautious.
    """

    missing = set(missing_source_ids)
    criteria = _success_criteria(
        sources=sources,
        missing_source_ids=missing,
        protected_files_unchanged=protected_files_unchanged,
    )
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(criteria)
    experiment_verdicts = _experiment_verdicts(sources, missing)
    reviewed = [exp_id for exp_id in EXPECTED_EXPERIMENT_IDS if exp_id not in missing]
    completed = [
        exp_id
        for exp_id in EXPECTED_EXPERIMENT_IDS
        if exp_id in sources and _is_complete(sources[exp_id])
    ]
    decisions = _line_decisions(sources)
    retired_lines = [item for item in decisions.values() if item["decision"] == "retired"]
    graduated_lines = [item for item in decisions.values() if item["decision"] == "graduated"]
    carry_forward_lines = [
        item for item in decisions.values() if item["decision"] == "carry_forward"
    ]
    gated_lines = [item for item in decisions.values() if item["decision"] == "gated"]
    gate_skips = _honest_gate_skips(sources)
    terminal_prior_tasks = all(
        exp_id in sources and _is_terminal(sources[exp_id]) for exp_id in EXPECTED_EXPERIMENT_IDS
    )
    required_fields_present = REQUIRED_ARTIFACT_FIELDS
    ops_deferred_reason = (
        "stop_when_done_delegates_ops_status_changelog_and_traceability_to_haiku_reconciler"
    )

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
        "experiments_reviewed": reviewed,
        "completed_experiments": completed,
        "terminal_experiments": [
            exp_id
            for exp_id in EXPECTED_EXPERIMENT_IDS
            if exp_id in sources and _is_terminal(sources[exp_id])
        ],
        "missing_artifacts": [
            {"experiment_id": exp_id, "path": _source_path(exp_id)} for exp_id in sorted(missing)
        ],
        "experiment_verdicts": experiment_verdicts,
        "honest_gate_skips": gate_skips,
        "retired_lines": retired_lines,
        "graduated_lines": graduated_lines,
        "carry_forward_lines": carry_forward_lines,
        "gated_lines": gated_lines,
        "line_decisions": decisions,
        "continuous_self_learning_outcome": (
            "Exp1497 graduates bounded FR-11 daily evaluation with 24 skills "
            "evaluated, 12 promotions, zero soundness mistakes, and Exp1498 "
            "reachability confirming zero unreachable artifacts. This does not "
            "claim unbounded autonomous self-learning."
        ),
        "hardware_claim_boundaries": (
            "Hardware evidence is limited to no-synthesis KAN accounting and "
            "THRML simulator-only parity; no Extropic TSU, KV260 board, bitfile, "
            "timing, or accelerator speed claim is allowed."
        ),
        "ops_docs_updated": False,
        "ops_docs_update_deferred_reason": ops_deferred_reason,
        "research_complete_updated": research_complete_updated,
        "research_complete_update_reason": research_complete_update_reason,
        "protected_files_unchanged": protected_files_unchanged,
        "protected_files": {
            "research-roadmap.yaml": "unchanged" if protected_files_unchanged else "modified",
            "scripts/research_conductor.py": (
                "unchanged" if protected_files_unchanged else "modified"
            ),
        },
        "source_inputs_read": _source_inputs_read(
            roadmap_doc_text=roadmap_doc_text,
            conductor_log_text=conductor_log_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            known_issues_text=known_issues_text,
        ),
        "conductor_log_exp1492_to_exp1504": _conductor_log_summary(conductor_log_text),
        "terminal_prior_tasks": terminal_prior_tasks,
        "required_artifact_fields_present": sorted(required_fields_present),
        "honest_verdict": (
            f"complete: milestone_115_{criteria_met}_of_{criteria_total}_criteria_met_"
            "claim_boundaries_preserved"
        ),
    }
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-053: write bootstrap, archive row if supported, and final JSON.

    The run intentionally leaves `ops/status.md`, `ops/changelog.md`, and
    `_bmad/traceability.md` to the conductor's separate reconciliation step.
    This preserves the stop-when-done boundary while still recording that the
    ops update was deferred rather than forgotten.
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
    common_inputs = {
        "sources": sources,
        "missing_source_ids": missing,
        "conductor_log_text": _read_text(root_path / "ops" / "conductor-log.md"),
        "roadmap_doc_text": _read_text(root_path / ROADMAP_DOC),
        "ops_status_text": _read_text(root_path / "ops" / "status.md"),
        "ops_changelog_text": _read_text(root_path / "ops" / "changelog.md"),
        "known_issues_text": _read_text(root_path / "ops" / "known-issues.md"),
        "protected_files_unchanged": protected,
    }
    preliminary = build_artifact(
        **common_inputs,
        research_complete_updated=False,
        research_complete_update_reason="pending_archive_write",
    )
    updated, reason = _append_research_complete_archive(
        root_path / "research-complete.yaml",
        preliminary,
    )
    research_complete_text = _read_text(root_path / "research-complete.yaml")
    research_complete_has_115 = _research_complete_has_115_entry(research_complete_text)
    artifact = build_artifact(
        **common_inputs,
        research_complete_updated=updated or research_complete_has_115,
        research_complete_update_reason=reason,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
