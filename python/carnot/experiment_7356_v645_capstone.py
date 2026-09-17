"""Close V645 while keeping five evidence classes separate.

The reducer reads exact current artifacts and exact conductor gate records. It
does not invoke a model, hardware, or an external service.

Spec refs: REQ-REPORT-7356 and SCENARIO-REPORT-7356-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7342_v644_capstone as prior_capstone
from carnot import experiment_7343_v645_contract as contract_source
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.645"
RUN_DATE = "20260916"
RANDOM_SEED = {
    "development": 7_356_202_609_16,
    "evaluation": 7_356_202_609_17,
    "resampling": 7_356_202_609_18,
}
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = deepcopy(prior_capstone.ZERO_INVOCATION_COUNTS)

EXPECTED_TASK_IDS = tuple(contract_source.EXPECTED_ID_ORDER)
VALID_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
CLAIM_NAMES = (
    "source_fidelity",
    "continuous_learning",
    "acquisition_cost",
    "live_arc_causality",
    "hardware_disposition",
)
DECISION_BRANCHES = (
    *CLAIM_NAMES,
    "native_tenfold_null",
    "retired_joint_claim_comparison",
    "delayed_feedback_mixture_retirement",
    "delayed_feedback_suffix_retirement",
    "unsupported_hardware_boundary",
)
PRIOR_DECISIONS = {
    "retire_exact_repeat",
    "preserve_prior_boundary_current_upstream_block",
    "preserve_record_mismatch_no_retirement",
    "changed_prerequisite_observed",
}
REQUIRED_SCOPED_CHECKS = scoped.REQUIRED_CHECK_NAMES
REQUIRED_SCIENCE_GATES = (
    ("exp7348-plan-capture", "plan_capture_complete_score", 1),
    ("exp7350-learning-audit", "learning_audit_complete_score", 1),
    ("exp7353-acquisition-audit", "acquisition_audit_complete_score", 1),
    ("exp7354-arc-transfer", "arc_capture_complete_score", 1),
)

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
V644_CAPSTONE_PATH = Path("results/experiment_7342_v644_capstone.json")
V643_CAPSTONE_PATH = Path("results/experiment_7328_v643_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7356_v645_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7356_v645_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7356_v645_capstone.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7356_v645_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7356_v645_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7356_v645_capstone")

CANONICAL_BLOCK_SPECS: dict[str, JsonDict] = {
    "exp7349-prospective-learning": {
        "marker": ("| 2026-09-16 17:36 UTC | Measure continuous structural learning on distinct"),
        "failure": {
            "upstream": "exp7346-learning-adapter",
            "failed_check": "learning_adapter_ready_score",
            "artifact_field": "learning_adapter_ready_score",
            "expected_value": 1,
            "observed_value": 0,
        },
    },
    "exp7350-learning-audit": {
        "marker": ("| 2026-09-16 17:38 UTC | Independently audit learning causality source fide"),
        "failure": {
            "upstream": "exp7349-prospective-learning",
            "failed_check": "terminal_producer_eligibility",
            "artifact_field": "verdict_class",
            "expected_value": ["positive", "circular_positive", "null"],
            "observed_value": "blocked",
        },
    },
    "exp7352-acquisition-cost": {
        "marker": ("| 2026-09-16 20:20 UTC | Compare acquisition strategies at the full executi"),
        "failure": {
            "upstream": "exp7351-acquisition-prototype",
            "failed_check": "acquisition_prototype_ready_score",
            "artifact_field": "acquisition_prototype_ready_score",
            "expected_value": 1,
            "observed_value": 0,
        },
    },
    "exp7353-acquisition-audit": {
        "marker": ("| 2026-09-16 20:22 UTC | Audit acquisition soundness assumptions and cost r"),
        "failure": {
            "upstream": "exp7352-acquisition-cost",
            "failed_check": "terminal_producer_eligibility",
            "artifact_field": "verdict_class",
            "expected_value": ["positive", "circular_positive", "null"],
            "observed_value": "blocked",
        },
    },
}

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input/resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; LLM tasks include unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
    "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
    "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding sets true and prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "capstone_complete_score": "Fourteen dispositions may be complete while science remains blocked.",
    "task_dispositions": "Carry each exact artifact path/hash/class or canonical pre-gate record.",
    "claim_matrix": "Separate source fidelity, learning, acquisition, ARC and hardware claims.",
    "publication_gate": "Keep stable G1-G4 unchanged by the new pilot.",
    "scope_reduction_compliance": "Explain why the next work addresses diagnosed barriers without branch proliferation.",
    "next_branch_decisions": "Each branch needs a falsifiable continuation or retirement condition.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

sha256_bytes = prior_capstone.sha256_bytes
sha256 = prior_capstone.sha256
read_json = prior_capstone.read_json
artifact_checksum = prior_capstone.artifact_checksum
atomic_write_json = prior_capstone.atomic_write_json
date_argument = prior_capstone.date_argument


def progress(phase: int, event: str, detail: str) -> None:
    """Flush a factual phase boundary so long validation stays observable."""

    print(f"[exp7356] phase={phase} event={event} {detail}", flush=True)


def phase_row(name: str, start: float, end: float, units: int, checkpoint: str | None) -> JsonDict:
    """Record one real disjoint span without adding artificial delay."""

    return {
        "phase": name,
        "start_s": start,
        "end_s": end,
        "duration_s": end - start,
        "units": units,
        "checkpoint": checkpoint,
        "pending_operations": [],
    }


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Use the V645 independent Markdown and YAML contract parsers."""

    return contract_source.evaluate_contract(markdown_text, yaml_document)


def load_contract(root: Path) -> JsonDict:
    """Select one V645 YAML authority and compare it with the Markdown bytes."""

    selected, document, content, candidates = contract_source.select_yaml_authority(root)
    if selected is None or document is None or content is None:
        raise ValueError("no selected V645 YAML authority")
    markdown_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
    result = evaluate_contract(markdown_text, document)
    result.update(
        {
            "selected_yaml_path": str(selected),
            "selected_yaml_sha256": sha256_bytes(content),
            "design_path": str(DESIGN_PATH),
            "design_sha256": sha256(root / DESIGN_PATH),
            "authority_candidates": candidates,
            "yaml_document": document,
            "markdown_text": markdown_text,
            "tasks": document.get("tasks", []),
        }
    )
    return result


def task_number(task_id: str) -> int | None:
    """Return the numeric experiment identity when the task ID has one."""

    match = re.match(r"exp(\d+)", task_id)
    return int(match.group(1)) if match else None


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept only the declared task identity or its ordinary numeric form."""

    number = task_number(task_id)
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def classify_payload(
    task_id: str, payload: Mapping[str, Any], *, quarantined: bool
) -> tuple[bool, str, bool, bool]:
    """Classify identity and terminal eligibility before reading numeric scores."""

    status = str(payload.get("status", ""))
    terminal_status = status in {"complete", "blocked", "disqualified"} or status.startswith(
        "complete_"
    )
    declared = str(payload.get("verdict_class", "disqualified"))
    authenticated = bool(
        terminal_status
        and _identity_matches(task_id, payload)
        and declared in VALID_VERDICT_CLASSES
    )
    flagged = payload.get("flagged_adversarial") is True
    if quarantined or flagged or not authenticated:
        disposition = "disqualified"
    elif status == "blocked" or declared == "blocked":
        disposition = "blocked"
    elif status == "disqualified" or declared == "disqualified":
        disposition = "disqualified"
    else:
        disposition = declared
    accepted = bool(
        authenticated
        and not quarantined
        and not flagged
        and disposition in {"positive", "circular_positive", "null"}
    )
    return authenticated, disposition, accepted, flagged


def canonical_record(root: Path, task_id: str) -> JsonDict | None:
    """Select one frozen GATE_BLOCK row and attach its structured failure."""

    spec = CANONICAL_BLOCK_SPECS.get(task_id)
    if spec is None:
        return None
    matches = [
        (number, line)
        for number, line in enumerate(
            (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8").splitlines(), 1
        )
        if line.startswith(str(spec["marker"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"one canonical conductor block required for {task_id}")
    line_number, line = matches[0]
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) != 4 or cells[2] != "GATE_BLOCK":
        raise ValueError(f"canonical conductor record is not GATE_BLOCK for {task_id}")
    return {
        "line_number": line_number,
        "timestamp": cells[0],
        "record_status": cells[2],
        "detail": cells[3],
        "record_text": line,
        "record_sha256": sha256_bytes((line + "\n").encode("utf-8")),
        "failure": deepcopy(spec["failure"]),
    }


def load_evidence(root: Path, task: Mapping[str, Any], manifest: object) -> JsonDict:
    """Read the declared artifact or its one allowed conductor gate record."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    path = root / declared
    if path.is_file():
        payload = read_json(path)
        quarantine = common.quarantine_receipt(payload, task_id, declared, manifest)
        authenticated, disposition, accepted, flagged = classify_payload(
            task_id, payload, quarantined=bool(quarantine.get("quarantined"))
        )
        return {
            "task_id": task_id,
            "declared_deliverable_path": declared,
            "selected_evidence_path": declared,
            "evidence_source": "declared_deliverable",
            "artifact_sha256": sha256(path),
            "record_sha256": None,
            "canonical_record": None,
            "payload": payload,
            "quarantine_receipt": quarantine,
            "quarantined": bool(quarantine.get("quarantined")),
            "adversarially_flagged": flagged,
            "authenticated": authenticated,
            "accepted_for_reduction": accepted,
            "disposition_class": disposition,
        }
    record = canonical_record(root, task_id)
    if record is not None:
        return {
            "task_id": task_id,
            "declared_deliverable_path": declared,
            "selected_evidence_path": f"{CONDUCTOR_LOG_PATH}:L{record['line_number']}",
            "evidence_source": "canonical_conductor_block",
            "artifact_sha256": record["record_sha256"],
            "record_sha256": record["record_sha256"],
            "canonical_record": record,
            "payload": {},
            "quarantine_receipt": {"quarantined": False, "reasons": []},
            "quarantined": False,
            "adversarially_flagged": False,
            "authenticated": True,
            "accepted_for_reduction": False,
            "disposition_class": "blocked",
        }
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "selected_evidence_path": None,
        "evidence_source": "missing",
        "artifact_sha256": None,
        "record_sha256": None,
        "canonical_record": None,
        "payload": {},
        "quarantine_receipt": {"quarantined": False, "reasons": []},
        "quarantined": False,
        "adversarially_flagged": False,
        "authenticated": False,
        "accepted_for_reduction": False,
        "disposition_class": "missing",
    }


def load_repository_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Create one exact evidence slot for every producer before this capstone."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {str(task["id"]): load_evidence(root, task, manifest) for task in tasks[:-1]}


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay all frozen gates while terminal eligibility outranks a score."""

    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            source = evidence[upstream]
            payload = source["payload"]
            field = str(gate["artifact_field"])
            observed = payload.get(field)
            disposition = source.get("disposition_class")
            if source.get("selected_evidence_path") is None:
                outcome = "missing_file"
            elif source.get("quarantined") is True:
                outcome = "quarantined"
            elif source.get("adversarially_flagged") is True:
                outcome = "flagged_adversarial"
            elif disposition in {"disqualified", "blocked", "partial"}:
                outcome = str(disposition)
            elif field not in payload:
                outcome = "missing_field"
            else:
                operator = str(gate.get("op"))
                expected = gate.get("value")
                passed = observed == expected if operator == "==" else observed in expected
                outcome = "passed" if passed and operator in {"==", "in"} else "value_mismatch"
            rows.append(
                {
                    "consumer": consumer["id"],
                    "upstream": upstream,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": deepcopy(gate.get("value")),
                    "observed_value": deepcopy(observed),
                    "artifact_path": source.get("selected_evidence_path"),
                    "artifact_sha256": source.get("artifact_sha256"),
                    "producer_disposition_class": disposition,
                    "producer_flagged_adversarial": source.get("adversarially_flagged"),
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def _list_of_rows(value: object) -> list[Mapping[str, Any]]:
    """Return only mapping rows so malformed producer data cannot crash reduction."""

    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _base_claim(
    claim: str,
    producer: str,
    source: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    costs: Mapping[str, Any],
    failures: Sequence[str],
) -> JsonDict:
    """Build a non-promoting claim row while retaining diagnostic measurements."""

    accepted = source.get("accepted_for_reduction") is True
    payload = source.get("payload", {})
    declared = str(payload.get("verdict_class", "blocked"))
    claim_class = declared if accepted else "blocked"
    if claim_class == "positive" and payload.get("verifier_is_oracle") is True:
        claim_class = "circular_positive"
    return {
        "unit_id": f"claim:{claim}",
        "arm": "independent_raw_reduction",
        "claim": claim,
        "producer": producer,
        "producer_sha256": source.get("artifact_sha256"),
        "verdict_class": claim_class,
        "metric": None,
        "metric_kind": "unavailable_for_promotion",
        "diagnostic_metrics": deepcopy(dict(diagnostics)),
        "costs": deepcopy(dict(costs)),
        "failures": list(failures),
        "abstention": not accepted,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _source_claim(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute source-field fidelity while respecting the flagged producer."""

    source = evidence["exp7348-plan-capture"]
    payload = source["payload"]
    rows = _list_of_rows(payload.get("source_fidelity_rows"))
    fidelity_fields = (
        "identity_fidelity",
        "assignment_field_fidelity",
        "assignment_type_fidelity",
        "duration_fidelity",
        "weight_fidelity",
        "window_fidelity",
    )
    fully_faithful = sum(all(row.get(field) is True for field in fidelity_fields) for row in rows)
    diagnostics = {
        "raw_unit_count": len(rows),
        "fully_faithful_unit_count": fully_faithful,
        "source_failure_count": sum(row.get("source_failure") is True for row in rows),
        "hidden_rule_accepted_count": sum(row.get("hidden_rule_accepted") is True for row in rows),
        "small_live_origin_cohort": True,
    }
    invocations = payload.get("invocation_counts", {})
    costs = {
        "upstream_generation_calls": invocations.get("generation_calls_attempted"),
        "upstream_model_loads": invocations.get("model_loads_attempted"),
    }
    failures = [f"producer_{source.get('disposition_class')}"]
    if source.get("adversarially_flagged") is True:
        failures.append("producer_flagged_adversarial")
    return _base_claim(
        "source_fidelity", "exp7348-plan-capture", source, diagnostics, costs, failures
    )


def _learning_claim(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute available adapter diagnostics without inventing prospective rows."""

    source = evidence["exp7350-learning-audit"]
    adapter = evidence["exp7346-learning-adapter"]["payload"]
    rows = _list_of_rows(adapter.get("rows"))
    diagnostics = {
        "adapter_raw_row_count": len(rows),
        "prospective_unit_count": 0,
        "future_use_witness_count": sum(row.get("future_use_witness") is True for row in rows),
        "returned_feasible_count": sum(row.get("returned_feasible") is True for row in rows),
        "shared_executor_authority": True,
    }
    costs = {
        "adapter_query_attempts": sum(
            row.get("query_attempts", 0)
            for row in rows
            if isinstance(row.get("query_attempts"), int)
        ),
        "prospective_rows": 0,
    }
    return _base_claim(
        "continuous_learning",
        "exp7350-learning-audit",
        source,
        diagnostics,
        costs,
        ["prospective_learning_gate_blocked", "independent_learning_audit_absent"],
    )


def _acquisition_claim(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute prototype full-cost totals while keeping the absent audit explicit."""

    source = evidence["exp7353-acquisition-audit"]
    prototype = evidence["exp7351-acquisition-prototype"]["payload"]
    rows = _list_of_rows(prototype.get("rows"))
    evaluation = [row for row in rows if row.get("panel") == "evaluation"]
    totals: dict[str, float] = {}
    for row in evaluation:
        arm = str(row.get("arm"))
        value = row.get("full_cost_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[arm] = totals.get(arm, 0.0) + float(value)
    diagnostics = {
        "prototype_raw_row_count": len(rows),
        "evaluation_row_count": len(evaluation),
        "challenge_row_count": sum(row.get("unsupported") is True for row in rows),
        "returned_infeasible_count": sum(
            int(row.get("returned_infeasible_count", 0))
            for row in rows
            if isinstance(row.get("returned_infeasible_count"), int)
        ),
        "evaluation_full_cost_s_by_arm": {
            arm: round(value, 12) for arm, value in sorted(totals.items())
        },
        "shared_executor_authority": True,
    }
    costs = {
        "query_attempts": sum(
            int(row.get("query_attempt_count", 0))
            for row in rows
            if isinstance(row.get("query_attempt_count"), int)
        ),
        "audit_available": False,
    }
    return _base_claim(
        "acquisition_cost",
        "exp7353-acquisition-audit",
        source,
        diagnostics,
        costs,
        ["acquisition_cost_gate_blocked", "independent_acquisition_audit_absent"],
    )


def _arc_claim(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute the fixed four-episode ARC comparison from per-game rows."""

    source = evidence["exp7354-arc-transfer"]
    payload = source["payload"]
    rows = _list_of_rows(payload.get("per_game_results", payload.get("rows")))
    chains = _list_of_rows(payload.get("causal_chain_rows"))
    treatment = [row for row in rows if row.get("arm") == "result_resume"]
    withheld = [row for row in rows if row.get("arm") == "result_withheld"]
    treatment_levels = sum(int(row.get("levels", 0)) for row in treatment)
    withheld_levels = sum(int(row.get("levels", 0)) for row in withheld)
    budget = payload.get("sample_size_budget", {})
    diagnostics = {
        "planned_episode_count": budget.get("planned_units"),
        "completed_episode_count": len([row for row in rows if row.get("censored") is not True]),
        "game_count": len({row.get("game") for row in rows}),
        "complete_causal_chain_count": sum(row.get("chain_complete") is True for row in chains),
        "treatment_level_total": treatment_levels,
        "withheld_level_total": withheld_levels,
        "paired_level_delta": treatment_levels - withheld_levels,
    }
    costs = {
        "generation_calls": sum(
            int(row.get("costs", {}).get("generation_calls", 0)) for row in rows
        ),
        "actions": sum(int(row.get("costs", {}).get("actions", 0)) for row in rows),
        "output_tokens": sum(int(row.get("costs", {}).get("output_tokens", 0)) for row in rows),
    }
    failures = [f"producer_{source.get('disposition_class')}"]
    if source.get("adversarially_flagged") is True:
        failures.append("producer_flagged_adversarial")
    if diagnostics["complete_causal_chain_count"] != 2:
        failures.append("successful_bound_result_absent")
    return _base_claim(
        "live_arc_causality", "exp7354-arc-transfer", source, diagnostics, costs, failures
    )


def _hardware_claim(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Count the expected GateMate block only as completed disposition accounting."""

    source = evidence["exp7355-board-state"]
    payload = source["payload"]
    rows = _list_of_rows(payload.get("board_rows", payload.get("rows")))
    complete = bool(
        source.get("authenticated") is True
        and source.get("disposition_class") == "blocked"
        and payload.get("board_disposition_complete_score") == 1
        and len(rows) == 3
    )
    return {
        "unit_id": "claim:hardware_disposition",
        "arm": "read_only_board_dispositions",
        "claim": "hardware_disposition",
        "producer": "exp7355-board-state",
        "producer_sha256": source.get("artifact_sha256"),
        "verdict_class": "blocked",
        "metric": 1 if complete else 0,
        "metric_kind": "disposition_accounting_only",
        "diagnostic_metrics": {
            "board_row_count": len(rows),
            "disposition_complete_score": payload.get("board_disposition_complete_score"),
            "hardware_readiness_score": payload.get("hardware_readiness_score"),
            "hardware_value_score": payload.get("hardware_value_score"),
            "hardware_promotion_score": payload.get("hardware_promotion_score"),
            "hardware_operations_issued_count": payload.get("hardware_operations_issued_count"),
        },
        "costs": {"hardware_operations": payload.get("hardware_operations_issued_count", 0)},
        "failures": ["gatemate_changed_physical_state_receipt_missing"],
        "abstention": True,
        "censored": False,
        "expected_external_block": True,
        "disposition_complete": complete,
        "science_prerequisite": False,
        "promotes_scientific_efficacy": False,
    }


def build_claim_matrix(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Independently reduce five claims without merging their authority classes."""

    return [
        _source_claim(evidence),
        _learning_claim(evidence),
        _acquisition_claim(evidence),
        _arc_claim(evidence),
        _hardware_claim(evidence),
    ]


def science_failures(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Name every exact failed prerequisite for required current science."""

    failures: list[JsonDict] = []
    for upstream, field, expected in REQUIRED_SCIENCE_GATES:
        source = evidence[upstream]
        observed = source["payload"].get(field)
        if source.get("accepted_for_reduction") is not True or observed != expected:
            failures.append(
                {
                    "upstream": upstream,
                    "failed_check": field,
                    "artifact_field": field,
                    "expected_value": expected,
                    "observed_value": observed,
                    "passed": False,
                    "terminal_blocking": True,
                }
            )
    return failures


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    validation_passed: bool,
) -> JsonDict:
    """Keep complete accounting separate from science and hardware availability."""

    if not validation_passed:
        failure = {
            "upstream": "exp7356-capstone",
            "failed_check": "required_scoped_and_terminal_validation",
            "artifact_field": "required_checks_passed/terminal_checks_passed",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
            "terminal_blocking": True,
        }
        return {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v645_capstone_required_validation_failed",
            "gate_check_summary": {"passed": False, "failures": [failure]},
        }
    failures = science_failures(evidence)
    if failures:
        first = str(failures[0]["upstream"]).replace("-", "_")
        return {
            "status": "complete",
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_{first}: all fourteen dispositions are represented; required "
                "V645 science is unavailable while the expected GateMate block remains independent"
            ),
            "gate_check_summary": {"passed": False, "failures": failures},
        }
    science = [row for row in claims if row.get("claim") != "hardware_disposition"]
    classes = {str(row.get("verdict_class")) for row in science}
    verdict_class = "circular_positive" if "circular_positive" in classes else "null"
    return {
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": (
            "complete_circular_positive_bounded_v645_claims_no_broad_learning_claim"
            if verdict_class == "circular_positive"
            else "complete_null_v645_scientific_claims"
        ),
        "gate_check_summary": {"passed": True, "failures": []},
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Represent thirteen producers and the non-recursive capstone self row."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        payload = source["payload"]
        record = source.get("canonical_record") or {}
        incoming = [deepcopy(row) for row in gate_rows if row["consumer"] == task_id]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task["title"],
                "phase": task["phase"],
                "declared_artifact_path": source["declared_deliverable_path"],
                "selected_evidence_path": source["selected_evidence_path"],
                "evidence_source": source["evidence_source"],
                "artifact_sha256": source["artifact_sha256"],
                "canonical_record_sha256": source["record_sha256"],
                "status": payload.get("status", record.get("record_status", "missing").lower()),
                "honest_verdict": payload.get(
                    "honest_verdict", record.get("detail", "blocked_missing_evidence")
                ),
                "verdict_class": payload.get("verdict_class", "blocked" if record else None),
                "disposition_class": source["disposition_class"],
                "authenticated": source["authenticated"],
                "quarantined": source["quarantined"],
                "flagged_adversarial": source["adversarially_flagged"],
                "incoming_gate_records": incoming,
                "skipped_gate_records": [row for row in incoming if not row["passed"]],
            }
        )
    self_task = tasks[-1]
    rows.append(
        {
            "order": 14,
            "task_id": self_task["id"],
            "title": self_task["title"],
            "phase": self_task["phase"],
            "declared_artifact_path": self_task["deliverable"],
            "selected_evidence_path": None,
            "evidence_source": "capstone_self",
            "artifact_sha256": None,
            "canonical_record_sha256": None,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "disposition_class": terminal["verdict_class"],
            "authenticated": True,
            "quarantined": False,
            "flagged_adversarial": terminal["verdict_class"] == "disqualified",
            "incoming_gate_records": [],
            "skipped_gate_records": [],
        }
    )
    return rows


def _predecessor_record(root: Path, experiment_id: str) -> JsonDict:
    """Resolve one exact predecessor artifact or its V644 disposition row."""

    number = task_number(experiment_id)
    matches = sorted((root / "results").glob(f"experiment_{number}_*.json")) if number else []
    if len(matches) == 1:
        payload = read_json(matches[0])
        return {
            "path": str(matches[0].relative_to(root)),
            "sha256": sha256(matches[0]),
            "honest_verdict": str(payload.get("honest_verdict", "")),
        }
    if len(matches) > 1:
        raise ValueError(f"one exact predecessor artifact required for {experiment_id}")
    historical = read_json(root / V644_CAPSTONE_PATH)
    rows = [
        row
        for row in historical.get("task_dispositions", [])
        if isinstance(row, Mapping) and row.get("task_id") == experiment_id
    ]
    if len(rows) != 1:
        raise ValueError(f"one recorded predecessor disposition required for {experiment_id}")
    encoded = json.dumps(rows[0], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": f"{V644_CAPSTONE_PATH}#task_dispositions:{experiment_id}",
        "sha256": sha256_bytes(encoded),
        "honest_verdict": str(rows[0].get("honest_verdict", "")),
    }


def prior_failure_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    root: Path,
) -> list[JsonDict]:
    """Compare declared, recorded predecessor, and current verdict bytes exactly."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id == "exp7356-capstone":
            current = str(terminal["honest_verdict"])
            disposition = str(terminal["verdict_class"])
        else:
            source = evidence[task_id]
            record = source.get("canonical_record") or {}
            current = str(source["payload"].get("honest_verdict", record.get("detail", "")))
            disposition = str(source["disposition_class"])
        for prior in task.get("prior_failures") or []:
            predecessor = _predecessor_record(root, str(prior["experiment_id"]))
            declared = str(prior["verdict"])
            recorded = predecessor["honest_verdict"]
            declaration_matches = declared == recorded
            exact_repeat = recorded == current
            if not declaration_matches:
                decision = "preserve_record_mismatch_no_retirement"
            elif exact_repeat and prior["retire_if_same_verdict"]:
                decision = "retire_exact_repeat"
            elif disposition in {"blocked", "missing"}:
                decision = "preserve_prior_boundary_current_upstream_block"
            else:
                decision = "changed_prerequisite_observed"
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior["experiment_id"],
                    "declared_prior_honest_verdict_bytes": declared,
                    "declared_prior_honest_verdict_sha256": sha256_bytes(declared.encode("utf-8")),
                    "recorded_predecessor_path": predecessor["path"],
                    "recorded_predecessor_sha256": predecessor["sha256"],
                    "recorded_predecessor_honest_verdict_bytes": recorded,
                    "declared_prior_matches_recorded_predecessor": declaration_matches,
                    "current_honest_verdict_bytes": current,
                    "current_honest_verdict_sha256": sha256_bytes(current.encode("utf-8")),
                    "exact_repeat": exact_repeat,
                    "retire_if_same_verdict": prior["retire_if_same_verdict"],
                    "addressed_by": prior["addressed_by"],
                    "decision": decision,
                }
            )
    return rows


def next_branch_decisions(evidence: Mapping[str, Mapping[str, Any]], root: Path) -> list[JsonDict]:
    """Give each current or preserved boundary one falsifiable next condition."""

    v644 = read_json(root / V644_CAPSTONE_PATH)
    old = {row["branch"]: row for row in v644["next_branch_decisions"]}
    suffix = old["v642_suffix_retirement"]
    mixture = suffix["evidence"]["evidence"]["prior_retirement_signals"][0]
    board = evidence["exp7355-board-state"]["payload"]
    return [
        {
            "branch": "source_fidelity",
            "action": "repair_artifact_validation_without_reusing_flagged_metrics",
            "evidence": deepcopy(evidence["exp7348-plan-capture"]["payload"]["gate_check_summary"]),
            "reopening_condition": "A corrected Exp7348 replay passes its frozen 128-call denominator, cold validation, and adversarial checks.",
        },
        {
            "branch": "continuous_learning",
            "action": "blocked_pending_eligible_prospective_measurement",
            "evidence": deepcopy(evidence["exp7350-learning-audit"]["canonical_record"]),
            "reopening_condition": "An eligible prospective study and independent audit complete the sealed future-request rows with total-cost and causality gates.",
        },
        {
            "branch": "acquisition_cost",
            "action": "blocked_pending_changed_full_cost_strategy",
            "evidence": deepcopy(evidence["exp7352-acquisition-cost"]["canonical_record"]),
            "reopening_condition": "A changed acquisition strategy passes soundness and a paired complete-cost gate before an independent audit.",
        },
        {
            "branch": "live_arc_causality",
            "action": "retire_unchanged_four_episode_resume_mechanism",
            "evidence": deepcopy(evidence["exp7354-arc-transfer"]["payload"]["causal_chain_rows"]),
            "reopening_condition": "A changed mechanism yields a successful bound result and later policy action within the same fixed episode budget.",
        },
        {
            "branch": "hardware_disposition",
            "action": "preserve_expected_external_block",
            "evidence": deepcopy(board["gate_check_summary"]),
            "reopening_condition": board["next_hardware_conditions"]["gatemate"]["condition"],
        },
        {
            "branch": "native_tenfold_null",
            "action": "preserve_retirement",
            "evidence": deepcopy(old["native_host_cost"]),
            "reopening_condition": old["native_host_cost"]["reopening_condition"],
        },
        {
            "branch": "retired_joint_claim_comparison",
            "action": "preserve_retirement",
            "evidence": deepcopy(old["v643_source_comparison"]),
            "reopening_condition": old["v643_source_comparison"]["reopening_condition"],
        },
        {
            "branch": "delayed_feedback_mixture_retirement",
            "action": "preserve_retirement",
            "evidence": deepcopy(mixture),
            "reopening_condition": "A different learner changes the delayed-feedback information contract and passes future-error controls.",
        },
        {
            "branch": "delayed_feedback_suffix_retirement",
            "action": "preserve_retirement",
            "evidence": deepcopy(suffix),
            "reopening_condition": suffix["reopening_condition"],
        },
        {
            "branch": "unsupported_hardware_boundary",
            "action": "preserve_boundary",
            "evidence": deepcopy(board["board_rows"]),
            "reopening_condition": "A separately authorized run supplies current hardware evidence; PolarFire CPU dispatch alone never qualifies as FPGA sampling.",
        },
    ]


def run_publication_gate(root: Path) -> tuple[JsonDict, JsonDict]:
    """Run the stable read-only publication gate and retain exact output bytes."""

    argv = [str(root / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"]
    started = time.monotonic()
    progress(2, "before_subprocess", "publication_gate")
    completed = subprocess.run(  # noqa: S603 - fixed repository script and arguments.
        argv,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started
    progress(2, "after_subprocess", f"publication_gate exit={completed.returncode}")
    output = completed.stdout
    payload = json.loads(output) if completed.returncode == 0 else {}
    return payload, {
        "name": "publication_gate",
        "command": shlex.join(argv),
        "command_argv": argv,
        "scope": "stable_historical_publication_status_read_only",
        "exit_code": completed.returncode,
        "duration_s": elapsed,
        "log_path": None,
        "log_sha256": sha256_bytes((output + completed.stderr).encode("utf-8")),
        "passed": completed.returncode == 0 and isinstance(payload, dict),
        "timed_out": False,
        "output_tail": (output + completed.stderr)[-4000:],
    }


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Bind authorities, implementation, exact producers, and historical decisions."""

    paths = {
        Path(str(contract["selected_yaml_path"])),
        DESIGN_PATH,
        SPEC_PATH,
        EXCLUSION_PATH,
        CONDUCTOR_LOG_PATH,
        V644_CAPSTONE_PATH,
        V643_CAPSTONE_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    }
    paths.update(
        Path(str(row["selected_evidence_path"]))
        for row in evidence.values()
        if row["evidence_source"] == "declared_deliverable"
    )
    return {
        str(path): {"sha256": sha256(root / path), "available": True}
        for path in sorted(paths, key=str)
    }


def _preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record source availability and eligibility before claim reduction."""

    rows: list[JsonDict] = [
        {
            "check": "driving_requirement",
            "upstream": str(SPEC_PATH),
            "artifact_field": "REQ-REPORT-7356",
            "expected_value": True,
            "observed_value": "REQ-REPORT-7356" in (root / SPEC_PATH).read_text(encoding="utf-8"),
        },
        {
            "check": "exact_contract",
            "upstream": str(contract["selected_yaml_path"]),
            "artifact_field": "fourteen Markdown/YAML rows",
            "expected_value": True,
            "observed_value": contract.get("passed") is True,
        },
    ]
    for task_id, source in evidence.items():
        available = source["evidence_source"] in {
            "declared_deliverable",
            "canonical_conductor_block",
        }
        rows.extend(
            [
                {
                    "check": "producer_identity_and_availability",
                    "upstream": task_id,
                    "artifact_field": source["declared_deliverable_path"],
                    "expected_value": "exact deliverable or canonical conductor block",
                    "observed_value": {
                        "source": source["evidence_source"],
                        "path": source["selected_evidence_path"],
                        "authenticated": source["authenticated"],
                    },
                    "passed": available and source["authenticated"] is True,
                },
                {
                    "check": "producer_terminal_eligibility",
                    "upstream": task_id,
                    "artifact_field": "verdict_class/flagged_adversarial/quarantine",
                    "expected_value": "positive|circular_positive|null and clean",
                    "observed_value": {
                        "disposition_class": source["disposition_class"],
                        "flagged_adversarial": source["adversarially_flagged"],
                        "quarantined": source["quarantined"],
                    },
                    "passed": source["accepted_for_reduction"] is True,
                },
            ]
        )
    for row in rows[:2]:
        row["passed"] = row["observed_value"] == row["expected_value"]
    return rows


def _acceptance_results(
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    publication: Mapping[str, Any],
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep accounting, science, validation, and publication checks separate."""

    science_available = all(
        row.get("verdict_class") not in {"blocked", "disqualified", "partial"}
        for row in claims
        if row.get("claim") != "hardware_disposition"
    )
    definitions = (
        (
            "exact_contract",
            True,
            contract.get("passed"),
            contract.get("passed") is True,
            "Both contract authorities must agree before any disposition is counted.",
        ),
        (
            "fourteen_dispositions",
            14,
            len(dispositions),
            len(dispositions) == 14,
            "Completion is exact roster accounting and does not imply scientific value.",
        ),
        (
            "five_claim_classes",
            list(CLAIM_NAMES),
            [row.get("claim") for row in claims],
            [row.get("claim") for row in claims] == list(CLAIM_NAMES),
            "Separate evidence classes prevent one success from laundering another claim.",
        ),
        (
            "required_science_available",
            True,
            science_available,
            science_available,
            "Blocked or disqualified current science cannot authorize value or promotion.",
        ),
        (
            "required_validation",
            True,
            validation_passed,
            validation_passed,
            "Only current affected checks can qualify the capstone implementation.",
        ),
        (
            "stable_publication_gate_shape",
            ["G1", "G2", "G3", "G4"],
            list(publication.get("gates", {})),
            list(publication.get("gates", {})) == ["G1", "G2", "G3", "G4"],
            "The capstone preserves the stable gate instead of redefining readiness.",
        ),
    )
    return [
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "principle": principle,
        }
        for name, expected, observed, passed, principle in definitions
    ]


def scope_reduction_compliance() -> JsonDict:
    """Record the bounded questions and prerequisites chosen for V645."""

    return {
        "focused_science_questions": [
            "continuous structural learning on distinct future requests",
            "complete acquisition cost with exact feedback",
            "live ARC result-to-later-action causality",
        ],
        "source_fidelity_is_cross_cutting_control": True,
        "independent_infrastructure_prerequisites": [
            "isolated exact executor fixture",
            "live ARC result-resume path",
        ],
        "new_model_families": [],
        "new_hardware_integrations": [],
        "compliant": True,
    }


def _historical_failures(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Retain upstream repository-health observations outside current checks."""

    failures: list[JsonDict] = []
    seen: set[str] = set()
    for task_id, source in evidence.items():
        health = source.get("payload", {}).get("repository_health", {})
        candidates = [*health.get("historical_failures", [])]
        current = health.get("current_observation")
        if isinstance(current, Mapping):
            candidates.append(current)
        for row in candidates:
            if not isinstance(row, Mapping):
                continue
            identity = str(row.get("log_sha256") or row.get("command") or row)
            if identity in seen:
                continue
            seen.add(identity)
            failures.append(
                {
                    "observed_at": row.get("observed_at", RUN_DATE),
                    "producer": task_id,
                    "name": row.get("name"),
                    "command": row.get("command"),
                    "exit_code": row.get("exit_code"),
                    "duration_s": row.get("duration_s"),
                    "log_sha256": row.get("log_sha256"),
                    "collection_errors": deepcopy(row.get("collection_errors", [])),
                    "resolved": row.get("resolved", False),
                }
            )
    return failures


def build_artifact(
    *,
    root: Path,
    run_date: str,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    publication_receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal artifact from authenticated current evidence."""

    claims = build_claim_matrix(evidence)
    required_passed = validation.get("required_checks_passed") is True
    terminal_passed = validation.get("terminal_checks_passed", True) is True
    terminal = terminal_state(evidence, claims, required_passed and terminal_passed)
    gates = replay_gates(contract["tasks"], evidence)
    dispositions = task_dispositions(contract["tasks"], evidence, terminal, gates)
    artifact: JsonDict = {
        "schema": "carnot.experiment_7356.v645_capstone.v1",
        "experiment_id": "exp7356-capstone",
        "milestone": MILESTONE,
        "status": terminal["status"],
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": _preconditions(root, contract, evidence),
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "rows": deepcopy(claims),
        "sample_size_budget": {
            "planned": 14,
            "attempted": 14,
            "completed": 14,
            "censored": 0,
            "stopping_rule": "classify each of the fourteen frozen V645 tasks exactly once",
        },
        "acceptance_gate_results": _acceptance_results(
            contract,
            dispositions,
            claims,
            publication_gate,
            required_passed and terminal_passed,
        ),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": validation.get("flagged_adversarial") is True,
        "validation_receipts": [
            *deepcopy(validation.get("validation_receipts", [])),
            deepcopy(dict(publication_receipt)),
        ],
        "required_checks_passed": required_passed,
        "terminal_checks_passed": terminal_passed,
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "capstone_complete_score": 1,
        "capstone_readiness_score": 0,
        "capstone_value_score": 0,
        "capstone_promotion_score": 0,
        "task_dispositions": dispositions,
        "contract_rows": deepcopy(contract["contract_rows"]),
        "same_milestone_gate_replay_rows": gates,
        "claim_matrix": deepcopy(claims),
        "prior_failure_rows": prior_failure_rows(contract["tasks"], evidence, terminal, root),
        "publication_gate": deepcopy(dict(publication_gate)),
        "scope_reduction_compliance": scope_reduction_compliance(),
        "next_branch_decisions": next_branch_decisions(evidence, root),
        "current_model_evidence_boundary": {
            "current_model_loads": 0,
            "current_generation_calls": 0,
            "upstream_model_shaped_evidence_is_historical_or_scripted": True,
            "authentication": "source_artifact_hashes",
        },
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_message_performed": False,
        "production_default_changed": False,
        "research_roadmap_modified": False,
        "conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def receipt_set_passes(receipts: object) -> bool:
    """Require one successful receipt for each fixed scoped command."""

    if not isinstance(receipts, list):
        return False
    return all(
        sum(row.get("name") == name for row in receipts if isinstance(row, Mapping)) == 1
        and next(
            row for row in receipts if isinstance(row, Mapping) and row.get("name") == name
        ).get("passed")
        is True
        and next(
            row for row in receipts if isinstance(row, Mapping) and row.get("name") == name
        ).get("exit_code")
        == 0
        for name in REQUIRED_SCOPED_CHECKS
    )


def validate_artifact(value: object, *, root: Path = REPO_ROOT, replay: bool = False) -> list[str]:
    """Cold-check identity, raw reductions, decisions, hashes, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    artifact = value
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.experiment_7356.v645_capstone.v1"
        or artifact.get("experiment_id") != "exp7356-capstone"
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS,
        "model_boundary",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    dispositions = artifact.get("task_dispositions")
    add(
        not isinstance(dispositions, list)
        or len(dispositions) != 14
        or [row.get("task_id") for row in dispositions] != list(EXPECTED_TASK_IDS),
        "task_dispositions",
    )
    claims = artifact.get("claim_matrix")
    add(
        not isinstance(claims, list)
        or [row.get("claim") for row in claims] != list(CLAIM_NAMES)
        or artifact.get("rows") != claims,
        "claim_matrix",
    )
    add(
        artifact.get("capstone_complete_score") != 1
        or artifact.get("capstone_readiness_score") != 0
        or artifact.get("capstone_value_score") != 0
        or artifact.get("capstone_promotion_score") != 0,
        "capstone_scores",
    )
    add(not receipt_set_passes(artifact.get("validation_receipts")), "validation_receipts")
    add(
        artifact.get("scope_reduction_compliance") != scope_reduction_compliance(),
        "scope_reduction_compliance",
    )
    add(
        any(
            artifact.get(field) is not False
            for field in (
                "publication_performed",
                "upload_performed",
                "submission_performed",
                "external_message_performed",
                "production_default_changed",
                "research_roadmap_modified",
                "conductor_modified",
            )
        ),
        "unauthorized_action",
    )
    add(
        artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if replay:
        contract = load_contract(root)
        evidence = load_repository_evidence(root, contract["tasks"])
        expected_claims = build_claim_matrix(evidence)
        expected_terminal = terminal_state(
            evidence,
            expected_claims,
            artifact.get("required_checks_passed") is True
            and artifact.get("terminal_checks_passed") is True,
        )
        gates = replay_gates(contract["tasks"], evidence)
        expected_dispositions = task_dispositions(
            contract["tasks"], evidence, expected_terminal, gates
        )
        add(artifact.get("contract_rows") != contract["contract_rows"], "contract_rows")
        add(claims != expected_claims, "claim_matrix")
        add(dispositions != expected_dispositions, "task_dispositions")
        add(artifact.get("same_milestone_gate_replay_rows") != gates, "gate_replay")
        add(
            artifact.get("status") != expected_terminal["status"]
            or artifact.get("verdict_class") != expected_terminal["verdict_class"]
            or artifact.get("honest_verdict") != expected_terminal["honest_verdict"]
            or artifact.get("gate_check_summary") != expected_terminal["gate_check_summary"],
            "terminal_state",
        )
        add(
            artifact.get("prior_failure_rows")
            != prior_failure_rows(contract["tasks"], evidence, expected_terminal, root),
            "prior_failure_rows",
        )
        add(
            artifact.get("next_branch_decisions") != next_branch_decisions(evidence, root),
            "next_branch_decisions",
        )
        add(
            artifact.get("source_artifact_hashes") != _source_hashes(root, contract, evidence),
            "source_artifact_hashes",
        )
        publication, _ = run_publication_gate(root)
        add(artifact.get("publication_gate") != publication, "publication_gate")
    return errors


def run_terminal_commands(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run cold replay and the two required terminal artifact checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys; from pathlib import Path; "
        "from carnot.experiment_7356_v645_capstone import validate_artifact; "
        "p=Path(sys.argv[1]); e=validate_artifact(json.loads(p.read_text()),"
        "root=Path.cwd(),replay=True); "
        "print(json.dumps({'errors':e}),flush=True); raise SystemExit(bool(e))"
    )
    commands = [
        scoped.CommandSpec(
            "independent_terminal_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    ]
    return scoped.run_commands(root, commands, log_dir=raw_dir / "terminal_validation")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E.
    """Run scoped checks, cold terminal checks, and one atomic final write."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    args = parser.parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []
    progress(0, "start", f"date={args.date}")
    checkpoint = REPO_ROOT / DEFAULT_CHECKPOINT_PATH
    atomic_write_json(
        checkpoint,
        {"status": "running", "experiment_id": "exp7356-capstone", "started_at_utc": started_at},
    )

    phase_start = time.monotonic() - started
    progress(1, "start", "authenticate exact contract and fourteen dispositions")
    contract = load_contract(REPO_ROOT)
    evidence = load_repository_evidence(REPO_ROOT, contract["tasks"])
    phase_end = time.monotonic() - started
    spans.append(
        phase_row("authentication", phase_start, phase_end, 14, str(DEFAULT_CHECKPOINT_PATH))
    )
    progress(1, "complete", "authenticated dispositions=14")

    phase_start = time.monotonic() - started
    publication, publication_receipt = run_publication_gate(REPO_ROOT)
    phase_end = time.monotonic() - started
    spans.append(phase_row("publication_gate", phase_start, phase_end, 1, None))

    raw_dir = REPO_ROOT / DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="exp7356-validation-", dir="/tmp"))
    private_basetemp = temporary_root / "pytest"
    private_basetemp.mkdir(parents=True, exist_ok=True)
    phase_start = time.monotonic() - started
    progress(3, "start", "run scoped affected validation")
    validation = scoped.run_scoped_validation(
        REPO_ROOT,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=private_basetemp,
        coverage_file=temporary_root / ".coverage",
        log_dir=raw_dir / "scoped_validation",
        historical_failures=_historical_failures(evidence),
    )
    phase_end = time.monotonic() - started
    spans.append(
        phase_row("scoped_validation", phase_start, phase_end, len(REQUIRED_SCOPED_CHECKS), None)
    )
    progress(3, "complete", f"required_checks_passed={validation['required_checks_passed']}")

    phase_start = time.monotonic() - started
    progress(4, "start", "reduce raw producer rows and write measured candidate")
    candidate = build_artifact(
        root=REPO_ROOT,
        run_date=args.date,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=publication_receipt,
        validation=validation,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured-terminal-candidate.json"
    atomic_write_json(candidate_path, candidate)
    raw_reduction_path = raw_dir / "independent-raw-reduction.json"
    atomic_write_json(
        raw_reduction_path,
        {
            "source_artifact_hashes": candidate["source_artifact_hashes"],
            "claim_matrix": candidate["claim_matrix"],
            "gate_replay": candidate["same_milestone_gate_replay_rows"],
        },
    )
    phase_end = time.monotonic() - started
    spans.append(
        phase_row(
            "candidate_reduction_and_write",
            phase_start,
            phase_end,
            5,
            str(
                candidate_path.relative_to(REPO_ROOT)
                if candidate_path.is_relative_to(REPO_ROOT)
                else candidate_path
            ),
        )
    )
    candidate_label = (
        candidate_path.relative_to(REPO_ROOT)
        if candidate_path.is_relative_to(REPO_ROOT)
        else candidate_path
    )
    progress(4, "complete", f"candidate={candidate_label}")

    phase_start = time.monotonic() - started
    progress(5, "start", "run cold reducer and terminal validators")
    terminal_receipts = run_terminal_commands(REPO_ROOT, candidate_path, raw_dir)
    validation = dict(validation)
    validation["validation_receipts"] = [
        *validation["validation_receipts"],
        *terminal_receipts,
    ]
    validation["terminal_checks_passed"] = all(row["passed"] is True for row in terminal_receipts)
    validation["flagged_adversarial"] = any(
        row.get("name") == "adversarial_verify" and row.get("passed") is not True
        for row in terminal_receipts
    )
    phase_end = time.monotonic() - started
    spans.append(
        phase_row("terminal_validation", phase_start, phase_end, len(terminal_receipts), None)
    )
    progress(
        5,
        "complete",
        f"terminal_checks_passed={validation['terminal_checks_passed']}",
    )

    final = build_artifact(
        root=REPO_ROOT,
        run_date=args.date,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=publication_receipt,
        validation=validation,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=REPO_ROOT, replay=True)
    if errors:
        progress(6, "failed", f"cold_replay_errors={errors}")
        return 1
    output = REPO_ROOT / DEFAULT_OUTPUT_PATH
    atomic_write_json(output, final)
    reloaded = read_json(output)
    reload_errors = validate_artifact(reloaded, root=REPO_ROOT, replay=True)
    progress(6, "complete", f"output={DEFAULT_OUTPUT_PATH} reload_errors={reload_errors}")
    return int(bool(reload_errors))


if __name__ == "__main__":  # pragma: no cover - wrapper is the declared entrypoint.
    raise SystemExit(main())
