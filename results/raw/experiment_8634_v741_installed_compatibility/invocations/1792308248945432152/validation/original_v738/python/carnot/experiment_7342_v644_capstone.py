"""Close V644 while keeping scientific and board dispositions independent.

The reducer reads exact current outputs or exact conductor block records. It
does not invoke a model, a board, or an external service.

Spec refs: REQ-REPORT-7342 and SCENARIO-REPORT-7342-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
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
from carnot import experiment_7329_v644_contract as contract_source
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.644"
RUN_DATE = "20260916"
RANDOM_SEED = {
    "development": 7_342_202_609_16,
    "evaluation": 7_342_202_609_17,
    "resampling": 7_342_202_609_18,
}
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

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
    "acquired_constraint_learning",
    "live_arc_causality",
    "native_host_cost",
    "board_evidence",
)
DECISION_BRANCHES = (
    *CLAIM_NAMES,
    "v643_source_comparison",
    "v642_suffix_retirement",
    "v642_storage_retirement",
)
PRIOR_DECISIONS = {
    "retire_exact_repeat",
    "preserve_prior_boundary_current_upstream_block",
    "reopen_on_independent_disposition_contract",
    "changed_prerequisite_observed",
}
REQUIRED_SCOPED_CHECKS = scoped.REQUIRED_CHECK_NAMES

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
V643_CAPSTONE_PATH = Path("results/experiment_7328_v643_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7342_v644_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7342_v644_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7342_v644_capstone.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7342_v644_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7342_v644_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7342_v644_capstone")

# The timestamp and title prefix identify one immutable conductor row. This is
# safer than selecting a similarly named JSON file that the task never produced.
CANONICAL_BLOCK_MARKERS = {
    "exp7331-learning-adapter": (
        "| 2026-09-16 02:48 UTC | Connect acquired constraints to the opt-in verific"
    ),
    "exp7332-plan-canary": (
        "| 2026-09-16 02:51 UTC | Qualify bounded Qwen3.8 plan proposals and identit"
    ),
    "exp7333-plan-capture": (
        "| 2026-09-16 02:53 UTC | Capture fresh Qwen3.8 proposals with source-fideli"
    ),
    "exp7334-prospective-learning": (
        "| 2026-09-16 02:53 UTC | Measure continuous learning on fresh plans and cha"
    ),
    "exp7335-learning-audit": (
        "| 2026-09-16 02:53 UTC | Audit executor independence learning causality and"
    ),
    "exp7337-arc-transfer": (
        "| 2026-09-16 04:29 UTC | Measure adapter-withheld ARC transfer with consume"
    ),
    "exp7338-arc-causal-audit": (
        "| 2026-09-16 04:31 UTC | Audit live feedback delivery and later action caus"
    ),
}

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the artifact while preserving ordinary top-level experiment_id and milestone.",
    "status": "Publish a terminal result only after current work and affected validation.",
    "run_date": "Use 20260916 and retain actual UTC timestamps.",
    "preconditions_checked": "Name input identity, availability and each failed check before work.",
    "MODEL_SPECS": "List actual intended/current executable identities; any LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True when any real load or generation is attempted, even if no usable answer arrives.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight loads and generations.",
    "inference_substrate": "Declare actual computation; historical model evidence is not a current invocation.",
    "inference_substrate_class": "Declare the real duration class; small fixed-token runs are model_bounded_generation.",
    "execution_venue": "Use host; old board receipts never imply current board execution.",
    "duration_s": "Measure real monotonic elapsed time; never pad a duration floor.",
    "phase_spans": "Disjoint stage spans, completed units, checkpoint positions and pending operations explain cost.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, public inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers rather than similarly named older results.",
    "rows": "Every comparative unit and arm carries metrics, costs, abstentions, failures and censoring.",
    "sample_size_budget": "Record planned, attempted, completed and censored units with a fixed stopping rule.",
    "acceptance_gate_results": "Each gate records expected, observed and passed; separate completion from value.",
    "gate_check_summary": "Every blocked_* must identify upstream, failed check, artifact field, expected and observed value.",
    "verifier_is_oracle": "True whenever the execution authority defines correctness; independent code alone does not remove circularity.",
    "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_ and names the check.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; external absence is blocked.",
    "validation_receipts": "Record exact command, affected scope, exit code, duration and log hash including failures.",
    "repository_health": "Keep unrelated repository failures as dated observations, separate from affected required checks.",
    "field_principles": "Explain each field without wrapping executable scores or ordinary dictionaries.",
    "capstone_complete_score": "One means fourteen exact dispositions, including expected blocks and this reducer.",
    "task_dispositions": "Carry each exact path/hash/class or canonical external-block record.",
    "claim_matrix": "Separate source semantics, learning authority, live causality, host performance and board evidence.",
    "publication_gate": "Carry stable G1-G4 without treating them as new science or publication authority.",
    "next_branch_decisions": "Each branch has an action and a falsifiable reopening condition.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


def progress(phase: int, event: str, detail: str) -> None:
    """Flush a factual boundary so a long validation remains observable."""

    print(f"[exp7342] phase={phase} event={event} {detail}", flush=True)


def sha256_bytes(content: bytes) -> str:
    """Hash exact bytes so a later text rewrite changes the identity."""

    return "sha256:" + hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    """Hash one exact file without normalizing its content."""

    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> JsonDict:
    """Read one JSON object and reject an array or scalar root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required at {path}")
    return value


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all artifact fields except the digest that stores this hash."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(encoded.encode("utf-8"))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Move a complete sibling document into place in one filesystem step."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Use the shipped independent V644 Markdown and YAML parsers."""

    return contract_source.evaluate_contract(markdown_text, yaml_document)


def load_contract(root: Path) -> JsonDict:
    """Select the V644 authority once and compare its two source formats."""

    selected, document, content, candidates = contract_source.select_yaml_authority(root)
    if selected is None or document is None or content is None:
        raise ValueError("no selected V644 YAML authority")
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


def _task_number(task_id: str) -> int | None:
    """Return the established numeric experiment identity when present."""

    match = re.match(r"exp(\d+)", task_id)
    return int(match.group(1)) if match else None


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept only the declared full task ID or its numeric experiment form."""

    number = _task_number(task_id)
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def classify_payload(
    task_id: str, payload: Mapping[str, Any], *, quarantined: bool
) -> tuple[bool, str, bool]:
    """Classify terminal identity before any readiness score is inspected."""

    authenticated = (
        payload.get("status") in {"complete", "blocked", "disqualified"}
        and _identity_matches(task_id, payload)
        and payload.get("verdict_class") in VALID_VERDICT_CLASSES
    )
    declared = str(payload.get("verdict_class", "disqualified"))
    if quarantined:
        disposition = "quarantined"
    elif not authenticated:
        disposition = "disqualified"
    elif payload.get("status") == "blocked":
        disposition = "blocked"
    else:
        disposition = declared
    accepted = bool(
        authenticated
        and not quarantined
        and payload.get("status") == "complete"
        and disposition in {"positive", "circular_positive", "null"}
    )
    return authenticated, disposition, accepted


def _canonical_record(root: Path, task_id: str) -> JsonDict | None:
    """Select one exact conductor block row by its frozen timestamp and title."""

    marker = CANONICAL_BLOCK_MARKERS.get(task_id)
    if marker is None:
        return None
    matches = [
        (number, line)
        for number, line in enumerate(
            (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8").splitlines(), 1
        )
        if line.startswith(marker)
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
    }


def load_evidence(root: Path, task: Mapping[str, Any], manifest: object) -> JsonDict:
    """Read one exact deliverable or its exact canonical conductor block."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    path = root / declared
    if path.is_file():
        payload = read_json(path)
        quarantine = common.quarantine_receipt(payload, task_id, declared, manifest)
        authenticated, disposition, accepted = classify_payload(
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
            "authenticated": authenticated,
            "accepted_for_reduction": accepted,
            "disposition_class": disposition,
        }
    record = _canonical_record(root, task_id)
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
        "authenticated": False,
        "accepted_for_reduction": False,
        "disposition_class": "missing",
    }


def load_repository_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Create exactly one evidence slot for each producer before the capstone."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {str(task["id"]): load_evidence(root, task, manifest) for task in tasks[:-1]}


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay structured gates while terminal eligibility outranks scores."""

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
            elif disposition == "disqualified":
                outcome = "disqualified"
            elif disposition == "blocked":
                outcome = "blocked"
            elif disposition == "partial":
                outcome = "partial"
            elif field not in payload:
                outcome = "missing_field"
            else:
                operator = str(gate.get("op"))
                expected = gate.get("value")
                if operator == "==":
                    passed = observed == expected
                elif operator == "in":
                    passed = observed in expected
                else:
                    passed = False
                outcome = "passed" if passed else "value_mismatch"
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
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def _blocked_claim(claim: str, producer: str, source: Mapping[str, Any]) -> JsonDict:
    """Keep unavailable science explicit instead of manufacturing a zero."""

    return {
        "unit_id": f"claim:{claim}",
        "arm": "authenticated_aggregation",
        "claim": claim,
        "producer": producer,
        "producer_sha256": source.get("artifact_sha256"),
        "verdict_class": "blocked",
        "metric": None,
        "metrics": {},
        "costs": {"model_loads": 0, "generation_calls": 0, "hardware_operations": 0},
        "failures": [f"producer_{source.get('disposition_class', 'missing')}"],
        "abstention": True,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _native_claim(source: Mapping[str, Any]) -> JsonDict:
    """Reduce complete host cost separately from native binding readiness."""

    payload = source["payload"]
    summary = payload["cost_summary"]
    intervals = {
        size: row["throughput_ratio_native_over_python"]
        for size, row in summary["by_batch_size"].items()
    }
    return {
        "unit_id": "claim:native_host_cost",
        "arm": "complete_in_process_boundary",
        "claim": "native_host_cost",
        "producer": "exp7340-native-cost",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "null",
        "mechanism_evidence_class": "circular_positive",
        "metric": min(row["ci95_lower"] for row in intervals.values()),
        "metrics": {
            "paired_blocks": summary["paired_blocks"],
            "parity_mismatches": summary["parity_mismatches"],
            "ten_x_lower_bound_passed": payload["native_ten_x_score"] == 1,
            "throughput_ratio_intervals": deepcopy(intervals),
        },
        "costs": {
            "rows": summary["row_count"],
            "setup_costs": deepcopy(payload["setup_costs"]),
            "amortization": deepcopy(payload["amortization"]),
        },
        "failures": ["unchanged_native_ten_x_lower_bound_gate_failed"],
        "abstention": False,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _board_claim(source: Mapping[str, Any]) -> JsonDict:
    """Count an expected board block without converting it to readiness."""

    payload = source["payload"]
    return {
        "unit_id": "claim:board_evidence",
        "arm": "read_only_board_dispositions",
        "claim": "board_evidence",
        "producer": "exp7341-board-continuity",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "blocked",
        "metric": payload["board_disposition_complete_score"],
        "metrics": {
            "disposition_complete_score": payload["board_disposition_complete_score"],
            "hardware_readiness_score": payload["hardware_readiness_score"],
            "hardware_promotion_score": payload["hardware_promotion_score"],
            "hardware_operations_issued_count": payload["hardware_operations_issued_count"],
        },
        "costs": {"hardware_operations": payload["hardware_operations_issued_count"]},
        "failures": ["gatemate_changed_physical_state_receipt_missing"],
        "abstention": True,
        "censored": False,
        "expected_external_block": True,
        "disposition_complete": True,
        "promotes_scientific_efficacy": False,
    }


def build_claim_matrix(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute five conclusions without merging their authority classes."""

    rows = [
        _blocked_claim("source_fidelity", "exp7333-plan-capture", evidence["exp7333-plan-capture"]),
        _blocked_claim(
            "acquired_constraint_learning",
            "exp7335-learning-audit",
            evidence["exp7335-learning-audit"],
        ),
        _blocked_claim(
            "live_arc_causality", "exp7338-arc-causal-audit", evidence["exp7338-arc-causal-audit"]
        ),
    ]
    native = evidence["exp7340-native-cost"]
    rows.append(
        _native_claim(native)
        if native.get("accepted_for_reduction") is True
        and native.get("disposition_class") == "null"
        else _blocked_claim("native_host_cost", "exp7340-native-cost", native)
    )
    board = evidence["exp7341-board-continuity"]
    rows.append(
        _board_claim(board)
        if board.get("authenticated") is True
        and board.get("disposition_class") == "blocked"
        and board.get("payload", {}).get("board_disposition_complete_score") == 1
        else _blocked_claim("board_evidence", "exp7341-board-continuity", board)
    )
    return rows


def _science_failures(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Name exact failed prerequisites for required current science."""

    definitions = (
        ("exp7330-executor-isolation", "executor_fixture_ready_score", 1),
        ("exp7336-arc-resume", "arc_resume_ready_score", 1),
    )
    failures: list[JsonDict] = []
    for upstream, field, expected in definitions:
        source = evidence[upstream]
        observed = source["payload"].get(field)
        if source.get("accepted_for_reduction") is not True or observed != expected:
            failures.append(
                {
                    "upstream": upstream,
                    "check": field,
                    "artifact_field": field,
                    "expected_value": expected,
                    "observed_value": observed,
                    "terminal_blocking": True,
                }
            )
    return failures


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    validation_passed: bool,
) -> JsonDict:
    """Keep complete accounting separate from scientific availability."""

    if not validation_passed:
        return {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v644_capstone_required_validation_failed",
            "gate_check_summary": {
                "passed": False,
                "failures": [
                    {
                        "upstream": "exp7342-capstone",
                        "check": "required_scoped_and_terminal_validation",
                        "artifact_field": "required_checks_passed/terminal_checks_passed",
                        "expected_value": True,
                        "observed_value": False,
                        "terminal_blocking": True,
                    }
                ],
            },
        }
    failures = _science_failures(evidence)
    if failures:
        first = str(failures[0]["upstream"]).replace("-", "_")
        return {
            "status": "complete",
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_{first}: all fourteen dispositions are represented; "
                "required current science remains unavailable while the expected board block stays independent"
            ),
            "gate_check_summary": {"passed": False, "failures": failures},
        }
    science = [row for row in claims if row.get("claim") != "board_evidence"]
    classes = {str(row.get("verdict_class")) for row in science}
    verdict_class = "circular_positive" if "circular_positive" in classes else "null"
    return {
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": (
            "complete_circular_positive_bounded_v644_claims_no_broad_learning_claim"
            if verdict_class == "circular_positive"
            else "complete_null_v644_scientific_claims"
        ),
        "gate_check_summary": {"passed": True, "failures": []},
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Represent every producer and the non-recursive capstone self row."""

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
            "incoming_gate_records": [],
            "skipped_gate_records": [],
        }
    )
    return rows


def prior_failure_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Compare every prior verdict as exact UTF-8 text and decide its scope."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id == "exp7342-capstone":
            current = str(terminal["honest_verdict"])
            disposition = str(terminal["verdict_class"])
        else:
            source = evidence[task_id]
            record = source.get("canonical_record") or {}
            current = str(source["payload"].get("honest_verdict", record.get("detail", "")))
            disposition = str(source["disposition_class"])
        for prior in task.get("prior_failures") or []:
            verdict = str(prior["verdict"])
            exact = verdict == current
            if exact and prior["retire_if_same_verdict"]:
                decision = "retire_exact_repeat"
            elif task_id == "exp7342-capstone":
                decision = "reopen_on_independent_disposition_contract"
            elif disposition in {"blocked", "missing"}:
                decision = "preserve_prior_boundary_current_upstream_block"
            else:
                decision = "changed_prerequisite_observed"
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior["experiment_id"],
                    "prior_honest_verdict_bytes": verdict,
                    "prior_honest_verdict_sha256": sha256_bytes(verdict.encode("utf-8")),
                    "current_honest_verdict_bytes": current,
                    "current_honest_verdict_sha256": sha256_bytes(current.encode("utf-8")),
                    "exact_repeat": exact,
                    "retire_if_same_verdict": prior["retire_if_same_verdict"],
                    "addressed_by": prior["addressed_by"],
                    "decision": decision,
                }
            )
    return rows


def next_branch_decisions(evidence: Mapping[str, Mapping[str, Any]], root: Path) -> list[JsonDict]:
    """Give each current and preserved branch one falsifiable next condition."""

    historical = read_json(root / V643_CAPSTONE_PATH)
    old = {row["branch"]: row for row in historical["next_branch_decisions"]}
    return [
        {
            "branch": "source_fidelity",
            "action": "blocked_repair_upstream",
            "evidence": evidence["exp7333-plan-capture"]["canonical_record"],
            "reopening_condition": "Exp7330 is terminal and eligible with executor_fixture_ready_score=1, then Exp7333 completes the frozen source-fidelity panel.",
        },
        {
            "branch": "acquired_constraint_learning",
            "action": "blocked_repair_upstream",
            "evidence": evidence["exp7335-learning-audit"]["canonical_record"],
            "reopening_condition": "Exp7331 and Exp7333 become eligible, and Exp7334 plus Exp7335 complete the frozen prospective cohorts and cold audit.",
        },
        {
            "branch": "live_arc_causality",
            "action": "blocked_repair_upstream",
            "evidence": evidence["exp7338-arc-causal-audit"]["canonical_record"],
            "reopening_condition": "Exp7336 passes arc_resume_ready_score=1 and a same-budget Exp7337 trace reaches a later policy action for Exp7338 audit.",
        },
        {
            "branch": "native_host_cost",
            "action": "retire_exact_boundary_ten_x_claim",
            "evidence": deepcopy(evidence["exp7340-native-cost"]["payload"]["retirement"]),
            "reopening_condition": "A changed complete boundary has a paired 95 percent lower speedup bound of at least 10 with zero parity mismatches.",
        },
        {
            "branch": "board_evidence",
            "action": "preserve_expected_external_block",
            "evidence": deepcopy(
                evidence["exp7341-board-continuity"]["payload"]["gate_check_summary"][
                    "first_failure"
                ]
            ),
            "reopening_condition": evidence["exp7341-board-continuity"]["payload"][
                "next_hardware_conditions"
            ]["gatemate"]["condition"],
        },
        {
            "branch": "v643_source_comparison",
            "action": "preserve_retirement",
            "evidence": deepcopy(old["source_cost"]),
            "reopening_condition": old["source_cost"]["reopening_condition"],
        },
        {
            "branch": "v642_suffix_retirement",
            "action": "preserve_retirement",
            "evidence": deepcopy(old["v642_suffix_retirement"]),
            "reopening_condition": old["v642_suffix_retirement"]["reopening_condition"],
        },
        {
            "branch": "v642_storage_retirement",
            "action": "preserve_retirement",
            "evidence": deepcopy(old["v642_storage_retirement"]),
            "reopening_condition": old["v642_storage_retirement"]["reopening_condition"],
        },
    ]


def run_publication_gate(root: Path) -> tuple[JsonDict, JsonDict]:
    """Run the stable read-only gate and retain its exact returned bytes."""

    argv = [str(root / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"]
    started = time.monotonic()
    progress(2, "before_subprocess", "publication_gate")
    completed = subprocess.run(  # noqa: S603 - fixed repository script and argument vector.
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
    """Bind contract, code, current producers, conductor blocks, and history."""

    paths = {
        Path(str(contract["selected_yaml_path"])),
        DESIGN_PATH,
        SPEC_PATH,
        EXCLUSION_PATH,
        CONDUCTOR_LOG_PATH,
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
    """Record every input identity and adverse state before score reduction."""

    rows = [
        {
            "check": "driving_requirement",
            "upstream": str(SPEC_PATH),
            "artifact_field": "REQ-REPORT-7342",
            "expected_value": True,
            "observed_value": "REQ-REPORT-7342" in (root / SPEC_PATH).read_text(encoding="utf-8"),
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
        rows.append(
            {
                "check": "producer_identity_and_availability",
                "upstream": task_id,
                "artifact_field": source["declared_deliverable_path"],
                "expected_value": "exact deliverable or canonical conductor block",
                "observed_value": {
                    "source": source["evidence_source"],
                    "path": source["selected_evidence_path"],
                    "disposition_class": source["disposition_class"],
                },
            }
        )
    for row in rows:
        row["passed"] = (
            row["observed_value"] == row["expected_value"]
            if isinstance(row["expected_value"], bool)
            else row["observed_value"] is not None
        )
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
        if row.get("claim") != "board_evidence"
    )
    definitions = (
        ("exact_contract", True, contract.get("passed"), contract.get("passed") is True),
        ("fourteen_dispositions", 14, len(dispositions), len(dispositions) == 14),
        (
            "five_claim_classes",
            list(CLAIM_NAMES),
            [row.get("claim") for row in claims],
            [row.get("claim") for row in claims] == list(CLAIM_NAMES),
        ),
        ("required_science_available", True, science_available, science_available),
        ("required_validation", True, validation_passed, validation_passed),
        (
            "stable_publication_gate_shape",
            ["G1", "G2", "G3", "G4"],
            list(publication.get("gates", {})),
            list(publication.get("gates", {})) == ["G1", "G2", "G3", "G4"],
        ),
    )
    return [
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
        }
        for name, expected, observed, passed in definitions
    ]


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
        "schema": "carnot.experiment_7342.v644_capstone.v1",
        "experiment_id": "exp7342-capstone",
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
            "stopping_rule": "classify each of the fourteen frozen V644 tasks exactly once",
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
        "prior_failure_rows": prior_failure_rows(contract["tasks"], evidence, terminal),
        "publication_gate": deepcopy(dict(publication_gate)),
        "next_branch_decisions": next_branch_decisions(evidence, root),
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


def _receipt_set_passes(receipts: object) -> bool:
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
    """Cold-check identity, evidence rows, decisions, hashes, and checksum."""

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
        artifact.get("schema") != "carnot.experiment_7342.v644_capstone.v1"
        or artifact.get("experiment_id") != "exp7342-capstone"
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
    add(not _receipt_set_passes(artifact.get("validation_receipts")), "validation_receipts")
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
            != prior_failure_rows(contract["tasks"], evidence, expected_terminal),
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


def date_argument(value: str) -> str:
    """Require the conductor's compact execution-date format."""

    if not re.fullmatch(r"\d{8}", value):
        raise ValueError("date must use YYYYMMDD")
    return value


def _phase_row(name: str, start: float, end: float, units: int, checkpoint: str | None) -> JsonDict:
    """Record one real disjoint span without a padded duration."""

    return {
        "phase": name,
        "start_s": start,
        "end_s": end,
        "duration_s": end - start,
        "units": units,
        "checkpoint": checkpoint,
        "pending_operations": [],
    }


def _run_terminal_commands(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run cold replay and both required terminal artifact checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys; from pathlib import Path; "
        "from carnot.experiment_7342_v644_capstone import validate_artifact; "
        "p=Path(sys.argv[1]); e=validate_artifact(json.loads(p.read_text()),root=Path.cwd(),replay=True); "
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
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
        {"status": "running", "experiment_id": "exp7342-capstone", "started_at_utc": started_at},
    )

    phase_start = time.monotonic() - started
    progress(1, "start", "authenticate contract and fourteen dispositions")
    contract = load_contract(REPO_ROOT)
    evidence = load_repository_evidence(REPO_ROOT, contract["tasks"])
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("authentication", phase_start, phase_end, 14, str(DEFAULT_CHECKPOINT_PATH))
    )

    phase_start = time.monotonic() - started
    publication, publication_receipt = run_publication_gate(REPO_ROOT)
    phase_end = time.monotonic() - started
    spans.append(_phase_row("publication_gate", phase_start, phase_end, 1, None))

    raw_dir = REPO_ROOT / DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="exp7342-validation-", dir="/tmp"))
    private_basetemp = temporary_root / "pytest"
    private_basetemp.mkdir()
    phase_start = time.monotonic() - started
    progress(3, "start", "scoped affected validation")
    validation = scoped.run_scoped_validation(
        REPO_ROOT,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=private_basetemp,
        coverage_file=temporary_root / ".coverage",
        log_dir=raw_dir / "scoped_validation",
        historical_failures=[],
    )
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("scoped_validation", phase_start, phase_end, len(REQUIRED_SCOPED_CHECKS), None)
    )

    phase_start = time.monotonic() - started
    progress(4, "start", "build measured terminal candidate")
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
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row(
            "candidate_reduction",
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

    phase_start = time.monotonic() - started
    progress(5, "start", "terminal candidate validation")
    terminal_receipts = _run_terminal_commands(REPO_ROOT, candidate_path, raw_dir)
    validation = dict(validation)
    validation["validation_receipts"] = [
        *validation["validation_receipts"],
        *terminal_receipts,
    ]
    validation["terminal_checks_passed"] = all(row["passed"] is True for row in terminal_receipts)
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("terminal_validation", phase_start, phase_end, len(terminal_receipts), None)
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
