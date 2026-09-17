"""Measure whether exact-feedback atoms change later distinct decisions.

The evaluator reuses the shipped opt-in adapter and exact Boolean executor. It
adds only proposal-list selection, paired stream reduction, and the terminal
experiment record needed for the V646 prospective study.

Spec refs: REQ-CL-7362 and SCENARIO-CL-7362-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot import experiment_7330_v644_public_learner as public
from carnot import experiment_7346_v645_learning_adapter as adapter_module
from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7360_v646_learning_fixture as fixture_module
from carnot.experiment_7346_v645_learning_adapter import (
    AdapterPipelineHarness,
    LearningAdapterError,
    LearningScheduleAdapter,
    QualifiedFixtureExecutor,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7362-prospective-learning"
SCHEMA = "carnot.exp7362.v646_prospective_learning.v1"

PERSISTENT_ARM = adapter_module.PERSISTENT_ARM
RESET_ARM = adapter_module.RESET_ARM
CACHE_ARM = adapter_module.CACHE_ARM
FROZEN_ARM = adapter_module.FROZEN_ARM
ARMS = adapter_module.ARMS
QUERY_BUDGET = adapter_module.QUERY_BUDGET
STATE_CAP_BYTES = adapter_module.STATE_CAP_BYTES
BOOTSTRAP_DRAWS = 10_000
RESAMPLING_SEED = 7_360_307

MODULE_PATH = Path("python/carnot/experiment_7362_v646_prospective_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7362_v646_prospective_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7362_v646_prospective_learning.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
FIXTURE_PATH = Path("results/experiment_7360_v646_learning_fixture.json")
CAPTURE_PATH = Path("results/experiment_7361_v646_fresh_plan_capture.json")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7362_v646_prospective_learning.json")
RAW_DIR = Path("results/raw/experiment_7362_v646_prospective_learning")

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ALL_REQUIRED_CHECK_NAMES = (*REQUIRED_CHECK_NAMES, "full_python_suite", *TERMINAL_CHECK_NAMES)

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

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version this record and retain ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and affected validation; never a success-shaped placeholder.",
    "run_date": "Use 20260917 and actual UTC timestamps.",
    "preconditions_checked": "Exact input, resource and required-field checks before dependent work.",
    "MODEL_SPECS": "Actual intended identities; every LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, even failure.",
    "invocation_counts": "Attempted/completed/failed/cancelled/in-flight calls; separate current from historical.",
    "inference_substrate": "Actual computation, with historical inference in explicitly labeled hash-bound sidecars.",
    "inference_substrate_class": "Actual closed duration class; no duration padding.",
    "execution_venue": "Host CPU or owned CUDA runtime as measured; no new board execution in V646.",
    "duration_s": "Measured monotonic elapsed, never synthetic elapsed or sleep to pass a floor.",
    "phase_spans": "Disjoint measured load, generation, evaluation, validation and write spans.",
    "random_seed": "Frozen development/evaluation/resampling seeds; null if truly inapplicable.",
    "reproducibility_checksum": "Bind exact code, settings, evaluator, inputs and raw evidence.",
    "source_artifact_hashes": "Exact producer paths and immutable byte hashes; preserve original classes/flags.",
    "rows": "Every comparative unit/arm/metric/cost/failure/censoring disposition, not only pooled means.",
    "sample_size_budget": "Frozen planned/attempted/completed/censored units and stopping rules.",
    "acceptance_gate_results": "Expected, observed and passed separately for required validation, safety and scientific value; never mark failed value as successful.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, expected and observed value, including missing paths.",
    "verifier_is_oracle": "True when the evaluator defines truth; independent code alone cannot remove circularity.",
    "honest_verdict": "Precise free-text terminal outcome; distinguish accounting, null science and unavailable work.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished retryable OWN work is partial; external unchanged absence is blocked.",
    "flagged_adversarial": "Current independent verification state; critical findings set true and prevent promotion.",
    "validation_receipts": "Exact command/scope/return code/elapsed/log hash for every required and diagnostic check, including failures.",
    "repository_health": "Dated unrelated failures kept separately from affected required validation.",
    "field_principles": "Explain each field without wrapping scalar gates or ordinary dictionaries.",
    "learning_capture_complete_score": "One when all frozen arms/cohorts and required controls/checks finish validly, regardless of benefit.",
    "learning_value_score": "One only if every frozen safety/utility/coverage/query/cost/causal gate passes; zero on a null result.",
    "structural_learning_witnesses": "Admitted atom provenance, later distinct request, snapshot, exact decision and atom-erasure counterfactual.",
    "per_stream_results": "Per-arm stream metrics, cohort, query categories, utility, coverage, full cost and drift state.",
    "confidence_intervals": "Paired stream bootstrap with frozen seeds and resample count; separate live and synthetic estimates.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        *REQUIRED_FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "phase",
        "started_at_utc",
        "completed_at_utc",
        "host_computation",
        "frozen_acceptance_manifest",
        "independent_reduction",
        "control_results",
        "promotion_score",
        "no_model_weight_mutation",
        "production_defaults_changed",
        "research_roadmap_changed",
        "historical_comparator",
        "raw_evidence_paths",
    }
)


def progress(phase: str, event: str, started: float, detail: str = "") -> None:
    """Emit a flushed phase boundary with measured monotonic time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7362] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a declared producer cannot silently drift."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object with a local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted rename leaves this.
            temporary.unlink()


def _load_object(path: Path) -> JsonDict:
    """Load a JSON object and map malformed external bytes to an empty object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
        "blocking": True,
    }


def _declared_hash(producer: Mapping[str, Any], path: Path) -> str | None:
    declared = dict(producer.get("source_artifact_hashes") or {})
    resolved = str(path.resolve())
    candidates = [
        value
        for name, value in declared.items()
        if str(name) == str(path) or str(Path(str(name)).resolve()) == resolved
    ]
    return str(candidates[0]) if len(candidates) == 1 else None


def _dependent_paths(
    fixture: Mapping[str, Any], capture: Mapping[str, Any]
) -> list[tuple[str, Path, str]]:
    fixture_manifest = dict(fixture.get("fixture_manifest") or {})
    rows: list[tuple[str, Path, str]] = []
    for field in ("public_manifest_path", "private_manifest_path", "acceptance_manifest_path"):
        value = fixture_manifest.get(field)
        if isinstance(value, str) and value:
            rows.append(("fixture_dependent_bytes", Path(value), field))
    candidate = capture.get("candidate_manifest_path")
    if isinstance(candidate, str) and candidate:
        rows.append(("capture_dependent_bytes", Path(candidate), "candidate_manifest_path"))
    for name in dict(capture.get("source_artifact_hashes") or {}):
        if str(name).endswith("private_evaluation.json"):
            rows.append(("capture_dependent_bytes", Path(str(name)), "private_evaluation"))
    return rows


def collect_preconditions(
    repo_root: Path,
    *,
    fixture_path: Path | None = None,
    capture_path: Path | None = None,
    exclusion_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Check exact producer fields and every consumed byte before evaluation."""

    root = repo_root.resolve()
    fixture_path = fixture_path or root / FIXTURE_PATH
    capture_path = capture_path or root / CAPTURE_PATH
    exclusion_path = exclusion_path or root / "ops/exclusion_manifest.yaml"
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    producers: dict[str, JsonDict] = {"fixture": {}, "capture": {}}
    for name, path in (("fixture", fixture_path), ("capture", capture_path)):
        present = path.is_file() and path.stat().st_size > 0
        value = _load_object(path) if present else {}
        producers[name] = value
        checks.append(
            _precondition(
                f"{name}_path",
                str(path),
                "bytes",
                "readable_json_object",
                "readable_json_object" if value else None,
                bool(value),
            )
        )
        if present:
            hashes[str(path)] = sha256_file(path)

    expected = {
        "fixture": (
            ("experiment_id", "exp7360-learning-fixture"),
            ("milestone", MILESTONE),
            ("run_date", RUN_DATE),
            ("learning_fixture_ready_score", 1),
            ("verdict_class", {"positive", "circular_positive", "null"}),
            ("flagged_adversarial", False),
        ),
        "capture": (
            ("experiment_id", "exp7361-fresh-plan-capture"),
            ("milestone", MILESTONE),
            ("run_date", RUN_DATE),
            ("plan_capture_complete_score", 1),
            ("verdict_class", {"positive", "circular_positive", "null"}),
            ("flagged_adversarial", False),
        ),
    }
    for name, pairs in expected.items():
        producer = producers[name]
        for field, wanted in pairs:
            observed = producer.get(field)
            passed = observed in wanted if isinstance(wanted, set) else observed == wanted
            display = sorted(wanted) if isinstance(wanted, set) else wanted
            checks.append(
                _precondition(
                    f"{name}_{field}",
                    str(fixture_path if name == "fixture" else capture_path),
                    field,
                    display,
                    observed,
                    passed,
                )
            )
        status = str(producer.get("status", ""))
        eligible = bool(status) and not any(
            token in status.lower()
            for token in ("blocked", "partial", "disqualified", "quarantined")
        )
        checks.append(
            _precondition(
                f"{name}_eligible_status",
                str(fixture_path if name == "fixture" else capture_path),
                "status",
                "terminal_not_blocked_partial_disqualified_or_quarantined",
                status or None,
                eligible,
            )
        )

    for check, path, field in _dependent_paths(producers["fixture"], producers["capture"]):
        owner = producers["fixture"] if check.startswith("fixture") else producers["capture"]
        expected_hash = _declared_hash(owner, path)
        present = path.is_file() and path.stat().st_size > 0
        observed_hash = sha256_file(path) if present else None
        checks.append(
            _precondition(
                check,
                str(path),
                field,
                expected_hash,
                observed_hash,
                expected_hash is not None and expected_hash == observed_hash,
            )
        )
        if observed_hash is not None:
            hashes[str(path)] = observed_hash

    required_paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/exclusion_manifest.yaml"),
        SPEC_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7346_v645_learning_adapter.py"),
        Path("python/carnot/memory/transactional_constraint_memory.py"),
        Path("python/carnot/pipeline/verify_repair.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in required_paths:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                path.stat().st_size if present else None,
                present,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7362",
            "REQ-CL-7362" if "REQ-CL-7362" in spec_text else None,
            "REQ-CL-7362" in spec_text,
        )
    )
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "exp7362" in exclusion_text.lower()
    checks.append(
        _precondition(
            "current_task_not_excluded",
            str(exclusion_path),
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
        )
    )
    if exclusion_path.is_file():
        hashes[str(exclusion_path)] = sha256_file(exclusion_path)
    return checks, hashes, producers


def _plan_errors(request: Mapping[str, Any], plan: object) -> list[str]:
    if not isinstance(plan, Mapping):
        return ["plan_not_object"]
    if set(plan) != {"request_id", "assignments"}:
        return ["plan_fields"]
    if plan.get("request_id") != request.get("request_id"):
        return ["request_identity"]
    assignments = plan.get("assignments")
    activities = list(request.get("activities") or [])
    if not isinstance(assignments, Mapping) or set(assignments) != set(activities):
        return ["assignment_entities"]
    errors: list[str] = []
    for name in activities:
        value = assignments.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"assignment_type:{name}")
        elif value not in request["allowed_starts"][name]:
            errors.append(f"assignment_domain:{name}")
    return errors


def _select_candidate(
    learner: public.PublicConstraintLearner,
    request: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict | None:
    atoms = learner.active_atoms()
    for candidate in candidates:
        plan = candidate.get("plan")
        if candidate.get("source_valid") is True and not _plan_errors(request, plan):
            assert isinstance(plan, Mapping)
            if all(public._atom_allows_plan(atom, request, plan) for atom in atoms):  # noqa: SLF001
                return deepcopy(dict(plan))
    return None


class CapturedProposalAdapter(LearningScheduleAdapter):
    """Select a sealed proposal list while retaining the adapter transaction rules."""

    def begin_candidate_request(
        self,
        request: Mapping[str, Any],
        executor: QualifiedFixtureExecutor,
        candidates: Sequence[Mapping[str, Any]],
        *,
        allow_learning: bool,
    ) -> JsonDict:
        """Freeze memory first, then select without reading exact feedback."""

        super().begin_request(request, executor, allow_learning=allow_learning)
        assert self._context is not None  # noqa: SLF001 - this subclass owns the extension.
        context = self._context  # noqa: SLF001
        learner = context["learner"]
        valid = [
            deepcopy(dict(row))
            for row in candidates
            if row.get("source_valid") is True and not _plan_errors(request, row.get("plan"))
        ]
        unconstrained = deepcopy(valid[0]["plan"]) if valid else None
        proposed = _select_candidate(learner, request, valid)
        influenced = [
            atom["atom_id"]
            for atom in learner.active_atoms()
            if unconstrained is not None
            and not public._atom_allows_plan(atom, request, unconstrained)  # noqa: SLF001
        ]
        context.update(
            {
                "unconstrained": unconstrained,
                "proposed": proposed,
                "influenced_atom_ids": influenced,
                "candidate_action": (
                    "rejected"
                    if proposed is None
                    else "redirected"
                    if unconstrained != proposed
                    else "accepted"
                ),
            }
        )
        return self.propose()

    def decision_without_atom(
        self,
        request: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        entry_records: Sequence[Mapping[str, Any]],
        atom_id: str,
    ) -> JsonDict | None:
        """Erase one atom from the entry snapshot and repeat candidate selection."""

        learner = self._learner(  # noqa: SLF001 - inherited decoder preserves exact records.
            str(request["version_token"]), entry_records, extra_invalidated=[atom_id]
        )
        return _select_candidate(learner, request, candidates)


def execute_candidate_request(
    adapter: CapturedProposalAdapter,
    harness: AdapterPipelineHarness,
    request: Mapping[str, Any],
    private_record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    warmup: bool,
    atom_provenance: dict[str, JsonDict],
    allow_learning: bool = True,
    exact_cache: dict[str, bool] | None = None,
) -> tuple[JsonDict, list[JsonDict]]:
    """Seal one decision, verify it exactly, then expose delayed atom effects."""

    started = time.monotonic()
    executor = QualifiedFixtureExecutor(request, private_record, exact_cache=exact_cache)
    proposal = adapter.begin_candidate_request(
        request, executor, candidates, allow_learning=allow_learning
    )
    assert adapter._context is not None  # noqa: SLF001 - needed for the saved entry snapshot.
    entry_records = deepcopy(adapter._context["records"])  # noqa: SLF001
    plan = proposal["plan"]
    if plan is None:
        adapter.cancel_request("no_source_valid_candidate")
        feedback: JsonDict = {
            "new_atom_count": 0,
            "new_atom_ids": [],
            "invalidated_atom_ids": [],
            "commit_count": 0,
            "state_bytes": len(adapter.state_bytes()),
            "entry_state_unchanged_during_request": True,
        }
        returned = False
        verified = False
    else:
        harness.extractor.bind(executor, plan)
        result = harness.pipeline.verify(
            f"Verify captured schedule {request['request_id']}",
            public.canonical_bytes(plan).decode("utf-8"),
            domain="schedule",
        )
        feedback = adapter.last_feedback
        verified = bool(result.verified)
        returned = verified

    for atom_id in feedback.get("new_atom_ids", []):
        atom_provenance[str(atom_id)] = {
            "admission_request_id": str(request["request_id"]),
            "entry_state_hash": proposal["entry_state_hash"],
            "exact_query_receipts": deepcopy(executor.receipts),
        }

    witnesses: list[JsonDict] = []
    if plan is not None:
        for atom_id in proposal["influenced_atom_ids"]:
            without = adapter.decision_without_atom(
                request, candidates, entry_records, str(atom_id)
            )
            provenance = atom_provenance.get(str(atom_id))
            if (
                provenance is not None
                and provenance["admission_request_id"] != request["request_id"]
                and without != plan
            ):
                witnesses.append(
                    {
                        "atom_id": str(atom_id),
                        "admission_request_id": provenance["admission_request_id"],
                        "later_request_id": str(request["request_id"]),
                        "entry_state_hash": proposal["entry_state_hash"],
                        "decision_with_atom": deepcopy(plan),
                        "decision_without_atom": deepcopy(without),
                        "single_atom_erasure": True,
                        "returned_feasible": verified,
                        "exact_final_receipt": deepcopy(
                            next(
                                (
                                    row
                                    for row in reversed(executor.receipts)
                                    if row["reason"] == "final"
                                ),
                                None,
                            )
                        ),
                    }
                )

    selected_hash = public.sha256_json(plan) if plan is not None else None
    dispositions: list[JsonDict] = []
    invalid_count = 0
    for candidate in candidates:
        errors = list(candidate.get("source_errors") or [])
        errors.extend(_plan_errors(request, candidate.get("plan")))
        source_valid = candidate.get("source_valid") is True and not errors
        if not source_valid:
            disposition = "source_invalid"
            invalid_count += 1
        elif public.sha256_json(candidate["plan"]) == selected_hash:
            disposition = "selected"
        else:
            disposition = "not_selected"
        dispositions.append(
            {
                "call_id": candidate.get("call_id"),
                "candidate_id": candidate.get("candidate_id"),
                "source_valid": source_valid,
                "source_errors": sorted(set(errors)),
                "disposition": disposition,
                "plan_hash": (
                    public.sha256_json(candidate["plan"])
                    if isinstance(candidate.get("plan"), Mapping)
                    else None
                ),
            }
        )
    model_cost = sum(float(row.get("historical_generation_cost_s", 0.0)) for row in candidates)
    elapsed = time.monotonic() - started
    query_categories = Counter(str(row["reason"]) for row in executor.receipts)
    final_receipt = next(
        (row for row in reversed(executor.receipts) if row["reason"] == "final"), None
    )
    row: JsonDict = {
        "row_type": "request",
        "request_id": str(request["request_id"]),
        "version_token": str(request["version_token"]),
        "warmup": bool(warmup),
        "candidate_action": proposal["candidate_action"],
        "source_proposals": len(candidates),
        "source_invalid_proposals": invalid_count,
        "proposal_dispositions": dispositions,
        "entry_state_hash": proposal["entry_state_hash"],
        "entry_state_unchanged_during_request": bool(
            feedback.get("entry_state_unchanged_during_request", True)
        ),
        "active_atom_count_before": proposal["active_atom_count_before"],
        "influenced_atom_ids": list(proposal["influenced_atom_ids"]),
        "decision_changed_by_memory": bool(proposal["candidate_action"] == "redirected"),
        "returned_plan": deepcopy(plan) if returned else None,
        "returned": returned,
        "returned_feasible": bool(returned and verified),
        "coverage": int(returned),
        "utility": adapter_module._utility(request, plan if returned else None),  # noqa: SLF001
        "paid_queries": executor.attempt_count,
        "query_attempts": executor.attempt_count,
        "executor_calls": executor.external_call_count,
        "cache_hits": executor.cache_hits,
        "query_categories": dict(sorted(query_categories.items())),
        "final_checks": int(final_receipt is not None),
        "final_receipt": deepcopy(final_receipt),
        "atoms_proposed": int(feedback.get("new_atom_count", 0)),
        "atoms_admitted": int(feedback.get("commit_count", 0)),
        "atoms_rejected": max(
            0, int(feedback.get("new_atom_count", 0)) - int(feedback.get("commit_count", 0))
        ),
        "atoms_invalidated": len(feedback.get("invalidated_atom_ids", [])),
        "new_atom_ids": list(feedback.get("new_atom_ids", [])),
        "invalidated_atom_ids": list(feedback.get("invalidated_atom_ids", [])),
        "state_bytes": int(feedback.get("state_bytes", len(adapter.state_bytes()))),
        "query_budget_exceeded": executor.attempt_count > QUERY_BUDGET,
        "stale_version_decision": False,
        "restarts": 0,
        "request_latency_s": elapsed,
        "model_amortization_s": model_cost,
        "complete_service_cost_s": elapsed + model_cost,
        "censored": False,
        "abstained": not returned,
        "failures": [] if returned else ["source_invalid_or_exact_rejection"],
    }
    return row, witnesses


def capture_candidate_groups(
    manifest: Mapping[str, Any],
    fidelity_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[str]]:
    """Join the frozen schedule, exact call bytes, fidelity, and historical cost."""

    schedule = manifest.get("schedule")
    calls = manifest.get("calls")
    if not isinstance(schedule, list) or not isinstance(calls, list):
        return [], ["schedule_or_calls_not_list"]
    errors: list[str] = []
    if len(schedule) != len(calls):
        errors.append("schedule_call_count_mismatch")
    call_ids = [str(row.get("call_id")) for row in calls]
    if len(set(call_ids)) != len(call_ids):
        errors.append("call_identity_mismatch")
    by_call = {str(row.get("call_id")): row for row in calls}
    fidelity = {str(row.get("call_id")): row for row in fidelity_rows}
    costs = {str(row.get("call_id")): row for row in cost_rows}
    groups: dict[tuple[str, str, str], JsonDict] = {}
    identity_fields = (
        "candidate_id",
        "candidate_index",
        "stream_id",
        "panel_id",
        "pair_side",
    )
    for scheduled in schedule:
        call_id = str(scheduled.get("call_id"))
        call = by_call.get(call_id)
        if call is None:
            if "schedule_call_count_mismatch" not in errors:
                errors.append("schedule_call_count_mismatch")
            continue
        if any(call.get(field) != scheduled.get(field) for field in identity_fields):
            errors.append("call_identity_mismatch")
            continue
        request = call.get("public_request")
        if not isinstance(request, Mapping):
            errors.append("public_request_missing")
            continue
        key = (
            str(call.get("stream_id")),
            str(call.get("panel_id")),
            str(call.get("pair_side")),
        )
        group = groups.setdefault(
            key,
            {
                "stream_id": key[0],
                "panel_id": key[1],
                "pair_side": key[2],
                "cohort": str(call.get("cohort", scheduled.get("cohort", "unknown"))),
                "warmup": bool(call.get("warmup", scheduled.get("warmup", False))),
                "request": deepcopy(dict(request)),
                "first_call_index": int(call.get("call_index", scheduled.get("call_index", 0))),
                "candidates": [],
            },
        )
        fidelity_row = fidelity.get(call_id, {})
        source_checks = {
            "terminal_response": call.get("terminal_state") == "response",
            "parse_valid": call.get("parse_status") == "valid",
            "schema_valid": fidelity_row.get("schema_valid") is True,
            "request_identity_fidelity": fidelity_row.get("request_identity_fidelity") is True,
            "entity_fidelity": fidelity_row.get("entity_fidelity") is True,
            "quantity_fidelity": fidelity_row.get("quantity_fidelity") is True,
            "ordering_fidelity": fidelity_row.get("ordering_fidelity") is True,
            "public_semantic_correct": fidelity_row.get("public_semantic_correct") is True,
        }
        cost = costs.get(call_id, {})
        group["candidates"].append(
            {
                "call_id": call_id,
                "candidate_id": call.get("candidate_id"),
                "candidate_index": int(call.get("candidate_index", 0)),
                "source_valid": all(source_checks.values()),
                "source_errors": sorted(
                    name for name, passed in source_checks.items() if not passed
                ),
                "plan": deepcopy(call.get("decoded_plan")),
                "raw_reply_sha256": call.get("raw_reply_sha256"),
                "historical_generation_cost_s": float(cost.get("generation_duration_s", 0.0))
                + float(cost.get("allocated_model_load_s", 0.0)),
            }
        )
    result = sorted(groups.values(), key=lambda row: row["first_call_index"])
    for group in result:
        group["candidates"].sort(key=lambda row: row["candidate_index"])
        if len(group["candidates"]) != 2:
            errors.append("candidate_group_size")
    return result, sorted(set(errors))


def _normalize_synthetic_row(row: Mapping[str, Any]) -> JsonDict:
    value = deepcopy(dict(row))
    value.update(
        {
            "cohort": "synthetic",
            "pair_side": "synthetic",
            "paid_queries": int(value.get("query_attempts", 0)),
            "query_categories": {
                "total": int(value.get("query_attempts", 0)),
                "final": int(value.get("final_checks", 0)),
            },
            "atoms_proposed": int(value.get("new_atom_count", 0)),
            "atoms_admitted": int(value.get("new_atom_count", 0)),
            "atoms_rejected": 0,
            "atoms_invalidated": len(value.get("invalidated_atom_ids", [])),
            "stale_version_decision": bool(value.get("stale_atom_returned", False)),
            "query_budget_exceeded": int(value.get("query_attempts", 0)) > QUERY_BUDGET,
            "request_latency_s": float(value.get("complete_wall_cost", 0.0)),
            "model_amortization_s": 0.0,
            "complete_service_cost_s": float(value.get("complete_wall_cost", 0.0)),
            "restarts": 0,
        }
    )
    return value


def run_measurement(
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    capture_artifact: Mapping[str, Any],
    state_root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run the four frozen arms over synthetic and live proposal streams."""

    synthetic_manifest = deepcopy(dict(public_manifest))
    synthetic_manifest["public_model_streams"] = []
    synthetic = adapter_module.run_measurement(
        synthetic_manifest, private_manifest, state_root / "synthetic"
    )
    rows = [_normalize_synthetic_row(row) for row in synthetic]
    groups, errors = capture_candidate_groups(
        proposal_manifest,
        capture_artifact.get("source_fidelity_rows", []),
        capture_artifact.get("generation_cost_rows", []),
    )
    if errors:
        raise LearningAdapterError("capture_manifest:" + ",".join(errors))
    by_stream: dict[str, list[JsonDict]] = defaultdict(list)
    for group in groups:
        by_stream[str(group["stream_id"])].append(group)
    witnesses: list[JsonDict] = []
    records = private_manifest["evaluator_records"]
    total = len(by_stream) * len(ARMS)
    completed = 0
    for stream_id, stream_groups in sorted(by_stream.items()):
        for arm in ARMS:
            learner = CapturedProposalAdapter(
                state_root / "live" / stream_id / arm,
                enabled=True,
                persist_feedback=arm in {PERSISTENT_ARM, FROZEN_ARM},
            )
            harness = AdapterPipelineHarness(learner)
            exact_cache: dict[str, bool] | None = {} if arm == CACHE_ARM else None
            provenance: dict[str, JsonDict] = {}
            for request_index, group in enumerate(stream_groups):
                request = group["request"]
                row, request_witnesses = execute_candidate_request(
                    learner,
                    harness,
                    request,
                    records[str(request["request_id"])],
                    group["candidates"],
                    warmup=bool(group["warmup"]),
                    atom_provenance=provenance,
                    allow_learning=arm != FROZEN_ARM or bool(group["warmup"]),
                    exact_cache=exact_cache,
                )
                row.update(
                    {
                        "cohort": "live",
                        "source_label": group["cohort"],
                        "stream_id": stream_id,
                        "panel_id": group["panel_id"],
                        "pair_side": group["pair_side"],
                        "request_index": request_index,
                        "arm": arm,
                    }
                )
                rows.append(row)
                for witness in request_witnesses:
                    witness.update(
                        {
                            "cohort": "live",
                            "stream_id": stream_id,
                            "panel_id": group["panel_id"],
                            "pair_side": group["pair_side"],
                            "arm": arm,
                        }
                    )
                    witnesses.append(witness)
            harness.close()
            completed += 1
            print(
                f"[exp7362] phase=evaluation event=unit_complete completed={completed}/{total} rows={len(rows)}",
                flush=True,
            )
    return rows, witnesses


def run_frozen_controls(
    state_root: Path,
    *,
    public_manifest: Mapping[str, Any] | None = None,
    private_manifest: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Replay drift, restart, rollback, cache, and higher-order controls."""

    if public_manifest is None:
        fixture = _load_object(REPO_ROOT / FIXTURE_PATH)
        public_manifest = _load_object(Path(fixture["fixture_manifest"]["public_manifest_path"]))
    if private_manifest is None:
        fixture = _load_object(REPO_ROOT / FIXTURE_PATH)
        private_manifest = _load_object(Path(fixture["fixture_manifest"]["private_manifest_path"]))
    rows = fixture_module.run_adapter_safety_controls(state_root / "adapter")
    challenge = public_manifest["compound_conflict_challenge"]
    request = challenge["request"]
    record = private_manifest["evaluator_records"][str(request["request_id"])]
    executor = QualifiedFixtureExecutor(request, record)
    learner = public.PublicConstraintLearner(str(request["version_token"]))
    plan = challenge["candidate_plan"]
    full_rejected = not executor.query(plan, "compound_full")
    atoms = learner.localize_rejection(request, plan, executor.query)
    rows.append(
        {
            "row_type": "control",
            "control": "higher_order_counterexample",
            "expected": {
                "full_rejected": True,
                "all_pair_projections_accepted": True,
                "learned_atom_count": 0,
            },
            "observed": {
                "full_rejected": full_rejected,
                "all_pair_projections_accepted": full_rejected and not atoms,
                "learned_atom_count": len(atoms),
            },
            "passed": full_rejected and not atoms,
            "query_attempts": executor.attempt_count,
            "state_bytes": len(learner.state_bytes()),
            "censored": False,
        }
    )
    return {"passed": all(row.get("passed") is True for row in rows), "rows": rows}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    stream_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    comparator: str,
    *,
    seed: int,
    draws: int,
) -> JsonDict:
    stream_ids = sorted(stream_metrics)
    if not stream_ids:
        return {
            "utility_difference_ci95": {"lower": math.nan, "upper": math.nan},
            "coverage_difference_ci95": {"lower": math.nan, "upper": math.nan},
            "query_ratio_ci95": {"lower": math.nan, "upper": math.nan},
            "complete_service_cost_ratio_ci95": {"lower": math.nan, "upper": math.nan},
            "stream_count": 0,
            "bootstrap_draws": draws,
            "cluster_unit": "stream",
        }
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        chosen = [rng.choice(stream_ids) for _ in stream_ids]
        persistent = [stream_metrics[key][PERSISTENT_ARM] for key in chosen]
        baseline = [stream_metrics[key][comparator] for key in chosen]
        samples["utility"].append(
            sum(left["utility"] - right["utility"] for left, right in zip(persistent, baseline))
            / len(chosen)
        )
        samples["coverage"].append(
            sum(left["coverage"] - right["coverage"] for left, right in zip(persistent, baseline))
            / len(chosen)
        )
        samples["query"].append(
            sum(row["queries"] for row in persistent)
            / max(1.0, sum(row["queries"] for row in baseline))
        )
        samples["cost"].append(
            sum(row["cost"] for row in persistent)
            / max(1e-12, sum(row["cost"] for row in baseline))
        )
    return {
        "utility_difference_ci95": {
            "lower": _quantile(samples["utility"], 0.025),
            "upper": _quantile(samples["utility"], 0.975),
        },
        "coverage_difference_ci95": {
            "lower": _quantile(samples["coverage"], 0.025),
            "upper": _quantile(samples["coverage"], 0.975),
        },
        "query_ratio_ci95": {
            "lower": _quantile(samples["query"], 0.025),
            "upper": _quantile(samples["query"], 0.975),
        },
        "complete_service_cost_ratio_ci95": {
            "lower": _quantile(samples["cost"], 0.025),
            "upper": _quantile(samples["cost"], 0.975),
        },
        "stream_count": len(stream_ids),
        "bootstrap_draws": draws,
        "cluster_unit": "stream",
    }


def test_acceptance_manifest() -> JsonDict:
    """Return the frozen thresholds for focused reducer tests."""

    return fixture_module.frozen_acceptance_manifest("sha256:" + "1" * 64)


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def reduce_rows(
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    acceptance_manifest: Mapping[str, Any],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> JsonDict:
    """Reduce paired whole streams and apply each frozen Exp7360 threshold."""

    evaluation = [
        row
        for row in rows
        if row.get("row_type") in {"request", "comparison"}
        and row.get("warmup") is False
        and row.get("censored") is False
    ]
    per_stream: list[JsonDict] = []
    confidence: dict[str, JsonDict] = {}
    for cohort_index, cohort in enumerate(("synthetic", "live")):
        cohort_rows = [row for row in evaluation if row.get("cohort") == cohort]
        metrics: dict[str, dict[str, JsonDict]] = {}
        for stream_id in sorted({str(row.get("stream_id")) for row in cohort_rows}):
            metrics[stream_id] = {}
            for arm in ARMS:
                selected = [
                    row
                    for row in cohort_rows
                    if str(row.get("stream_id")) == stream_id and row.get("arm") == arm
                ]
                if not selected:
                    continue
                query_categories: Counter[str] = Counter()
                for row in selected:
                    query_categories.update(
                        {
                            str(key): int(value)
                            for key, value in row.get("query_categories", {}).items()
                        }
                    )
                metric = {
                    "cohort": cohort,
                    "stream_id": stream_id,
                    "arm": arm,
                    "request_rows": len(selected),
                    "pair_sides": sorted({str(row.get("pair_side")) for row in selected}),
                    "utility": sum(float(row.get("utility", 0.0)) for row in selected)
                    / len(selected),
                    "coverage": sum(float(row.get("coverage", 0.0)) for row in selected)
                    / len(selected),
                    "queries": sum(
                        float(row.get("paid_queries", row.get("query_attempts", 0)))
                        for row in selected
                    ),
                    "query_categories": dict(sorted(query_categories.items())),
                    "cost": sum(float(row.get("complete_service_cost_s", 0.0)) for row in selected),
                    "model_amortization_s": sum(
                        float(row.get("model_amortization_s", 0.0)) for row in selected
                    ),
                    "atoms_proposed": sum(int(row.get("atoms_proposed", 0)) for row in selected),
                    "atoms_admitted": sum(int(row.get("atoms_admitted", 0)) for row in selected),
                    "atoms_rejected": sum(int(row.get("atoms_rejected", 0)) for row in selected),
                    "atoms_invalidated": sum(
                        int(row.get("atoms_invalidated", 0)) for row in selected
                    ),
                    "restarts": sum(int(row.get("restarts", 0)) for row in selected),
                    "failure_count": sum(bool(row.get("failures")) for row in selected),
                    "censored_count": sum(bool(row.get("censored")) for row in selected),
                    "drift_labels": sorted(
                        {str(row.get("source_label", "unknown")) for row in selected}
                    ),
                }
                metrics[stream_id][arm] = metric
                per_stream.append(deepcopy(metric))
        complete = {key: value for key, value in metrics.items() if set(value) == set(ARMS)}
        confidence[cohort] = {
            comparator: _paired_bootstrap(
                complete,
                comparator,
                seed=int(acceptance_manifest.get("resampling_seed", RESAMPLING_SEED))
                + cohort_index * 10
                + comparator_index,
                draws=bootstrap_draws,
            )
            for comparator_index, comparator in enumerate((RESET_ARM, CACHE_ARM), start=1)
        }

    gate_specs = {
        str(row["check"]): dict(row) for row in acceptance_manifest.get("value_gates", [])
    }
    unsafe = sum(
        bool(row.get("returned")) and row.get("returned_feasible") is not True for row in evaluation
    )
    stale = sum(bool(row.get("stale_version_decision")) for row in evaluation)
    intervals = [
        confidence[cohort][comparator]
        for cohort in ("synthetic", "live")
        for comparator in (RESET_ARM, CACHE_ARM)
    ]
    utility_lower = min(
        (value["utility_difference_ci95"]["lower"] for value in intervals), default=math.nan
    )
    coverage_lower = min(
        (value["coverage_difference_ci95"]["lower"] for value in intervals), default=math.nan
    )
    reset_query_upper = max(
        (confidence[cohort][RESET_ARM]["query_ratio_ci95"]["upper"] for cohort in confidence),
        default=math.nan,
    )
    cache_query_upper = max(
        (confidence[cohort][CACHE_ARM]["query_ratio_ci95"]["upper"] for cohort in confidence),
        default=math.nan,
    )
    cost_upper = max(
        (value["complete_service_cost_ratio_ci95"]["upper"] for value in intervals),
        default=math.nan,
    )
    causal = sum(
        row.get("single_atom_erasure") is True
        and row.get("admission_request_id") != row.get("later_request_id")
        and row.get("decision_with_atom") != row.get("decision_without_atom")
        for row in witnesses
    )
    utility_threshold = float(gate_specs["normalized_utility_noninferiority"]["lower_95_inclusive"])
    coverage_threshold = float(gate_specs["coverage_noninferiority"]["lower_95_inclusive"])
    reset_threshold = float(gate_specs["query_ratio_vs_reset"]["upper_95_exclusive"])
    cache_threshold = float(gate_specs["query_ratio_vs_exact_cache"]["upper_95_exclusive"])
    cost_threshold = float(gate_specs["complete_service_cost_ratio"]["upper_95_inclusive"])
    witness_minimum = int(gate_specs["later_distinct_request_erasure_witness"]["minimum"])
    gates = {
        "unsafe_accepted_plans": _gate(
            0, unsafe, unsafe == 0, "No unsafe accepted plan may be returned."
        ),
        "stale_version_decisions": _gate(
            0, stale, stale == 0, "Stale atoms cannot control a decision."
        ),
        "normalized_utility_noninferiority": _gate(
            {"lower_95_inclusive": utility_threshold},
            {"minimum_lower_95": utility_lower},
            utility_lower >= utility_threshold,
            "Every cohort and comparator must preserve normalized utility.",
        ),
        "coverage_noninferiority": _gate(
            {"lower_95_inclusive": coverage_threshold},
            {"minimum_lower_95": coverage_lower},
            coverage_lower >= coverage_threshold,
            "Every cohort and comparator must preserve feasibility coverage.",
        ),
        "query_ratio_vs_reset": _gate(
            {"upper_95_exclusive": reset_threshold},
            {"maximum_upper_95": reset_query_upper},
            reset_query_upper < reset_threshold,
            "Persistent memory must reduce paid queries against reset.",
        ),
        "query_ratio_vs_exact_cache": _gate(
            {"upper_95_exclusive": cache_threshold},
            {"maximum_upper_95": cache_query_upper},
            cache_query_upper < cache_threshold,
            "Structural reuse must outperform exact-request caching.",
        ),
        "complete_service_cost_ratio": _gate(
            {"upper_95_inclusive": cost_threshold},
            {"maximum_upper_95": cost_upper},
            cost_upper <= cost_threshold,
            "CPU work plus historical model amortization cannot exceed the controls.",
        ),
        "later_distinct_request_erasure_witness": _gate(
            {"minimum": witness_minimum},
            {"witness_count": causal},
            causal >= witness_minimum,
            "One individually necessary atom must change a later distinct request.",
        ),
    }
    return {
        "row_count": len(rows),
        "evaluation_row_count": len(evaluation),
        "per_stream_results": per_stream,
        "confidence_intervals": confidence,
        "gate_results": gates,
        "learning_value_passed": all(value["passed"] for value in gates.values()),
        "structural_witness_count": causal,
        "bootstrap_draws": bootstrap_draws,
        "resampling_seed": int(acceptance_manifest.get("resampling_seed", RESAMPLING_SEED)),
    }


def passing_test_preconditions() -> list[JsonDict]:
    """Return one explicit passing prerequisite for reducer unit tests."""

    return [_precondition("test", "fixture", "ready", True, True, True)]


def passing_test_receipts() -> list[JsonDict]:
    """Return one successful receipt for each required terminal command."""

    return [
        {
            "name": name,
            "command": f"test:{name}",
            "command_argv": ["test", name],
            "scope": "unit_test",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in ALL_REQUIRED_CHECK_NAMES
    ]


def test_phase_spans() -> list[JsonDict]:
    """Return disjoint measured-shape spans for terminal reducer tests."""

    return [
        {"phase": "load", "start_elapsed_s": 0.0, "end_elapsed_s": 0.0, "duration_s": 0.0},
        {"phase": "generation", "start_elapsed_s": 0.0, "end_elapsed_s": 0.0, "duration_s": 0.0},
        {"phase": "evaluation", "start_elapsed_s": 0.0, "end_elapsed_s": 0.5, "duration_s": 0.5},
        {"phase": "validation", "start_elapsed_s": 0.5, "end_elapsed_s": 0.9, "duration_s": 0.4},
        {"phase": "write", "start_elapsed_s": 0.9, "end_elapsed_s": 1.0, "duration_s": 0.1},
    ]


def _receipt_passed(receipts: Sequence[Mapping[str, Any]], name: str) -> bool:
    selected = [row for row in receipts if row.get("name") == name]
    return (
        len(selected) == 1
        and selected[0].get("passed") is True
        and selected[0].get("exit_code") == 0
    )


def _gate_summary(
    preconditions: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    failed_precondition = next((row for row in preconditions if not row.get("passed")), None)
    if failed_precondition is not None:
        return {
            "upstream": failed_precondition["upstream"],
            "failed_check": failed_precondition["check"],
            "artifact_field": failed_precondition["artifact_field"],
            "expected_value": failed_precondition["expected_value"],
            "observed_value": failed_precondition["observed_value"],
            "passed": False,
        }
    failed_gate = next((name for name, row in gates.items() if not row.get("passed")), None)
    if failed_gate is not None:
        value = gates[failed_gate]
        return {
            "upstream": EXPERIMENT_ID,
            "failed_check": failed_gate,
            "artifact_field": f"acceptance_gate_results.{failed_gate}",
            "expected_value": value["expected"],
            "observed_value": value["observed"],
            "passed": False,
        }
    return {
        "upstream": EXPERIMENT_ID,
        "failed_check": None,
        "artifact_field": "learning_value_score",
        "expected_value": 1,
        "observed_value": 1,
        "passed": True,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind source, settings, rows, controls, and reduced claims."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "frozen_acceptance_manifest",
        "structural_learning_witnesses",
        "control_results",
        "per_stream_results",
        "confidence_intervals",
        "acceptance_gate_results",
        "learning_capture_complete_score",
        "learning_value_score",
        "verdict_class",
    )
    payload = {key: artifact.get(key) for key in fields}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
        ).hexdigest()
    )


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    return {
        key: REQUIRED_FIELD_PRINCIPLES.get(
            key,
            "Retain this measured field in its ordinary JSON type without changing gate meaning.",
        )
        for key in fields
    }


def artifact_from_evidence(
    *,
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    expected_row_count: int,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    acceptance_manifest: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> JsonDict:
    """Build a terminal candidate while keeping completion separate from value."""

    acceptance = deepcopy(dict(acceptance_manifest or test_acceptance_manifest()))
    reduction = reduce_rows(rows, witnesses, acceptance, bootstrap_draws=bootstrap_draws)
    receipts = [deepcopy(dict(row)) for row in validation_receipts]
    prerequisites_pass = all(row.get("passed") is True for row in preconditions)
    controls_pass = controls.get("passed") is True
    rows_complete = len(rows) == expected_row_count and all(
        row.get("censored") is False for row in rows
    )
    validation_pass = all(_receipt_passed(receipts, name) for name in ALL_REQUIRED_CHECK_NAMES)
    flagged = not _receipt_passed(receipts, "adversarial_verify")
    capture_complete = prerequisites_pass and controls_pass and rows_complete and validation_pass
    value_pass = capture_complete and reduction["learning_value_passed"] and not flagged
    if not prerequisites_pass:
        verdict = "blocked"
        status = "blocked_external_precondition"
        honest = "blocked_external_precondition_no_dependent_evaluation"
    elif not rows_complete:
        verdict = "partial"
        status = "partial_resumable_evaluation"
        honest = "partial_own_work_with_real_checkpoints"
    elif not controls_pass or not validation_pass or flagged:
        verdict = "disqualified"
        status = "complete_disqualified_required_check"
        honest = "complete_disqualified_required_safety_or_validation_failure"
    elif value_pass:
        verdict = "circular_positive"
        status = "complete_circular_positive_structural_learning"
        honest = "complete_circular_positive_exact_executor_structural_learning"
    else:
        verdict = "null"
        status = "complete_null_structural_learning_value_gate_miss"
        honest = "complete_null_no_joint_structural_learning_benefit"
    operational_gates = {
        "preconditions": _gate(
            True, prerequisites_pass, prerequisites_pass, "Exact producers gate work."
        ),
        "safety_controls": _gate(
            True, controls_pass, controls_pass, "All frozen controls must pass."
        ),
        "row_completion": _gate(
            expected_row_count,
            len(rows),
            rows_complete,
            "Every fixed-denominator arm and request must finish.",
        ),
        "affected_validation": _gate(
            list(ALL_REQUIRED_CHECK_NAMES),
            [name for name in ALL_REQUIRED_CHECK_NAMES if _receipt_passed(receipts, name)],
            validation_pass,
            "Every required affected and terminal command must pass.",
        ),
        "adversarial_clear": _gate(
            False, flagged, not flagged, "Critical findings prevent promotion."
        ),
    }
    all_gates = {**operational_gates, **deepcopy(reduction["gate_results"])}
    now = datetime.now(UTC).isoformat()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at or now,
        "completed_at_utc": now,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": "CPython exact Boolean execution, transactional structural memory, and paired stream bootstrap",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "development": 7_360_101,
            "evaluation": 7_360_211,
            "resampling": int(acceptance.get("resampling_seed", RESAMPLING_SEED)),
        },
        "source_artifact_hashes": dict(source_hashes),
        "rows": [deepcopy(dict(row)) for row in rows],
        "sample_size_budget": {
            "planned_synthetic_streams": 32,
            "planned_live_streams": 8,
            "planned_arms": list(ARMS),
            "planned_rows": expected_row_count,
            "attempted_rows": len(rows),
            "completed_rows": sum(row.get("censored") is False for row in rows),
            "censored_rows": sum(bool(row.get("censored")) for row in rows),
            "query_budget_per_request": QUERY_BUDGET,
            "state_cap_bytes": STATE_CAP_BYTES,
            "stopping_rule": "Run each frozen stream-arm-request unit once; never extend from outcomes.",
        },
        "frozen_acceptance_manifest": acceptance,
        "acceptance_gate_results": all_gates,
        "gate_check_summary": _gate_summary(preconditions, all_gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": flagged,
        "validation_receipts": receipts,
        "repository_health": {
            "status": "historical_failures_retained",
            "affects_required_checks": False,
            "date": RUN_DATE,
            "historical_failures": [
                {
                    "source": "exp7346-learning-adapter",
                    "classification": "historical_full_suite_timeout_and_value_null",
                    "affects_current_required_checks": False,
                }
            ],
        },
        "learning_capture_complete_score": int(capture_complete),
        "learning_value_score": int(value_pass),
        "promotion_score": int(value_pass),
        "structural_learning_witnesses": [deepcopy(dict(row)) for row in witnesses],
        "per_stream_results": deepcopy(reduction["per_stream_results"]),
        "confidence_intervals": deepcopy(reduction["confidence_intervals"]),
        "independent_reduction": reduction,
        "control_results": deepcopy(dict(controls)),
        "no_model_weight_mutation": True,
        "production_defaults_changed": False,
        "research_roadmap_changed": False,
        "historical_comparator": {
            "path": "results/experiment_7346_v645_learning_adapter.json",
            "verdict_class": "disqualified",
            "query_ratio_and_wall_cost_null_preserved_as_history": True,
            "counts_as_current_row": False,
        },
        "raw_evidence_paths": {
            "rows": str(RAW_DIR / "rows.json"),
            "historical_inference_sidecar": str(RAW_DIR / "historical_inference_sidecar.json"),
            "measured_candidate": str(RAW_DIR / "measured-terminal-candidate.json"),
        },
    }
    artifact["field_principles"] = _field_principles(
        sorted({*artifact, "field_principles", "reproducibility_checksum"})
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    started_at: str,
) -> JsonDict:
    """Create a row-free blocked record without starting dependent work."""

    return artifact_from_evidence(
        rows=[],
        witnesses=[],
        controls={"passed": False, "rows": [], "not_run": "external_precondition"},
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[],
        expected_row_count=0,
        duration_s=duration_s,
        phase_spans=[
            {
                "phase": "load",
                "start_elapsed_s": 0.0,
                "end_elapsed_s": duration_s,
                "duration_s": duration_s,
            },
            {
                "phase": "generation",
                "start_elapsed_s": duration_s,
                "end_elapsed_s": duration_s,
                "duration_s": 0.0,
            },
            {
                "phase": "evaluation",
                "start_elapsed_s": duration_s,
                "end_elapsed_s": duration_s,
                "duration_s": 0.0,
            },
            {
                "phase": "validation",
                "start_elapsed_s": duration_s,
                "end_elapsed_s": duration_s,
                "duration_s": 0.0,
            },
            {
                "phase": "write",
                "start_elapsed_s": duration_s,
                "end_elapsed_s": duration_s,
                "duration_s": 0.0,
            },
        ],
        started_at=started_at,
        bootstrap_draws=1,
    )


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute raw-row claims without trusting stored aggregates."""

    rows = artifact.get("rows")
    witnesses = artifact.get("structural_learning_witnesses")
    acceptance = artifact.get("frozen_acceptance_manifest")
    stored = artifact.get("independent_reduction")
    if (
        not isinstance(rows, list)
        or not isinstance(witnesses, list)
        or not isinstance(acceptance, Mapping)
    ):
        return ["raw reduction inputs are malformed"]
    draws = (
        int(stored.get("bootstrap_draws", BOOTSTRAP_DRAWS))
        if isinstance(stored, Mapping)
        else BOOTSTRAP_DRAWS
    )
    reduced = reduce_rows(rows, witnesses, acceptance, bootstrap_draws=draws)
    errors: list[str] = []
    if stored != reduced:
        errors.append("stored reduction differs from rows")
    if artifact.get("per_stream_results") != reduced["per_stream_results"]:
        errors.append("per-stream results differ from rows")
    if artifact.get("confidence_intervals") != reduced["confidence_intervals"]:
        errors.append("confidence intervals differ from rows")
    expected_value = int(
        artifact.get("learning_capture_complete_score") == 1
        and reduced["learning_value_passed"]
        and artifact.get("flagged_adversarial") is False
    )
    if artifact.get("learning_value_score") != expected_value:
        errors.append("learning value score differs from rows")
    return errors


def validate_artifact(value: object) -> list[str]:
    """Cold-check schema, exact limits, classification, and row-derived claims."""

    if not isinstance(value, Mapping):
        return ["artifact is not an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(value))
    if missing:
        return [f"missing fields: {missing}"]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current invocation counts are nonzero")
    if (
        value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("inference substrate mismatch")
    if value.get("execution_venue") != "host" or value.get("verifier_is_oracle") is not True:
        errors.append("execution or oracle disclosure mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid verdict class")
    rows = value.get("rows")
    if not isinstance(rows, list):
        errors.append("rows are not a list")
    else:
        if any(
            int(row.get("paid_queries", row.get("query_attempts", 0))) > QUERY_BUDGET
            for row in rows
        ):
            errors.append("query budget exceeded")
        if any(int(row.get("state_bytes", 0)) > STATE_CAP_BYTES for row in rows):
            errors.append("state cap exceeded")
    errors.extend(independent_reduce(value))
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(field not in principles for field in value):
        errors.append("field principles incomplete")
    verdict = value.get("verdict_class")
    if verdict in {"blocked", "disqualified", "partial"} and any(
        value.get(field) != 0
        for field in ("learning_capture_complete_score", "learning_value_score", "promotion_score")
    ):
        errors.append("unavailable or disqualified scores must be zero")
    if verdict == "blocked" and rows:
        errors.append("blocked artifact contains rows")
    if value.get("flagged_adversarial") is True and value.get("promotion_score") != 0:
        errors.append("adversarial finding did not zero promotion")
    if value.get("verdict_class") == "positive":
        errors.append("exact executor result cannot use positive class")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility checksum mismatch")
    return errors


def _affected_manifest() -> validation_contract.AffectedManifest:
    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def scoped_command_plan(
    repo_root: Path, temporary_root: Path
) -> list[validation_scope.CommandSpec]:
    """Build the explicit Exp7358-bounded affected command plan."""

    return validation_contract.build_command_plan(repo_root, _affected_manifest(), temporary_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any affected command that escapes the declared V646 files."""

    return validation_contract.validate_command_plan(repo_root, _affected_manifest(), commands)


def _run_scoped_commands(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec], log_dir: Path
) -> list[JsonDict]:  # pragma: no cover - exercised by the declared entrypoint.
    planned = [
        validation_contract.PlannedCommand(command, command.scope, True) for command in commands
    ]
    return validation_contract.run_categorized_commands(repo_root, planned, log_dir=log_dir)


def _full_suite_command(repo_root: Path) -> validation_scope.CommandSpec:
    return validation_scope.CommandSpec(
        "full_python_suite",
        (str(repo_root / ".venv/bin/pytest"), "tests/python", "-q"),
        "mandated_python_suite",
        timeout_s=3_600.0,
    )


def _terminal_commands(repo_root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    python = str(repo_root / ".venv/bin/python")
    resolved = str(candidate.resolve())
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                "-c",
                "import json,pathlib;from carnot.experiment_7362_v646_prospective_learning import independent_reduce;"
                f"a=json.loads(pathlib.Path(r'{resolved}').read_text());"
                "e=independent_reduce(a);print(e,flush=True);raise SystemExit(bool(e))",
            ),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", resolved),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", resolved),
            "measured_candidate",
        ),
    ]


def _span(phase: str, start: float, end: float, run_start: float) -> JsonDict:
    return {
        "phase": phase,
        "start_elapsed_s": start - run_start,
        "end_elapsed_s": end - run_start,
        "duration_s": end - start,
    }


def _write_historical_sidecar(path: Path, capture: Mapping[str, Any]) -> None:
    _atomic_json(
        path,
        {
            "schema": SCHEMA + ".historical_inference_sidecar.v1",
            "label": "historical_exp7361_llm_receipts_not_current_inference",
            "source_path": str(CAPTURE_PATH),
            "source_sha256": sha256_file(REPO_ROOT / CAPTURE_PATH),
            "historical_MODEL_SPECS": deepcopy(capture.get("MODEL_SPECS", [])),
            "historical_invocation_counts": deepcopy(capture.get("invocation_counts", {})),
            "current_invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        },
    )


def run_experiment(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    run_full_suite: bool = True,
) -> JsonDict:  # pragma: no cover - executed through the declared entrypoint.
    """Execute the fixed measurement, validation, cold checks, and atomic write."""

    run_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []
    progress("preconditions", "start", run_start)
    load_start = time.monotonic()
    preconditions, source_hashes, producers = collect_preconditions(REPO_ROOT)
    load_end = time.monotonic()
    spans.append(_span("load", load_start, load_end, run_start))
    spans.append(_span("generation", load_end, load_end, run_start))
    passed = all(row["passed"] for row in preconditions)
    progress("preconditions", "end", run_start, f"passed={passed}")
    if not passed:
        blocked = blocked_artifact(
            preconditions=preconditions,
            source_hashes=source_hashes,
            duration_s=time.monotonic() - run_start,
            started_at=started_at,
        )
        progress("write", "before_atomic_publish", run_start, str(output_path))
        _atomic_json(REPO_ROOT / output_path, blocked)
        progress("write", "after_atomic_publish", run_start, blocked["status"])
        return blocked

    fixture = producers["fixture"]
    capture = producers["capture"]
    fixture_manifest = fixture["fixture_manifest"]
    public_manifest = _load_object(Path(fixture_manifest["public_manifest_path"]))
    private_manifest = _load_object(Path(fixture_manifest["private_manifest_path"]))
    acceptance = _load_object(Path(fixture_manifest["acceptance_manifest_path"]))
    proposal_manifest = _load_object(Path(capture["candidate_manifest_path"]))
    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_historical_sidecar(raw_dir / "historical_inference_sidecar.json", capture)
    source_hashes[str(raw_dir / "historical_inference_sidecar.json")] = sha256_file(
        raw_dir / "historical_inference_sidecar.json"
    )

    progress("evaluation", "start", run_start)
    evaluation_start = time.monotonic()
    state_root = Path(tempfile.mkdtemp(prefix="state-", dir=raw_dir))
    rows, witnesses = run_measurement(
        public_manifest, private_manifest, proposal_manifest, capture, state_root / "measurement"
    )
    controls = run_frozen_controls(
        state_root / "controls", public_manifest=public_manifest, private_manifest=private_manifest
    )
    evaluation_end = time.monotonic()
    spans.append(_span("evaluation", evaluation_start, evaluation_end, run_start))
    _atomic_json(
        raw_dir / "rows.json",
        {"schema": SCHEMA + ".rows.v1", "rows": rows, "witnesses": witnesses, "controls": controls},
    )
    source_hashes[str(raw_dir / "rows.json")] = sha256_file(raw_dir / "rows.json")
    progress(
        "evaluation",
        "end",
        run_start,
        f"rows={len(rows)} witnesses={len(witnesses)}",
    )

    validation_start = time.monotonic()
    validation_root = Path(tempfile.mkdtemp(prefix="exp7362-validation-", dir="/tmp"))
    commands = scoped_command_plan(REPO_ROOT, validation_root)
    plan_errors = validate_scoped_command_plan(REPO_ROOT, commands)
    if plan_errors:
        raise LearningAdapterError("validation_plan:" + ",".join(plan_errors))
    progress("validation", "before_scoped_subprocesses", run_start)
    scoped_receipts = _run_scoped_commands(REPO_ROOT, commands, raw_dir / "validation" / "scoped")
    full_receipts: list[JsonDict] = []
    if run_full_suite:
        progress("validation", "before_full_suite", run_start)
        full_receipts = validation_scope.run_commands(
            REPO_ROOT,
            [_full_suite_command(REPO_ROOT)],
            log_dir=raw_dir / "validation" / "full",
        )
        progress("validation", "after_full_suite", run_start)
    expected_rows = len(rows)
    preliminary = artifact_from_evidence(
        rows=rows,
        witnesses=witnesses,
        controls=controls,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*scoped_receipts, *full_receipts],
        expected_row_count=expected_rows,
        duration_s=time.monotonic() - run_start,
        phase_spans=[*spans, _span("validation", validation_start, time.monotonic(), run_start)],
        acceptance_manifest=acceptance,
        started_at=started_at,
    )
    candidate = raw_dir / "measured-terminal-candidate.json"
    _atomic_json(candidate, preliminary)
    progress("validation", "before_terminal_subprocesses", run_start)
    terminal_receipts = validation_scope.run_commands(
        REPO_ROOT,
        _terminal_commands(REPO_ROOT, candidate),
        log_dir=raw_dir / "validation" / "terminal",
    )
    validation_end = time.monotonic()
    spans.append(_span("validation", validation_start, validation_end, run_start))
    progress(
        "validation",
        "after_terminal_subprocesses",
        run_start,
        f"receipts={len(scoped_receipts) + len(full_receipts) + len(terminal_receipts)}",
    )
    write_start = time.monotonic()
    spans.append(_span("write", write_start, write_start, run_start))
    final = artifact_from_evidence(
        rows=rows,
        witnesses=witnesses,
        controls=controls,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*scoped_receipts, *full_receipts, *terminal_receipts],
        expected_row_count=expected_rows,
        duration_s=time.monotonic() - run_start,
        phase_spans=spans,
        acceptance_manifest=acceptance,
        started_at=started_at,
    )
    errors = validate_artifact(final)
    if errors:
        final["status"] = "complete_disqualified_cold_validation"
        final["honest_verdict"] = "complete_disqualified_cold_artifact_validation"
        final["verdict_class"] = "disqualified"
        final["flagged_adversarial"] = True
        final["learning_capture_complete_score"] = 0
        final["learning_value_score"] = 0
        final["promotion_score"] = 0
        final["gate_check_summary"] = {
            "upstream": EXPERIMENT_ID,
            "failed_check": "cold_artifact_validation",
            "artifact_field": "validate_artifact",
            "expected_value": [],
            "observed_value": errors,
            "passed": False,
        }
        final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress("write", "before_atomic_publish", run_start, str(output_path))
    final["completed_at_utc"] = datetime.now(UTC).isoformat()
    final["duration_s"] = time.monotonic() - run_start
    final["phase_spans"][-1] = _span("write", write_start, time.monotonic(), run_start)
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    _atomic_json(REPO_ROOT / output_path, final)
    progress("write", "after_atomic_publish", run_start, final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the exact run date and optional output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI orchestration.
    args = parse_args(argv)
    if args.validate:
        artifact = _load_object(REPO_ROOT / args.output)
        errors = validate_artifact(artifact)
        print(errors, flush=True)
        return int(bool(errors))
    artifact = run_experiment(output_path=args.output)
    return int(bool(validate_artifact(artifact)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
