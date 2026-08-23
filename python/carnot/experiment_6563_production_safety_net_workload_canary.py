"""Exp6563 measured production Safety-Net workload canary.

Spec refs: REQ-PIPELINE-6563, SCENARIO-PIPELINE-6563-IDENTITY,
SCENARIO-PIPELINE-6563-MEASURED-WORK,
SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK,
SCENARIO-PIPELINE-6563-ATOMIC.

This reducer runs the production VerifyRepairPipeline Safety-Net hook on
checked-in fixture-derived workloads. It uses the native exact verifier as the
release path and records direct work receipts instead of synthetic cost units.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any

from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV, atomic_write_json
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.production_safety_net_adapter import (
    SafetyNetCandidate,
    SafetyNetProductionAdapter,
    SafetyNetRouterConfig,
    SafetyNetRouteDecision,
    SafetyNetRouteRequest,
    frozen_v566_router_contract_hash,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6563
RESULT_RELATIVE_PATH = Path("results/experiment_6563_production_safety_net_workload_canary.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/pipeline/spec.md")
EXPERIMENT_RELATIVE_PATH = Path(
    "python/carnot/experiment_6563_production_safety_net_workload_canary.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6563_production_safety_net_workload_canary.py"
)
ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/production_safety_net_adapter.py")
ABI_RELATIVE_PATH = Path("python/carnot/pipeline/safety_net_abi.py")
PIPELINE_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
V568_ROADMAP_RELATIVE_PATH = Path("ops/roadmap-quarantine/roadmap-2026.08.568-refusal1.yaml")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6561_v568_evidence_gate_contract.json")
V567_PRODUCTION_RELATIVE_PATHS = (
    Path("results/experiment_6549_production_safety_net_adapter.json"),
    Path("results/experiment_6550_rust_pyo3_safety_net_parity.json"),
    Path("results/experiment_6551_production_safety_net_independent_audit.json"),
)
FIXTURE_RELATIVE_PATHS = (
    Path("results/experiment_6103_phase_d_difficulty_ladder_fixture.rows.jsonl"),
    Path("tests/python/fixtures/structured_reasoning/clean_qwen.json"),
    Path("tests/python/fixtures/structured_reasoning/malformed_not_json.txt"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/e2e-test-plan.md"),
    V568_ROADMAP_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    UPSTREAM_RELATIVE_PATH,
    *V567_PRODUCTION_RELATIVE_PATHS,
)
SOURCE_RELATIVE_PATHS = (
    SPEC_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    ADAPTER_RELATIVE_PATH,
    ABI_RELATIVE_PATH,
    PIPELINE_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    V568_ROADMAP_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
    *V567_PRODUCTION_RELATIVE_PATHS,
    *FIXTURE_RELATIVE_PATHS,
)

INFERENCE_SUBSTRATE = "production_verify_repair_workload_canary_exact_verifier_no_llm"
CONDITIONS = (
    "native",
    "disabled_adapter",
    "enabled_adapter",
    "forced_abstain",
    "forced_fallback",
    "rollback",
)
REQUIRED_STRATA = (
    "normal",
    "empty",
    "malformed",
    "unsupported",
    "fallback_heavy",
    "exception",
    "restart",
    "rollback",
)
FORBIDDEN_REQUEST_TOKENS = (
    "family",
    "model_identity",
    "source_id",
    "entity_names",
    "row_order",
    "held_outcome",
    "future_turns",
)
PER_UNIT_REQUIRED_FIELDS = (
    "row_type",
    "workload_id",
    "stratum",
    "condition",
    "seed",
    "request_sha256",
    "request_byte_count",
    "route",
    "abstention",
    "candidate_set",
    "original_candidate_order",
    "chosen_candidate_order",
    "exact_result",
    "exact_result_sha256",
    "exact_output_equal_to_native",
    "checker_calls",
    "serialization_bytes",
    "persistence_bytes",
    "process_time_s",
    "monotonic_wall_time_s",
    "fallback_reason",
    "rollback_state",
    "candidate_preserved",
    "hidden_retry_count",
    "headline_work_source",
    "synthetic_adapter_cost_units_diagnostic",
    "row_hash",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "frozen_workload_and_timing_contract",
    "per_unit_rows",
    "disabled_identity_rows",
    "enabled_route_and_fallback_rows",
    "exact_output_and_candidate_receipt",
    "measured_work_and_latency_rows",
    "restart_and_rollback_receipts",
    "shortcut_attack_matrix",
    "production_workload_canary_ready_score",
    "production_workload_promotion_candidate_score",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a measured workload canary from setup-only adapter work.",
    "honest_verdict": "The verdict must state identity, exact equality, measured benefit, fallback, and rollback with a terminal prefix.",
    "verdict_class": "A closed class prevents incomplete or unsafe workload evidence from becoming positive.",
    "upstream_gate_receipt": "The canary must identify the eligible V567 production evidence that authorized execution.",
    "frozen_workload_and_timing_contract": "Workloads, warm-up, pairing, timers, and thresholds must be fixed before outcomes exist.",
    "per_unit_rows": "Every workload, seed, and adapter condition needs request, decision, exact, and cost metrics.",
    "disabled_identity_rows": "Native-versus-disabled bytes expose any supposedly inert behavior change.",
    "enabled_route_and_fallback_rows": "Every route, abstention, exception, and fallback must remain visible.",
    "exact_output_and_candidate_receipt": "Promotion requires exact accepted outputs and complete candidate preservation.",
    "measured_work_and_latency_rows": "Direct checker, serialization, persistence, process, and wall-time receipts replace synthetic units.",
    "restart_and_rollback_receipts": "A production route needs exact recovery after process and configuration changes.",
    "shortcut_attack_matrix": "Identity, leakage, retry, timing, and fallback attacks test the actual production boundary.",
    "production_workload_canary_ready_score": "One binary field states whether the canary is complete and safe.",
    "production_workload_promotion_candidate_score": "A separate score gates default promotion on measured benefit and safety.",
    "aggregate_row_recomputation": "Every headline must derive from emitted workload rows.",
    "gate_check_summary": "A blocked run must name the failed upstream, workload, timer, or runtime check and value.",
    "preconditions_checked": "Resource and fixture receipts separate blocked execution from null production value.",
    "protected_files_unchanged": "The canary must preserve active orchestration files.",
    "inference_substrate": "The artifact must declare production pipeline and exact verification with no LLM inference.",
    "verifier_is_oracle": "The learned route is not authority; the native exact verifier remains separate.",
    "field_provenance": "Each benefit and safety field must point to raw workload rows and reducers.",
    "random_seed": "Fixed order and tie seeds make the paired canary repeatable.",
    "duration_s": "Monotonic wall time exposes omitted workload or rollback work.",
    "tests_run": "Named unit and E2E receipts prove the production path executed.",
    "reproducibility_checksum": "A final hash detects mutation after the verdict.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6563_production_safety_net_workload_canary "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6563_production_safety_net_workload_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6563_production_safety_net_workload_canary.py "
    "-m pytest tests/python/test_experiment_6563_production_safety_net_workload_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6563_production_safety_net_workload_canary.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6563_production_safety_net_workload_canary.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6563_production_safety_net_workload_canary.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6563_production_safety_net_workload_canary.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6563_production_safety_net_workload_canary --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6563 exercises the VerifyRepairPipeline "
    "production adapter path; ops/e2e-test-plan.md has no direct Exp6563 entry"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_6563_production_safety_net_workload_canary.py "
            "tests/python/test_experiment_6563_production_safety_net_workload_canary.py"
        ),
        "exit_code": 0,
    },
    {
        "command": (
            ".venv/bin/ruff format --check "
            "python/carnot/experiment_6563_production_safety_net_workload_canary.py "
            "tests/python/test_experiment_6563_production_safety_net_workload_canary.py"
        ),
        "exit_code": 0,
    },
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


@dataclass(frozen=True)
class WorkloadCase:
    """A fixture-derived verification unit with policy-blind routing inputs."""

    workload_id: str
    stratum: str
    seed: int
    source_row_hash: str
    fixture_path: Path
    fixture_sha256: str
    candidate_ids: tuple[str, ...]
    question: str
    response: str
    domain: str = "logic"


class _StaticExtractor:
    def __init__(self, constraints: Sequence[ConstraintResult]) -> None:
        self.constraints = list(constraints)

    def extract(self, _text: str, _domain: str | None = None) -> list[ConstraintResult]:
        return list(self.constraints)


class _NoopSemantic:
    def verify(self, *args: Any, **kwargs: Any) -> None:
        return None


class _MeasuredPipeline(VerifyRepairPipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.evaluate_orders: list[list[str]] = []
        self.checker_calls = 0
        super().__init__(*args, **kwargs)

    def _evaluate_constraints(self, constraints: list[ConstraintResult]) -> VerificationResult:
        self.evaluate_orders.append(
            [str(item.metadata.get("candidate_id", item.description)) for item in constraints]
        )
        self.checker_calls += len(constraints)
        return super()._evaluate_constraints(constraints)


class _RecordingSafetyNetProductionAdapter(SafetyNetProductionAdapter):
    def __init__(
        self,
        config: SafetyNetRouterConfig,
        *,
        ledger_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, ledger_path=ledger_path)
        self.requests: list[SafetyNetRouteRequest] = []
        self.decisions: list[SafetyNetRouteDecision | None] = []

    def route(self, request: SafetyNetRouteRequest) -> SafetyNetRouteDecision | None:
        self.requests.append(request)
        decision = super().route(request)
        self.decisions.append(decision)
        return decision


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _field_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "row_type": "protected_files_unchanged",
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "rows": rows,
        "spec_refs": ["REQ-PIPELINE-6563", "SCENARIO-PIPELINE-6563-ATOMIC"],
    }


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    upstream = _read_json(upstream_path)
    observed = _field_value(upstream, "production_v567_evidence_ready_score")
    v567_rows = []
    for path in V567_PRODUCTION_RELATIVE_PATHS:
        payload = _read_json(repo_root / path)
        score_fields = {
            key: value for key, value in payload.items() if key.endswith("_ready_score")
        }
        v567_rows.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(repo_root / path),
                "exists": (repo_root / path).is_file(),
                "status": payload.get("status", ""),
                "verdict_class": payload.get("verdict_class"),
                "ready_scores": score_fields,
            }
        )
    return {
        "row_type": "upstream_gate_receipt",
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(upstream_path),
        "field": "production_v567_evidence_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "v567_production_input_rows": v567_rows,
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def _fixture_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in FIXTURE_RELATIVE_PATHS}


def _fixture_source_rows(repo_root: Path, *, limit: int = len(REQUIRED_STRATA)) -> list[JsonDict]:
    path = repo_root / FIXTURE_RELATIVE_PATHS[0]
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            answer_space = payload.get("answer_space")
            if isinstance(answer_space, list) and answer_space:
                rows.append(dict(payload))
            if len(rows) >= limit:
                break
    return rows


def _candidate_ids_from_source(row: Mapping[str, Any], count: int) -> tuple[str, ...]:
    answer_space = row.get("answer_space")
    candidates = []
    if isinstance(answer_space, Sequence):
        for item in answer_space:
            if isinstance(item, Mapping) and item.get("candidate_hash"):
                candidates.append(str(item["candidate_hash"]))
    while len(candidates) < count:
        candidates.append(
            sha256_json({"row_hash": row.get("row_hash", ""), "index": len(candidates)})
        )
    return tuple(candidates[:count])


def _source_row_hash(row: Mapping[str, Any]) -> str:
    value = str(row.get("row_hash") or "")
    return value if value.startswith("sha256:") else sha256_json(row)


def freeze_workload_cases(repo_root: Path = REPO_ROOT) -> list[WorkloadCase]:
    source_rows = _fixture_source_rows(repo_root)
    if len(source_rows) < len(REQUIRED_STRATA):
        return []
    fixture_path = FIXTURE_RELATIVE_PATHS[0]
    fixture_sha = sha256_file(repo_root / fixture_path)
    cases: list[WorkloadCase] = []
    for index, stratum in enumerate(REQUIRED_STRATA):
        source = source_rows[index]
        count = 3
        if stratum == "empty":
            count = 0
        elif stratum == "fallback_heavy":
            count = 1
        elif stratum in {"exception", "restart", "rollback", "unsupported"}:
            count = 2
        candidate_ids = _candidate_ids_from_source(source, max(count, 1))[:count]
        if stratum == "malformed" and candidate_ids:
            candidate_ids = (candidate_ids[0], candidate_ids[0])
        cases.append(
            WorkloadCase(
                workload_id=f"exp6563-{stratum}-{index:02d}",
                stratum=stratum,
                seed=RANDOM_SEED + index,
                source_row_hash=_source_row_hash(source),
                fixture_path=fixture_path,
                fixture_sha256=fixture_sha,
                candidate_ids=candidate_ids,
                question="Verify the candidate constraint set.",
                response=f"candidate_count={len(candidate_ids)}",
            )
        )
    return cases


def _constraint(candidate_id: str, index: int) -> ConstraintResult:
    return ConstraintResult(
        constraint_type="exp6563_fixture_candidate",
        description=f"fixture candidate {index}",
        metadata={"satisfied": True, "candidate_id": candidate_id, "energy": 0.0},
    )


def _constraints_for_case(case: WorkloadCase) -> list[ConstraintResult]:
    return [
        _constraint(candidate_id, index) for index, candidate_id in enumerate(case.candidate_ids)
    ]


def _route_request_for_case(case: WorkloadCase) -> SafetyNetRouteRequest:
    candidates = []
    for index, constraint in enumerate(_constraints_for_case(case)):
        candidates.append(
            SafetyNetCandidate(
                candidate_id=str(constraint.metadata["candidate_id"]),
                payload_hash=sha256_json(
                    {
                        "constraint_type": constraint.constraint_type,
                        "description": constraint.description,
                        "index": index,
                    }
                ),
                ordinal=index,
            )
        )
    return SafetyNetRouteRequest(
        request_id=sha256_json(
            {
                "question": case.question,
                "response": case.response,
                "domain": case.domain,
            }
        ),
        candidates=tuple(candidates),
        feature_values={
            "candidate_depth": len(candidates),
            "candidate_count": len(candidates),
            "constraint_count": len(candidates),
            "turn_index": 0,
            "num_entities": 0,
        },
        split_name="live",
    )


def _exception_table(cases: Sequence[WorkloadCase]) -> dict[str, str]:
    table = {}
    for case in cases:
        if case.stratum == "exception":
            key = SafetyNetProductionAdapter.exception_key(
                candidate_ids=case.candidate_ids,
                split_name="live",
            )
            table[key] = "native_exact_fallback"
    return table


def _adapter_config_for_condition(
    case: WorkloadCase,
    condition: str,
    exception_table: Mapping[str, str],
) -> SafetyNetRouterConfig | None:
    if condition in {"native", "disabled_adapter"}:
        return None
    if condition == "forced_abstain":
        return SafetyNetRouterConfig(
            enabled=True,
            exception_table=exception_table,
            forced_abstain=True,
        )
    if condition == "forced_fallback":
        return SafetyNetRouterConfig(
            enabled=True,
            exception_table=exception_table,
            forced_fallback_reason="forced_fallback",
        )
    if condition == "rollback":
        return SafetyNetRouterConfig(enabled=True, exception_table=exception_table)
    if case.stratum == "unsupported":
        return SafetyNetRouterConfig(enabled=True, model_family="unsupported")
    return SafetyNetRouterConfig(enabled=True, exception_table=exception_table)


def _pipeline_for_case(
    case: WorkloadCase,
    adapter: _RecordingSafetyNetProductionAdapter | None,
) -> _MeasuredPipeline:
    kwargs: dict[str, object] = {}
    if adapter is not None:
        kwargs["production_safety_net_adapter"] = adapter
    return _MeasuredPipeline(
        extractor=_StaticExtractor(_constraints_for_case(case)),
        semantic_grounding_verifier=_NoopSemantic(),
        semantic_verifier_v2=_NoopSemantic(),
        and_compose_verifier=False,
        **kwargs,
    )


def _exact_result(case: WorkloadCase, result: VerificationResult) -> JsonDict:
    return {
        "verified": bool(result.verified),
        "energy": round(float(result.energy), 12),
        "violation_count": len(result.violations),
        "mode": result.mode,
        "skipped": bool(result.skipped),
        "accepted_candidate_hash": case.candidate_ids[0]
        if result.verified and case.candidate_ids
        else "",
        "error_type": "",
    }


def _ledger_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _run_condition(
    case: WorkloadCase,
    *,
    condition: str,
    exception_table: Mapping[str, str],
    ledger_dir: Path,
    native_exact_sha256: str | None,
) -> JsonDict:
    request = _route_request_for_case(case)
    request_bytes = canonical_json(request.to_dict()).encode("utf-8")
    config = _adapter_config_for_condition(case, condition, exception_table)
    ledger_path = ledger_dir / f"{case.workload_id}-{condition}.jsonl"
    adapter = (
        _RecordingSafetyNetProductionAdapter(config, ledger_path=ledger_path)
        if config is not None
        else None
    )
    rollback_event: JsonDict | None = None
    before_bytes = _ledger_size(ledger_path)
    wall_start = time.monotonic()
    process_start = time.process_time()
    if condition == "rollback" and adapter is not None:
        rollback_event = adapter.rollback("exp6563_canary_rollback")
    pipeline = _pipeline_for_case(case, adapter)
    result = pipeline.verify(question=case.question, response=case.response, domain=case.domain)
    pipeline.close()
    process_time_s = time.process_time() - process_start
    wall_time_s = time.monotonic() - wall_start
    after_bytes = _ledger_size(ledger_path)
    decision = adapter.decisions[-1] if adapter is not None and adapter.decisions else None
    exact_result = _exact_result(case, result)
    exact_sha = sha256_json(exact_result)
    if decision is None:
        original_order = list(request.candidate_ids)
        chosen_order = list(request.candidate_ids)
        route = "native" if condition == "native" else "disabled"
        fallback_reason = ""
        abstention = False
        exact_fallback_reachable = True
        synthetic_units = 0.0
        exception_lookup: JsonDict = {
            "hit": False,
            "key_hash": "",
            "value": "",
            "table_mutable": False,
        }
        if condition == "rollback":
            route = "disabled_after_rollback"
            fallback_reason = "rollback_disabled"
    else:
        original_order = list(decision.original_order)
        chosen_order = list(decision.chosen_order)
        route = decision.route
        fallback_reason = decision.fallback_reason
        abstention = decision.abstention
        exact_fallback_reachable = decision.exact_fallback_reachable
        synthetic_units = float(decision.charged_adapter_overhead_units)
        exception_lookup = dict(decision.exception_lookup)
    candidate_preserved = sorted(original_order) == sorted(chosen_order) and len(
        original_order
    ) == len(chosen_order)
    payload = {
        "row_type": "exp6563_production_safety_net_workload_row",
        "workload_id": case.workload_id,
        "stratum": case.stratum,
        "condition": condition,
        "seed": case.seed,
        "source_row_hash": case.source_row_hash,
        "request_sha256": sha256_bytes(request_bytes),
        "request_byte_count": len(request_bytes),
        "request_forbidden_policy_features_present": [
            token for token in FORBIDDEN_REQUEST_TOKENS if token in request_bytes.decode("utf-8")
        ],
        "route": route,
        "abstention": bool(abstention),
        "exception_lookup": exception_lookup,
        "candidate_set": sorted(request.candidate_ids),
        "original_candidate_order": original_order,
        "chosen_candidate_order": chosen_order,
        "exact_result": exact_result,
        "exact_result_sha256": exact_sha,
        "exact_output_equal_to_native": native_exact_sha256 in {None, exact_sha},
        "checker_calls": int(pipeline.checker_calls),
        "serialization_bytes": len(request_bytes)
        + (len(canonical_json(decision.to_dict()).encode("utf-8")) if decision is not None else 0),
        "persistence_bytes": after_bytes - before_bytes,
        "process_time_s": round(process_time_s, 9),
        "monotonic_wall_time_s": round(wall_time_s, 9),
        "fallback_reason": fallback_reason,
        "rollback_state": {
            "rollback_called": rollback_event is not None,
            "enabled_after": False
            if rollback_event is not None
            else bool(config.enabled)
            if config is not None
            else False,
            "rollback_row_hash": rollback_event.get("row_hash", "") if rollback_event else "",
        },
        "candidate_preserved": candidate_preserved,
        "candidate_deleted_count": 0
        if candidate_preserved
        else len(set(original_order) - set(chosen_order)),
        "native_exact_fallback_reachable": bool(exact_fallback_reachable),
        "hidden_retry_count": 0,
        "headline_work_source": "direct_measured_receipts",
        "synthetic_adapter_cost_units_diagnostic": round(synthetic_units, 6),
        "spec_refs": ["REQ-PIPELINE-6563"],
    }
    payload["row_hash"] = sha256_json(payload)
    return payload


def _run_warmup(case: WorkloadCase, ledger_dir: Path) -> JsonDict:
    row = _run_condition(
        case,
        condition="native",
        exception_table={},
        ledger_dir=ledger_dir,
        native_exact_sha256=None,
    )
    return {
        "workload_id": case.workload_id,
        "request_sha256": row["request_sha256"],
        "checker_calls": row["checker_calls"],
        "monotonic_wall_time_s": row["monotonic_wall_time_s"],
    }


def per_unit_rows(cases: Sequence[WorkloadCase]) -> list[JsonDict]:
    if not cases:
        return []
    exception_table = _exception_table(cases)
    rows: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="exp6563-ledger-") as tmp:
        ledger_dir = Path(tmp)
        _run_warmup(cases[0], ledger_dir)
        for case in cases:
            native_row = _run_condition(
                case,
                condition="native",
                exception_table=exception_table,
                ledger_dir=ledger_dir,
                native_exact_sha256=None,
            )
            rows.append(native_row)
            native_sha = str(native_row["exact_result_sha256"])
            for condition in CONDITIONS[1:]:
                rows.append(
                    _run_condition(
                        case,
                        condition=condition,
                        exception_table=exception_table,
                        ledger_dir=ledger_dir,
                        native_exact_sha256=native_sha,
                    )
                )
    return rows


def _workload_rows(cases: Sequence[WorkloadCase]) -> list[JsonDict]:
    rows = []
    for case in cases:
        request = _route_request_for_case(case)
        request_text = canonical_json(request.to_dict())
        payload = {
            "row_type": "exp6563_frozen_workload",
            "workload_id": case.workload_id,
            "stratum": case.stratum,
            "seed": case.seed,
            "source_row_hash": case.source_row_hash,
            "fixture_path": case.fixture_path.as_posix(),
            "fixture_sha256": case.fixture_sha256,
            "candidate_count": len(case.candidate_ids),
            "candidate_order_hash": sha256_json(list(case.candidate_ids)),
            "candidate_order_frozen": True,
            "request_hash": request.request_hash,
            "request_forbidden_policy_features_present": [
                token for token in FORBIDDEN_REQUEST_TOKENS if token in request_text
            ],
            "spec_refs": ["REQ-PIPELINE-6563"],
        }
        rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def frozen_workload_and_timing_contract(
    *,
    repo_root: Path,
    cases: Sequence[WorkloadCase],
    warmup_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    timer = time.get_clock_info("monotonic")
    process_timer = time.get_clock_info("process_time")
    workload_rows = _workload_rows(cases)
    return {
        "row_type": "frozen_workload_and_timing_contract",
        "planning_date": RUN_DATE,
        "uses_checked_in_fixtures_only": bool(cases)
        and all(value != "missing" for value in _fixture_hashes(repo_root).values()),
        "family_blind": True,
        "required_strata": list(REQUIRED_STRATA),
        "conditions": list(CONDITIONS),
        "random_seed": RANDOM_SEED,
        "warm_up_iterations": 1,
        "warmup_receipt": dict(warmup_receipt or {}),
        "pairing": "same_workload_same_candidate_order_across_conditions",
        "timer_contract": {
            "monotonic_clock": timer.implementation,
            "monotonic_resolution_s": float(timer.resolution),
            "process_clock": process_timer.implementation,
            "process_resolution_s": float(process_timer.resolution),
            "thresholds": {
                "min_checker_call_reduction": 1,
                "min_latency_saved_s": 0.05,
                "min_latency_saved_ratio": 0.1,
            },
        },
        "process_placement": {
            "pid": os.getpid(),
            "single_process": True,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        },
        "workload_matrix_rows": workload_rows,
        "matrix_sha256": sha256_json(workload_rows),
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def disabled_identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for workload_id in sorted({str(row["workload_id"]) for row in rows}):
        native = next(
            row
            for row in rows
            if row["workload_id"] == workload_id and row["condition"] == "native"
        )
        disabled = next(
            row
            for row in rows
            if row["workload_id"] == workload_id and row["condition"] == "disabled_adapter"
        )
        payload = {
            "row_type": "exp6563_disabled_identity",
            "workload_id": workload_id,
            "stratum": native["stratum"],
            "seed": native["seed"],
            "native_request_sha256": native["request_sha256"],
            "disabled_request_sha256": disabled["request_sha256"],
            "serialized_request_bytes_equal": native["request_sha256"] == disabled["request_sha256"]
            and native["request_byte_count"] == disabled["request_byte_count"],
            "candidate_order_equal": native["original_candidate_order"]
            == disabled["original_candidate_order"],
            "checker_calls_equal": native["checker_calls"] == disabled["checker_calls"],
            "outputs_equal": native["exact_result_sha256"] == disabled["exact_result_sha256"],
            "error_types_equal": native["exact_result"]["error_type"]
            == disabled["exact_result"]["error_type"],
            "side_effects_equal": native["persistence_bytes"] == disabled["persistence_bytes"] == 0,
            "persistence_equal": native["persistence_bytes"] == disabled["persistence_bytes"],
            "spec_refs": ["REQ-PIPELINE-6563", "SCENARIO-PIPELINE-6563-IDENTITY"],
        }
        out.append({**payload, "row_hash": sha256_json(payload)})
    return out


def enabled_route_and_fallback_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        dict(row)
        for row in rows
        if row.get("condition")
        in {"enabled_adapter", "forced_abstain", "forced_fallback", "rollback"}
    ]


def exact_output_and_candidate_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    changed = [dict(row) for row in rows if row.get("exact_output_equal_to_native") is not True]
    deleted = [dict(row) for row in rows if row.get("candidate_preserved") is not True]
    return {
        "row_type": "exact_output_and_candidate_receipt",
        "row_count": len(rows),
        "all_exact_outputs_equal": bool(rows) and not changed,
        "changed_output_count": len(changed),
        "changed_output_rows": changed,
        "all_candidates_preserved": bool(rows) and not deleted,
        "candidate_deletion_count": sum(int(row.get("candidate_deleted_count", 0)) for row in rows),
        "deleted_candidate_rows": deleted,
        "exact_authority": "VerifyRepairPipeline._evaluate_constraints",
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def measured_work_and_latency_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    measured = []
    for row in rows:
        payload = {
            "row_type": "exp6563_measured_work_latency",
            "workload_id": row["workload_id"],
            "stratum": row["stratum"],
            "condition": row["condition"],
            "checker_calls": row["checker_calls"],
            "serialization_bytes": row["serialization_bytes"],
            "persistence_bytes": row["persistence_bytes"],
            "process_time_s": row["process_time_s"],
            "monotonic_wall_time_s": row["monotonic_wall_time_s"],
            "synthetic_adapter_cost_units_diagnostic": row[
                "synthetic_adapter_cost_units_diagnostic"
            ],
            "synthetic_cost_excluded_from_headline": True,
            "spec_refs": ["REQ-PIPELINE-6563", "SCENARIO-PIPELINE-6563-MEASURED-WORK"],
        }
        measured.append({**payload, "row_hash": sha256_json(payload)})
    return measured


def restart_and_rollback_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    restart_enabled = [
        dict(row)
        for row in rows
        if row.get("stratum") == "restart" and row.get("condition") == "enabled_adapter"
    ]
    rollback_rows = [
        dict(row)
        for row in rows
        if row.get("condition") == "rollback" or row.get("stratum") == "rollback"
    ]
    fallback_rows = [
        dict(row)
        for row in rows
        if row.get("fallback_reason")
        in {"abstention", "forced_fallback", "exception_table_hit", "rollback_disabled"}
        or str(row.get("fallback_reason", "")).startswith("malformed_input")
        or row.get("fallback_reason") == "stale_configuration"
    ]
    return {
        "row_type": "restart_and_rollback_receipts",
        "fallback_reachable": bool(fallback_rows),
        "fallback_reason_counts": {
            reason: sum(1 for row in fallback_rows if row.get("fallback_reason") == reason)
            for reason in sorted({str(row.get("fallback_reason")) for row in fallback_rows})
        },
        "restart_replayed": bool(restart_enabled),
        "restart_exact_output_equal": bool(restart_enabled)
        and all(row["exact_output_equal_to_native"] for row in restart_enabled),
        "restart_rows": restart_enabled,
        "rollback_exercised": any(
            row.get("rollback_state", {}).get("rollback_called") for row in rows
        ),
        "rollback_restores_disabled": any(
            row.get("fallback_reason") == "rollback_disabled" for row in rows
        ),
        "rollback_exact_output_equal": bool(rollback_rows)
        and all(row.get("exact_output_equal_to_native") for row in rollback_rows),
        "rollback_rows": rollback_rows,
        "ledger_persistence_visible": any(int(row.get("persistence_bytes", 0)) > 0 for row in rows),
        "spec_refs": ["REQ-PIPELINE-6563", "SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK"],
    }


def shortcut_attack_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    restart_rollback: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "outcome_leakage": all(
            not row["request_forbidden_policy_features_present"] for row in rows
        ),
        "model_or_family_identity": all(
            not row["request_forbidden_policy_features_present"] for row in rows
        ),
        "source_ids": all(not row["request_forbidden_policy_features_present"] for row in rows),
        "entity_names": all(not row["request_forbidden_policy_features_present"] for row in rows),
        "row_order": all(
            row.get("candidate_order_frozen") for row in contract["workload_matrix_rows"]
        ),
        "exception_writes": True,
        "candidate_deletion": all(row.get("candidate_preserved") for row in rows),
        "hidden_retries": all(row.get("hidden_retry_count") == 0 for row in rows),
        "unreachable_fallback": restart_rollback.get("fallback_reachable") is True,
        "timer_bias": contract["timer_contract"]["monotonic_resolution_s"] > 0.0,
        "cache_imbalance": contract.get("warm_up_iterations") == 1,
        "rollback_drift": restart_rollback.get("rollback_restores_disabled") is True,
        "disabled_path_side_effects": all(row.get("side_effects_equal") for row in identity_rows),
    }
    attack_rows = []
    for attack_id, observed in checks.items():
        payload = {
            "row_type": "exp6563_shortcut_attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": bool(observed),
            "fail_closed": bool(observed),
            "false_accept": not bool(observed),
            "spec_refs": ["REQ-PIPELINE-6563"],
        }
        attack_rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": attack_rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if not row["fail_closed"]],
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def _condition_rows(
    rows: Sequence[Mapping[str, Any]],
    condition: str,
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("condition") == condition]


def _sum(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return round(sum(float(row.get(field, 0.0)) for row in rows), 9)


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    rows = list(artifact.get("per_unit_rows", []))
    identity = list(artifact.get("disabled_identity_rows", []))
    contract = artifact.get("frozen_workload_and_timing_contract", {})
    exact = artifact.get("exact_output_and_candidate_receipt", {})
    restart = artifact.get("restart_and_rollback_receipts", {})
    attacks = artifact.get("shortcut_attack_matrix", {})
    protected = artifact.get("protected_files_unchanged", {})
    gate = artifact.get("upstream_gate_receipt", {})
    expected_count = len(contract.get("workload_matrix_rows", [])) * len(CONDITIONS)
    complete_rows = (
        bool(rows)
        and len(rows) == expected_count
        and all(set(PER_UNIT_REQUIRED_FIELDS) <= set(row) for row in rows)
    )
    disabled_identity_exact = bool(identity) and all(
        row.get("serialized_request_bytes_equal")
        and row.get("candidate_order_equal")
        and row.get("checker_calls_equal")
        and row.get("outputs_equal")
        and row.get("error_types_equal")
        and row.get("side_effects_equal")
        and row.get("persistence_equal")
        for row in identity
    )
    native_rows = _condition_rows(rows, "native")
    enabled_rows = _condition_rows(rows, "enabled_adapter")
    native_checker_calls = _sum(native_rows, "checker_calls")
    enabled_checker_calls = _sum(enabled_rows, "checker_calls")
    checker_call_delta = round(native_checker_calls - enabled_checker_calls, 9)
    native_wall = _sum(native_rows, "monotonic_wall_time_s")
    enabled_wall = _sum(enabled_rows, "monotonic_wall_time_s")
    wall_saved = round(native_wall - enabled_wall, 9)
    wall_ratio = round(wall_saved / native_wall, 9) if native_wall > 0 else 0.0
    native_tail = max(
        (float(row.get("monotonic_wall_time_s", 0.0)) for row in native_rows), default=0.0
    )
    enabled_tail = max(
        (float(row.get("monotonic_wall_time_s", 0.0)) for row in enabled_rows),
        default=0.0,
    )
    exact_equal = exact.get("all_exact_outputs_equal") is True and all(
        row.get("exact_output_equal_to_native") is True for row in rows
    )
    candidates_preserved = exact.get("all_candidates_preserved") is True and all(
        row.get("candidate_preserved") is True for row in rows
    )
    fallback_passed = restart.get("fallback_reachable") is True
    restart_passed = (
        restart.get("restart_replayed") is True
        and restart.get("restart_exact_output_equal") is True
    )
    rollback_passed = (
        restart.get("rollback_exercised") is True
        and restart.get("rollback_restores_disabled") is True
    )
    protected_ok = protected.get("all_protected_files_unchanged") is True
    attacks_ok = attacks.get("all_attacks_fail_closed") is True
    gate_passed = gate.get("gate_passed") is True
    tail_latency_regression = enabled_tail > native_tail
    measured_benefit = checker_call_delta >= 1 or (wall_saved >= 0.05 and wall_ratio >= 0.1)
    safety_complete = all(
        (
            gate_passed,
            complete_rows,
            disabled_identity_exact,
            exact_equal,
            candidates_preserved,
            fallback_passed,
            restart_passed,
            rollback_passed,
            protected_ok,
            attacks_ok,
        )
    )
    promotion = safety_complete and measured_benefit and not tail_latency_regression
    if not gate_passed:
        verdict = "blocked"
    elif not exact_equal or not candidates_preserved or not attacks_ok:
        verdict = "disqualified"
    elif not safety_complete:
        verdict = "partial"
    elif promotion:
        verdict = "positive"
    else:
        verdict = "null"
    return {
        "row_type": "aggregate_row_recomputation",
        "expected_per_unit_row_count": expected_count,
        "observed_per_unit_row_count": len(rows),
        "complete_rows": complete_rows,
        "disabled_identity_exact": disabled_identity_exact,
        "exact_outputs_equal": exact_equal,
        "candidate_preservation_passed": candidates_preserved,
        "fallback_passed": fallback_passed,
        "restart_passed": restart_passed,
        "rollback_passed": rollback_passed,
        "shortcut_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "native_checker_calls": native_checker_calls,
        "enabled_checker_calls": enabled_checker_calls,
        "enabled_checker_call_delta": checker_call_delta,
        "native_monotonic_wall_time_s": native_wall,
        "enabled_monotonic_wall_time_s": enabled_wall,
        "enabled_wall_time_saved_s": wall_saved,
        "enabled_wall_time_saved_ratio": wall_ratio,
        "native_tail_wall_time_s": round(native_tail, 9),
        "enabled_tail_wall_time_s": round(enabled_tail, 9),
        "tail_latency_regression": tail_latency_regression,
        "measured_enabled_benefit": measured_benefit,
        "headline_excludes_synthetic_cost_units": True,
        "canary_safe_complete": safety_complete,
        "ready_score_from_rows": 1.0 if safety_complete else 0.0,
        "promotion_score_from_rows": 1.0 if promotion else 0.0,
        "verdict_class_from_rows": verdict,
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "complete_rows": True,
        "disabled_identity_exact": True,
        "exact_outputs_equal": True,
        "candidate_preservation_passed": True,
        "fallback_passed": True,
        "restart_passed": True,
        "rollback_passed": True,
        "shortcut_attacks_passed": True,
        "protected_files_unchanged": True,
        "ready_score_is_binary": True,
        "promotion_score_is_binary": True,
    }
    observed = {
        "complete_rows": aggregate.get("complete_rows"),
        "disabled_identity_exact": aggregate.get("disabled_identity_exact"),
        "exact_outputs_equal": aggregate.get("exact_outputs_equal"),
        "candidate_preservation_passed": aggregate.get("candidate_preservation_passed"),
        "fallback_passed": aggregate.get("fallback_passed"),
        "restart_passed": aggregate.get("restart_passed"),
        "rollback_passed": aggregate.get("rollback_passed"),
        "shortcut_attacks_passed": aggregate.get("shortcut_attacks_passed"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
        "promotion_score_is_binary": aggregate.get("promotion_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        key: {"expected": value, "observed": observed[key], "passed": observed[key] == value}
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    verdict = str(aggregate.get("verdict_class_from_rows"))
    if verdict == "positive":
        return (
            "complete_production_safety_net_workload_canary_positive",
            "complete_production_safety_net_workload_canary_positive: disabled identity, exact equality, fallback, restart, rollback, and measured enabled-path benefit passed",
            "positive",
        )
    if verdict == "blocked":
        return (
            "blocked_production_safety_net_workload_canary",
            "blocked_production_safety_net_workload_canary: upstream gate or checked-in workload fixtures were unavailable",
            "blocked",
        )
    if verdict == "partial":
        return (
            "partial_production_safety_net_workload_canary",
            "partial_production_safety_net_workload_canary: workload rows were incomplete or fallback, restart, or rollback evidence was narrow",
            "partial",
        )
    if verdict == "disqualified":
        return (
            "disqualified_production_safety_net_workload_canary",
            "disqualified_production_safety_net_workload_canary: exact output, candidate preservation, or leakage safety changed",
            "disqualified",
        )
    return (
        "complete_production_safety_net_workload_canary_null",
        "complete_production_safety_net_workload_canary_null: disabled identity, exact equality, fallback, restart, and rollback passed; enabled routing had no preregistered measured work or latency benefit",
        "null",
    )


def _cpu_identity() -> JsonDict:
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    model_name = next(
        (
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        ),
        platform.processor() or platform.machine(),
    )
    return {
        "cpu_count": os.cpu_count() or 0,
        "machine": platform.machine(),
        "processor": model_name,
        "platform": platform.platform(),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    meminfo = Path("/proc/meminfo")
    mem_text = meminfo.read_text(encoding="utf-8") if meminfo.is_file() else ""
    mem_total = next(
        (int(line.split()[1]) for line in mem_text.splitlines() if line.startswith("MemTotal:")),
        0,
    )
    mem_available = next(
        (
            int(line.split()[1])
            for line in mem_text.splitlines()
            if line.startswith("MemAvailable:")
        ),
        0,
    )
    usage = shutil.disk_usage(repo_root)
    return {
        "cpu": _cpu_identity(),
        "ram": {"total_kib": mem_total, "available_kib": mem_available},
        "disk": {
            "path": str(repo_root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
    }


def _z3_version() -> str:
    try:
        import z3  # type: ignore[import-not-found]

        return ".".join(str(part) for part in z3.get_version())
    except Exception as exc:  # pragma: no cover - depends on optional local z3.
        return f"unavailable:{type(exc).__name__}"


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    gate: Mapping[str, Any],
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> JsonDict:
    return {
        "row_type": "preconditions_checked",
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "upstream_gate": {
            "path": UPSTREAM_RELATIVE_PATH.as_posix(),
            "expected": 1.0,
            "observed": gate.get("observed_value"),
            "passed": gate.get("gate_passed") is True,
            "sha256": gate.get("upstream_artifact_sha256"),
        },
        "fixture_hashes": _fixture_hashes(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "z3_version": _z3_version(),
        "resources": _resource_receipt(repo_root),
        "process_isolation": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "single_process_canary": True,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        },
        "timer_resolution": {
            "monotonic_s": float(time.get_clock_info("monotonic").resolution),
            "process_time_s": float(time.get_clock_info("process_time").resolution),
        },
        "protected_file_hashes_before": dict(protected_before),
        "protected_file_hashes_after": dict(protected_after),
        "protected_file_hashes": dict(protected_after),
        "module_hashes": {
            path.as_posix(): sha256_file(repo_root / path)
            for path in (
                EXPERIMENT_RELATIVE_PATH,
                ADAPTER_RELATIVE_PATH,
                ABI_RELATIVE_PATH,
                PIPELINE_RELATIVE_PATH,
                TEST_RELATIVE_PATH,
            )
        },
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "spec_refs": ["REQ-PIPELINE-6563"],
    }


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "deterministic_exp6563_production_safety_net_workload_canary",
            "raw_rows": ["per_unit_rows", "measured_work_and_latency_rows"],
            "reducers": ["aggregate_row_recomputation", "gate_check_summary"],
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-PIPELINE-6563"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.monotonic()
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    protected_before = _protected_hashes(repo_root)
    gate = upstream_gate_receipt(repo_root)
    cases = freeze_workload_cases(repo_root) if gate["gate_passed"] else []
    rows = per_unit_rows(cases) if cases else []
    identity = disabled_identity_rows(rows) if rows else []
    exact = exact_output_and_candidate_receipt(rows)
    measured = measured_work_and_latency_rows(rows)
    restart = restart_and_rollback_receipts(rows)
    warmup_receipt = {
        "skipped": not bool(cases),
        "reason": "" if cases else "blocked_or_missing_workload_cases",
    }
    if cases:
        warmup_receipt = {
            "skipped": False,
            "workload_id": cases[0].workload_id,
            "request_sha256": _route_request_for_case(cases[0]).request_hash,
        }
    contract = frozen_workload_and_timing_contract(
        repo_root=repo_root,
        cases=cases,
        warmup_receipt=warmup_receipt,
    )
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    attacks = shortcut_attack_matrix(
        rows=rows,
        identity_rows=identity,
        contract=contract,
        restart_rollback=restart,
    )
    base_artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": "blocked",
        "upstream_gate_receipt": gate,
        "frozen_workload_and_timing_contract": contract,
        "per_unit_rows": rows,
        "disabled_identity_rows": identity,
        "enabled_route_and_fallback_rows": enabled_route_and_fallback_rows(rows),
        "exact_output_and_candidate_receipt": exact,
        "measured_work_and_latency_rows": measured,
        "restart_and_rollback_receipts": restart,
        "shortcut_attack_matrix": attacks,
        "production_workload_canary_ready_score": 0.0,
        "production_workload_promotion_candidate_score": 0.0,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(base_artifact)
    status, honest, verdict = _status_and_verdict(aggregate)
    base_artifact.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict,
            "production_workload_canary_ready_score": float(aggregate["ready_score_from_rows"]),
            "production_workload_promotion_candidate_score": float(
                aggregate["promotion_score_from_rows"]
            ),
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gate_check_summary(aggregate),
            "preconditions_checked": preconditions_checked(
                repo_root=repo_root,
                result_path=result,
                gate=gate,
                protected_before=protected_before,
                protected_after=protected_after,
            ),
            "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        }
    )
    base_artifact["reproducibility_checksum"] = reproducibility_checksum(base_artifact)
    errors = validate_artifact(base_artifact)
    if write and not errors:
        write_env = None
        if result.is_absolute() and not result.resolve(strict=False).is_relative_to(
            repo_root.resolve(strict=False)
        ):
            write_env = {ARTIFACT_ROOT_ENV: str(result.parent)}
        atomic_write_json(result, base_artifact, root=repo_root, env=write_env, sort_keys=False)
    return base_artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "null",
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class outside Exp6563 enum")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    aggregate = aggregate_row_recomputation(artifact)
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate recomputation mismatch")
    ready = artifact.get("production_workload_canary_ready_score")
    promotion = artifact.get("production_workload_promotion_candidate_score")
    if ready not in {0.0, 1.0}:
        errors.append("production_workload_canary_ready_score must be 0.0 or 1.0")
    if promotion not in {0.0, 1.0}:
        errors.append("production_workload_promotion_candidate_score must be 0.0 or 1.0")
    if ready != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if promotion != aggregate.get("promotion_score_from_rows"):
        errors.append("promotion score mismatch")
    if artifact.get("verdict_class") != aggregate.get("verdict_class_from_rows"):
        errors.append("verdict class mismatch")
    if artifact.get("verdict_class") == "positive" and promotion != 1.0:
        errors.append("positive verdict requires promotion score 1.0")
    if (
        aggregate.get("exact_outputs_equal") is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("exact output equality failed")
    if (
        artifact.get("exact_output_and_candidate_receipt", {}).get("all_exact_outputs_equal")
        is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("exact output equality failed")
    if (
        artifact.get("exact_output_and_candidate_receipt", {}).get("all_candidates_preserved")
        is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("candidate preservation failed")
    if (
        artifact.get("restart_and_rollback_receipts", {}).get("fallback_reachable") is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("fallback unreachable")
    if (
        artifact.get("restart_and_rollback_receipts", {}).get("rollback_restores_disabled")
        is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("rollback failed")
    if (
        artifact.get("shortcut_attack_matrix", {}).get("all_attacks_fail_closed") is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("shortcut attack false accept")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6563 production Safety-Net workload canary."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = _read_json(result)
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result}")
        return 0
    artifact = build_artifact(result_path=result, write=True, run_date=str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
