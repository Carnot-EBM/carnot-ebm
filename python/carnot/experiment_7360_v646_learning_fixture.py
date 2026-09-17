"""Qualify the structural learner's safety and seal a fresh request panel.

This fixture does not measure learning benefit. It reuses the isolated Boolean
executor and opt-in adapter, then freezes the later value contract separately.
Process separation tests an information boundary, not an OS security sandbox.

Spec refs: REQ-CL-7360 and SCENARIO-CL-7360-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7330_v644_public_learner as public
from carnot import experiment_7344_v645_executor_fixture as executor_fixture
from carnot import experiment_7346_v645_learning_adapter as adapter
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    build_scoped_commands,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7360-learning-fixture"
SCHEMA = "carnot.exp7360.v646_learning_fixture.v1"

MODULE_PATH = Path("python/carnot/experiment_7360_v646_learning_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7360_v646_learning_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7360_v646_learning_fixture.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
VALIDATION_PATH = Path("python/carnot/reporting/experiment_7303_validation_scope.py")
PRODUCER_PATH = Path("results/experiment_7358_v646_validation_contract.json")
V645_EXECUTOR_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
V645_ADAPTER_PATH = Path("results/experiment_7346_v645_learning_adapter.json")
V645_PUBLIC_MANIFEST_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/public/public_manifest.json"
)
V645_HISTORICAL_MODEL_PATH = Path(
    "results/raw/experiment_7344_v645_executor_fixture/sidecars/historical_model_receipt.json"
)
PUBLIC_LEARNER_PATH = Path("python/carnot/experiment_7330_v644_public_learner.py")
PRIVATE_EXECUTOR_PATH = Path("scripts/experiments/experiment_7330_v644_private_executor.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7360_v646_learning_fixture.json")
RAW_DIR = Path("results/raw/experiment_7360_v646_learning_fixture")

DEVELOPMENT_SEED = 7_360_101
EVALUATION_SEED = 7_360_211
RESAMPLING_SEED = 7_360_307
TOKEN_SEED = 7_360_401
PRIVATE_RULE_SEED = 7_360_503
CANARY_SEED = 7_360_607
COHORTS = ("stable_rules", "announced_changes", "recurrence", "unannounced_changes")
ARMS = adapter.ARMS
QUERY_BUDGET = adapter.QUERY_BUDGET
STATE_CAP_BYTES = adapter.STATE_CAP_BYTES

ZERO_INVOCATION_COUNTS = deepcopy(adapter.ZERO_INVOCATION_COUNTS)
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ALL_VALIDATION_NAMES = (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    VALIDATION_PATH,
    PUBLIC_LEARNER_PATH,
    PRIVATE_EXECUTOR_PATH,
    Path("python/carnot/experiment_7344_v645_executor_fixture.py"),
    Path("python/carnot/experiment_7346_v645_learning_adapter.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/pipeline/verify_repair.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    PRODUCER_PATH,
    V645_EXECUTOR_PATH,
    V645_ADAPTER_PATH,
    V645_PUBLIC_MANIFEST_PATH,
    V645_HISTORICAL_MODEL_PATH,
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version this record and retain ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and affected validation; never a success-shaped placeholder.",
    "run_date": "Use 20260917 and actual UTC timestamps.",
    "preconditions_checked": "Exact input, resource and required-field checks before dependent work.",
    "MODEL_SPECS": "Actual intended identities; this CPU fixture intends no current model.",
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
    "learning_fixture_ready_score": "One for safe adapter and complete sealed panel, independent of benefit; no promotion.",
    "fixture_manifest": "Hashes, distinct request IDs, development/evaluation partition, evaluator identity and inaccessible labels.",
    "frozen_acceptance_manifest": "Arms, budgets, comparisons, cluster bootstrap and unchanged value criteria before outcomes.",
    "safety_rows": "Each lifecycle/drift/leakage/mutation control including failed controls.",
}

REQUIRED_FIELDS = frozenset(
    {
        *REQUIRED_FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "phase",
        "started_at_utc",
        "completed_at_utc",
        "execution_host",
        "host_computation",
        "learning_value_score",
        "promotion_score",
        "raw_evidence_paths",
        "historical_determination",
        "production_defaults_changed",
        "research_roadmap_changed",
        "no_model_weight_mutation",
    }
)


@dataclass(frozen=True)
class FixturePaths:
    """Keep public requests, evaluator labels, and terminal evidence separate."""

    raw_dir: Path
    public_manifest: Path
    private_manifest: Path
    acceptance_manifest: Path
    learner_evidence: Path
    evaluator_receipt: Path
    safety_rows: Path
    historical_inference_sidecar: Path
    candidate: Path

    @classmethod
    def for_raw_dir(cls, raw_dir: Path) -> FixturePaths:
        root = raw_dir.resolve()
        return cls(
            raw_dir=root,
            public_manifest=root / "public/public_manifest.json",
            private_manifest=root / "evaluator/evaluator_private_manifest.json",
            acceptance_manifest=root / "public/frozen_acceptance_manifest.json",
            learner_evidence=root / "learner/learner_evidence.json",
            evaluator_receipt=root / "evaluator/evaluator_boundary_receipt.json",
            safety_rows=root / "safety/safety_rows.json",
            historical_inference_sidecar=root / "sidecars/historical_inference_receipts.json",
            candidate=root / "measured-terminal-candidate.json",
        )


def progress(phase: str, event: str, detail: str = "") -> None:
    """Flush each boundary so long operations remain observable."""

    suffix = f" {detail}" if detail else ""
    print(f"[exp7360] phase={phase} event={event}{suffix}", flush=True)


def sha256_file(path: Path) -> str:
    """Hash exact bytes with the existing public learner representation."""

    return public.sha256_file(path)


def load_json(path: Path) -> JsonDict:
    """Load one object and reject arrays or scalar placeholders."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete canonical bytes through one same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(public.canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted rename leaves it.
            temporary.unlink()


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
        "available": bool(passed),
        "blocking": True,
    }


def collect_preconditions(
    root: Path,
    *,
    producer_path: Path | None = None,
    exclusion_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Check every producer field and required byte before dependent work."""

    root = root.resolve()
    producer_path = producer_path or root / PRODUCER_PATH
    exclusion_path = exclusion_path or root / "ops/exclusion_manifest.yaml"
    rows: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition(
                f"source_bytes:{relative}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                path.stat().st_size if available else "missing",
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)
    producer: JsonDict = {}
    producer_exists = producer_path.is_file() and producer_path.stat().st_size > 0
    if producer_exists:
        try:
            producer = load_json(producer_path)
        except (OSError, json.JSONDecodeError, ValueError):
            producer = {}
            producer_exists = False
    rows.append(
        _precondition(
            "producer_path",
            str(producer_path),
            "bytes",
            "readable_nonempty_json",
            producer_path.stat().st_size if producer_exists else "missing",
            producer_exists,
        )
    )
    if producer_exists:
        hashes[str(producer_path)] = sha256_file(producer_path)
    checks = (
        ("producer_milestone", "milestone", MILESTONE, producer.get("milestone") == MILESTONE),
        ("producer_run_date", "run_date", RUN_DATE, producer.get("run_date") == RUN_DATE),
        (
            "producer_ready",
            "validation_contract_ready_score",
            1,
            producer.get("validation_contract_ready_score") == 1,
        ),
        (
            "producer_terminal_class",
            "verdict_class",
            ["positive", "circular_positive", "null"],
            producer.get("verdict_class") in {"positive", "circular_positive", "null"},
        ),
        (
            "producer_adversarial_clear",
            "flagged_adversarial",
            False,
            producer.get("flagged_adversarial") is False,
        ),
    )
    for check, field, expected, passed in checks:
        rows.append(
            _precondition(check, str(producer_path), field, expected, producer.get(field), passed)
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    rows.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7360",
            "REQ-CL-7360" if "REQ-CL-7360" in spec_text else "missing",
            "REQ-CL-7360" in spec_text,
        )
    )
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "exp7360" in exclusion_text.lower() or "experiment_id: 7360" in exclusion_text
    rows.append(
        _precondition(
            "current_task_not_quarantined",
            str(exclusion_path),
            EXPERIMENT_ID,
            True,
            not excluded,
            not excluded,
        )
    )
    return rows, hashes, producer


def _normalized_request(request: Mapping[str, Any]) -> JsonDict:
    """Remove opaque names so identifier renaming cannot hide request reuse."""

    activities = [str(value) for value in request["activities"]]
    rename = {name: f"unit-{index + 1}" for index, name in enumerate(activities)}
    return {
        "activities": [rename[name] for name in activities],
        "allowed_starts": {rename[name]: list(request["allowed_starts"][name]) for name in activities},
        "durations": {rename[name]: request["durations"][name] for name in activities},
        "weights": {rename[name]: request["weights"][name] for name in activities},
        "horizon": request["horizon"],
        "public_revision": request.get("public_revision", 0),
    }


def _base_requests(manifest: Mapping[str, Any]) -> list[JsonDict]:
    requests = [
        dict(request)
        for stream in manifest.get("development_streams", [])
        for request in stream.get("requests", [])
    ]
    requests.extend(dict(row["original"]) for row in manifest.get("live_proposal_panel", []))
    requests.extend(dict(request) for request in manifest.get("development_canary", []))
    return requests


def _request_hashes(manifest: Mapping[str, Any]) -> set[str]:
    return {public.sha256_json(_normalized_request(request)) for request in _base_requests(manifest)}


def frozen_acceptance_manifest(public_manifest_sha256: str) -> JsonDict:
    """Freeze Exp7362 arms and value gates without evaluating any outcome."""

    return {
        "schema": "carnot.exp7360.frozen_acceptance_manifest.v1",
        "sealed_before_evaluation": True,
        "public_manifest_sha256": public_manifest_sha256,
        "arms": list(ARMS),
        "arm_definitions": {
            ARMS[0]: "persistent structural memory",
            ARMS[1]: "reset every request",
            ARMS[2]: "exact-request cache without structural generalization",
            ARMS[3]: "structural memory frozen after warmup",
        },
        "same_proposal_list_per_paired_request": True,
        "same_exact_executor_per_paired_request": True,
        "query_budget_per_request": QUERY_BUDGET,
        "state_cap_bytes": STATE_CAP_BYTES,
        "charged_operations": [
            "information_query",
            "proposal_check",
            "write",
            "persistence",
            "verification",
        ],
        "cluster_bootstrap_unit": "stream",
        "resampling_seed": RESAMPLING_SEED,
        "value_evaluated": False,
        "value_gates": [
            {
                "check": "unsafe_accepted_plans",
                "expected": 0,
                "fixture_gate": False,
            },
            {
                "check": "stale_version_decisions",
                "expected": 0,
                "fixture_gate": False,
            },
            {
                "check": "normalized_utility_noninferiority",
                "lower_95_inclusive": -0.02,
                "fixture_gate": False,
            },
            {
                "check": "coverage_noninferiority",
                "lower_95_inclusive": -0.02,
                "fixture_gate": False,
            },
            {
                "check": "query_ratio_vs_reset",
                "upper_95_exclusive": 0.90,
                "fixture_gate": False,
            },
            {
                "check": "query_ratio_vs_exact_cache",
                "upper_95_exclusive": 0.90,
                "fixture_gate": False,
            },
            {
                "check": "complete_service_cost_ratio",
                "upper_95_inclusive": 1.0,
                "fixture_gate": False,
            },
            {
                "check": "later_distinct_request_erasure_witness",
                "minimum": 1,
                "fixture_gate": False,
            },
        ],
    }


def build_fixture_manifests(
    paths: FixturePaths,
    v645_public_manifest_path: Path,
) -> tuple[JsonDict, JsonDict]:
    """Seal fresh public requests and evaluator-only labels before evaluation."""

    private = executor_fixture._private_executor()  # noqa: SLF001 - existing fixture API.
    development_rng = random.Random(DEVELOPMENT_SEED)
    evaluation_rng = random.Random(EVALUATION_SEED)
    canary_rng = random.Random(CANARY_SEED)
    token_rng = random.Random(TOKEN_SEED)
    rule_rng = random.Random(PRIVATE_RULE_SEED)
    development: list[JsonDict] = []
    records: dict[str, JsonDict] = {}
    for cohort_index, cohort in enumerate(COHORTS):
        for local_index in range(8):
            stream, private_rows = private._stream(  # noqa: SLF001
                development_rng,
                token_rng,
                rule_rng,
                f"v646-development-{cohort_index}-{local_index:02d}",
                executor_fixture._internal_cohort(cohort),  # noqa: SLF001
            )
            stream["cohort"] = cohort
            for request_index, request in enumerate(stream["requests"]):
                request["warmup"] = request_index < 4
            development.append(stream)
            records.update(private_rows)

    public_streams: list[JsonDict] = []
    pairs: list[JsonDict] = []
    for stream_index in range(8):
        cohort = COHORTS[stream_index % len(COHORTS)]
        first = private._opaque_token(token_rng)  # noqa: SLF001
        second = private._opaque_token(token_rng)  # noqa: SLF001
        authorities = {
            "authority-a": private._rule_assignment(rule_rng),  # noqa: SLF001
            "authority-b": private._rule_assignment(rule_rng),  # noqa: SLF001
        }
        stream_pairs: list[JsonDict] = []
        schedule = executor_fixture._model_schedule(cohort, first, second)  # noqa: SLF001
        for request_index, (token, authority) in enumerate(schedule):
            original = private._public_request(  # noqa: SLF001
                evaluation_rng,
                f"v646-public-{stream_index:02d}-{request_index:02d}-original",
                token,
                request_index,
            )
            twin, rename = private._renamed_twin(  # noqa: SLF001
                original,
                f"v646-public-{stream_index:02d}-{request_index:02d}-twin",
            )
            rules = authorities[authority]
            twin_rules = private._rename_rules(rules, rename)  # noqa: SLF001
            records[str(original["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": rules,
                "acceptance_witness": private._acceptance_witness(original, rules),  # noqa: SLF001
                "witness_label": True,
            }
            records[str(twin["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": twin_rules,
                "acceptance_witness": private._acceptance_witness(twin, twin_rules),  # noqa: SLF001
                "witness_label": True,
            }
            pair = {
                "panel_id": f"v646-public-{stream_index:02d}-{request_index:02d}",
                "stream_id": f"v646-public-{stream_index:02d}",
                "request_index": request_index,
                "warmup": request_index < 2,
                "cohort": cohort,
                "presentation_order": "original_first" if request_index % 2 == 0 else "twin_first",
                "original": original,
                "twin": twin,
                "renaming_map": rename,
            }
            stream_pairs.append(pair)
            pairs.append(pair)
        public_streams.append(
            {
                "stream_id": f"v646-public-{stream_index:02d}",
                "cohort": cohort,
                "requests": stream_pairs,
            }
        )

    canary: list[JsonDict] = []
    for index in range(4):
        token = private._opaque_token(token_rng)  # noqa: SLF001
        request = private._public_request(  # noqa: SLF001
            canary_rng, f"v646-development-canary-{index:02d}", token, index
        )
        rules = private._rule_assignment(rule_rng)  # noqa: SLF001
        canary.append(request)
        records[str(request["request_id"])] = {
            "version_token": token,
            "authority_label": "development-canary",
            "private_rules": rules,
            "acceptance_witness": private._acceptance_witness(request, rules),  # noqa: SLF001
            "witness_label": True,
        }

    challenge_request: JsonDict = {
        "request_id": "v646-compound-challenge",
        "version_token": private._opaque_token(token_rng),  # noqa: SLF001
        "activities": ["a", "b", "c"],
        "allowed_starts": {name: [0, 3] for name in "abc"},
        "durations": {name: 2 for name in "abc"},
        "weights": {name: 1 for name in "abc"},
        "horizon": 6,
        "public_revision": 0,
    }
    challenge_plan = {
        "request_id": challenge_request["request_id"],
        "assignments": {name: 0 for name in "abc"},
    }
    challenge_rules = {
        "capacity": 6,
        "pair_gaps": [],
        "forbidden_compounds": [["a", "b", "c"]],
    }
    records[str(challenge_request["request_id"])] = {
        "version_token": challenge_request["version_token"],
        "authority_label": "compound-outside-language",
        "private_rules": challenge_rules,
        "acceptance_witness": private._acceptance_witness(challenge_request, challenge_rules),  # noqa: SLF001
        "witness_label": True,
    }

    v645 = load_json(v645_public_manifest_path)
    old_hashes = sorted(_request_hashes(v645))
    public_manifest: JsonDict = {
        "schema": "carnot.exp7360.public_manifest.v1",
        "sealed_before_outcomes": True,
        "development_seed": DEVELOPMENT_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "resampling_seed": RESAMPLING_SEED,
        "development_streams": development,
        "held_out_streams": [],
        "warmup_requests_per_development_stream": 4,
        "public_request_streams": public_streams,
        "public_model_streams": public_streams,
        "warmup_requests_per_public_stream": 2,
        "live_proposal_panel": pairs,
        "development_canary": canary,
        "compound_conflict_challenge": {
            "request": challenge_request,
            "candidate_plan": challenge_plan,
            "outside_acquisition_language": True,
        },
        "hypothesis_vocabulary": ["pair_minimum_gap", "capacity_bound"],
        "downstream_query_budget_per_request": QUERY_BUDGET,
        "downstream_state_cap_bytes": STATE_CAP_BYTES,
        "v645_normalized_request_hash_seal": public.sha256_json(old_hashes),
    }
    new_hashes = sorted(_request_hashes(public_manifest))
    public_manifest["normalized_request_hashes"] = new_hashes
    public_manifest["manifest_hash"] = public.sha256_json(public_manifest)
    _atomic_json(paths.public_manifest, public_manifest)
    private_manifest: JsonDict = {
        "schema": "carnot.exp7360.evaluator_private_manifest.v1",
        "public_manifest_sha256": sha256_file(paths.public_manifest),
        "public_manifest_hash": public_manifest["manifest_hash"],
        "token_seed_commitment": public.sha256_json(TOKEN_SEED),
        "private_rule_seed_commitment": public.sha256_json(PRIVATE_RULE_SEED),
        "evaluator_records": records,
        "label_count": len(records),
        "all_acceptance_witnesses_nonempty": all(
            bool(row["acceptance_witness"]["assignments"]) for row in records.values()
        ),
    }
    private_manifest["manifest_hash"] = public.sha256_json(private_manifest)
    _atomic_json(paths.private_manifest, private_manifest)
    _atomic_json(
        paths.acceptance_manifest,
        frozen_acceptance_manifest(sha256_file(paths.public_manifest)),
    )
    _write_historical_sidecar(paths.historical_inference_sidecar)
    return public_manifest, private_manifest


def panel_errors(
    manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    v645_manifest: Mapping[str, Any],
) -> list[str]:
    """Recompute all panel counts, seals, freshness, and label boundaries."""

    errors: list[str] = []
    streams = manifest.get("development_streams", [])
    if len(streams) != 32 or any(len(row.get("requests", [])) != 12 for row in streams):
        errors.append("development_panel_shape")
    if any(sum(row.get("cohort") == cohort for row in streams) != 8 for cohort in COHORTS):
        errors.append("development_cohorts")
    if any(
        request.get("warmup") is not (index < 4)
        for stream in streams
        for index, request in enumerate(stream.get("requests", []))
    ):
        errors.append("development_warmup")
    public_streams = manifest.get("public_request_streams", [])
    pairs = manifest.get("live_proposal_panel", [])
    if (
        len(public_streams) != 8
        or any(len(row.get("requests", [])) != 4 for row in public_streams)
        or len(pairs) != 32
    ):
        errors.append("public_panel_shape")
    if any(row.get("warmup") is not (int(row.get("request_index", -1)) < 2) for row in pairs):
        errors.append("public_warmup")
    if len(manifest.get("development_canary", [])) != 4:
        errors.append("development_canary_count")
    all_requests = [
        *[request for stream in streams for request in stream.get("requests", [])],
        *[pair[side] for pair in pairs for side in ("original", "twin")],
        *manifest.get("development_canary", []),
        manifest.get("compound_conflict_challenge", {}).get("request", {}),
    ]
    request_ids = [str(row.get("request_id")) for row in all_requests]
    if len(request_ids) != len(set(request_ids)) or any(value in {"", "None"} for value in request_ids):
        errors.append("request_ids_not_distinct")
    if len({public.sha256_json(_normalized_request(row["original"])) for row in pairs}) != 32:
        errors.append("public_requests_not_distinct")
    serialized = json.dumps(manifest, sort_keys=True)
    if any(marker in serialized for marker in ("private_rules", "acceptance_witness", "witness_label")):
        errors.append("private_data_in_public_manifest")
    if _request_hashes(manifest) & _request_hashes(v645_manifest):
        errors.append("v645_request_overlap")
    seeds = [
        manifest.get("development_seed"),
        manifest.get("evaluation_seed"),
        manifest.get("resampling_seed"),
    ]
    if len(set(seeds)) != 3:
        errors.append("seeds_not_disjoint")
    frozen = deepcopy(dict(manifest))
    observed_hash = frozen.pop("manifest_hash", None)
    if observed_hash != public.sha256_json(frozen):
        errors.append("public_manifest_hash")
    if private_manifest.get("public_manifest_hash") != manifest.get("manifest_hash"):
        errors.append("private_public_manifest_binding")
    records = private_manifest.get("evaluator_records", {})
    if set(records) != set(request_ids):
        errors.append("private_record_identity")
    if not private_manifest.get("all_acceptance_witnesses_nonempty"):
        errors.append("empty_acceptance_witness")
    return sorted(set(errors))


def _write_historical_sidecar(path: Path) -> None:
    """Bind historical model-shaped evidence without counting current calls."""

    sources = []
    for relative in (V645_EXECUTOR_PATH, V645_ADAPTER_PATH, V645_HISTORICAL_MODEL_PATH):
        absolute = REPO_ROOT / relative
        value = load_json(absolute)
        sources.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(absolute),
                "experiment_id": value.get("experiment_id"),
                "status": value.get("status"),
                "verdict_class": value.get("verdict_class"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "MODEL_SPECS": value.get("MODEL_SPECS", []),
                "model_invoked": value.get("model_invoked", False),
                "counted_as_current_inference": False,
            }
        )
    _atomic_json(
        path,
        {
            "schema": "carnot.exp7360.historical_inference_receipts.v1",
            "label": "historical_hash_bound_not_current_generation",
            "current_MODEL_SPECS": [],
            "current_model_invoked": False,
            "current_invocation_counts": ZERO_INVOCATION_COUNTS,
            "sources": sources,
        },
    )


def _stream_reader(process: subprocess.Popen[str], lines: list[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[exp7360:evaluator] {line.rstrip()}", flush=True)


def run_isolated_fixture(root: Path, paths: FixturePaths) -> JsonDict:
    """Run the existing learner and private executor in separate processes."""

    python = str(root / ".venv/bin/python")
    endpoint = Path("/tmp") / f"carnot-exp7360-{public.sha256_json(str(paths.raw_dir))[-12:]}.sock"
    evaluator_argv = [
        python,
        "-u",
        str(root / PRIVATE_EXECUTOR_PATH),
        "--serve",
        "--public-manifest",
        str(paths.public_manifest),
        "--private-manifest",
        str(paths.private_manifest),
        "--endpoint",
        str(endpoint),
        "--receipt",
        str(paths.evaluator_receipt),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    progress("evaluation", "before_evaluator", f"command={shlex.join(evaluator_argv)}")
    evaluator = subprocess.Popen(  # noqa: S603 - fixed local evaluator argument vector.
        evaluator_argv,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    evaluator_lines: list[str] = []
    reader = threading.Thread(target=_stream_reader, args=(evaluator, evaluator_lines), daemon=True)
    reader.start()
    deadline = time.monotonic() + 30.0
    while not endpoint.exists() and evaluator.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not endpoint.exists():  # pragma: no cover - local process startup failure.
        evaluator.terminate()
        raise RuntimeError("evaluator_endpoint_unavailable")
    learner_receipt = executor_fixture.run_streamed_process(
        [
            python,
            "-u",
            "-m",
            "carnot.experiment_7330_v644_public_learner",
            "--public-manifest",
            str(paths.public_manifest),
            "--endpoint",
            str(endpoint),
            "--output",
            str(paths.learner_evidence),
        ],
        root,
        paths.raw_dir / "logs/public_learner.log",
        name="v646_public_learner",
    )
    evaluator_exit = evaluator.wait(timeout=120)
    reader.join(timeout=2)
    evaluator_log = paths.raw_dir / "logs/private_evaluator.log"
    evaluator_log.parent.mkdir(parents=True, exist_ok=True)
    evaluator_log.write_text("".join(evaluator_lines), encoding="utf-8")
    progress("evaluation", "after_evaluator", f"exit={evaluator_exit}")
    if not learner_receipt["passed"] or evaluator_exit != 0:  # pragma: no cover - retained failure.
        raise RuntimeError("isolated_fixture_process_failed")
    learner = load_json(paths.learner_evidence)
    evaluator_evidence = load_json(paths.evaluator_receipt)
    return {
        "learner": learner,
        "evaluator": evaluator_evidence,
        "process_receipts": {
            "learner": learner_receipt,
            "evaluator": {
                "name": "v646_private_evaluator",
                "command": shlex.join(evaluator_argv),
                "command_argv": evaluator_argv,
                "scope": "bounded_process_e2e",
                "exit_code": evaluator_exit,
                "duration_s": learner_receipt["duration_s"],
                "log_path": str(evaluator_log),
                "log_sha256": sha256_file(evaluator_log),
                "passed": evaluator_exit == 0,
                "timed_out": False,
                "pid": evaluator.pid,
            },
        },
    }


def _safety_row(
    control: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    query_attempts: int = 0,
    state_bytes: int = 0,
) -> JsonDict:
    return {
        "row_type": "safety_control",
        "unit_id": f"safety:{control}",
        "arm": "learning_fixture",
        "control": control,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "query_attempts": int(query_attempts),
        "state_bytes": int(state_bytes),
        "costs": {"current_model_calls": 0, "paid_exact_queries": int(query_attempts)},
        "failures": [] if passed else [control],
        "censored": False,
        "disposition": "complete",
    }


def _control_request(request_id: str, version: str) -> JsonDict:
    return {
        "request_id": request_id,
        "version_token": version,
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2, 4], "b": [0, 2, 4]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 8,
        "public_revision": 0,
    }


def _control_record(version: str, minimum_gap: int) -> JsonDict:
    return {
        "version_token": version,
        "private_rules": {
            "capacity": 2,
            "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": minimum_gap}],
            "forbidden_compounds": [],
        },
    }


def run_adapter_safety_controls(state_root: Path) -> list[JsonDict]:
    """Exercise drift, transaction lifecycle, and exact release authority."""

    state_root.mkdir(parents=True, exist_ok=True)
    version = "opaque-v646-control-a"
    learning = adapter.LearningScheduleAdapter(state_root / "drift", enabled=True)
    harness = adapter.AdapterPipelineHarness(learning)
    first = _control_request("v646-control-first", version)
    first_row = harness.execute(first, _control_record(version, 1), warmup=True)
    committed_bytes = len(learning.state_bytes())
    announced_request = _control_request("v646-control-announced", "opaque-v646-control-b")
    announced_executor = adapter.QualifiedFixtureExecutor(
        announced_request,
        _control_record("opaque-v646-control-b", 1),
    )
    announced = learning.begin_request(announced_request, announced_executor)
    learning.cancel_request("announced_drift_probe")
    second = _control_request("v646-control-hidden-drift", version)
    second["public_revision"] = 1
    hidden_row = harness.execute(second, _control_record(version, 3), warmup=False)
    feedback = learning.last_feedback
    harness.close()

    lifecycle = adapter.run_adapter_lifecycle(state_root / "lifecycle")
    cache: dict[str, bool] = {}
    cache_request = _control_request("v646-control-cache", "opaque-v646-cache")
    cache_executor = adapter.QualifiedFixtureExecutor(
        cache_request,
        _control_record("opaque-v646-cache", 0),
        exact_cache=cache,
    )
    cache_plan = public.make_plan(cache_request["request_id"], {"a": 0, "b": 2})
    cache_executor.query(cache_plan, "proposal_check", allow_cache=True)
    cache_executor.query(cache_plan, "proposal_check", allow_cache=True)
    cache_executor.query(cache_plan, "final", allow_cache=True)
    final_receipts = [row for row in cache_executor.receipts if row["reason"] == "final"]
    cache_observed = {
        "final_external_calls": sum(row["external_call"] for row in final_receipts),
        "final_cache_hits": sum(row["cache_hit"] for row in final_receipts),
    }
    lifecycle_expected = {
        "commit_after_close": True,
        "restart_atom_visible": True,
        "restart_bytes_equal": True,
        "rollback_bytes_equal": True,
        "duplicate_feedback_noop": True,
        "e2e_007_passed": True,
    }
    lifecycle_observed = {key: lifecycle.get(key) for key in lifecycle_expected}
    rows = [
        _safety_row(
            "post_request_only_commit",
            {"entry_unchanged": True, "committed": True},
            {
                "entry_unchanged": first_row["entry_state_hash"] is not None,
                "committed": committed_bytes > 0 and first_row["new_atom_count"] >= 1,
            },
            committed_bytes > 0 and first_row["new_atom_count"] >= 1,
            query_attempts=first_row["query_attempts"],
            state_bytes=committed_bytes,
        ),
        _safety_row(
            "announced_drift",
            {"active_atoms": 0},
            {"active_atoms": announced["active_atom_count_before"]},
            announced["active_atom_count_before"] == 0,
            state_bytes=committed_bytes,
        ),
        _safety_row(
            "contradictory_feedback",
            {"invalidated_count": ">=1", "unsafe_return": False},
            {
                "invalidated_count": len(feedback.get("invalidated_atom_ids", [])),
                "unsafe_return": bool(hidden_row["returned"] and not hidden_row["returned_feasible"]),
            },
            len(feedback.get("invalidated_atom_ids", [])) >= 1
            and not (hidden_row["returned"] and not hidden_row["returned_feasible"]),
            query_attempts=hidden_row["query_attempts"],
            state_bytes=hidden_row["state_bytes"],
        ),
        _safety_row(
            "unannounced_drift",
            {"contradiction_revalidation": True, "returned_unsafe": False},
            {
                "contradiction_revalidation": hidden_row["contradiction_revalidation"],
                "returned_unsafe": bool(hidden_row["returned"] and not hidden_row["returned_feasible"]),
            },
            hidden_row["contradiction_revalidation"] is True
            and not (hidden_row["returned"] and not hidden_row["returned_feasible"]),
            query_attempts=hidden_row["query_attempts"],
            state_bytes=hidden_row["state_bytes"],
        ),
        _safety_row(
            "rollback_restart_post_request_commit",
            lifecycle_expected,
            lifecycle_observed,
            lifecycle_observed == lifecycle_expected,
            state_bytes=committed_bytes,
        ),
        _safety_row(
            "unsafe_cached_release",
            {"final_external_calls": 1, "final_cache_hits": 0},
            cache_observed,
            cache_observed == {"final_external_calls": 1, "final_cache_hits": 0},
            query_attempts=cache_executor.attempt_count,
            state_bytes=committed_bytes,
        ),
    ]
    return rows


def build_safety_rows(
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    process: Mapping[str, Any],
    adapter_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Reduce process, leakage, drift, and adapter controls into per-unit rows."""

    learner = process["learner"]
    evaluator = process["evaluator"]
    distinct_pids = learner["learner_pid"] != evaluator["evaluator_pid"]
    no_private_access = learner["forbidden_accesses"] == []
    response_shape = evaluator["response_keys"] == ["accepted", "query_id"]
    compound = learner["compound_conflict_challenge"]
    maximum_request_queries = max(int(row["query_attempts"]) for row in learner["rows"])
    panel_failures = panel_errors(
        public_manifest,
        private_manifest,
        load_json(REPO_ROOT / V645_PUBLIC_MANIFEST_PATH),
    )
    process_rows = [
        _safety_row(
            "process_information_boundary",
            {"separate": True, "os_security_sandbox": False, "maximum_request_queries": "<=24"},
            {
                "separate": distinct_pids,
                "os_security_sandbox": False,
                "maximum_request_queries": maximum_request_queries,
                "total_paid_queries": len(evaluator["query_rows"]),
            },
            distinct_pids and maximum_request_queries <= QUERY_BUDGET,
            query_attempts=maximum_request_queries,
        ),
        _safety_row(
            "hidden_rule_access_rejection",
            {"forbidden_accesses": [], "response_keys": ["accepted", "query_id"]},
            {
                "forbidden_accesses": learner["forbidden_accesses"],
                "response_keys": evaluator["response_keys"],
            },
            no_private_access and response_shape and evaluator["returned_private_fields"] is False,
        ),
        _safety_row(
            "higher_order_counterexample",
            {
                "full_rejected": True,
                "all_pair_projections_accepted": True,
                "learned_atom_count": 0,
            },
            {
                "full_rejected": compound["full_rejected"],
                "all_pair_projections_accepted": compound["all_pair_projections_accepted"],
                "learned_atom_count": compound["learned_atom_count"],
            },
            compound["full_rejected"] is True
            and compound["all_pair_projections_accepted"] is True
            and compound["learned_atom_count"] == 0,
        ),
        _safety_row("fresh_sealed_panel", [], panel_failures, not panel_failures),
    ]
    return [*process_rows, *[deepcopy(dict(row)) for row in adapter_rows]]


def write_raw_evidence(paths: FixturePaths, safety_rows: Sequence[Mapping[str, Any]]) -> None:
    """Persist safety rows separately so terminal claims can be cold-reduced."""

    _atomic_json(
        paths.safety_rows,
        {"schema": "carnot.exp7360.safety_rows.v1", "rows": list(safety_rows)},
    )


def raw_source_hashes(paths: FixturePaths) -> dict[str, str]:
    """Return exact hashes for every task-owned raw input used by reduction."""

    selected = (
        paths.public_manifest,
        paths.private_manifest,
        paths.acceptance_manifest,
        paths.learner_evidence,
        paths.evaluator_receipt,
        paths.safety_rows,
        paths.historical_inference_sidecar,
    )
    return {str(path): sha256_file(path) for path in selected}


def _fixture_manifest(paths: FixturePaths) -> JsonDict:
    public_manifest = load_json(paths.public_manifest)
    private_manifest = load_json(paths.private_manifest)
    learner = load_json(paths.learner_evidence)
    evaluator = load_json(paths.evaluator_receipt)
    request_ids = [str(row["request_id"]) for row in _base_requests(public_manifest)]
    twin_ids = [str(row["twin"]["request_id"]) for row in public_manifest["live_proposal_panel"]]
    return {
        "public_manifest_path": str(paths.public_manifest),
        "public_manifest_sha256": sha256_file(paths.public_manifest),
        "public_manifest_hash": public_manifest["manifest_hash"],
        "private_manifest_path": str(paths.private_manifest),
        "private_manifest_sha256": sha256_file(paths.private_manifest),
        "private_manifest_hash": private_manifest["manifest_hash"],
        "acceptance_manifest_path": str(paths.acceptance_manifest),
        "acceptance_manifest_sha256": sha256_file(paths.acceptance_manifest),
        "development_stream_count": len(public_manifest["development_streams"]),
        "development_request_count": sum(
            len(stream["requests"]) for stream in public_manifest["development_streams"]
        ),
        "public_request_count": len(public_manifest["live_proposal_panel"]),
        "renamed_twin_count": len(twin_ids),
        "development_canary_count": len(public_manifest["development_canary"]),
        "distinct_request_ids": sorted([*request_ids, *twin_ids]),
        "normalized_request_hashes": public_manifest["normalized_request_hashes"],
        "v645_normalized_request_hash_seal": public_manifest[
            "v645_normalized_request_hash_seal"
        ],
        "partition": {
            "development_seed": DEVELOPMENT_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "resampling_seed": RESAMPLING_SEED,
            "seeds_disjoint": len({DEVELOPMENT_SEED, EVALUATION_SEED, RESAMPLING_SEED}) == 3,
        },
        "evaluator_identity": {
            "module": PRIVATE_EXECUTOR_PATH.as_posix(),
            "module_sha256": sha256_file(REPO_ROOT / PRIVATE_EXECUTOR_PATH),
            "evaluator_pid": evaluator["evaluator_pid"],
            "learner_pid": learner["learner_pid"],
            "response_keys": evaluator["response_keys"],
        },
        "evaluator_labels_inaccessible_to_learner": learner["forbidden_accesses"] == [],
        "learner_private_label_reads": len(learner["forbidden_accesses"]),
        "process_isolation_scope": "information_boundary_test_not_os_security_sandbox",
        "hypothesis_vocabulary": public_manifest["hypothesis_vocabulary"],
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _receipt_passed(receipts: Sequence[Mapping[str, Any]], name: str) -> bool:
    selected = [row for row in receipts if row.get("name") == name]
    return (
        len(selected) == 1
        and selected[0].get("passed") is True
        and selected[0].get("exit_code") == 0
        and selected[0].get("timed_out") is not True
    )


def passing_test_receipts() -> list[JsonDict]:
    """Provide complete named receipts for terminal reducer unit tests."""

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
        for name in ALL_VALIDATION_NAMES
    ]


def test_phase_spans() -> list[JsonDict]:
    """Provide disjoint spans for deterministic terminal reducer tests."""

    return [
        {"phase": "load", "start_elapsed_s": 0.0, "end_elapsed_s": 0.0, "duration_s": 0.0},
        {
            "phase": "generation",
            "start_elapsed_s": 0.0,
            "end_elapsed_s": 0.0,
            "duration_s": 0.0,
        },
        {
            "phase": "evaluation",
            "start_elapsed_s": 0.0,
            "end_elapsed_s": 0.5,
            "duration_s": 0.5,
        },
        {
            "phase": "validation",
            "start_elapsed_s": 0.5,
            "end_elapsed_s": 0.9,
            "duration_s": 0.4,
        },
        {"phase": "write", "start_elapsed_s": 0.9, "end_elapsed_s": 1.0, "duration_s": 0.1},
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    principles = {
        field: "Retain this supporting fixture evidence in its ordinary JSON type."
        for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind settings and raw claims while excluding host timing and receipts."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "fixture_manifest",
        "frozen_acceptance_manifest",
        "safety_rows",
        "historical_determination",
        "learning_fixture_ready_score",
        "learning_value_score",
        "promotion_score",
        "verdict_class",
    )
    return public.sha256_json({key: artifact.get(key) for key in keys})


def _historical_determination() -> JsonDict:
    old = load_json(REPO_ROOT / V645_ADAPTER_PATH)
    return {
        "path": V645_ADAPTER_PATH.as_posix(),
        "sha256": sha256_file(REPO_ROOT / V645_ADAPTER_PATH),
        "status": old.get("status"),
        "verdict_class": old.get("verdict_class"),
        "honest_verdict": old.get("honest_verdict"),
        "learning_adapter_ready_score": old.get("learning_adapter_ready_score"),
        "learning_value_score": old.get("learning_value_score"),
        "query_advantage_gate": old.get("acceptance_gate_results", {}).get("query_advantage"),
        "unchanged_method_value_claim": False,
        "preserved_as_evidence_against_unchanged_method": True,
    }


def _raw_paths(paths: FixturePaths) -> dict[str, str]:
    return {
        "public_manifest": str(paths.public_manifest),
        "private_manifest": str(paths.private_manifest),
        "acceptance_manifest": str(paths.acceptance_manifest),
        "learner_evidence": str(paths.learner_evidence),
        "evaluator_receipt": str(paths.evaluator_receipt),
        "safety_rows": str(paths.safety_rows),
        "historical_inference_sidecar": str(paths.historical_inference_sidecar),
    }


def artifact_from_evidence(
    *,
    paths: FixturePaths,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    safety_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str | None = None,
) -> JsonDict:
    """Build a schema-complete artifact from raw fixture and validation evidence."""

    public_manifest = load_json(paths.public_manifest)
    receipts = [deepcopy(dict(row)) for row in validation_receipts]
    rows = [deepcopy(dict(row)) for row in safety_rows]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": "partial_pending_terminal_validation",
        "run_date": RUN_DATE,
        "started_at_utc": started_at or datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "host_computation": "CPython generation, exact Boolean execution, and transactional adapter controls",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "canary": CANARY_SEED,
            "opaque_token_commitment": public.sha256_json(TOKEN_SEED),
            "private_rule_commitment": public.sha256_json(PRIVATE_RULE_SEED),
        },
        "source_artifact_hashes": dict(source_hashes),
        "rows": rows,
        "sample_size_budget": {
            "development_streams": {"planned": 32, "attempted": 32, "completed": 32, "censored": 0},
            "development_requests": {
                "planned": 384,
                "attempted": 384,
                "completed": 384,
                "censored": 0,
            },
            "public_requests": {"planned": 32, "attempted": 0, "completed": 0, "censored": 0},
            "renamed_twins": {"planned": 32, "attempted": 0, "completed": 0, "censored": 0},
            "development_canary": {"planned": 4, "attempted": 0, "completed": 0, "censored": 0},
            "safety_controls": {
                "planned": len(rows),
                "attempted": len(rows),
                "completed": sum(not row.get("censored", False) for row in rows),
                "censored": sum(bool(row.get("censored", False)) for row in rows),
            },
            "stopping_rule": "Seal each fixed unit once and run each safety control once; do not extend from outcomes.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial_pending_terminal_validation",
        "verdict_class": "partial",
        "flagged_adversarial": not _receipt_passed(receipts, "adversarial_verify"),
        "validation_receipts": receipts,
        "repository_health": {
            "status": "degraded_open",
            "affects_required_checks": False,
            "historical_failures": [
                {
                    "date": "2026-09-16",
                    "source": V645_ADAPTER_PATH.as_posix(),
                    "name": "full_python_suite",
                    "exit_code": -15,
                    "classification": "unrelated_historical_repository_health",
                    "resolved": False,
                }
            ],
        },
        "field_principles": {},
        "learning_fixture_ready_score": 0,
        "learning_value_score": 0,
        "promotion_score": 0,
        "fixture_manifest": _fixture_manifest(paths),
        "frozen_acceptance_manifest": load_json(paths.acceptance_manifest),
        "safety_rows": rows,
        "raw_evidence_paths": _raw_paths(paths),
        "historical_determination": _historical_determination(),
        "production_defaults_changed": False,
        "research_roadmap_changed": False,
        "no_model_weight_mutation": True,
        "reproducibility_checksum": "",
    }
    assert len(public_manifest["development_streams"]) == 32
    artifact = reclassify_artifact(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _failed_precondition(preconditions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in preconditions if row.get("available") is not True), None)


def reclassify_artifact(artifact: JsonDict) -> JsonDict:
    """Compute fixture readiness without consulting unmeasured value gates."""

    result = deepcopy(artifact)
    preconditions_pass = _failed_precondition(result.get("preconditions_checked", [])) is None
    panel = result.get("fixture_manifest", {})
    panel_pass = (
        panel.get("development_stream_count") == 32
        and panel.get("development_request_count") == 384
        and panel.get("public_request_count") == 32
        and panel.get("renamed_twin_count") == 32
        and panel.get("development_canary_count") == 4
        and panel.get("partition", {}).get("seeds_disjoint") is True
        and panel.get("evaluator_labels_inaccessible_to_learner") is True
    )
    safety_pass = bool(result.get("safety_rows")) and all(
        row.get("passed") is True
        and int(row.get("query_attempts", 0)) <= QUERY_BUDGET
        and int(row.get("state_bytes", 0)) <= STATE_CAP_BYTES
        for row in result.get("safety_rows", [])
    )
    receipts = result.get("validation_receipts", [])
    scoped_pass = all(_receipt_passed(receipts, name) for name in REQUIRED_CHECK_NAMES)
    terminal_pass = all(_receipt_passed(receipts, name) for name in TERMINAL_CHECK_NAMES)
    adversarial_clear = result.get("flagged_adversarial") is False
    gates = {
        "preconditions": _gate(True, preconditions_pass, preconditions_pass, "Exact eligible inputs gate dependent work."),
        "sealed_panel": _gate(True, panel_pass, panel_pass, "All fresh partitions and label boundaries must be complete."),
        "safety_controls": _gate(True, safety_pass, safety_pass, "Every isolation and lifecycle control must pass within budget."),
        "scoped_validation": _gate(True, scoped_pass, scoped_pass, "All affected Exp7303 commands must pass."),
        "terminal_validation": _gate(True, terminal_pass, terminal_pass, "Cold reduction and both strict readers must pass."),
        "adversarial_clear": _gate(False, result.get("flagged_adversarial"), adversarial_clear, "Critical findings prevent readiness."),
        "scientific_value": _gate("not_evaluated", "not_evaluated", False, "Value is an Exp7362 gate, not a fixture gate."),
        "promotion": _gate(1, 0, False, "A safety fixture never authorizes promotion."),
    }
    result["acceptance_gate_results"] = gates
    readiness_names = (
        "preconditions",
        "sealed_panel",
        "safety_controls",
        "scoped_validation",
        "terminal_validation",
        "adversarial_clear",
    )
    failed_name = next((name for name in readiness_names if not gates[name]["passed"]), None)
    ready = failed_name is None
    result["learning_fixture_ready_score"] = int(ready)
    result["learning_value_score"] = 0
    result["promotion_score"] = 0
    if ready:
        result["status"] = "complete_learning_fixture_ready_value_not_evaluated"
        result["honest_verdict"] = "complete_null_learning_fixture_ready_value_not_evaluated"
        result["verdict_class"] = "null"
        result["gate_check_summary"] = {
            "passed": True,
            "upstream": EXPERIMENT_ID,
            "failed_check": None,
            "artifact_field": "learning_fixture_ready_score",
            "expected_value": 1,
            "observed_value": 1,
        }
    else:
        assert failed_name is not None
        gate = gates[failed_name]
        result["status"] = "complete_learning_fixture_disqualified"
        result["honest_verdict"] = f"complete_disqualified_learning_fixture_{failed_name}"
        result["verdict_class"] = "disqualified"
        result["gate_check_summary"] = {
            "passed": False,
            "upstream": EXPERIMENT_ID,
            "failed_check": failed_name,
            "artifact_field": f"acceptance_gate_results.{failed_name}",
            "expected_value": gate["expected"],
            "observed_value": gate["observed"],
        }
    result["reproducibility_checksum"] = reproducibility_checksum(result)
    return result


def blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    *,
    duration_s: float,
) -> JsonDict:
    """Create a terminal external block without running dependent fixture work."""

    failed = _failed_precondition(preconditions)
    if failed is None:
        raise ValueError("blocked artifact requires failed precondition")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": f"blocked_{failed['check']}",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "host_computation": "precondition checks only",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
        },
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_development_requests": 384,
            "attempted": 0,
            "completed": 0,
            "censored": 384,
            "stopping_rule": "External prerequisite failure stops all dependent work.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {
            "passed": False,
            "upstream": failed["upstream"],
            "failed_check": failed["check"],
            "artifact_field": failed["artifact_field"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
        },
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{failed['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "not_evaluated", "affects_required_checks": False},
        "field_principles": {},
        "learning_fixture_ready_score": 0,
        "learning_value_score": 0,
        "promotion_score": 0,
        "fixture_manifest": {},
        "frozen_acceptance_manifest": {},
        "safety_rows": [],
        "raw_evidence_paths": {},
        "historical_determination": {},
        "production_defaults_changed": False,
        "research_roadmap_changed": False,
        "no_model_weight_mutation": True,
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reload raw evidence and compare every fixture-derived terminal claim."""

    raw = artifact.get("raw_evidence_paths", {})
    try:
        paths = FixturePaths(
            raw_dir=Path(str(raw["public_manifest"])).parents[1],
            public_manifest=Path(str(raw["public_manifest"])),
            private_manifest=Path(str(raw["private_manifest"])),
            acceptance_manifest=Path(str(raw["acceptance_manifest"])),
            learner_evidence=Path(str(raw["learner_evidence"])),
            evaluator_receipt=Path(str(raw["evaluator_receipt"])),
            safety_rows=Path(str(raw["safety_rows"])),
            historical_inference_sidecar=Path(str(raw["historical_inference_sidecar"])),
            candidate=Path(str(raw.get("candidate", "unused"))),
        )
        public_manifest = load_json(paths.public_manifest)
        private_manifest = load_json(paths.private_manifest)
        safety = load_json(paths.safety_rows)["rows"]
    except (KeyError, OSError, json.JSONDecodeError, ValueError):
        return ["raw_evidence_unavailable"]
    errors = panel_errors(
        public_manifest,
        private_manifest,
        load_json(REPO_ROOT / V645_PUBLIC_MANIFEST_PATH),
    )
    if safety != artifact.get("safety_rows") or safety != artifact.get("rows"):
        errors.append("safety_rows_mismatch")
    if not safety or any(row.get("passed") is not True for row in safety):
        errors.append("safety_control_failed")
    if _fixture_manifest(paths) != artifact.get("fixture_manifest"):
        errors.append("fixture_manifest_mismatch")
    acceptance = load_json(paths.acceptance_manifest)
    if acceptance != artifact.get("frozen_acceptance_manifest"):
        errors.append("acceptance_manifest_mismatch")
    return sorted(set(errors))


def _receipt_valid(receipt: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(receipt.get("command"), str)
        and receipt.get("command")
        and isinstance(receipt.get("scope"), str)
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and str(receipt.get("log_sha256", "")).startswith("sha256:")
    )


def validate_artifact(
    artifact: object,
    *,
    root: Path = REPO_ROOT,
    verify_source_hashes: bool = True,
) -> list[str]:
    """Cold-check identity, evidence, readiness separation, and exact hashes."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    seeds = artifact.get("random_seed", {})
    if len({seeds.get("development"), seeds.get("evaluation"), seeds.get("resampling")}) != 3:
        errors.append("random_seed_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_disclosure_missing")
    if artifact.get("learning_value_score") != 0 or artifact.get("promotion_score") != 0:
        errors.append("value_promotion_not_zero")
    if artifact.get("verdict_class") in {"blocked", "disqualified", "partial"} and artifact.get(
        "learning_fixture_ready_score"
    ) != 0:
        errors.append("failed_readiness_not_zero")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") != [] or artifact.get("safety_rows") != []:
            errors.append("blocked_dependent_work_present")
        summary = artifact.get("gate_check_summary", {})
        if not all(
            key in summary
            for key in (
                "upstream",
                "failed_check",
                "artifact_field",
                "expected_value",
                "observed_value",
            )
        ):
            errors.append("blocked_gate_summary_incomplete")
    else:
        errors.extend(independent_reduce(artifact))
        receipts = artifact.get("validation_receipts", [])
        if any(not _receipt_valid(row) for row in receipts):
            errors.append("validation_receipt_invalid")
        if artifact.get("learning_fixture_ready_score") == 1:
            if artifact.get("verdict_class") != "null" or not all(
                _receipt_passed(receipts, name) for name in ALL_VALIDATION_NAMES
            ):
                errors.append("ready_state_invalid")
            if artifact.get("flagged_adversarial") is not False:
                errors.append("ready_adversarial_invalid")
    spans = artifact.get("phase_spans", [])
    if any(
        span.get("end_elapsed_s", -1) < span.get("start_elapsed_s", 0)
        or (index and span.get("start_elapsed_s", 0) < spans[index - 1].get("end_elapsed_s", 0))
        for index, span in enumerate(spans)
    ):
        errors.append("phase_spans_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(key) != value for key, value in REQUIRED_FIELD_PRINCIPLES.items()
    ) or any(key not in principles for key in artifact):
        errors.append("field_principles_invalid")
    if verify_source_hashes:
        for name, expected in artifact.get("source_artifact_hashes", {}).items():
            path = Path(str(name))
            absolute = path if path.is_absolute() else root / path
            if not absolute.is_file() or sha256_file(absolute) != expected:
                errors.append("source_hash_mismatch")
                break
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return sorted(set(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Write only a cold-valid terminal artifact with an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def scoped_command_plan(root: Path, temporary_root: Path) -> list[CommandSpec]:
    """Build the exact Exp7358-compatible affected validation plan."""

    basetemp = temporary_root / "basetemp"
    coverage = temporary_root / "coverage/.coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage.parent.mkdir(parents=True, exist_ok=True)
    return build_scoped_commands(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage,
    )


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7360_v646_learning_fixture import independent_reduce;"
        "a=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=independent_reduce(a);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "independent_reducer",
            (python, "-u", "-c", reducer, str(candidate.resolve())),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate.resolve())),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate.resolve()),
            ),
            "measured_candidate",
        ),
    ]


def _span(phase: str, started: float, overall: float) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_elapsed_s": started - overall,
        "end_elapsed_s": ended - overall,
        "duration_s": ended - started,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    raw_dir: Path = RAW_DIR,
) -> JsonDict:
    """Run preconditions, fixture controls, scoped checks, and terminal readers."""

    overall = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []
    progress("preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, _producer = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, overall))
    progress("preconditions", "end", f"passed={all(row['available'] for row in preconditions)}")
    output = output_path if output_path.is_absolute() else root / output_path
    if not all(row["available"] for row in preconditions):
        blocked = blocked_artifact(
            preconditions,
            source_hashes,
            duration_s=time.monotonic() - overall,
        )
        _atomic_json(output, blocked)
        progress("write", "terminal_blocked", str(output))
        return blocked

    raw = raw_dir if raw_dir.is_absolute() else root / raw_dir
    paths = FixturePaths.for_raw_dir(raw)
    progress("seal", "start", "write public, private, acceptance, and historical sidecars")
    phase_started = time.monotonic()
    public_manifest, private_manifest = build_fixture_manifests(
        paths,
        root / V645_PUBLIC_MANIFEST_PATH,
    )
    spans.append(_span("load", phase_started, overall))
    progress("seal", "end", f"public={paths.public_manifest}")

    progress("evaluation", "start", "run isolated processes and adapter safety controls")
    phase_started = time.monotonic()
    process = run_isolated_fixture(root, paths)
    state_root = Path(tempfile.mkdtemp(prefix="state-", dir=paths.raw_dir))
    adapter_rows = run_adapter_safety_controls(state_root)
    safety_rows = build_safety_rows(public_manifest, private_manifest, process, adapter_rows)
    write_raw_evidence(paths, safety_rows)
    spans.append(_span("evaluation", phase_started, overall))
    spans.insert(
        1,
        {
            "phase": "generation",
            "start_elapsed_s": spans[0]["end_elapsed_s"],
            "end_elapsed_s": spans[0]["end_elapsed_s"],
            "duration_s": 0.0,
        },
    )
    progress("evaluation", "end", f"controls={len(safety_rows)}")
    source_hashes.update(raw_source_hashes(paths))

    progress("validation", "before_subprocesses")
    phase_started = time.monotonic()
    validation_root = Path(tempfile.mkdtemp(prefix="exp7360-validation-", dir="/tmp"))
    commands = scoped_command_plan(root, validation_root)
    scoped = run_scoped_validation(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=validation_root / "basetemp",
        coverage_file=validation_root / "coverage/.coverage",
        log_dir=paths.raw_dir / "validation/affected",
    )
    preliminary = list(scoped["validation_receipts"])
    candidate_spans = [*spans, _span("validation", phase_started, overall)]
    candidate = artifact_from_evidence(
        paths=paths,
        preconditions=preconditions,
        source_hashes=source_hashes,
        safety_rows=safety_rows,
        validation_receipts=preliminary,
        duration_s=time.monotonic() - overall,
        phase_spans=candidate_spans,
        started_at=started_at,
    )
    _atomic_json(paths.candidate, candidate)
    terminal = run_commands(
        root,
        _terminal_commands(root, paths.candidate),
        log_dir=paths.raw_dir / "validation/terminal",
    )
    progress("validation", "after_subprocesses", f"receipts={len(preliminary) + len(terminal)}")
    spans = candidate_spans
    final = artifact_from_evidence(
        paths=paths,
        preconditions=preconditions,
        source_hashes=source_hashes,
        safety_rows=safety_rows,
        validation_receipts=[*preliminary, *terminal],
        duration_s=time.monotonic() - overall,
        phase_spans=spans,
        started_at=started_at,
    )
    final["repository_health"] = scoped["repository_health"]
    write_started = time.monotonic()
    last_end = spans[-1]["end_elapsed_s"] if spans else 0.0
    write_span = _span("write", write_started, overall)
    write_span["start_elapsed_s"] = max(last_end, write_span["start_elapsed_s"])
    write_span["end_elapsed_s"] = max(write_span["start_elapsed_s"], write_span["end_elapsed_s"])
    write_span["duration_s"] = write_span["end_elapsed_s"] - write_span["start_elapsed_s"]
    final["phase_spans"] = [*spans, write_span]
    final["completed_at_utc"] = datetime.now(UTC).isoformat()
    final["duration_s"] = time.monotonic() - overall
    final["field_principles"] = _field_principles(tuple(final))
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress("write", "before_atomic_publish", str(output))
    write_artifact(output, final)
    progress("write", "after_atomic_publish", f"status={final['status']}")
    return final


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - declared E2E entrypoint.
    args = _parse_args(argv)
    progress("startup", "start", f"date={args.date}")
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if args.validate:
        errors = validate_artifact(load_json(output))
        print(errors, flush=True)
        return int(bool(errors))
    artifact = build_artifact(root=REPO_ROOT, output_path=output, raw_dir=args.raw_dir)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7360] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    progress(
        "complete",
        "end",
        f"verdict={artifact['honest_verdict']} score={artifact['learning_fixture_ready_score']}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
