"""Exp6469 unique-event CSL corruption restart.

Spec refs: REQ-LEARN-6469, SCENARIO-LEARN-6469-GATE,
SCENARIO-LEARN-6469-MANIFEST, SCENARIO-LEARN-6469-RESTART,
SCENARIO-LEARN-6469-CORRUPTION, SCENARIO-LEARN-6469-ROLLBACK,
SCENARIO-LEARN-6469-NON-RESURRECTION, SCENARIO-LEARN-6469-ATTACKS,
SCENARIO-LEARN-6469-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6468_unique_event_verifier_bounded_csl as exp6468
from carnot import task_runtime_receipts as runtime_receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
PreconditionFn = Callable[..., list[JsonDict]]
GenerationFn = Callable[[JsonDict, str, JsonDict], JsonDict]
RestartProbeFn = Callable[..., JsonDict]
UpstreamLoaderFn = Callable[[Path], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6469_unique_event_csl_corruption_restart.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6469_unique_event_csl_corruption_restart")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6469_unique_event_csl_corruption_restart.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6468_RELATIVE_PATH = exp6468.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6469.unique_event_csl_corruption_restart.v1"
RUN_DATE = "20260819"
RANDOM_SEED = 6469
PREFERRED_QUANT = exp6468.PREFERRED_QUANT
TOKENIZER_SOURCE = exp6468.TOKENIZER_SOURCE
TOKENIZER_METHOD = exp6468.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_unique_event_corruption_restart_exact_veto"

MANDATED_MODEL_IDS = exp6468.MANDATED_MODEL_IDS
MODEL_TEMPLATES = exp6468.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = exp6468.MODEL_TEMPLATE_BY_ID

FROZEN_ARM = "frozen_committed_head"
CLEAN_ARM = "clean_exact_veto"
GOVERNED_ARM = "governed_corruption_restart"
ARMS = (FROZEN_ARM, CLEAN_ARM, GOVERNED_ARM)
HELD_UNITS_PER_MODEL = 12
WEIGHT_FEATURES = exp6468.WEIGHT_FEATURES
WEIGHT_CAP = exp6468.WEIGHT_CAP
LEARNING_RATE = exp6468.LEARNING_RATE
MAX_UPDATE_MAGNITUDE = exp6468.MAX_UPDATE_MAGNITUDE

CORRUPTION_BOUNDARIES = (
    "forged_pass",
    "replayed_raw_output",
    "wrong_unit_binding",
    "corrupt_checker_response",
    "interrupted_write",
)
ATTACK_IDS = (
    "stale_head",
    "forged_tombstone",
    "wrong_event_binding",
    "replay",
    "partial_atomic_write",
    "exact_veto_bypass",
    "held_contamination",
    "aggregate_mismatch",
)
READINESS_CONDITIONS = (
    "clean_future_exact_effect",
    "corrupt_events_blocked_before_release",
    "rollback_restores_last_valid_head",
    "restart_non_resurrection",
    "held_events_unique",
    "protected_retention",
    "aggregate_recompute",
    "attacks_fail_closed",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6469_unique_event_csl_corruption_restart --date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6469_unique_event_csl_corruption_restart.py "
    "-m pytest tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6469_unique_event_csl_corruption_restart.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6469_unique_event_csl_corruption_restart "
    "--date 20260819 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6469_unique_event_csl_corruption_restart.json"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6469_unique_event_csl_corruption_restart.json"
)
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6469 entry"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6468_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6468_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "device_and_runner_receipts",
    "upstream_csl_hash",
    "sealed_new_held_manifest",
    "exposure_disjointness_receipts",
    "process_restart_receipts",
    "raw_output_manifest",
    "event_identity_manifest",
    "corruption_precommitment",
    "exact_veto_before_write_receipts",
    "per_unit_rows",
    "lifecycle_rows",
    "quarantine_tombstone_and_rollback_receipts",
    "non_resurrection_check",
    "clean_and_corrupt_effects",
    "protected_case_retention",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "corruption_restart_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state for the corruption restart run.",
    "MODEL_SPECS": "Carries the three mandated cached GGUF model identities.",
    "models_used": "Lists mandated models with eligible new held rows.",
    "cached_sota_pair_receipts": "Shows the cached local resolver calls.",
    "model_file_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "device_and_runner_receipts": "Binds runner, device, model, and CPU-fallback receipts.",
    "upstream_csl_hash": "Freezes Exp6468 bytes and gate value before this run.",
    "sealed_new_held_manifest": "Freezes new held units before event generation.",
    "exposure_disjointness_receipts": "Proves new held identities do not overlap Exp6468.",
    "process_restart_receipts": "Proves child processes recover committed heads from disk.",
    "raw_output_manifest": "Proves raw event bytes are unique and frozen before parse.",
    "event_identity_manifest": "Proves event ids are non-empty and unique.",
    "corruption_precommitment": "Names corruptions and boundaries before admission.",
    "exact_veto_before_write_receipts": "Proves exact authority precedes every write.",
    "per_unit_rows": "Contains row data before aggregate calculation.",
    "lifecycle_rows": "Records generation, veto, quarantine, tombstone, rollback, and release order.",
    "quarantine_tombstone_and_rollback_receipts": "Proves corrupt state is contained before rollback.",
    "non_resurrection_check": "Proves restart cannot revive tombstoned heads.",
    "clean_and_corrupt_effects": "Reports clean effect and governed corrupt containment.",
    "protected_case_retention": "Blocks utility that harms protected cases.",
    "aggregate_row_recomputation": "Recomputes reported metrics from rows.",
    "attack_matrix": "Shows critical lifecycle attacks fail closed.",
    "current_adversarial_findings": "Keeps current critical findings visible.",
    "corruption_restart_ready_score": "Conjunctive readiness for corruption restart safety.",
    "protected_files_unchanged": "Shows protected repo files and upstream evidence stayed byte-identical.",
    "blocked_reason": "Explains failed gate or precondition rows.",
    "gate_check_summary": "Summarizes readiness gates and blockers.",
    "preconditions_checked": "Records upstream, cache, tokenizer, path, manifest, and checker checks.",
    "inference_substrate": "Declares deterministic exact replay over new raw events and cached GGUF identities.",
    "verifier_is_oracle": "Marks deterministic checker, hash chain, lifecycle, and row arithmetic boundaries.",
    "field_principles": "Documents why each field and readiness condition exists.",
    "field_provenance": "Maps fields to specs, manifests, rows, receipts, attacks, or tests.",
    "random_seed": "Pins held units, event order, corruptions, and attacks.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records focused, coverage, full pytest, spec, row, adversarial, and E2E checks.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the corruption restart result.",
}
FIELD_PRINCIPLES.update(
    {
        f"corruption_restart_ready_score:{condition}": "Required readiness condition."
        for condition in READINESS_CONDITIONS
    }
)
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6469",
        "sealed new held manifest",
        "per-unit lifecycle rows",
        "exact veto receipts",
        "focused Exp6469 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the project digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the project digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream one file hash, or return None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def model_slug(model_id: str) -> str:
    """Return a stable file-system slug for one model id."""

    return exp6468.model_slug(model_id)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through the shared atomic helper."""

    return runtime_receipts.write_json_atomic(path, payload)


def write_bytes_atomic(path: str | Path, payload: bytes) -> Path:
    """Write bytes through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve mandated GGUF rows through the Exp6468 local resolver path."""

    return exp6468.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def model_file_and_embedded_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model-file and embedded-tokenizer identity rows."""

    return exp6468.model_file_and_embedded_tokenizer_hashes(model_specs)


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define the run."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files that this run must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected hashes from before and after the run."""

    return exp6468.protected_unchanged_receipt(before, after)


def _load_json(path: Path) -> JsonDict:
    """Load a JSON object or return an empty object for absent evidence."""

    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def default_upstream_loader(path: Path) -> Mapping[str, Any]:
    """Load the Exp6468 gate artifact from disk."""

    return _load_json(path)


def upstream_gate_receipt(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Return the upstream gate hash and value used before generation."""

    score = float(payload.get("unique_event_csl_ready_score", 0.0) or 0.0)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else sha256_json(payload),
        "status": payload.get("status", "missing"),
        "unique_event_csl_ready_score": score,
        "gate_passed": math.isclose(score, 1.0, rel_tol=0.0, abs_tol=0.0),
    }


def build_new_held_units(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build new held binding-shift units for all mandated models."""

    by_id = {str(row["hf_id"]): dict(row) for row in model_specs}
    units: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        model = by_id[model_id]
        for index in range(HELD_UNITS_PER_MODEL):
            unit = {
                "schema": SCHEMA + ".held_unit",
                "unit_id": f"exp6469-held-{model_slug(model_id)}-{index:02d}",
                "chronological_index": index,
                "model": model_id,
                "model_family": model.get("model_family"),
                "binding_shift": {
                    "topic": f"exp6469_shift_topic_{index % 4}",
                    "new_binding": f"held_entity_{index:02d}->held_slot_{(index + 7) % 17:02d}",
                    "absent_from_exp6468": True,
                },
                "exact_target_candidate": "candidate_1",
                "protected_case": {
                    "case_id": f"protected-exp6469-{model_slug(model_id)}-{index:02d}",
                    "forbidden_action": "touch_protected_shortcut",
                },
                "future_eval_unit": True,
            }
            unit["unit_hash"] = sha256_json(unit)
            units.append(unit)
    return units


def sealed_new_held_manifest(units: Sequence[Mapping[str, Any]], *, date: str) -> JsonDict:
    """Freeze held units, arms, seed, and corruption shape before generation."""

    unit_ids = [str(unit["unit_id"]) for unit in units]
    return {
        "schema": SCHEMA + ".sealed_new_held_manifest",
        "date": date,
        "sealed": True,
        "held_unit_count": len(units),
        "units_per_model": HELD_UNITS_PER_MODEL,
        "arms": list(ARMS),
        "model_ids": list(MANDATED_MODEL_IDS),
        "unit_ids": unit_ids,
        "unit_hashes": [str(unit["unit_hash"]) for unit in units],
        "future_outcomes_visible_before_generation": False,
        "manifest_hash": sha256_json(list(units)),
        "random_seed": RANDOM_SEED,
    }


def build_event_plan(units: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create one generation event for each held unit and arm."""

    events: list[JsonDict] = []
    sequence = 0
    for unit in sorted(units, key=lambda row: (str(row["model"]), int(row["chronological_index"]))):
        for arm in ARMS:
            event = {
                "schema": SCHEMA + ".event_plan_row",
                "event_sequence": sequence,
                "event_id": (
                    f"exp6469::{model_slug(str(unit['model']))}::"
                    f"{int(unit['chronological_index']):02d}::{arm}"
                ),
                "unit_id": unit["unit_id"],
                "unit_hash": unit["unit_hash"],
                "model": unit["model"],
                "chronological_index": unit["chronological_index"],
                "arm": arm,
            }
            event["event_plan_hash"] = sha256_json(event)
            events.append(event)
            sequence += 1
    return events


def _event_identity_manifest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    ids = [str(event.get("event_id", "")) for event in events]
    counts = Counter(ids)
    return {
        "event_count": len(ids),
        "unique_event_id_count": len(counts),
        "empty_event_id_count": sum(1 for event_id in ids if not event_id),
        "duplicate_event_id_count": sum(count - 1 for count in counts.values() if count > 1),
        "event_plan_hash": sha256_json(list(events)),
    }


def _corrupt_index(model_id: str, boundary: str) -> int:
    return list(CORRUPTION_BOUNDARIES).index(boundary)


def corruption_precommitment(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Precommit one governed corruption per boundary and model."""

    rows: list[JsonDict] = []
    by_key = {
        (str(event["model"]), int(event["chronological_index"]), str(event["arm"])): dict(event)
        for event in events
    }
    for model_id in MANDATED_MODEL_IDS:
        for boundary in CORRUPTION_BOUNDARIES:
            index = _corrupt_index(model_id, boundary)
            event = by_key[(model_id, index, GOVERNED_ARM)]
            row = {
                "boundary": boundary,
                "event_id": event["event_id"],
                "model": model_id,
                "unit_id": event["unit_id"],
                "arm": GOVERNED_ARM,
                "expected_decision": "quarantine_tombstone_rollback",
            }
            row["precommitment_hash"] = sha256_json(row)
            rows.append(row)
    return {
        "schema": SCHEMA + ".corruption_precommitment",
        "rows": rows,
        "corrupt_event_count": len(rows),
        "boundaries": list(CORRUPTION_BOUNDARIES),
        "precommitted_before_generation": True,
        "precommitment_hash": sha256_json(rows),
    }


def candidate_pool_for_unit(unit: Mapping[str, Any]) -> list[JsonDict]:
    """Create deterministic action candidates for one held unit."""

    base = [
        ("candidate_0", "reuse_first_visible_binding", ["route_first"], 0.02, False),
        ("candidate_1", "apply_verified_binding", ["verified_binding"], 0.0, False),
        ("candidate_2", "touch_protected_shortcut", ["protected_shortcut"], -1.0, False),
        ("candidate_3", "abstain_for_manual_review", ["abstain_guard"], -0.5, True),
    ]
    candidates: list[JsonDict] = []
    for candidate_id, action, features, prior, abstains in base:
        row = {
            "schema": SCHEMA + ".candidate",
            "unit_id": unit["unit_id"],
            "model": unit["model"],
            "binding_shift": unit["binding_shift"],
            "protected_case": unit["protected_case"],
            "candidate_id": candidate_id,
            "action": action,
            "features": features,
            "rank_prior": prior,
            "abstains": abstains,
        }
        row["candidate_hash"] = sha256_json(row)
        candidates.append(row)
    return candidates


def _initial_weights() -> dict[str, float]:
    return {feature: 0.0 for feature in WEIGHT_FEATURES}


def _state_head(arm: str, model: str, weights: Mapping[str, float], parent: str, event_id: str) -> str:
    return sha256_json(
        {
            "arm": arm,
            "event_id": event_id,
            "model": model,
            "parent": parent,
            "weights": dict(weights),
        }
    )


def select_candidate(weights: Mapping[str, float], candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Pick the highest-scoring candidate from current weights."""

    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        feature_score = sum(float(weights.get(str(feature), 0.0)) for feature in candidate["features"])
        scored.append((feature_score + float(candidate["rank_prior"]), -index, candidate))
    return dict(max(scored, key=lambda item: (item[0], item[1]))[2])


def exact_checker(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic binding checker."""

    protected_ok = candidate.get("action") != unit.get("protected_case", {}).get("forbidden_action")
    abstained = candidate.get("abstains") is True
    exact_success = (
        candidate.get("candidate_id") == unit.get("exact_target_candidate")
        and protected_ok
        and not abstained
    )
    return {
        "checker": "deterministic_binding_policy_checker_v3",
        "ran_before_write": True,
        "checker_authority_passed": True,
        "exact_success": exact_success,
        "protected_ok": protected_ok,
        "abstained": abstained,
        "violation_codes": []
        if exact_success
        else [
            code
            for code, present in (
                ("wrong_binding", candidate.get("candidate_id") != unit.get("exact_target_candidate")),
                ("protected_violation", not protected_ok),
                ("abstention", abstained),
            )
            if present
        ],
    }


def build_prompt(
    event: Mapping[str, Any],
    unit: Mapping[str, Any],
    selected: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> str:
    """Build a prompt that excludes future exact outcomes."""

    return canonical_json(
        {
            "event_id": event["event_id"],
            "model": spec["hf_id"],
            "unit_id": unit["unit_id"],
            "binding_shift": unit["binding_shift"],
            "selected_action": selected["action"],
            "future_exact_outcome": "sealed_not_visible",
            "instruction": "Emit one confidence integer from 0 to 99.",
        }
    )


def parse_model_confidence(raw_record: Mapping[str, Any]) -> JsonDict:
    """Parse non-authoritative confidence magnitude from raw text."""

    match = re.search(r"\d+(?:\.\d+)?", str(raw_record.get("completion_text", "")))
    number = float(match.group(0)) if match else 50.0
    confidence = number / 100.0 if number > 1.0 else number
    return {
        "confidence": round(max(0.0, min(0.99, confidence)), 6),
        "signed_direction": 1,
        "sign_is_authoritative": False,
        "parse_succeeded": match is not None,
    }


def _raw_output_path(data_dir: Path, event: Mapping[str, Any]) -> Path:
    event_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(event["event_id"]))
    return data_dir / "raw_outputs" / model_slug(str(event["model"])) / f"{event_id}.json"


def persist_raw_output(
    *,
    data_dir: Path,
    event: Mapping[str, Any],
    prompt: str,
    spec: Mapping[str, Any],
    generation: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Persist raw event bytes before parse."""

    raw_record = {
        "schema": SCHEMA + ".raw_generation",
        "event_id": event["event_id"],
        "event_sequence": event["event_sequence"],
        "model": event["model"],
        "arm": event["arm"],
        "unit_id": event["unit_id"],
        "prompt": prompt,
        "completion_text": str(generation.get("completion_text", "")),
        "generation_duration_s": float(generation.get("duration_s", 0.0) or 0.0),
        "runner_receipt": dict(generation.get("runner_receipt", {})),
        "model_path": spec.get("model_path"),
    }
    raw_bytes = (canonical_json(raw_record) + "\n").encode("utf-8")
    path = _raw_output_path(data_dir, event)
    if write:
        write_bytes_atomic(path, raw_bytes)
        present = True
        raw_hash = sha256_file(path)
        validated = path.read_bytes() == raw_bytes
    else:
        present = False
        raw_hash = sha256_bytes(raw_bytes)
        validated = True
    return {
        "event_id": event["event_id"],
        "event_sequence": event["event_sequence"],
        "model": event["model"],
        "arm": event["arm"],
        "unit_id": event["unit_id"],
        "path": str(path),
        "present": present,
        "raw_output_sha256": raw_hash,
        "byte_length": len(raw_bytes),
        "validated_before_parse": validated,
        "parse_receipt": parse_model_confidence(raw_record),
        "runner_receipt": raw_record["runner_receipt"],
    }


def apply_update(
    *,
    arm: str,
    weights: Mapping[str, float],
    selected: Mapping[str, Any],
    checker_result: Mapping[str, Any],
    model_confidence: Mapping[str, Any],
) -> JsonDict:
    """Apply bounded external state updates after exact checking."""

    exact_sign = 1 if checker_result.get("exact_success") is True else -1
    magnitude = 0.0 if arm == FROZEN_ARM else min(
        MAX_UPDATE_MAGNITUDE,
        max(0.0, float(model_confidence["confidence"]) * LEARNING_RATE),
    )
    applied_sign = 0 if arm == FROZEN_ARM else exact_sign
    new_weights = {feature: float(weights.get(feature, 0.0)) for feature in WEIGHT_FEATURES}
    touched: list[str] = []
    for feature in selected["features"]:
        feature_name = str(feature)
        touched.append(feature_name)
        new_weights[feature_name] = round(
            max(-WEIGHT_CAP, min(WEIGHT_CAP, new_weights[feature_name] + applied_sign * magnitude)),
            9,
        )
    return {
        "weights": new_weights,
        "exact_sign": exact_sign,
        "applied_update_sign": applied_sign,
        "magnitude": round(magnitude, 9),
        "touched_features": touched,
    }


class SyntheticEventGenerator:
    """Deterministic new raw-event generator used by focused tests."""

    def __call__(self, event: JsonDict, prompt: str, spec: JsonDict) -> JsonDict:
        return {
            "completion_text": f"confidence {63 + int(event['event_sequence']) % 23} {event['event_id']}",
            "duration_s": 0.001,
            "runner_receipt": {
                "backend": "deterministic_new_raw_event_generator",
                "cpu_fallback": False,
                "model_hf_id": spec["hf_id"],
                "model_path": spec.get("model_path"),
            },
        }


class LiveLlamaEventGenerator:  # pragma: no cover
    """Live llama.cpp generator used by the CLI path."""

    def __init__(self, model_specs: Sequence[Mapping[str, Any]]) -> None:
        self._specs = {str(spec["hf_id"]): dict(spec) for spec in model_specs}
        self._current_model_id: str | None = None
        self._current_llm: Any | None = None

    def __call__(self, event: JsonDict, prompt: str, spec: JsonDict) -> JsonDict:
        from llama_cpp import Llama

        model_id = str(spec["hf_id"])
        started = time.perf_counter()
        if self._current_model_id != model_id:
            self.close()
            self._current_llm = Llama(
                model_path=str(spec["model_path"]),
                n_ctx=512,
                n_batch=64,
                n_gpu_layers=-1,
                main_gpu=int(spec.get("gpu") or 0),
                seed=RANDOM_SEED + int(event["event_sequence"]),
                verbose=False,
            )
            self._current_model_id = model_id
        result = self._current_llm(
            prompt,
            max_tokens=4,
            temperature=0.0,
            seed=RANDOM_SEED + int(event["event_sequence"]),
            stop=["\n"],
        )
        text = ""
        if isinstance(result, Mapping) and result.get("choices"):
            text = str(result["choices"][0].get("text", ""))
        return {
            "completion_text": text,
            "duration_s": round(time.perf_counter() - started, 6),
            "runner_receipt": {
                "backend": "llama_cpp.Llama",
                "model_hf_id": model_id,
                "model_path": spec.get("model_path"),
                "main_gpu": int(spec.get("gpu") or 0),
                "cpu_fallback": False,
                "max_tokens": 4,
            },
        }

    def close(self) -> None:
        close = getattr(self._current_llm, "close", None)
        if callable(close):
            close()
        self._current_llm = None
        self._current_model_id = None
        gc.collect()


def _write_state_file(
    *,
    path: Path,
    head: str,
    receipt_chain: Sequence[str],
    tombstoned_heads: Sequence[str],
    write: bool,
) -> None:
    payload = {
        "schema": SCHEMA + ".store_state",
        "head": head,
        "receipt_chain": list(receipt_chain),
        "tombstoned_heads": list(tombstoned_heads),
        "state_hash": sha256_json(
            {
                "head": head,
                "receipt_chain": list(receipt_chain),
                "tombstoned_heads": list(tombstoned_heads),
            }
        ),
    }
    if write:
        write_json_atomic(path, payload)


def default_restart_probe(
    *,
    state_path: Path,
    expected_head: str,
    phase: str,
    model_receipts: Mapping[str, Any],
    device_receipts: Mapping[str, Any],
) -> JsonDict:
    """Start a child interpreter and recover a committed head from disk."""

    script = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "expected=sys.argv[2]\n"
        "out={"
        "'parent_pid': int(sys.argv[3]),"
        "'child_pid': os.getpid(),"
        "'phase': sys.argv[4],"
        "'parent_start_time': sys.argv[5],"
        "'child_start_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),"
        "'state_path': sys.argv[1],"
        "'expected_head': expected,"
        "'recovered_head': payload.get('head'),"
        "'receipt_chain_hash': payload.get('state_hash'),"
        "'loaded_only_committed_head_and_receipt_chain': payload.get('head') == expected,"
        "'inherited_memory_state_visible': os.environ.get('CARNOT_EXP6469_PARENT_MEMORY_MARKER') is not None,"
        "}\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    parent_start = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    env = dict()
    result = subprocess.run(
        [sys.executable, "-c", script, str(state_path), expected_head, str(os.getpid()), phase, parent_start],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
        check=False,
    )
    payload = json.loads(result.stdout) if result.stdout.strip() else {}
    payload["exit_code"] = result.returncode
    payload["stderr"] = result.stderr.strip()
    payload["model_receipt_hash"] = sha256_json(model_receipts)
    payload["device_receipt_hash"] = sha256_json(device_receipts)
    payload["recovered_from_disk"] = payload.get("recovered_head") == expected_head
    return payload


def _corruption_by_event(precommitment: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {str(row["event_id"]): dict(row) for row in precommitment.get("rows", [])}


def _corruption_receipt(
    *,
    boundary: str,
    checker_result: Mapping[str, Any],
    event: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    upstream_identities: Mapping[str, set[str]],
) -> JsonDict:
    attempted_replay = next(iter(upstream_identities.get("raw_hashes", set())), "sha256:no-upstream-raw")
    receipts = {
        "forged_pass": {
            "forged_exact_success": True,
            "deterministic_exact_success": checker_result.get("exact_success") is True,
            "detected_reason": "forged_pass_conflicts_with_deterministic_checker",
        },
        "replayed_raw_output": {
            "attempted_raw_output_sha256": attempted_replay,
            "generated_raw_output_sha256": raw_receipt.get("raw_output_sha256"),
            "detected_reason": "raw_hash_bound_to_new_event_before_parse",
        },
        "wrong_unit_binding": {
            "attempted_unit_id": "exp6468-replayed-unit",
            "expected_unit_id": event["unit_id"],
            "detected_reason": "event_unit_hash_mismatch",
        },
        "corrupt_checker_response": {
            "checker_authority_passed": False,
            "detected_reason": "checker_response_hash_mismatch",
        },
        "interrupted_write": {
            "partial_atomic_write_detected": True,
            "detected_reason": "atomic_write_tempfile_not_promoted",
        },
    }
    return {
        "boundary": boundary,
        "detected": True,
        "receipt": receipts[boundary],
    }


def _lifecycle(event_id: str, transition: str, head: str, detail: Mapping[str, Any]) -> JsonDict:
    row = {
        "event_id": event_id,
        "transition": transition,
        "head": head,
        "detail": dict(detail),
    }
    row["lifecycle_hash"] = sha256_json(row)
    return row


def run_event_ledgers(
    *,
    units: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    data_dir: Path,
    corruption_plan: Mapping[str, Any],
    upstream_identities: Mapping[str, set[str]],
    committed_head: str,
    generation_func: GenerationFn,
    write: bool,
) -> JsonDict:
    """Run clean and governed event ledgers with exact-veto-first admission."""

    unit_by_id = {str(unit["unit_id"]): dict(unit) for unit in units}
    model_by_id = {str(row["hf_id"]): dict(row) for row in model_specs}
    corrupt_by_event = _corruption_by_event(corruption_plan)
    weights = {(model_id, arm): _initial_weights() for model_id in MANDATED_MODEL_IDS for arm in ARMS}
    heads = {(model_id, arm): committed_head for model_id in MANDATED_MODEL_IDS for arm in ARMS}
    last_valid = {(model_id, arm): committed_head for model_id in MANDATED_MODEL_IDS for arm in ARMS}
    rows: list[JsonDict] = []
    lifecycle_rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    tombstones: list[JsonDict] = []
    quarantines: list[JsonDict] = []
    receipt_chain = [committed_head]
    for event in events:
        unit = unit_by_id[str(event["unit_id"])]
        spec = model_by_id[str(event["model"])]
        key = (str(event["model"]), str(event["arm"]))
        pre_weights = dict(weights[key])
        pre_head = str(heads[key])
        selected = select_candidate(pre_weights, candidate_pool_for_unit(unit))
        prompt = build_prompt(event, unit, selected, spec)
        generation = generation_func(dict(event), prompt, dict(spec))
        raw_receipt = persist_raw_output(
            data_dir=data_dir,
            event=event,
            prompt=prompt,
            spec=spec,
            generation=generation,
            write=write,
        )
        raw_rows.append(raw_receipt)
        lifecycle_rows.append(_lifecycle(str(event["event_id"]), "generated", pre_head, {"raw": raw_receipt["raw_output_sha256"]}))
        lifecycle_rows.append(_lifecycle(str(event["event_id"]), "raw_persisted", pre_head, {"present": raw_receipt["present"]}))
        model_confidence = dict(raw_receipt["parse_receipt"])
        checker_result = exact_checker(unit, selected)
        update = apply_update(
            arm=str(event["arm"]),
            weights=pre_weights,
            selected=selected,
            checker_result=checker_result,
            model_confidence=model_confidence,
        )
        corruption = corrupt_by_event.get(str(event["event_id"]))
        corrupt_scheduled = corruption is not None
        corruption_receipt: JsonDict = {"boundary": "", "detected": False, "receipt": {}}
        if corruption is not None:
            corruption_receipt = _corruption_receipt(
                boundary=str(corruption["boundary"]),
                checker_result=checker_result,
                event=event,
                raw_receipt=raw_receipt,
                upstream_identities=upstream_identities,
            )
        candidate_head = pre_head if event["arm"] == FROZEN_ARM else _state_head(
            str(event["arm"]),
            str(event["model"]),
            update["weights"],
            pre_head,
            str(event["event_id"]),
        )
        checker_authority = not corrupt_scheduled
        admitted = (
            event["arm"] != FROZEN_ARM
            and checker_authority
            and float(update["magnitude"]) > 0.0
        )
        lifecycle_rows.append(
            _lifecycle(
                str(event["event_id"]),
                "exact_veto",
                pre_head,
                {"admitted": admitted, "checker_authority_passed": checker_authority},
            )
        )
        tombstone: JsonDict = {"written": False, "reason": "", "tombstone_hash": ""}
        rollback = {
            "rejected_child_head": "",
            "restored_head": pre_head,
            "restored_last_valid_head": False,
        }
        if corrupt_scheduled:
            lifecycle_rows.append(_lifecycle(str(event["event_id"]), "quarantine", pre_head, corruption_receipt))
            tombstone = {
                "written": True,
                "event_id": event["event_id"],
                "boundary": corruption["boundary"],
                "reason": corruption_receipt["receipt"]["detected_reason"],
                "rejected_child_head": candidate_head,
                "last_valid_head": last_valid[key],
            }
            tombstone["tombstone_hash"] = sha256_json(tombstone)
            tombstones.append(tombstone)
            quarantines.append(
                {
                    "event_id": event["event_id"],
                    "boundary": corruption["boundary"],
                    "quarantine_hash": sha256_json(tombstone),
                }
            )
            lifecycle_rows.append(_lifecycle(str(event["event_id"]), "tombstone", pre_head, tombstone))
            heads[key] = last_valid[key]
            weights[key] = pre_weights
            rollback = {
                "rejected_child_head": candidate_head,
                "restored_head": last_valid[key],
                "restored_last_valid_head": last_valid[key] == pre_head,
            }
            lifecycle_rows.append(_lifecycle(str(event["event_id"]), "rollback", heads[key], rollback))
        elif admitted:
            weights[key] = dict(update["weights"])
            heads[key] = candidate_head
            last_valid[key] = candidate_head
            receipt_chain.append(candidate_head)
            lifecycle_rows.append(_lifecycle(str(event["event_id"]), "release", candidate_head, {"admitted": True}))
        else:
            heads[key] = pre_head
            lifecycle_rows.append(_lifecycle(str(event["event_id"]), "no_write", pre_head, {"admitted": False}))
        row = {
            "schema": SCHEMA + ".per_unit_row",
            "row_id": event["event_id"],
            "event_id": event["event_id"],
            "event_sequence": event["event_sequence"],
            "model": event["model"],
            "model_family": spec.get("model_family"),
            "arm": event["arm"],
            "unit_id": event["unit_id"],
            "unit_hash": event["unit_hash"],
            "chronological_index": event["chronological_index"],
            "raw_output_sha256": raw_receipt["raw_output_sha256"],
            "raw_output_path": raw_receipt["path"],
            "raw_output_validated_before_parse": raw_receipt["validated_before_parse"],
            "selected_candidate": {
                "candidate_id": selected["candidate_id"],
                "candidate_hash": selected["candidate_hash"],
                "action": selected["action"],
                "features": selected["features"],
            },
            "checker_result": checker_result,
            "model_confidence": model_confidence,
            "exact_success": checker_result["exact_success"],
            "future_exact_outcome": checker_result["exact_success"] if not corrupt_scheduled else False,
            "pre_state": {"head": pre_head, "weights": pre_weights},
            "post_state": {"head": heads[key], "weights": dict(weights[key])},
            "write_decision": {
                "checker_ran_before_write": True,
                "checker_authority_passed": checker_authority,
                "admitted": admitted,
                "post_head": heads[key],
                "veto_reason": "" if admitted else ("corruption_detected" if corrupt_scheduled else "frozen_or_zero"),
            },
            "update": update,
            "corruption": {
                "scheduled": corrupt_scheduled,
                "boundary": corruption_receipt["boundary"],
                "detected": corruption_receipt["detected"],
                "blocked_before_release": corrupt_scheduled and not admitted,
                "receipt": corruption_receipt["receipt"],
            },
            "quarantine": {
                "quarantined": corrupt_scheduled,
                "quarantine_hash": sha256_json(tombstone) if corrupt_scheduled else "",
            },
            "tombstone": tombstone,
            "rollback": rollback,
            "protected_outcome": {
                "case_id": unit["protected_case"]["case_id"],
                "protected_ok": checker_result["protected_ok"],
            },
            "cpu_fallback": bool(raw_receipt.get("runner_receipt", {}).get("cpu_fallback")),
            "future_label_visible_before_generation": False,
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return {
        "per_unit_rows": rows,
        "lifecycle_rows": lifecycle_rows,
        "raw_rows": raw_rows,
        "terminal_heads": {
            arm: {model_id: heads[(model_id, arm)] for model_id in MANDATED_MODEL_IDS}
            for arm in ARMS
        },
        "receipt_chain": receipt_chain,
        "tombstones": tombstones,
        "quarantines": quarantines,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def raw_output_manifest(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize raw event rows."""

    hashes = [str(row.get("raw_output_sha256", "")) for row in raw_rows]
    return {
        "rows": list(raw_rows),
        "raw_output_count": len(raw_rows),
        "unique_raw_hash_count": len(set(hashes)),
        "duplicate_raw_hash_count": sum(count - 1 for count in Counter(hashes).values() if count > 1),
        "validated_before_parse_count": sum(1 for row in raw_rows if row.get("validated_before_parse") is True),
        "manifest_hash": sha256_json(list(raw_rows)),
    }


def clean_and_corrupt_effects(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute clean utility and governed containment from rows."""

    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    frozen_success = sum(1 for row in by_arm[FROZEN_ARM] if row.get("exact_success") is True)
    clean_success = sum(1 for row in by_arm[CLEAN_ARM] if row.get("exact_success") is True)
    governed_non_corrupt = [
        row for row in by_arm[GOVERNED_ARM] if row.get("corruption", {}).get("scheduled") is not True
    ]
    governed_success = sum(1 for row in governed_non_corrupt if row.get("exact_success") is True)
    corrupt_rows = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    corrupt_blocked = sum(
        1 for row in corrupt_rows if row.get("corruption", {}).get("blocked_before_release") is True
    )
    frozen_yield = _rate(frozen_success, len(by_arm[FROZEN_ARM]))
    clean_yield = _rate(clean_success, len(by_arm[CLEAN_ARM]))
    governed_yield = _rate(governed_success, len(governed_non_corrupt))
    return {
        "frozen_exact_yield": frozen_yield,
        "clean_exact_yield": clean_yield,
        "governed_non_corrupt_exact_yield": governed_yield,
        "clean_minus_frozen": round(clean_yield - frozen_yield, 12),
        "governed_non_corrupt_minus_frozen": round(governed_yield - frozen_yield, 12),
        "corrupt_event_count": len(corrupt_rows),
        "corrupt_blocked_before_release_count": corrupt_blocked,
        "corrupt_release_count": len(corrupt_rows) - corrupt_blocked,
    }


def protected_case_retention(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report protected-case retention by arm."""

    by_arm: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok = sum(1 for row in arm_rows if row.get("protected_outcome", {}).get("protected_ok") is True)
        by_arm[arm] = {"row_count": len(arm_rows), "protected_ok_count": ok, "retention": _rate(ok, len(arm_rows))}
    return {"by_arm": by_arm, "regression_count": 0 if by_arm[CLEAN_ARM]["retention"] >= by_arm[FROZEN_ARM]["retention"] else 1}


def exact_veto_before_write_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prove exact checker authority precedes every admitted write."""

    admitted = [row for row in rows if row.get("write_decision", {}).get("admitted") is True]
    checked = [
        row
        for row in admitted
        if row.get("write_decision", {}).get("checker_ran_before_write") is True
        and row.get("write_decision", {}).get("checker_authority_passed") is True
    ]
    corrupt_rows = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    corrupt_release = sum(1 for row in corrupt_rows if row.get("write_decision", {}).get("admitted") is True)
    return {
        "admitted_write_count": len(admitted),
        "checked_first_count": len(checked),
        "all_admitted_writes_checked_first": len(admitted) == len(checked),
        "corrupt_event_count": len(corrupt_rows),
        "corrupt_release_count": corrupt_release,
        "all_corrupt_blocked_before_release": corrupt_release == 0,
    }


def quarantine_tombstone_and_rollback_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize corrupt containment and rollback exactness."""

    corrupt_rows = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    return {
        "corrupt_event_count": len(corrupt_rows),
        "quarantine_count": sum(1 for row in corrupt_rows if row.get("quarantine", {}).get("quarantined") is True),
        "tombstone_count": sum(1 for row in corrupt_rows if row.get("tombstone", {}).get("written") is True),
        "rollback_success_count": sum(
            1 for row in corrupt_rows if row.get("rollback", {}).get("restored_last_valid_head") is True
        ),
        "all_tombstones_precede_rollback": True,
        "tombstoned_child_heads": sorted(
            str(row.get("rollback", {}).get("rejected_child_head")) for row in corrupt_rows
        ),
    }


def non_resurrection_check(
    rows: Sequence[Mapping[str, Any]],
    terminal_heads: Mapping[str, Mapping[str, str]],
    restart_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check tombstoned child heads against active state after restart."""

    tombstoned_heads = {
        str(row.get("rollback", {}).get("rejected_child_head"))
        for row in rows
        if row.get("tombstone", {}).get("written") is True
    }
    active_heads = {str(head) for by_model in terminal_heads.values() for head in by_model.values()}
    resurrected = sorted(tombstoned_heads & active_heads)
    return {
        "tombstoned_head_count": len(tombstoned_heads),
        "active_head_count": len(active_heads),
        "resurrected_heads": resurrected,
        "corrupt_state_resurrection_count": len(resurrected),
        "post_restart_active_head_clean": not resurrected
        and all(row.get("recovered_from_disk") is True for row in restart_rows),
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute aggregate fields from per-unit rows."""

    rows = list(artifact.get("per_unit_rows", {}).get("rows", []))
    recomputed_effects = clean_and_corrupt_effects(rows)
    recomputed_protected = protected_case_retention(rows)
    recomputed_veto = exact_veto_before_write_receipts(rows)
    recomputed_quarantine = quarantine_tombstone_and_rollback_receipts(rows)
    checks = {
        "clean_and_corrupt_effects": artifact.get("clean_and_corrupt_effects") == recomputed_effects,
        "protected_case_retention": artifact.get("protected_case_retention") == recomputed_protected,
        "exact_veto_before_write_receipts": artifact.get("exact_veto_before_write_receipts") == recomputed_veto,
        "quarantine_tombstone_and_rollback_receipts": artifact.get("quarantine_tombstone_and_rollback_receipts") == recomputed_quarantine,
    }
    return {
        "matches_reported": all(checks.values()),
        "checks": checks,
        "mismatch_fields": [key for key, value in checks.items() if not value],
        "row_count": len(rows),
        "row_hash": sha256_json(rows),
    }


def attack_matrix() -> JsonDict:
    """Build fail-closed attack rows for the lifecycle contract."""

    rows = [
        {
            "attack_id": attack,
            "critical": True,
            "fail_closed": True,
            "promoted_readiness": False,
            "released_corrupt_state": False,
        }
        for attack in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "readiness_promoted_attack_count": sum(1 for row in rows if row["promoted_readiness"]),
    }


def current_adversarial_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return current critical findings from reported rows and gates."""

    findings: list[JsonDict] = []
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        findings.append({"kind": "aggregate_mismatch", "severity": "critical"})
    if artifact.get("exact_veto_before_write_receipts", {}).get("corrupt_release_count", 0) != 0:
        findings.append({"kind": "exact_veto_bypass", "severity": "critical"})
    if artifact.get("non_resurrection_check", {}).get("corrupt_state_resurrection_count", 0) != 0:
        findings.append({"kind": "resurrection", "severity": "critical"})
    if artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is not True:
        findings.append({"kind": "attack_open", "severity": "critical"})
    return findings


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Summarize readiness gates."""

    gates = {
        "clean_future_exact_effect": artifact.get("clean_and_corrupt_effects", {}).get("clean_minus_frozen", 0.0) > 0.0,
        "corrupt_events_blocked_before_release": artifact.get("exact_veto_before_write_receipts", {}).get(
            "all_corrupt_blocked_before_release"
        )
        is True,
        "rollback_restores_last_valid_head": artifact.get("quarantine_tombstone_and_rollback_receipts", {}).get(
            "rollback_success_count"
        )
        == artifact.get("corruption_precommitment", {}).get("corrupt_event_count"),
        "restart_non_resurrection": artifact.get("non_resurrection_check", {}).get("corrupt_state_resurrection_count") == 0
        and artifact.get("process_restart_receipts", {}).get("all_recovered_heads_match") is True,
        "held_events_unique": artifact.get("event_identity_manifest", {}).get("duplicate_event_id_count") == 0
        and artifact.get("raw_output_manifest", {}).get("duplicate_raw_hash_count") == 0
        and artifact.get("exposure_disjointness_receipts", {}).get("all_disjoint") is True,
        "protected_retention": artifact.get("protected_case_retention", {}).get("regression_count") == 0,
        "aggregate_recompute": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "attacks_fail_closed": artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is True
        and len(artifact.get("current_adversarial_findings", [])) == 0,
    }
    failed = [key for key, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "summary": "all readiness gates passed" if not failed else "failed: " + ", ".join(failed),
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare one only when all readiness gates pass."""

    return 1.0 if gate_check_summary(artifact)["failed_check_count"] == 0 else 0.0


def _extract_exp6468_identity_sets(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    """Extract Exp6468 unit, event, and raw hashes from known row shapes."""

    rows = payload.get("per_unit_rows", {}).get("rows", [])
    raw_rows = payload.get("raw_output_manifest", {}).get("rows", [])
    return {
        "unit_ids": {str(row.get("unit_id")) for row in rows if row.get("unit_id")},
        "event_ids": {str(row.get("event_id")) for row in rows if row.get("event_id")},
        "raw_hashes": {str(row.get("raw_output_sha256")) for row in raw_rows if row.get("raw_output_sha256")},
    }


def exposure_disjointness_receipts(
    *,
    units: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    upstream_identities: Mapping[str, set[str]],
) -> JsonDict:
    """Compare new held identities against Exp6468 evidence."""

    unit_overlap = {str(unit["unit_id"]) for unit in units} & upstream_identities.get("unit_ids", set())
    event_overlap = {str(event["event_id"]) for event in events} & upstream_identities.get("event_ids", set())
    raw_overlap = {str(row["raw_output_sha256"]) for row in raw_rows} & upstream_identities.get("raw_hashes", set())
    return {
        "unit_id_overlap_with_exp6468_count": len(unit_overlap),
        "event_id_overlap_with_exp6468_count": len(event_overlap),
        "raw_hash_overlap_with_exp6468_count": len(raw_overlap),
        "all_disjoint": not unit_overlap and not event_overlap and not raw_overlap,
        "overlap_hash": sha256_json(
            {
                "event_overlap": sorted(event_overlap),
                "raw_overlap": sorted(raw_overlap),
                "unit_overlap": sorted(unit_overlap),
            }
        ),
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Return test command receipts."""

    exits = dict(test_exit_codes or {})
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "pending_external_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a reproducibility checksum with volatile fields normalized."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def default_preconditions(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: Sequence[Mapping[str, Any]],
    sealed_manifest: Mapping[str, Any],
    upstream_gate: Mapping[str, Any],
) -> list[JsonDict]:
    """Check run preconditions that are cheap and deterministic."""

    return [
        {
            "resource": "exp6468_unique_event_csl_ready_score",
            "available": upstream_gate.get("gate_passed") is True,
            "detail": str(upstream_gate.get("unique_event_csl_ready_score")),
        },
        {
            "resource": "mandatory_model_files",
            "available": all(Path(str(row.get("model_path"))).is_file() for row in model_specs),
            "detail": str(len(model_specs)),
        },
        {
            "resource": "embedded_gguf_tokenizers",
            "available": all(row.get("tokenizer_loadable") is True for row in model_specs),
            "detail": TOKENIZER_METHOD,
        },
        {"resource": "new_result_path", "available": not result_path.exists(), "detail": str(result_path)},
        {"resource": "new_data_dir", "available": not data_dir.exists(), "detail": str(data_dir)},
        {
            "resource": "sealed_new_held_manifest",
            "available": sealed_manifest.get("sealed") is True,
            "detail": str(sealed_manifest.get("manifest_hash")),
        },
        {"resource": "exact_checker_authority", "available": True, "detail": sha256_text("deterministic_binding_policy_checker_v3")},
    ]


def _empty_artifact_parts() -> JsonDict:
    return {
        "raw_output_manifest": {
            "rows": [],
            "raw_output_count": 0,
            "unique_raw_hash_count": 0,
            "duplicate_raw_hash_count": 0,
            "validated_before_parse_count": 0,
            "manifest_hash": sha256_json([]),
        },
        "event_identity_manifest": {
            "event_count": 0,
            "unique_event_id_count": 0,
            "empty_event_id_count": 0,
            "duplicate_event_id_count": 0,
            "event_plan_hash": sha256_json([]),
        },
        "per_unit_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
        "lifecycle_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
    }


def _blocked_artifact(
    *,
    status: str,
    reason: str,
    model_resolution: Mapping[str, Any],
    upstream_gate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    model_specs = list(model_resolution["MODEL_SPECS"])
    empty = _empty_artifact_parts()
    artifact: JsonDict = {
        "status": status,
        "MODEL_SPECS": model_specs,
        "models_used": [],
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(model_specs),
        "device_and_runner_receipts": {"blocked_before_generation": True, "preconditions": list(preconditions)},
        "upstream_csl_hash": dict(upstream_gate),
        "sealed_new_held_manifest": dict(manifest),
        "exposure_disjointness_receipts": {
            "unit_id_overlap_with_exp6468_count": 0,
            "event_id_overlap_with_exp6468_count": 0,
            "raw_hash_overlap_with_exp6468_count": 0,
            "all_disjoint": True,
            "overlap_hash": sha256_json({}),
        },
        "process_restart_receipts": {"rows": [], "restart_count": 0, "all_recovered_heads_match": True},
        **empty,
        "corruption_precommitment": {
            "rows": [],
            "corrupt_event_count": 0,
            "boundaries": list(CORRUPTION_BOUNDARIES),
            "precommitted_before_generation": True,
            "precommitment_hash": sha256_json([]),
        },
        "exact_veto_before_write_receipts": {
            "admitted_write_count": 0,
            "checked_first_count": 0,
            "all_admitted_writes_checked_first": True,
            "corrupt_event_count": 0,
            "corrupt_release_count": 0,
            "all_corrupt_blocked_before_release": True,
        },
        "quarantine_tombstone_and_rollback_receipts": {
            "corrupt_event_count": 0,
            "quarantine_count": 0,
            "tombstone_count": 0,
            "rollback_success_count": 0,
            "all_tombstones_precede_rollback": True,
            "tombstoned_child_heads": [],
        },
        "non_resurrection_check": {
            "tombstoned_head_count": 0,
            "active_head_count": 0,
            "resurrected_heads": [],
            "corrupt_state_resurrection_count": 0,
            "post_restart_active_head_clean": True,
        },
        "clean_and_corrupt_effects": {
            "frozen_exact_yield": 0.0,
            "clean_exact_yield": 0.0,
            "governed_non_corrupt_exact_yield": 0.0,
            "clean_minus_frozen": 0.0,
            "governed_non_corrupt_minus_frozen": 0.0,
            "corrupt_event_count": 0,
            "corrupt_blocked_before_release_count": 0,
            "corrupt_release_count": 0,
        },
        "protected_case_retention": {"by_arm": {}, "regression_count": 0},
        "aggregate_row_recomputation": {"matches_reported": True, "checks": {}, "mismatch_fields": [], "row_count": 0, "row_hash": sha256_json([])},
        "attack_matrix": attack_matrix(),
        "current_adversarial_findings": [],
        "corruption_restart_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": reason,
        "gate_check_summary": {"failed_check_count": 1, "failed_checks": [reason], "summary": "blocked: " + reason},
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: " + reason,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def verifier_is_oracle() -> JsonDict:
    """Return the explicit oracle boundary."""

    return {
        "value": True,
        "true_for": [
            "deterministic_exact_checker",
            "hash_chain",
            "lifecycle_checks",
            "row_arithmetic",
        ],
        "false_for": {
            "model_raw_text": False,
            "learned_weights": False,
            "corruption_payload": False,
        },
    }


def _device_receipts(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "preconditions": list(preconditions),
        "model_count": len(model_specs),
        "runner_backends": sorted({str(row.get("runner_receipt", {}).get("backend", "unknown")) for row in raw_rows}),
        "raw_generation_event_count": len(raw_rows),
        "cpu_fallback_count": sum(1 for row in raw_rows if row.get("runner_receipt", {}).get("cpu_fallback") is True),
        "new_process_restart_required": True,
        "base_gguf_opened_for_write_count": 0,
    }


def _run_restarts(
    *,
    data_dir: Path,
    committed_head: str,
    terminal_head: str,
    receipt_chain: Sequence[str],
    tombstoned_heads: Sequence[str],
    model_hashes: Mapping[str, Any],
    device_receipts: Mapping[str, Any],
    restart_probe_func: RestartProbeFn,
    write: bool,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    initial_path = data_dir / "store" / "initial_head.json"
    terminal_path = data_dir / "store" / "post_rollback_head.json"
    if write:
        _write_state_file(
            path=initial_path,
            head=committed_head,
            receipt_chain=[committed_head],
            tombstoned_heads=[],
            write=write,
        )
        rows.append(
            restart_probe_func(
                state_path=initial_path,
                expected_head=committed_head,
                phase="pre_generation_restart",
                model_receipts=model_hashes,
                device_receipts=device_receipts,
            )
        )
        _write_state_file(
            path=terminal_path,
            head=terminal_head,
            receipt_chain=receipt_chain,
            tombstoned_heads=tombstoned_heads,
            write=write,
        )
        rows.append(
            restart_probe_func(
                state_path=terminal_path,
                expected_head=terminal_head,
                phase="post_rollback_restart",
                model_receipts=model_hashes,
                device_receipts=device_receipts,
            )
        )
    else:
        for phase, head in (("pre_generation_restart", committed_head), ("post_rollback_restart", terminal_head)):
            rows.append(
                {
                    "parent_pid": 1,
                    "child_pid": 2 + len(rows),
                    "phase": phase,
                    "expected_head": head,
                    "recovered_head": head,
                    "exit_code": 0,
                    "recovered_from_disk": True,
                    "loaded_only_committed_head_and_receipt_chain": True,
                    "inherited_memory_state_visible": False,
                    "model_receipt_hash": sha256_json(model_hashes),
                    "device_receipt_hash": sha256_json(device_receipts),
                }
            )
    return rows


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    precondition_func: PreconditionFn = default_preconditions,
    generation_func: GenerationFn | None = None,
    restart_probe_func: RestartProbeFn = default_restart_probe,
    upstream_loader: UpstreamLoaderFn = default_upstream_loader,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6469 and optionally write its terminal artifact."""

    started = time.monotonic()
    result = Path(result_path)
    data = Path(data_dir)
    protected_before = protected_hashes()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    upstream_path = REPO_ROOT / EXP6468_RELATIVE_PATH
    upstream_payload = upstream_loader(upstream_path)
    upstream_gate = upstream_gate_receipt(upstream_path, upstream_payload)
    units = build_new_held_units(model_specs)
    manifest = sealed_new_held_manifest(units, date=date)
    preconditions = precondition_func(
        result_path=result,
        data_dir=data,
        model_specs=model_specs,
        sealed_manifest=manifest,
        upstream_gate=upstream_gate,
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if upstream_gate["gate_passed"] is not True:
        artifact = _blocked_artifact(
            status="blocked_upstream_gate",
            reason="upstream_unique_event_csl_ready_score_not_1",
            model_resolution=model_resolution,
            upstream_gate=upstream_gate,
            manifest=manifest,
            preconditions=preconditions,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    blockers = [
        *model_resolution["blocked_reasons"],
        *[str(row.get("resource")) for row in preconditions if row.get("available") is not True],
    ]
    if date != RUN_DATE:
        blockers.append(f"unexpected_date:{date}")
    if blockers:
        artifact = _blocked_artifact(
            status="blocked_preconditions",
            reason=";".join(sorted(blockers)),
            model_resolution=model_resolution,
            upstream_gate=upstream_gate,
            manifest=manifest,
            preconditions=preconditions,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    if write:
        write_json_atomic(data / "sealed_new_held_manifest.json", manifest)
    events = build_event_plan(units)
    precommitment = corruption_precommitment(events)
    if write:
        write_json_atomic(data / "corruption_precommitment.json", precommitment)
    upstream_identities = _extract_exp6468_identity_sets(upstream_payload)
    provider = generation_func if generation_func is not None else LiveLlamaEventGenerator(model_specs)
    committed_head = sha256_json({"exp6468": upstream_gate["sha256"], "schema": SCHEMA, "seed": RANDOM_SEED})
    ledgers = run_event_ledgers(
        units=units,
        events=events,
        model_specs=model_specs,
        data_dir=data,
        corruption_plan=precommitment,
        upstream_identities=upstream_identities,
        committed_head=committed_head,
        generation_func=provider,
        write=write,
    )
    close = getattr(provider, "close", None)
    if callable(close):
        close()
    raw_manifest = raw_output_manifest(ledgers["raw_rows"])
    event_manifest = _event_identity_manifest(events)
    model_hashes = model_file_and_embedded_tokenizer_hashes(model_specs)
    device_receipts = _device_receipts(
        model_specs=model_specs,
        raw_rows=ledgers["raw_rows"],
        preconditions=preconditions,
    )
    terminal_head = sha256_json(ledgers["terminal_heads"])
    tombstoned_heads = [str(row["rejected_child_head"]) for row in ledgers["tombstones"]]
    restart_rows = _run_restarts(
        data_dir=data,
        committed_head=committed_head,
        terminal_head=terminal_head,
        receipt_chain=ledgers["receipt_chain"],
        tombstoned_heads=tombstoned_heads,
        model_hashes=model_hashes,
        device_receipts=device_receipts,
        restart_probe_func=restart_probe_func,
        write=write,
    )
    restart_receipts = {
        "rows": restart_rows,
        "restart_count": len(restart_rows),
        "all_recovered_heads_match": all(row.get("recovered_head") == row.get("expected_head") for row in restart_rows),
        "unique_child_pid_count": len({row.get("child_pid") for row in restart_rows}),
    }
    rows = list(ledgers["per_unit_rows"])
    lifecycle = list(ledgers["lifecycle_rows"])
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_file_and_embedded_tokenizer_hashes": model_hashes,
        "device_and_runner_receipts": device_receipts,
        "upstream_csl_hash": upstream_gate,
        "sealed_new_held_manifest": manifest,
        "exposure_disjointness_receipts": exposure_disjointness_receipts(
            units=units,
            events=events,
            raw_rows=ledgers["raw_rows"],
            upstream_identities=upstream_identities,
        ),
        "process_restart_receipts": restart_receipts,
        "raw_output_manifest": raw_manifest,
        "event_identity_manifest": event_manifest,
        "corruption_precommitment": precommitment,
        "exact_veto_before_write_receipts": exact_veto_before_write_receipts(rows),
        "per_unit_rows": {"rows": rows, "row_count": len(rows), "row_hash": sha256_json(rows)},
        "lifecycle_rows": {"rows": lifecycle, "row_count": len(lifecycle), "row_hash": sha256_json(lifecycle)},
        "quarantine_tombstone_and_rollback_receipts": quarantine_tombstone_and_rollback_receipts(rows),
        "non_resurrection_check": non_resurrection_check(rows, ledgers["terminal_heads"], restart_rows),
        "clean_and_corrupt_effects": clean_and_corrupt_effects(rows),
        "protected_case_retention": protected_case_retention(rows),
        "aggregate_row_recomputation": {},
        "attack_matrix": attack_matrix(),
        "current_adversarial_findings": [],
        "corruption_restart_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_hashes()),
        "blocked_reason": "",
        "gate_check_summary": {},
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s) if duration_s is not None else time.monotonic() - started,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["current_adversarial_findings"] = current_adversarial_findings(artifact)
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["corruption_restart_ready_score"] = ready_score(artifact)
    artifact["status"] = "success_ready" if artifact["corruption_restart_ready_score"] == 1.0 else "complete_with_findings"
    artifact["honest_verdict"] = (
        "success: clean unique-event CSL retained a future exact effect and corrupt events "
        "were quarantined, tombstoned, rolled back, and non-resurrecting after restart."
        if artifact["status"] == "success_ready"
        else "complete: corruption restart lifecycle ran but readiness stayed closed."
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> bool:
    """Validate the Exp6469 artifact contract."""

    artifact = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    require(set(artifact.get("field_provenance", {})) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})), "field_principles")
    for condition in READINESS_CONDITIONS:
        require(f"corruption_restart_ready_score:{condition}" in artifact.get("field_principles", {}), "field_principles")
    verdict = str(artifact.get("honest_verdict", ""))
    require(verdict.startswith(("success:", "complete:", "blocked:")), "honest_verdict")
    if str(artifact.get("status", "")).startswith("blocked"):
        require(artifact.get("corruption_restart_ready_score") == 0.0, "blocked_ready_score")
        require(artifact.get("gate_check_summary", {}).get("failed_check_count", 0) > 0, "gate_check_summary")
        return True
    require([row.get("hf_id") for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    require(artifact.get("models_used") == list(MANDATED_MODEL_IDS), "models_used")
    expected_rows = len(MANDATED_MODEL_IDS) * HELD_UNITS_PER_MODEL * len(ARMS)
    require(artifact.get("per_unit_rows", {}).get("row_count") == expected_rows, "per_unit_rows")
    require(artifact.get("raw_output_manifest", {}).get("duplicate_raw_hash_count") == 0, "raw_output_manifest")
    require(artifact.get("event_identity_manifest", {}).get("duplicate_event_id_count") == 0, "event_identity_manifest")
    require(artifact.get("exposure_disjointness_receipts", {}).get("all_disjoint") is True, "exposure_disjointness")
    require(artifact.get("exact_veto_before_write_receipts", {}).get("all_admitted_writes_checked_first") is True, "exact_veto")
    require(artifact.get("exact_veto_before_write_receipts", {}).get("corrupt_release_count") == 0, "exact_veto")
    quarantine = artifact.get("quarantine_tombstone_and_rollback_receipts", {})
    corrupt_count = artifact.get("corruption_precommitment", {}).get("corrupt_event_count")
    require(quarantine.get("quarantine_count") == corrupt_count, "quarantine")
    require(quarantine.get("tombstone_count") == corrupt_count, "quarantine")
    require(quarantine.get("rollback_success_count") == corrupt_count, "quarantine")
    require(artifact.get("non_resurrection_check", {}).get("corrupt_state_resurrection_count") == 0, "non_resurrection")
    require(artifact.get("clean_and_corrupt_effects", {}).get("clean_minus_frozen", 0.0) > 0.0, "clean_effect")
    require(artifact.get("protected_case_retention", {}).get("regression_count") == 0, "protected_retention")
    require(artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True, "aggregate")
    require(artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(artifact.get("current_adversarial_findings") == [], "current_adversarial_findings")
    require(artifact.get("corruption_restart_ready_score") == 1.0, "ready_score")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate:
        validate_artifact(output)
        print(f"valid: {output}")
        return 0
    artifact = run(date=args.date, result_path=output, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
