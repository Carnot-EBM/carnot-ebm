"""Audit the ARC producer evidence contract in a restricted fresh process.

The controller creates deterministic producer fixtures in a temporary store.
It then starts a read-only child with no network or game source. The child uses
the independent replay code in this module. Producer validation helpers are not
imported by the child.

Spec refs: REQ-ARC-WMTE-6994 and SCENARIO-ARC-WMTE-6994-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260904"
EXPERIMENT_ID = 6994
SCHEMA = "carnot.exp6994.arc_producer_cold_audit.v1"
INFERENCE_SUBSTRATE = "fresh_process_arc_contract_replay_no_llm"
RANDOM_SEED = 69_942_026_090_4
STAMP = "20260904T120000_000000"

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6994_arc_producer_cold_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6994_arc_producer_cold_audit.py")
RESULT_PATH = Path("results/experiment_6994_arc_producer_cold_audit.json")
EXP6993_PATH = Path("results/experiment_6993_arc_producer_evidence_contract.json")
EXPECTED_EXP6993_SHA256 = "sha256:cd34bbb63249d8aeafc8e763baf3e3cf1ffd7f9d01e212a97d8051d8c085ca7d"

SOURCE_PATHS = {
    "exp6993": EXP6993_PATH,
    "audit_module": MODULE_PATH,
    "audit_wrapper": WRAPPER_PATH,
    "spec": SPEC_PATH,
    "producer": Path("python/carnot/agentic/arc_producer_evidence.py"),
    "world_model": Path("python/carnot/agentic/arc_executable_world_model.py"),
    "factory_policy": Path("python/carnot/agentic/arc_competition_agent.py"),
    "scorer": Path("scripts/arc_e3_induced_model_quality.py"),
    "solve_registry": Path("ops/arc_solve_registry.yaml"),
}

ENVELOPE_SCHEMA = "carnot.arc.live_engine_evidence.v1"
MANIFEST_SCHEMA = "carnot.arc.live_engine_manifest.v1"
LIVE_SOURCE_KIND = "live_agent_attempts"
REQUIRED_ENVELOPE_FIELDS = (
    "schema",
    "run_id",
    "game",
    "created_at",
    "published_at",
    "raw_prompt_path",
    "raw_prompt_sha256",
    "transition_jsonl_path",
    "transition_sha256",
    "transition_count",
    "transition_source_kind",
    "engine_path",
    "engine_sha256",
    "environment_receipt_path",
    "environment_receipt_sha256",
    "scorer_path",
    "scorer_sha256",
    "live_policy_path",
    "live_policy_sha256",
    "agent_factory_path",
    "agent_factory_sha256",
    "manifest_row_path",
    "manifest_row_sha256",
    "envelope_sha256",
)
HASH_FIELDS = {
    "raw_prompt_path": ("raw_prompt_sha256", "prompt_hash_replay_rows"),
    "transition_jsonl_path": ("transition_sha256", "transition_hash_replay_rows"),
    "engine_path": ("engine_sha256", "engine_hash_replay_rows"),
    "environment_receipt_path": (
        "environment_receipt_sha256",
        "environment_hash_replay_rows",
    ),
    "scorer_path": ("scorer_sha256", "scorer_hash_replay_rows"),
    "live_policy_path": ("live_policy_sha256", "policy_hash_replay_rows"),
    "agent_factory_path": ("agent_factory_sha256", "factory_hash_replay_rows"),
}

PROMPT = b"fixture raw prompt\x00\xff\r\n"
ENGINE = b"""import numpy as np

def engine(grid, action, data):
    result = np.asarray(grid).copy()
    if int(action) == 2:
        result[0, 0] = 7
    return result

def is_level_complete(grid):
    return False
"""
TRANSITIONS = (
    {
        "grid": [[0, 0], [0, 0]],
        "action": 1,
        "data": None,
        "next_grid": [[0, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
    {
        "grid": [[0, 0], [0, 0]],
        "action": 2,
        "data": {"x": 0, "y": 0},
        "next_grid": [[7, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
)

FAKE_SOURCE_KINDS = (
    "game_source",
    "hand_adapter",
    "offline_bfs",
    "synthetic_hidden_transition",
    None,
    "unknown_source",
)
FAILPOINTS = (
    "after_engine_temp",
    "after_bundle_publish",
    "after_engine_publish",
    "before_manifest",
)
TAMPER_TARGETS = {
    "tamper_prompt": "raw_prompt_path",
    "tamper_engine": "engine_path",
    "tamper_environment": "environment_receipt_path",
    "tamper_scorer": "scorer_path",
    "tamper_policy": "live_policy_path",
    "tamper_factory": "agent_factory_path",
}
REQUIRED_FIXTURE_IDS = (
    "success",
    *TAMPER_TARGETS,
    "reordered_transitions",
    "missing_prompt",
    "tamper_envelope",
    "tamper_manifest",
    "legacy",
    "duplicate",
    *(f"source_{kind or 'missing'}" for kind in FAKE_SOURCE_KINDS),
    "unreachable_engine",
    *(f"interrupt_{failpoint}" for failpoint in FAILPOINTS),
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_fixture_rows",
    "prompt_hash_replay_rows",
    "transition_hash_replay_rows",
    "engine_hash_replay_rows",
    "environment_hash_replay_rows",
    "scorer_hash_replay_rows",
    "policy_hash_replay_rows",
    "factory_hash_replay_rows",
    "manifest_hash_replay_rows",
    "envelope_hash_replay_rows",
    "transition_order_rows",
    "source_kind_rows",
    "eligibility_replay_rows",
    "atomicity_replay_rows",
    "tamper_replay_rows",
    "legacy_replay_rows",
    "agent_factory_trace_rows",
    "routing_replay_rows",
    "action_influence_replay_rows",
    "source_disagreement_rows",
    "read_only_enforcement_receipt",
    "arc_contract_audit_complete_score",
    "arc_producer_contract_confirmed_score",
    "solve_provenance_applicable",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
    "model_quality_claimed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for every field makes the audit contract reviewable.",
    "preconditions_checked": "Exact preflight checks stop missing source evidence from looking audited.",
    "inference_substrate": "The substrate separates a byte replay from LLM or game execution.",
    "duration_s": "Wall time shows that the fresh audit process executed.",
    "source_artifact_hashes": "Input hashes bind the audit to exact source and fixture bytes.",
    "rows": "Terminal row records preserve every decision denominator.",
    "per_fixture_rows": "One summary per fixture exposes omissions and duplicate credit.",
    "prompt_hash_replay_rows": "Prompt replay binds synthesis to the raw request bytes.",
    "transition_hash_replay_rows": "Transition hashes bind values and serialization order.",
    "engine_hash_replay_rows": "Engine hashes identify the executable candidate exactly.",
    "environment_hash_replay_rows": "Environment hashes bind the producer runtime receipt.",
    "scorer_hash_replay_rows": "Scorer hashes fix the measurement code used by the producer.",
    "policy_hash_replay_rows": "Policy hashes bind evidence to the shipped routing code.",
    "factory_hash_replay_rows": "Factory hashes bind evidence to the shipped constructor.",
    "manifest_hash_replay_rows": "Manifest replay checks the final eligibility marker.",
    "envelope_hash_replay_rows": "Envelope replay detects content drift without a hash cycle.",
    "transition_order_rows": "Order rows prevent a valid set from hiding an invalid sequence.",
    "source_kind_rows": "Source rows reject evidence not produced by live attempts.",
    "eligibility_replay_rows": "Independent decisions show which records can support later science.",
    "atomicity_replay_rows": "Atomicity rows prove stopped writes have no final marker.",
    "tamper_replay_rows": "Attack rows show that changed evidence fails closed.",
    "legacy_replay_rows": "Legacy rows stay readable without receiving current eligibility.",
    "agent_factory_trace_rows": "Executed calls prove reachability beyond a source reference.",
    "routing_replay_rows": "Routing rows bind the envelope engine to the real candidate seam.",
    "action_influence_replay_rows": "Influence rows require a measured deterministic score change.",
    "source_disagreement_rows": "Contradictions between replay and source claims remain visible.",
    "read_only_enforcement_receipt": "Sandbox evidence proves the audit could not change its sources.",
    "arc_contract_audit_complete_score": "Completion requires a terminal result for every fixture row.",
    "arc_producer_contract_confirmed_score": "Confirmation requires integrity, routing, and source agreement.",
    "solve_provenance_applicable": "A fixture contract has no solve provenance to assess.",
    "solve_claimed": "An evidence replay does not prove a game solve.",
    "level_claimed": "A score change does not prove level completion.",
    "registry_updated": "A read-only fixture audit must preserve the solve registry.",
    "submitted_to_leaderboard": "A local byte audit is not a leaderboard submission.",
    "model_quality_claimed": "Contract integrity does not establish model quality.",
    "random_seed": "A fixed seed identifies the deterministic fixture run.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "Exact expected and observed values explain any blocked audit.",
    "verifier_is_oracle": "False separates contract validation from a game oracle.",
    "verdict_class": "A closed verdict class prevents blocked work from reading as positive.",
    "honest_verdict": "A class-specific prefix gives automation a stable terminal result.",
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "complete_disqualified_",
    "partial": "partial_",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable bytes for hashes that cover JSON values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash one file without parsing it, or return None when it is absent."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def compute_envelope_hash(envelope: Mapping[str, Any]) -> str:
    """Rebuild the producer's acyclic envelope self-hash projection."""

    projected = dict(envelope)
    projected.pop("envelope_sha256", None)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def compute_manifest_hash(row: Mapping[str, Any]) -> str:
    """Rebuild the final-row hash without its own digest field."""

    projected = dict(row)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record one fail-closed comparison with exact values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and promote the first failure."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def write_json_atomic(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    """Replace one JSON file only after its full content is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _fixture_row(
    fixture_id: str,
    *,
    game: str,
    scenario: str,
    expected_eligible: bool,
) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "game": game,
        "scenario": scenario,
        "expected_eligible": expected_eligible,
        "terminal": True,
    }


def materialize_fixtures(root: Path) -> list[JsonDict]:
    """Create raw producer stores before the independent child starts.

    This controller-only function may import the producer. The child receives
    only the resulting bytes and never imports the producer validator.
    """

    from carnot.agentic.arc_producer_evidence import (
        EvidencePublishInterrupted,
        produce_engine_evidence,
    )

    root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []

    def produce(fixture_id: str, source_kind: str | None = LIVE_SOURCE_KIND, **kwargs: Any):
        return produce_engine_evidence(
            store_root=root,
            game=fixture_id,
            raw_prompt=PROMPT,
            transitions=TRANSITIONS,
            transition_source_kind=source_kind,
            environment_receipt={"environment": "deterministic_stub", "game": fixture_id},
            synthesize_engine=lambda: ENGINE,
            run_id=f"{fixture_id}-run",
            timestamp=STAMP,
            **kwargs,
        )

    produce("success")
    rows.append(_fixture_row("success", game="success", scenario="success", expected_eligible=True))

    for fixture_id, envelope_field in TAMPER_TARGETS.items():
        published = produce(fixture_id)
        envelope = json.loads(published.envelope_path.read_text(encoding="utf-8"))
        target = root / envelope[envelope_field]
        target.write_bytes(target.read_bytes() + b"\nchanged")
        rows.append(
            _fixture_row(
                fixture_id,
                game=fixture_id,
                scenario="tampered",
                expected_eligible=False,
            )
        )

    reordered = produce("reordered_transitions")
    transition_lines = reordered.transition_path.read_bytes().splitlines(keepends=True)
    reordered.transition_path.write_bytes(b"".join(reversed(transition_lines)))
    rows.append(
        _fixture_row(
            "reordered_transitions",
            game="reordered_transitions",
            scenario="reordered",
            expected_eligible=False,
        )
    )

    missing = produce("missing_prompt")
    missing.prompt_path.unlink()
    rows.append(
        _fixture_row(
            "missing_prompt",
            game="missing_prompt",
            scenario="missing_prompt",
            expected_eligible=False,
        )
    )

    changed_envelope = produce("tamper_envelope")
    envelope = json.loads(changed_envelope.envelope_path.read_text(encoding="utf-8"))
    envelope["published_at"] = "20260904T120001_000000"
    changed_envelope.envelope_path.write_bytes(canonical_json_bytes(envelope))
    rows.append(
        _fixture_row(
            "tamper_envelope",
            game="tamper_envelope",
            scenario="tampered",
            expected_eligible=False,
        )
    )

    changed_manifest = produce("tamper_manifest")
    manifest_row = json.loads(changed_manifest.manifest_path.read_text(encoding="utf-8"))
    manifest_row["note"] = "changed after publication"
    changed_manifest.manifest_path.write_bytes(canonical_json_bytes(manifest_row) + b"\n")
    rows.append(
        _fixture_row(
            "tamper_manifest",
            game="tamper_manifest",
            scenario="tampered",
            expected_eligible=False,
        )
    )

    legacy_manifest = root / "legacy" / "attempts" / "manifest.jsonl"
    legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest.write_bytes(
        canonical_json_bytes(
            {"ts": "20260901T000000_000000", "file": "wm_old.py", "sha256_16": "0" * 16}
        )
        + b"\n"
    )
    rows.append(_fixture_row("legacy", game="legacy", scenario="legacy", expected_eligible=False))

    duplicate = produce("duplicate")
    original_line = duplicate.manifest_path.read_bytes()
    duplicate.manifest_path.write_bytes(original_line + original_line)
    rows.append(
        _fixture_row("duplicate", game="duplicate", scenario="duplicate", expected_eligible=False)
    )

    for source_kind in FAKE_SOURCE_KINDS:
        suffix = source_kind or "missing"
        fixture_id = f"source_{suffix}"
        produce(fixture_id, source_kind)
        rows.append(
            _fixture_row(
                fixture_id,
                game=fixture_id,
                scenario="fake_source",
                expected_eligible=False,
            )
        )

    unreachable = produce("unreachable_engine")
    unreachable.canonical_engine_path.unlink()
    rows.append(
        _fixture_row(
            "unreachable_engine",
            game="unreachable_engine",
            scenario="unreachable",
            expected_eligible=False,
        )
    )

    for failpoint in FAILPOINTS:
        fixture_id = f"interrupt_{failpoint}"
        try:
            produce(fixture_id, failpoint=failpoint)
        except EvidencePublishInterrupted:
            pass
        rows.append(
            _fixture_row(
                fixture_id,
                game=fixture_id,
                scenario="interrupted",
                expected_eligible=False,
            )
        )

    write_json_atomic(root / "fixture_index.json", rows)
    return rows


def _relative_fixture_hashes(fixture_root: Path) -> list[JsonDict]:
    """Hash all raw fixture files in path order before parsing any of them."""

    return [
        {
            "path": str(path.relative_to(fixture_root)),
            "sha256": sha256_path(path),
            "terminal": True,
        }
        for path in sorted(value for value in fixture_root.rglob("*") if value.is_file())
    ]


def load_hashed_inputs(
    repo_root: Path,
    fixture_root: Path,
    *,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_source_hash: str = EXPECTED_EXP6993_SHA256,
) -> JsonDict:
    """Hash every source and fixture file before parsing aggregate claims."""

    observed = {
        source_id: sha256_path(repo_root / relative_path)
        for source_id, relative_path in source_paths.items()
    }
    fixture_hashes = _relative_fixture_hashes(fixture_root)
    checks = [gate_check("source_hash:exp6993", expected_source_hash, observed.get("exp6993"))]
    if checks[0]["passed"] is not True:
        return {
            "passed": False,
            "sources": {},
            "preconditions": checks,
            "hashes": {"repository_sources": observed, "fixture_inputs": fixture_hashes},
        }
    try:
        source_artifact = json.loads((repo_root / source_paths["exp6993"]).read_text("utf-8"))
        fixture_rows = json.loads((fixture_root / "fixture_index.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        checks.append(gate_check("input_json_parse", "all inputs parse", type(exc).__name__))
        return {
            "passed": False,
            "sources": {},
            "preconditions": checks,
            "hashes": {"repository_sources": observed, "fixture_inputs": fixture_hashes},
        }
    checks.append(gate_check("source_artifact_object", True, isinstance(source_artifact, dict)))
    checks.append(gate_check("raw_fixture_rows_list", True, isinstance(fixture_rows, list)))
    return {
        "passed": all(row["passed"] for row in checks),
        "sources": {"exp6993": source_artifact, "fixture_rows": fixture_rows},
        "preconditions": checks,
        "hashes": {"repository_sources": observed, "fixture_inputs": fixture_hashes},
    }


def _safe_path(root: Path, value: Any) -> Path | None:
    """Resolve one relative path and reject absolute or escaping values."""

    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _reason_rows(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in checks
        if row.get("passed") is not True
    ]


def _hash_replay_row(
    fixture_id: str,
    expected_hash: Any,
    observed_hash: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "expected_hash": expected_hash,
        "observed_hash": observed_hash,
        "passed": bool(expected_hash == observed_hash if passed is None else passed),
        "terminal": True,
    }


def _empty_replay() -> JsonDict:
    return {
        field: []
        for field in (
            "rows",
            "per_fixture_rows",
            "prompt_hash_replay_rows",
            "transition_hash_replay_rows",
            "engine_hash_replay_rows",
            "environment_hash_replay_rows",
            "scorer_hash_replay_rows",
            "policy_hash_replay_rows",
            "factory_hash_replay_rows",
            "manifest_hash_replay_rows",
            "envelope_hash_replay_rows",
            "transition_order_rows",
            "source_kind_rows",
            "eligibility_replay_rows",
            "atomicity_replay_rows",
            "tamper_replay_rows",
            "legacy_replay_rows",
            "source_disagreement_rows",
        )
    }


def _replay_current_row(
    root: Path,
    fixture: Mapping[str, Any],
    row: Mapping[str, Any],
    replay: JsonDict,
) -> JsonDict:
    """Independently validate one current-schema manifest row."""

    fixture_id = str(fixture["fixture_id"])
    checks = [
        gate_check("manifest_schema", MANIFEST_SCHEMA, row.get("schema")),
        gate_check("manifest_complete", True, row.get("complete")),
    ]
    observed_manifest_hash = compute_manifest_hash(row)
    expected_manifest_hash = row.get("manifest_row_sha256")
    envelope_record = row.get("envelope")
    envelope_path = (
        _safe_path(root, envelope_record.get("path"))
        if isinstance(envelope_record, Mapping)
        else None
    )
    envelope = None
    envelope_raw = None
    if envelope_path is not None:
        try:
            envelope_raw = envelope_path.read_bytes()
            parsed = json.loads(envelope_raw)
            envelope = parsed if isinstance(parsed, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            envelope = None
    checks.append(gate_check("envelope_readable", True, envelope is not None))

    manifest_file_hash = None
    if envelope is not None:
        manifest_file = _safe_path(root, envelope.get("manifest_row_path"))
        manifest_file_hash = sha256_path(manifest_file) if manifest_file is not None else None
    manifest_passed = bool(expected_manifest_hash == observed_manifest_hash == manifest_file_hash)
    replay["manifest_hash_replay_rows"].append(
        _hash_replay_row(
            fixture_id,
            expected_manifest_hash,
            observed_manifest_hash,
            passed=manifest_passed,
        )
        | {"manifest_file_hash": manifest_file_hash}
    )
    checks.append(
        gate_check(
            "manifest_row_hash",
            expected_manifest_hash,
            {"row_projection": observed_manifest_hash, "manifest_file": manifest_file_hash},
            passed=manifest_passed,
        )
    )

    if envelope is None:
        replay["envelope_hash_replay_rows"].append(
            _hash_replay_row(fixture_id, None, None, passed=False)
        )
        checks.append(gate_check("canonical_engine_reachable", True, False))
        return {
            "classification": "evidence_envelope",
            "run_id": row.get("run_id"),
            "eligible": False,
            "checks": checks,
            "rejection_reasons": _reason_rows(checks),
            "row": dict(row),
            "terminal": True,
        }

    missing = [field for field in REQUIRED_ENVELOPE_FIELDS if field not in envelope]
    checks.extend(
        (
            gate_check("envelope_schema", ENVELOPE_SCHEMA, envelope.get("schema")),
            gate_check("envelope_required_fields", [], missing),
            gate_check("run_id_binding", row.get("run_id"), envelope.get("run_id")),
            gate_check("game_binding", row.get("game"), envelope.get("game")),
            gate_check(
                "manifest_envelope_binding",
                row.get("manifest_row_sha256"),
                envelope.get("manifest_row_sha256"),
            ),
        )
    )
    observed_envelope_hash = compute_envelope_hash(envelope)
    expected_envelope_hash = envelope.get("envelope_sha256")
    row_envelope_hash = (
        envelope_record.get("sha256") if isinstance(envelope_record, Mapping) else None
    )
    canonical_envelope_hash = sha256_bytes(canonical_json_bytes(envelope))
    raw_envelope_hash = sha256_bytes(envelope_raw) if envelope_raw is not None else None
    envelope_passed = bool(
        expected_envelope_hash == observed_envelope_hash == row_envelope_hash
        and canonical_envelope_hash == raw_envelope_hash
    )
    replay["envelope_hash_replay_rows"].append(
        _hash_replay_row(
            fixture_id,
            expected_envelope_hash,
            observed_envelope_hash,
            passed=envelope_passed,
        )
        | {
            "row_envelope_hash": row_envelope_hash,
            "canonical_file_hash": canonical_envelope_hash,
            "raw_file_hash": raw_envelope_hash,
        }
    )
    checks.append(
        gate_check(
            "envelope_hash",
            expected_envelope_hash,
            {
                "projection": observed_envelope_hash,
                "row": row_envelope_hash,
                "canonical_file": canonical_envelope_hash,
                "raw_file": raw_envelope_hash,
            },
            passed=envelope_passed,
        )
    )

    for path_field, (hash_field, replay_field) in HASH_FIELDS.items():
        source = _safe_path(root, envelope.get(path_field))
        observed_hash = sha256_path(source) if source is not None else None
        expected_hash = envelope.get(hash_field)
        passed = source is not None and expected_hash == observed_hash
        replay[replay_field].append(
            _hash_replay_row(fixture_id, expected_hash, observed_hash, passed=passed)
        )
        checks.append(gate_check(f"{path_field}_safe", True, source is not None))
        checks.append(gate_check(hash_field, expected_hash, observed_hash, passed=passed))

    transition_path = _safe_path(root, envelope.get("transition_jsonl_path"))
    transition_rows: list[Any] = []
    transition_parse_ok = True
    if transition_path is None:
        transition_parse_ok = False
    else:
        try:
            transition_rows = [
                json.loads(line) for line in transition_path.read_bytes().splitlines()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError):
            transition_parse_ok = False
    indices = [value.get("index") for value in transition_rows if isinstance(value, Mapping)]
    expected_order = list(range(len(transition_rows)))
    order_passed = transition_parse_ok and indices == expected_order
    count_passed = transition_parse_ok and envelope.get("transition_count") == len(transition_rows)
    replay["transition_order_rows"].append(
        {
            "fixture_id": fixture_id,
            "expected_order": expected_order,
            "observed_order": indices,
            "expected_count": envelope.get("transition_count"),
            "observed_count": len(transition_rows),
            "passed": order_passed and count_passed,
            "terminal": True,
        }
    )
    checks.append(gate_check("transition_order", expected_order, indices, passed=order_passed))
    checks.append(
        gate_check(
            "transition_count",
            envelope.get("transition_count"),
            len(transition_rows),
            passed=count_passed,
        )
    )

    source_kind = envelope.get("transition_source_kind")
    source_passed = source_kind == LIVE_SOURCE_KIND
    replay["source_kind_rows"].append(
        {
            "fixture_id": fixture_id,
            "expected_source_kind": LIVE_SOURCE_KIND,
            "observed_source_kind": source_kind,
            "passed": source_passed,
            "terminal": True,
        }
    )
    checks.append(gate_check("transition_source_kind", LIVE_SOURCE_KIND, source_kind))

    canonical_engine = root / str(row.get("game")) / "world_model.py"
    canonical_hash = sha256_path(canonical_engine)
    engine_hash = envelope.get("engine_sha256")
    reachable = canonical_hash == engine_hash
    checks.append(
        gate_check("canonical_engine_reachable", engine_hash, canonical_hash, passed=reachable)
    )
    reasons = _reason_rows(checks)
    return {
        "classification": "evidence_envelope",
        "run_id": row.get("run_id"),
        "eligible": not reasons,
        "checks": checks,
        "rejection_reasons": reasons,
        "row": dict(row),
        "terminal": True,
    }


def replay_fixtures(root: Path, fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay every fixture without calling the producer's reader."""

    replay = _empty_replay()
    for fixture in fixture_rows:
        fixture_id = str(fixture.get("fixture_id"))
        game = str(fixture.get("game"))
        scenario = str(fixture.get("scenario"))
        manifest = root / game / "attempts" / "manifest.jsonl"
        try:
            lines = manifest.read_bytes().splitlines()
        except OSError:
            lines = []
        validations: list[JsonDict] = []
        for line_number, line in enumerate(lines):
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                value = None
            if not isinstance(value, Mapping):
                validation = {
                    "classification": "malformed",
                    "run_id": None,
                    "eligible": False,
                    "checks": [],
                    "rejection_reasons": [
                        {
                            "failed_check": "manifest_row_json",
                            "expected_value": "json_object",
                            "observed_value": line_number,
                        }
                    ],
                    "row": None,
                    "terminal": True,
                }
            elif value.get("schema") != MANIFEST_SCHEMA:
                validation = {
                    "classification": "legacy",
                    "run_id": value.get("run_id"),
                    "eligible": False,
                    "checks": [],
                    "rejection_reasons": [
                        {
                            "failed_check": "evidence_envelope_schema",
                            "expected_value": MANIFEST_SCHEMA,
                            "observed_value": value.get("schema"),
                        }
                    ],
                    "row": dict(value),
                    "terminal": True,
                }
            else:
                validation = _replay_current_row(root, fixture, value, replay)
            validations.append(validation)

        counts = Counter(row.get("run_id") for row in validations if row.get("run_id"))
        for validation in validations:
            run_id = validation.get("run_id")
            if run_id and counts[run_id] > 1:
                reason = {
                    "failed_check": "unique_run_id",
                    "expected_value": 1,
                    "observed_value": counts[run_id],
                }
                validation["checks"].append({"check": "unique_run_id", **reason, "passed": False})
                validation["rejection_reasons"].append(reason)
                validation["eligible"] = False

        replay["rows"].extend(
            {**validation, "fixture_id": fixture_id, "scenario": scenario}
            for validation in validations
        )
        eligible_count = sum(row["eligible"] is True for row in validations)
        observed_eligible = len(validations) == 1 and eligible_count == 1
        failed_checks = sorted(
            {
                reason["failed_check"]
                for validation in validations
                for reason in validation["rejection_reasons"]
            }
        )
        if not lines:
            failed_checks.append("manifest_marker_present")
        classification = (
            validations[0]["classification"]
            if validations
            else ("interrupted" if scenario == "interrupted" else "missing")
        )
        summary = {
            "fixture_id": fixture_id,
            "game": game,
            "scenario": scenario,
            "classification": classification,
            "manifest_marker_count": len(lines),
            "eligible": observed_eligible,
            "expected_eligible": fixture.get("expected_eligible"),
            "expected_outcome_matches": observed_eligible is fixture.get("expected_eligible"),
            "failed_checks": failed_checks,
            "terminal": True,
        }
        replay["per_fixture_rows"].append(summary)
        replay["eligibility_replay_rows"].append(
            {
                "fixture_id": fixture_id,
                "eligible": observed_eligible,
                "expected_eligible": fixture.get("expected_eligible"),
                "passed": summary["expected_outcome_matches"],
                "terminal": True,
            }
        )
        if scenario in {"success", "interrupted"}:
            replay["atomicity_replay_rows"].append(
                {
                    "fixture_id": fixture_id,
                    "scenario": scenario,
                    "manifest_marker_count": len(lines),
                    "eligible_count": eligible_count,
                    "passed": (len(lines), eligible_count)
                    == ((1, 1) if scenario == "success" else (0, 0)),
                    "terminal": True,
                }
            )
        if scenario in {"tampered", "reordered", "missing_prompt", "unreachable"}:
            replay["tamper_replay_rows"].append(
                {
                    "fixture_id": fixture_id,
                    "scenario": scenario,
                    "eligible": observed_eligible,
                    "detected": not observed_eligible,
                    "failed_checks": failed_checks,
                    "terminal": True,
                }
            )
        if scenario in {"legacy", "duplicate"}:
            replay["legacy_replay_rows"].append(
                {
                    "fixture_id": fixture_id,
                    "scenario": scenario,
                    "classification": classification,
                    "eligible": observed_eligible,
                    "failed_checks": failed_checks,
                    "terminal": True,
                }
            )
        if summary["expected_outcome_matches"] is not True:
            replay["source_disagreement_rows"].append(
                {
                    "kind": "fixture_eligibility_disagreement",
                    "fixture_id": fixture_id,
                    "expected_value": fixture.get("expected_eligible"),
                    "observed_value": observed_eligible,
                    "terminal": True,
                }
            )
    return replay


def _trace_row(symbol: str, value: Any, called: bool) -> JsonDict:
    path = inspect.getsourcefile(value)
    try:
        line = inspect.getsourcelines(value)[1]
    except (OSError, TypeError):
        line = None
    relative = None
    if path is not None:
        try:
            relative = str(Path(path).resolve().relative_to(REPO_ROOT))
        except ValueError:
            relative = str(path)
    return {"symbol": symbol, "path": relative, "line": line, "called": called, "terminal": True}


def trace_factory_route(fixture_root: Path, game: str) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Execute the shipped factory, loader, candidate route, and score seam."""

    import numpy as np

    from carnot.agentic import arc_executable_world_model as world
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

    called = {
        "make_carnot_agent": False,
        "E3AgentPolicy": False,
        "load_engine": False,
        "_world_model_candidates": False,
    }

    class FixtureBase:
        def __init__(self) -> None:
            self.game_id = game

    agent_type = make_carnot_agent(FixtureBase, cascade=True, proposer=object())
    called["make_carnot_agent"] = True
    agent = agent_type()
    called["E3AgentPolicy"] = isinstance(agent._policy, E3AgentPolicy)
    old_root = world.E3_DIR
    world.E3_DIR = fixture_root
    try:
        engine, goal = world.load_engine(game)
        called["load_engine"] = True
    finally:
        world.E3_DIR = old_root
    candidates = agent._policy._world_model_candidates(engine, goal)
    called["_world_model_candidates"] = True
    routed = next(row for row in candidates if row.name == "loaded_world_model.py")

    manifest = json.loads(
        (fixture_root / game / "attempts" / "manifest.jsonl").read_text(encoding="utf-8")
    )
    envelope_path = _safe_path(fixture_root, manifest["envelope"]["path"])
    if envelope_path is None:
        raise ValueError("fixture_envelope_path_unsafe")
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    grid = np.zeros((2, 2), dtype=np.int64)
    control_score = int(np.count_nonzero(grid != grid))
    routed_score = int(np.count_nonzero(np.asarray(routed.engine(grid, 2, None)) != grid))
    canonical_hash = sha256_path(fixture_root / game / "world_model.py")
    trace = [
        _trace_row("make_carnot_agent", make_carnot_agent, called["make_carnot_agent"]),
        _trace_row("E3AgentPolicy", E3AgentPolicy, called["E3AgentPolicy"]),
        _trace_row("load_engine", world.load_engine, called["load_engine"]),
        _trace_row(
            "_world_model_candidates",
            E3AgentPolicy._world_model_candidates,
            called["_world_model_candidates"],
        ),
    ]
    routing = {
        "envelope_run_id": envelope.get("run_id"),
        "factory_constructed_e3_policy": called["E3AgentPolicy"],
        "selected_candidate_name": routed.name,
        "expected_engine_hash": envelope.get("engine_sha256"),
        "observed_engine_hash": canonical_hash,
        "engine_hash_matches_envelope": canonical_hash == envelope.get("engine_sha256"),
        "terminal": True,
    }
    influence = {
        "action": 2,
        "score_definition": "changed_cell_count",
        "control_score": control_score,
        "routed_score": routed_score,
        "score_changed": routed_score != control_score,
        "future_frame_used": False,
        "terminal": True,
    }
    return trace, routing, influence


def sandbox_runtime_receipt(repo_root: Path, fixture_root: Path) -> JsonDict:
    """Measure the child restrictions instead of trusting environment labels."""

    parent_netns = os.environ.get("CARNOT_EXP6994_PARENT_NETNS", "")
    child_netns = os.readlink("/proc/self/ns/net")
    source_rows = []
    paths = [repo_root / value for value in SOURCE_PATHS.values()]
    paths.extend(path for path in fixture_root.rglob("*") if path.is_file())
    for path in paths:
        denied = False
        error = None
        try:
            with path.open("rb+"):
                pass
        except OSError as exc:
            denied = True
            error = f"{type(exc).__name__}:{exc.errno}"
        try:
            relative = str(path.relative_to(fixture_root))
            source_type = "fixture"
        except ValueError:
            relative = str(path.relative_to(repo_root))
            source_type = "repository"
        source_rows.append(
            {
                "source_type": source_type,
                "path": relative,
                "write_denied": denied,
                "error": error,
            }
        )
    game_source = repo_root / "environment_files"
    game_source_empty = game_source.is_dir() and not any(game_source.iterdir())
    gpu_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    llm_modules = sorted(
        name for name in ("llama_cpp", "transformers", "vllm") if name in sys.modules
    )
    receipt = {
        "fresh_process": os.getpid() != os.getppid(),
        "fresh_process_pid": os.getpid(),
        "parent_process_pid": os.getppid(),
        "network_namespace": child_netns,
        "parent_network_namespace": parent_netns,
        "network_namespace_isolated": bool(parent_netns and child_netns != parent_netns),
        "gpu_devices_visible": gpu_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "llm_modules_loaded": llm_modules,
        "llm_disabled": os.environ.get("CARNOT_DISABLE_LLM") == "1" and not llm_modules,
        "arc_service_disabled": os.environ.get("CARNOT_DISABLE_ARC_SERVICE") == "1",
        "game_source_disabled": os.environ.get("CARNOT_DISABLE_GAME_SOURCE") == "1"
        and game_source_empty,
        "source_write_rows": source_rows,
        "source_tree_read_only": bool(source_rows)
        and all(row["write_denied"] for row in source_rows),
        "output_is_temporary": not str(
            os.environ.get("CARNOT_EXP6994_CHILD_OUTPUT", "")
        ).startswith(str(repo_root / "results")),
    }
    receipt["passed"] = bool(
        receipt["fresh_process"]
        and receipt["network_namespace_isolated"]
        and not gpu_devices
        and receipt["cuda_visible_devices"] == ""
        and receipt["llm_disabled"]
        and receipt["arc_service_disabled"]
        and receipt["game_source_disabled"]
        and receipt["source_tree_read_only"]
        and receipt["output_is_temporary"]
    )
    return receipt


def _fixture_rows_complete(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    required = {"fixture_id", "game", "scenario", "expected_eligible", "terminal"}
    ids = [row.get("fixture_id") for row in rows if isinstance(row, Mapping)]
    return bool(
        len(ids) == len(rows)
        and set(ids) == set(REQUIRED_FIXTURE_IDS)
        and len(ids) == len(set(ids))
        and all(
            isinstance(row, Mapping) and required <= set(row) and row.get("terminal") is True
            for row in rows
        )
    )


def _source_claim_rows(
    source: Mapping[str, Any], contract_score: int, path_score: int
) -> tuple[list[JsonDict], list[JsonDict]]:
    rows = []
    disagreements = []
    for field, observed in (
        ("arc_producer_contract_complete_score", contract_score),
        ("arc_live_path_fixture_ready_score", path_score),
    ):
        expected = source.get(field)
        passed = expected == observed
        row = {
            "kind": "source_readiness_replay",
            "field": field,
            "expected_value": expected,
            "observed_value": observed,
            "passed": passed,
            "terminal": True,
        }
        rows.append(row)
        if not passed:
            disagreements.append(
                {
                    "kind": "source_readiness_disagreement",
                    "field": field,
                    "expected_value": expected,
                    "observed_value": observed,
                    "terminal": True,
                }
            )
    return rows, disagreements


def _all_terminal(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, Mapping) and row.get("terminal") is True for row in value
    )


def build_from_sources(
    repo_root: Path,
    fixture_root: Path,
    *,
    run_date: str = RUN_DATE,
    loaded: Mapping[str, Any] | None = None,
    read_only_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the audit inside the restricted child after all hashes exist."""

    started = time.perf_counter()
    loaded_inputs = dict(loaded or load_hashed_inputs(repo_root, fixture_root))
    receipt = dict(read_only_receipt or sandbox_runtime_receipt(repo_root, fixture_root))
    preconditions = [dict(row) for row in loaded_inputs.get("preconditions", [])]
    preconditions.extend(
        (
            gate_check("execution_date", RUN_DATE, run_date),
            gate_check("read_only_sandbox", True, receipt.get("passed")),
        )
    )
    if loaded_inputs.get("passed") is not True:
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=preconditions,
            evidence={
                "source_artifact_hashes": loaded_inputs.get("hashes", {}),
                "read_only_enforcement_receipt": receipt,
            },
        )

    sources = loaded_inputs["sources"]
    source = sources["exp6993"]
    fixture_rows = sources["fixture_rows"]
    source_hashes = loaded_inputs["hashes"]["repository_sources"]
    source_factory_hash = (
        source.get("source_artifact_hashes", {}).get("live_policy_and_factory", {}).get("sha256")
    )
    success_dir = fixture_root / "success" / "attempts" / "evidence" / "success-run"
    preconditions.extend(
        (
            gate_check(
                "upstream_arc_producer_contract_complete_score",
                1,
                source.get("arc_producer_contract_complete_score"),
            ),
            gate_check(
                "upstream_arc_live_path_fixture_ready_score",
                1,
                source.get("arc_live_path_fixture_ready_score"),
            ),
            gate_check(
                "raw_fixture_rows",
                "complete",
                "complete" if _fixture_rows_complete(fixture_rows) else "missing",
            ),
            gate_check("complete_envelope_file", True, (success_dir / "envelope.json").is_file()),
            gate_check(
                "complete_manifest_file",
                True,
                (fixture_root / "success" / "attempts" / "manifest.jsonl").is_file(),
            ),
            gate_check(
                "shipped_factory_code", source_factory_hash, source_hashes.get("factory_policy")
            ),
        )
    )
    if any(row["passed"] is not True for row in preconditions):
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=preconditions,
            evidence={
                "source_artifact_hashes": loaded_inputs["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )

    replay = replay_fixtures(fixture_root, fixture_rows)
    success_summary = next(
        row for row in replay["per_fixture_rows"] if row["fixture_id"] == "success"
    )
    trace: list[JsonDict] = []
    routing_rows: list[JsonDict] = []
    influence_rows: list[JsonDict] = []
    if success_summary["eligible"]:
        trace, routing, influence = trace_factory_route(fixture_root, "success")
        routing_rows.append(routing)
        influence_rows.append(influence)

    only_success_eligible = [
        row["fixture_id"] for row in replay["eligibility_replay_rows"] if row["eligible"]
    ] == ["success"]
    success_hashes_pass = all(
        row["passed"]
        for field in HASH_FIELDS.values()
        for row in replay[field[1]]
        if row["fixture_id"] == "success"
    ) and all(
        row["passed"]
        for field in ("manifest_hash_replay_rows", "envelope_hash_replay_rows")
        for row in replay[field]
        if row["fixture_id"] == "success"
    )
    atomicity_pass = bool(replay["atomicity_replay_rows"]) and all(
        row["passed"] for row in replay["atomicity_replay_rows"]
    )
    expected_outcomes_pass = all(
        row["expected_outcome_matches"] for row in replay["per_fixture_rows"]
    )
    contract_score = int(
        success_hashes_pass
        and atomicity_pass
        and expected_outcomes_pass
        and only_success_eligible
        and not replay["source_disagreement_rows"]
    )
    path_score = int(
        bool(trace)
        and all(row["called"] for row in trace)
        and bool(routing_rows)
        and routing_rows[0]["engine_hash_matches_envelope"]
        and routing_rows[0]["selected_candidate_name"] == "loaded_world_model.py"
        and bool(influence_rows)
        and influence_rows[0]["score_changed"]
    )
    source_claim_rows, claim_disagreements = _source_claim_rows(source, contract_score, path_score)
    disagreements = [
        *replay["source_disagreement_rows"],
        *claim_disagreements,
    ]
    replay["rows"].extend(source_claim_rows)
    terminal_fields = (
        "rows",
        "per_fixture_rows",
        "prompt_hash_replay_rows",
        "transition_hash_replay_rows",
        "engine_hash_replay_rows",
        "environment_hash_replay_rows",
        "scorer_hash_replay_rows",
        "policy_hash_replay_rows",
        "factory_hash_replay_rows",
        "manifest_hash_replay_rows",
        "envelope_hash_replay_rows",
        "transition_order_rows",
        "source_kind_rows",
        "eligibility_replay_rows",
        "atomicity_replay_rows",
        "tamper_replay_rows",
        "legacy_replay_rows",
    )
    audit_complete = bool(
        len(replay["per_fixture_rows"]) == len(fixture_rows)
        and all(_all_terminal(replay[field]) for field in terminal_fields)
        and _all_terminal(trace)
        and _all_terminal(routing_rows)
        and _all_terminal(influence_rows)
    )
    confirmed = bool(
        audit_complete
        and contract_score == source.get("arc_producer_contract_complete_score") == 1
        and path_score == source.get("arc_live_path_fixture_ready_score") == 1
        and not disagreements
        and receipt.get("passed") is True
    )
    if confirmed:
        verdict_class = "positive"
        honest_verdict = "complete_positive_arc_producer_contract_confirmed"
    elif audit_complete:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_arc_producer_contract_disagreement"
    else:
        verdict_class = "partial"
        honest_verdict = "partial_arc_producer_contract_audit"
    evidence = {
        **{field: replay[field] for field in replay if field != "source_disagreement_rows"},
        "agent_factory_trace_rows": trace,
        "routing_replay_rows": routing_rows,
        "action_influence_replay_rows": influence_rows,
        "source_disagreement_rows": disagreements,
        "read_only_enforcement_receipt": receipt,
        "source_artifact_hashes": loaded_inputs["hashes"],
        "source_readiness_replay_rows": source_claim_rows,
        "arc_contract_audit_complete_score": int(audit_complete),
        "arc_producer_contract_confirmed_score": int(confirmed),
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    return build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        preconditions_checked=preconditions,
        evidence=evidence,
    )


def _checksum_value(artifact: Mapping[str, Any]) -> JsonDict:
    stable = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    receipt = stable.get("read_only_enforcement_receipt")
    if isinstance(receipt, dict):
        for key in (
            "fresh_process_pid",
            "parent_process_pid",
            "network_namespace",
            "parent_network_namespace",
            "sandbox_command_hash",
        ):
            receipt.pop(key, None)
    return stable


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable scientific content while excluding runtime identities."""

    return sha256_bytes(canonical_json_bytes(_checksum_value(artifact)))


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> JsonDict:
    """Build one complete blocked, partial, disqualified, or positive artifact."""

    checks = [deepcopy(dict(row)) for row in preconditions_checked]
    preconditions_pass = bool(checks) and all(row.get("passed") is True for row in checks)
    defaults = _empty_replay()
    defaults.update(
        {
            "source_artifact_hashes": {},
            "agent_factory_trace_rows": [],
            "routing_replay_rows": [],
            "action_influence_replay_rows": [],
            "read_only_enforcement_receipt": {},
            "arc_contract_audit_complete_score": 0,
            "arc_producer_contract_confirmed_score": 0,
            "verdict_class": "partial",
            "honest_verdict": "partial_arc_producer_contract_audit",
        }
    )
    defaults.update(deepcopy(dict(evidence)))
    if not preconditions_pass:
        defaults.update(
            {
                "arc_contract_audit_complete_score": 0,
                "arc_producer_contract_confirmed_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_arc_producer_cold_audit",
            }
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        **defaults,
        "solve_provenance_applicable": False,
        "solve_claimed": False,
        "level_claimed": False,
        "registry_updated": False,
        "submitted_to_leaderboard": False,
        "model_quality_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing evidence, overclaims, invalid verdicts, and checksum drift."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"required_fields_missing:{sorted(missing)}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in ("arc_contract_audit_complete_score", "arc_producer_contract_confirmed_score"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field}_not_bare_int")
        elif artifact.get(field) not in {0, 1}:
            errors.append(f"{field}_outside_binary_range")
    for field in (
        "solve_provenance_applicable",
        "solve_claimed",
        "level_claimed",
        "registry_updated",
        "submitted_to_leaderboard",
        "model_quality_claimed",
        "verifier_is_oracle",
    ):
        if artifact.get(field) is not False:
            errors.append(f"{field}_must_be_false")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    elif not verdict.startswith(VERDICT_PREFIXES[verdict_class]):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("arc_producer_contract_confirmed_score") == 1 and (
        artifact.get("arc_contract_audit_complete_score") != 1 or verdict_class != "positive"
    ):
        errors.append("confirmation_state_mismatch")
    checks = artifact.get("preconditions_checked", [])
    failed_precondition = not checks or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in checks
    )
    if failed_precondition and verdict_class != "blocked":
        errors.append("failed_precondition_not_blocked")
    if (
        verdict_class == "blocked"
        and artifact.get("gate_check_summary", {}).get("passed") is not False
    ):
        errors.append("blocked_without_failed_gate")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def fresh_process_command(
    *,
    executable: Path,
    wrapper: Path,
    repo_root: Path,
    fixture_root: Path,
    writable_root: Path,
    output_path: Path,
    run_date: str,
    parent_netns: str,
) -> list[str]:
    """Build a sandbox command with one narrow writable output directory."""

    game_source_mask = writable_root / "game-source-disabled"
    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--ro-bind",
        str(fixture_root),
        str(fixture_root),
        "--ro-bind",
        str(game_source_mask),
        str(repo_root / "environment_files"),
        "--bind",
        str(output_path.parent),
        str(output_path.parent),
        "--chdir",
        str(repo_root),
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        "--setenv",
        "JAX_PLATFORMS",
        "cpu",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "TMPDIR",
        str(output_path.parent),
        "--setenv",
        "CARNOT_DISABLE_LLM",
        "1",
        "--setenv",
        "CARNOT_DISABLE_ARC_SERVICE",
        "1",
        "--setenv",
        "CARNOT_DISABLE_GAME_SOURCE",
        "1",
        "--setenv",
        "CARNOT_EXP6994_PARENT_NETNS",
        parent_netns,
        "--setenv",
        "CARNOT_EXP6994_CHILD_OUTPUT",
        str(output_path),
        "--",
        str(executable),
        str(wrapper),
        "--fresh-child",
        "--date",
        run_date,
        "--fixture-root",
        str(fixture_root),
        "--output",
        str(output_path),
    ]


def _tree_hashes(root: Path) -> dict[str, str | None]:
    return {
        str(path.relative_to(root)): sha256_path(path)
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def run_controller(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:  # pragma: no cover - exercised by the required command.
    """Materialize fixtures, run the child, and publish its validated artifact."""

    started = time.perf_counter()
    final_path = result_path or (repo_root / RESULT_PATH)
    if shutil.which("bwrap") is None:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=[gate_check("bubblewrap_available", True, False)],
            evidence={},
        )
        write_json_atomic(final_path, artifact)
        return artifact

    registry = repo_root / SOURCE_PATHS["solve_registry"]
    registry_before = sha256_path(registry)
    with tempfile.TemporaryDirectory(prefix="carnot-exp6994-") as directory:
        temporary_root = Path(directory)
        fixture_root = temporary_root / "fixtures"
        output_root = temporary_root / "output"
        output_root.mkdir()
        game_source_mask = temporary_root / "game-source-disabled"
        game_source_mask.mkdir()
        child_output = output_root / "child-result.json"
        materialize_fixtures(fixture_root)
        repo_before = {
            source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()
        }
        fixture_before = _tree_hashes(fixture_root)
        parent_netns = os.readlink("/proc/self/ns/net")
        command = fresh_process_command(
            executable=Path(sys.executable),
            wrapper=repo_root / WRAPPER_PATH,
            repo_root=repo_root,
            fixture_root=fixture_root,
            writable_root=temporary_root,
            output_path=child_output,
            run_date=run_date,
            parent_netns=parent_netns,
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and child_output.is_file():
            artifact = json.loads(child_output.read_text(encoding="utf-8"))
        else:
            artifact = build_artifact(
                run_date=run_date,
                duration_s=time.perf_counter() - started,
                preconditions_checked=[
                    gate_check("fresh_process_exit_code", 0, completed.returncode),
                    gate_check("fresh_process_output", True, child_output.is_file()),
                ],
                evidence={},
            )
        repo_after = {
            source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()
        }
        fixture_after = _tree_hashes(fixture_root)
    registry_after = sha256_path(registry)
    unchanged = repo_before == repo_after and fixture_before == fixture_after
    receipt = dict(artifact.get("read_only_enforcement_receipt", {}))
    receipt.update(
        {
            "controller_source_hashes_before": repo_before,
            "controller_source_hashes_after": repo_after,
            "controller_fixture_hashes_unchanged": fixture_before == fixture_after,
            "controller_source_hashes_unchanged": unchanged,
            "fresh_child_exit_code": completed.returncode,
            "fresh_child_stderr_hash": sha256_bytes(completed.stderr.encode("utf-8")),
            "sandbox_command_hash": sha256_bytes(canonical_json_bytes(command)),
        }
    )
    receipt["passed"] = bool(receipt.get("passed") and unchanged and completed.returncode == 0)
    artifact["read_only_enforcement_receipt"] = receipt
    artifact["registry_updated"] = registry_before != registry_after
    artifact["duration_s"] = time.perf_counter() - started
    if not receipt["passed"] or artifact["registry_updated"]:
        checks = list(artifact.get("preconditions_checked", []))
        checks.extend(
            (
                gate_check("controller_read_only_enforcement", True, receipt["passed"]),
                gate_check("solve_registry_unchanged", registry_before, registry_after),
            )
        )
        artifact["preconditions_checked"] = checks
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["arc_contract_audit_complete_score"] = 0
        artifact["arc_producer_contract_confirmed_score"] = 0
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_arc_producer_cold_audit"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(final_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or its private restricted-child mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--fresh-child", action="store_true")
    args = parser.parse_args(argv)
    if args.fresh_child:
        if args.output is None or args.fixture_root is None:
            parser.error("--output and --fixture-root are required with --fresh-child")
        artifact = build_from_sources(REPO_ROOT, args.fixture_root, run_date=args.date)
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"child_artifact_validation_failed:{errors}")
        write_json_atomic(args.output, artifact)
        return 0
    run_controller(result_path=args.output, run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
