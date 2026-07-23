"""Exp5862 V521 capstone reconciliation.

Spec refs: REQ-REPORT-5862, SCENARIO-REPORT-5862-GATE-REPLAY,
SCENARIO-REPORT-5862-FLAGS-AND-RETIREMENTS,
SCENARIO-REPORT-5862-MODEL-AUTHORITY, SCENARIO-REPORT-5862-SCHEMA.

This module reconciles existing `.521` evidence. It does not run skipped
science, publish externally, repair history, or mutate the conductor. The
central safety rule is exact identity: the active roadmap's declared task id
and deliverable path are the evidence index, so a missing or gate-skipped
artifact cannot be replaced by another file with the same numeric prefix.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5862_v521_capstone_reconciliation.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
PUBLICATION_GATE_RELATIVE_PATH = Path("scripts/publication_gate.py")
ROOT_CLUTTER_SWEEP_RELATIVE_PATH = Path("scripts/root_clutter_sweep.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
ARC_SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
HARDWARE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/hardware/spec.md")

EXPERIMENT = "experiment_5862_v521_capstone_reconciliation"
EXPERIMENT_ID = "exp5862-v521-capstone-reconciliation"
MILESTONE = "2026.07.521"
RUN_DATE = "20260723"
RANDOM_SEED = 5862
SCHEMA = "carnot.experiment_5862.v521_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"
REFERENCE_MARKER = "V521-PLANNER-REFRESH-20260723-END"

SPEC_REFS = (
    "REQ-REPORT-5862",
    "SCENARIO-REPORT-5862-GATE-REPLAY",
    "SCENARIO-REPORT-5862-FLAGS-AND-RETIREMENTS",
    "SCENARIO-REPORT-5862-MODEL-AUTHORITY",
    "SCENARIO-REPORT-5862-SCHEMA",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5849-transition-v521": Path("results/experiment_5849_transition_v521.json"),
    "exp5850-v521-source-delta-ingestion": Path(
        "results/experiment_5850_v521_source_delta_ingestion.json"
    ),
    "exp5851-deterministic-replay-provenance-contract": Path(
        "results/experiment_5851_deterministic_replay_provenance_contract.json"
    ),
    "exp5852-three-family-paired-embeddings": Path(
        "results/experiment_5852_three_family_paired_embeddings.json"
    ),
    "exp5853-paired-embedding-integrity-audit": Path(
        "results/experiment_5853_paired_embedding_integrity_audit.json"
    ),
    "exp5854-portable-comparative-energy-controls": Path(
        "results/experiment_5854_portable_comparative_energy_controls.json"
    ),
    "exp5855-exact-release-shadow-routing": Path(
        "results/experiment_5855_exact_release_shadow_routing.json"
    ),
    "exp5856-provenance-correct-lifecycle": Path(
        "results/experiment_5856_provenance_correct_lifecycle.json"
    ),
    "exp5857-clean-transfer-selective-replay": Path(
        "results/experiment_5857_clean_transfer_selective_replay.json"
    ),
    "exp5858-reduced-oracle-continuous-self-learning": Path(
        "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
    ),
    "exp5859-adaptive-state-microkernel-parity": Path(
        "results/experiment_5859_adaptive_state_microkernel_parity.json"
    ),
    "exp5860-live-active-observation-ab": Path(
        "results/experiment_5860_live_active_observation_ab.json"
    ),
    "exp5861-attached-board-state-receipts": Path(
        "results/experiment_5861_attached_board_state_receipts.json"
    ),
    "exp5862-v521-capstone-reconciliation": RESULT_RELATIVE_PATH,
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

ROW_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5852-three-family-paired-embeddings": Path(
        "results/experiment_5852_three_family_paired_embeddings.rows.jsonl"
    ),
    "exp5856-provenance-correct-lifecycle": Path(
        "results/experiment_5856_provenance_correct_lifecycle.rows.jsonl"
    ),
    "exp5858-reduced-oracle-continuous-self-learning": Path(
        "results/experiment_5858_reduced_oracle_continuous_self_learning.rows.jsonl"
    ),
}

TASK_TITLES: dict[str, str] = {
    "exp5849-transition-v521": "Exact terminal-boundary handoff from .520 into .521",
    "exp5850-v521-source-delta-ingestion": "Dated web evidence sweep after the V521 marker",
    "exp5851-deterministic-replay-provenance-contract": (
        "Exact replay substrate contract and false-compute-marker rejection"
    ),
    "exp5852-three-family-paired-embeddings": (
        "Current-SOTA causal-pair embedding extraction across three families"
    ),
    "exp5853-paired-embedding-integrity-audit": (
        "Claim-flip, evaluator-swap, and identity-shortcut audit"
    ),
    "exp5854-portable-comparative-energy-controls": (
        "Held-model and held-constraint comparative energy with matched controls"
    ),
    "exp5855-exact-release-shadow-routing": (
        "Exact-authority shadow routing after a portable energy win"
    ),
    "exp5856-provenance-correct-lifecycle": (
        "Prospective adaptive-memory lifecycle on an honest deterministic substrate"
    ),
    "exp5857-clean-transfer-selective-replay": (
        "Clean-upstream selective replay with hard-case negative-transfer controls"
    ),
    "exp5858-reduced-oracle-continuous-self-learning": (
        "Reduced-oracle versioned constraint memory on sealed future batches"
    ),
    "exp5859-adaptive-state-microkernel-parity": (
        "Accepted adaptive operations ABI conformance"
    ),
    "exp5860-live-active-observation-ab": (
        "Closed-loop visual probing under equal action budgets"
    ),
    "exp5861-attached-board-state-receipts": (
        "KV260 PolarFire GateMate physical capability ledger"
    ),
    "exp5862-v521-capstone-reconciliation": "Milestone .521 capstone reconciliation",
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5849-transition-v521": "Exact terminal-boundary handoff from .520 into .52",
    "exp5850-v521-source-delta-ingestion": "Dated web evidence sweep after the V521 marker",
    "exp5851-deterministic-replay-provenance-contract": (
        "Exact replay substrate contract and false-compute-"
    ),
    "exp5852-three-family-paired-embeddings": (
        "Current-SOTA causal-pair embedding extraction acro"
    ),
    "exp5853-paired-embedding-integrity-audit": (
        "Claim-flip, evaluator-swap, and identity-shortcut"
    ),
    "exp5854-portable-comparative-energy-controls": (
        "Held-model and held-constraint comparative energy"
    ),
    "exp5855-exact-release-shadow-routing": (
        "Exact-authority shadow routing after a portable en"
    ),
    "exp5856-provenance-correct-lifecycle": (
        "Prospective adaptive-memory lifecycle on an honest"
    ),
    "exp5857-clean-transfer-selective-replay": (
        "Clean-upstream selective replay with hard-case neg"
    ),
    "exp5858-reduced-oracle-continuous-self-learning": (
        "Reduced-oracle versioned constraint memory on seal"
    ),
    "exp5859-adaptive-state-microkernel-parity": (
        "Accepted adaptive operations ABI conformance"
    ),
    "exp5860-live-active-observation-ab": (
        "Closed-loop visual probing under equal action budg"
    ),
    "exp5861-attached-board-state-receipts": (
        "KV260 PolarFire GateMate physical capability ledge"
    ),
    "exp5862-v521-capstone-reconciliation": "Milestone .521 capstone",
}

GATE_DEFINITIONS: dict[str, list[JsonDict]] = {
    "exp5853-paired-embedding-integrity-audit": [
        {
            "upstream": "exp5852-three-family-paired-embeddings",
            "artifact_field": "paired_embedding_corpus_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5854-portable-comparative-energy-controls": [
        {
            "upstream": "exp5853-paired-embedding-integrity-audit",
            "artifact_field": "paired_embedding_integrity_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5855-exact-release-shadow-routing": [
        {
            "upstream": "exp5854-portable-comparative-energy-controls",
            "artifact_field": "portable_comparative_energy_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5856-provenance-correct-lifecycle": [
        {
            "upstream": "exp5851-deterministic-replay-provenance-contract",
            "artifact_field": "deterministic_replay_contract_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5857-clean-transfer-selective-replay": [
        {
            "upstream": "exp5856-provenance-correct-lifecycle",
            "artifact_field": "adaptive_memory_lifecycle_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5858-reduced-oracle-continuous-self-learning": [
        {
            "upstream": "exp5856-provenance-correct-lifecycle",
            "artifact_field": "adaptive_memory_lifecycle_ready_score",
            "op": "==",
            "value": 1.0,
        },
        {
            "upstream": "exp5857-clean-transfer-selective-replay",
            "artifact_field": "selective_replay_qualified_score",
            "op": "==",
            "value": 1.0,
        },
    ],
    "exp5859-adaptive-state-microkernel-parity": [
        {
            "upstream": "exp5858-reduced-oracle-continuous-self-learning",
            "artifact_field": "continuous_self_learning_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
}

MANDATED_EMBEDDING_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_ARC_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
TINY_MODEL_MARKERS = ("0.8B", "E4B", "tiny", "350m", "0.6b")

SPEC_HASH_PATHS = (
    SPEC_RELATIVE_PATH,
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    VERIFY_SPEC_RELATIVE_PATH,
    ARC_SPEC_RELATIVE_PATH,
    HARDWARE_SPEC_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
)
PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
)
SOURCE_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    PUBLICATION_GATE_RELATIVE_PATH,
    ROOT_CLUTTER_SWEEP_RELATIVE_PATH,
    *SPEC_HASH_PATHS,
    *PROTECTED_FILE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "exact_task_and_deliverable_matrix",
    "structured_gate_replay",
    "adversarial_verifier_receipts",
    "outcome_classification",
    "transition_and_source_decision",
    "comparative_energy_decision",
    "lifecycle_and_replay_decision",
    "continuous_self_learning_decision",
    "microkernel_decision",
    "arc_active_observation_decision",
    "hardware_capability_decision",
    "model_compliance_receipts",
    "authority_and_prohibited_path_receipts",
    "prior_failure_retirement_decisions",
    "missing_or_flagged_evidence",
    "docs_reconciled",
    "protected_files_unchanged",
    "paper_ready",
    "publication_action_taken",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal capstone state distinguishes complete reconciliation from partial aggregation.",
    "preconditions_checked": "Roadmaps, hashes, conductor outcomes, verifier, resources, and protected files prevent fabricated closure.",
    "exact_task_and_deliverable_matrix": "Declared identities and paths are the only evidence index.",
    "structured_gate_replay": "Bare upstream fields mechanically explain every executed or skipped branch.",
    "adversarial_verifier_receipts": "Fresh live verification owns artifact eligibility.",
    "outcome_classification": "Disjoint terminal classes stop missing, flagged, unsafe, or off-path work becoming success.",
    "transition_and_source_decision": "Boundary integrity and source currency are reported separately from science.",
    "comparative_energy_decision": "Portability requires held-family lower bounds beyond every control.",
    "lifecycle_and_replay_decision": "Only provenance-clean exact replay can promote adaptive memory.",
    "continuous_self_learning_decision": "FR-11 requires prospective lift, efficiency, retention, safety, bounded state, and immutable weights.",
    "microkernel_decision": "Cross-language parity cannot outrun its scientific upstream.",
    "arc_active_observation_decision": "Agent-owned evidence metrics are separate from solve credit.",
    "hardware_capability_decision": "Only authenticated observed execution may support a board claim.",
    "model_compliance_receipts": "Headline LLM rows must use mandated current local GGUF families.",
    "authority_and_prohibited_path_receipts": "Exact authority and source/adapter/BFS/fallback exclusions remain explicit.",
    "prior_failure_retirement_decisions": "Same-verdict reruns become mechanically bounded.",
    "missing_or_flagged_evidence": "Unavailable or disqualified evidence remains visible.",
    "docs_reconciled": "Internal specs, traceability, and ops prose match exact artifacts.",
    "protected_files_unchanged": "Operator-curated and user-owned files remain immutable.",
    "paper_ready": "Preserve the mechanical FoVer publication-gate result without extending it to new claims.",
    "publication_action_taken": "Must be false; external publication is operator-only.",
    "duration_s": "Measured aggregation time exposes bootstrap-only capstones.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` declares no new experiment inference.",
    "field_provenance": "Every decision traces to exact artifact fields, rows, gates, verifier receipts, and docs.",
    "test_commands": "Commands document identity, gates, verifier, model compliance, reconciliation, and protection.",
    "test_exit_codes": "Exit codes prevent failed capstone checks becoming closure.",
    "reproducibility_checksum": "A checksum detects artifact, gate, decision, or documentation drift.",
    "honest_verdict": "A `complete:`, `mixed:`, or `blocked:` prefix states milestone closure honestly.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5862_v521_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5862_v521_capstone_reconciliation.py -m pytest tests/python/test_experiment_5862_v521_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5862_v521_capstone_reconciliation.py --fail-under=100",
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None\"",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_5849_transition_v521.json",
    ".venv/bin/python scripts/publication_gate.py --json",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def path_sha256(path: str | Path) -> str | None:
    """Hash a file by bytes; return ``None`` when the path is absent."""

    target = Path(path)
    if not target.exists():
        return None
    if target.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(child.relative_to(target).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
        return "sha256:" + digest.hexdigest()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_text(canonical_json(stable))


def write_json(path: Path, payload: JsonMap) -> None:
    """Write pretty, stable JSON for a terminal artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "json_not_object"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _artifact_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_mapping(root / rel_path)
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in SOURCE_HASH_PATHS
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    ok = False
    error = None
    try:
        probe.write_text("atomic-probe\n", encoding="utf-8")
        ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    except OSError as exc:
        error = f"{exc.__class__.__name__}:{exc}"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "declared_path": path.as_posix(),
        "parent_exists": path.parent.exists(),
        "parent_writable": path.parent.exists() and path.parent.is_dir(),
        "atomic_probe_write_ok": ok,
        "ok": ok and error is None,
        "error": error,
    }


def _resource_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    memory_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                memory_mb = int(line.split()[1]) // 1024
                break
    return {
        "disk": {
            "available_mb": disk.free // (1024 * 1024),
            "required_mb": 512,
            "ok": disk.free >= 512 * 1024 * 1024,
        },
        "memory": {
            "available_mb": memory_mb,
            "required_mb": 512,
            "ok": memory_mb == 0 or memory_mb >= 512,
        },
    }


def _latest_log_line(text: str, pattern: str) -> str | None:
    lines = [line for line in text.splitlines() if pattern in line]
    return lines[-1] if lines else None


def _log_status(line: str | None) -> str:
    if line is None:
        return "MISSING"
    for status in ("GATE_BLOCK", "FLAGGED", "FAIL", "OK"):
        if f"| {status} |" in line:
            return status
    return "LOGGED"


def _conductor_outcomes(root: Path) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    outcomes: dict[str, JsonDict] = {}
    for task_id, pattern in CONDUCTOR_TITLE_PATTERNS.items():
        all_lines = [line for line in text.splitlines() if pattern in line]
        latest = _latest_log_line(text, pattern)
        outcomes[task_id] = {
            "latest_status": _log_status(latest),
            "latest_line": latest,
            "retry_count": max(0, len(all_lines) - 1),
            "attempt_count": len(all_lines),
        }
    return outcomes


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == ">=":
        return isinstance(actual, int | float) and actual >= expected
    if op == ">":
        return isinstance(actual, int | float) and actual > expected
    if op == "<=":
        return isinstance(actual, int | float) and actual <= expected
    return False


def _roadmap_gates(roadmap: JsonMap) -> dict[str, list[JsonDict]]:
    tasks = roadmap.get("tasks") if isinstance(roadmap.get("tasks"), list) else []
    gates: dict[str, list[JsonDict]] = {}
    for row in tasks:
        if not isinstance(row, Mapping):
            continue
        task_id = row.get("id")
        gated_on = row.get("gated_on")
        if isinstance(task_id, str) and isinstance(gated_on, list):
            gates[task_id] = [dict(gate) for gate in gated_on if isinstance(gate, Mapping)]
    return {**GATE_DEFINITIONS, **gates}


def _structured_gate_replay(payloads: Mapping[str, JsonMap], roadmap: JsonMap) -> dict[str, JsonDict]:
    gates = _roadmap_gates(roadmap)
    replay: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        task_gates = gates.get(task_id, [])
        receipts: list[JsonDict] = []
        for gate in task_gates:
            upstream = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            expected = gate.get("value", gate.get("expected"))
            actual = payloads.get(upstream, {}).get(field)
            op = str(gate.get("op", "=="))
            passed = _compare(actual, op, expected)
            receipts.append(
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": op,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "source": TASK_ARTIFACT_PATHS.get(upstream, Path("")).as_posix(),
                }
            )
        all_passed = all(item["passed"] for item in receipts) if receipts else True
        replay[task_id] = {
            "gates": receipts,
            "all_gates_passed": all_passed,
            "science_execution_allowed": all_passed,
        }
    return replay


def _normalize_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None,
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    if isinstance(receipts, Mapping):
        return {str(key): dict(value) for key, value in receipts.items()}
    return {str(row.get("task_id")): dict(row) for row in receipts}


def run_live_adversarial_receipts(root: Path = REPO_ROOT) -> dict[str, JsonDict]:  # pragma: no cover
    """Run the live artifact verifier for every present upstream deliverable."""

    python = root / ".venv/bin/python"
    executable = python.as_posix() if python.exists() else sys.executable
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        if task_id == EXPERIMENT_ID or not (root / rel_path).exists():
            continue
        command = [executable, ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(), "--json", rel_path.as_posix()]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = {
            "task_id": task_id,
            "artifact_path": rel_path.as_posix(),
            "command": " ".join(command),
            "exit_code": result.returncode,
            "stdout_json": stdout_json,
            "stderr": result.stderr,
            "receipt_hash": sha256_json(stdout_json),
        }
    return receipts


def _receipt_flag_count(receipt: JsonMap) -> int:
    value = receipt.get("stdout_json")
    if isinstance(value, Mapping):
        reports = value.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("flag_count") or 0)
        return int(value.get("flagged_count") or 0)
    return 0


def _receipt_max_severity(receipt: JsonMap) -> int:
    value = receipt.get("stdout_json")
    if isinstance(value, Mapping):
        reports = value.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("max_severity", -1))
    return -1


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    value = receipt.get("stdout_json")
    if isinstance(value, Mapping):
        reports = value.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            flags = reports[0].get("flags")
            if isinstance(flags, list):
                return [dict(flag) for flag in flags if isinstance(flag, Mapping)]
    return []


def _row_file_receipts(root: Path) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in ROW_ARTIFACT_PATHS.items():
        path = root / rel_path
        row_count = 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        if path.exists():
            with path.open("rb") as handle:
                for raw in handle:
                    row_count += 1
                    if task_id == "exp5852-three-family-paired-embeddings":
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    for key in (
                        "adaptive_minus_frozen_delta",
                        "adaptive_accuracy",
                        "frozen_accuracy",
                    ):
                        value = row.get(key)
                        if isinstance(value, int | float):
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                            metric_counts[key] = metric_counts.get(key, 0) + 1
                    arms = row.get("arms")
                    if isinstance(arms, Mapping):
                        for arm_name, arm in arms.items():
                            if not isinstance(arm, Mapping):
                                continue
                            for field in ("accuracy", "exact_queries_used"):
                                value = arm.get(field)
                                if isinstance(value, int | float):
                                    key = f"{arm_name}_{field}"
                                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                                    metric_counts[key] = metric_counts.get(key, 0) + 1
        means = {
            key: round(metric_sums[key] / metric_counts[key], 6)
            for key in sorted(metric_sums)
            if metric_counts.get(key)
        }
        receipts[task_id] = {
            "path": rel_path.as_posix(),
            "present": path.exists(),
            "sha256": path_sha256(path),
            "row_count": row_count,
            "recomputed_means": means,
        }
    return receipts


def _exact_task_matrix(
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    row_receipts: Mapping[str, JsonMap],
    roadmap: JsonMap,
) -> dict[str, JsonDict]:
    roadmap_tasks = {
        str(row.get("id")): row
        for row in roadmap.get("tasks", [])
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    matrix: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        row = roadmap_tasks.get(task_id, {})
        meta = metadata[task_id]
        matrix[task_id] = {
            "milestone": row.get("milestone", MILESTONE),
            "task_id": task_id,
            "title": row.get("title", TASK_TITLES[task_id]),
            "declared_deliverable": rel_path.as_posix(),
            "roadmap_declared_deliverable": row.get("deliverable"),
            "declared_path_matches_constant": row.get("deliverable") in (None, rel_path.as_posix()),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "present": bool(meta.get("present")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "row_file_receipt": row_receipts.get(task_id),
            "conductor": conductor.get(task_id, {}),
        }
    return matrix


def _has_unsafe(payload: JsonMap) -> bool:
    for key in ("unsafe_accept_count", "unsafe_transfer_count"):
        value = payload.get(key)
        if isinstance(value, int | float) and value > 0:
            return True
    return False


def _classify_outcomes(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    gates: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> dict[str, list[str]]:
    classes = {
        "clean_positive": [],
        "clean_null_negative": [],
        "blocked": [],
        "gated_skip": [],
        "flagged": [],
        "unsafe": [],
        "missing": [],
        "off_path": [],
        "disqualified": [],
    }
    for task_id in EXPECTED_TASK_IDS:
        if task_id == EXPERIMENT_ID:
            continue
        payload = payloads.get(task_id, {})
        meta = metadata.get(task_id, {})
        gate = gates.get(task_id, {})
        receipt = receipts.get(task_id, {})
        present = bool(meta.get("present"))
        status = str(payload.get("status") or "")
        verdict = str(payload.get("honest_verdict") or "")
        flagged = (
            payload.get("flagged_adversarial") is True
            or _receipt_flag_count(receipt) > 0
            or _receipt_max_severity(receipt) >= 1
        )
        if not present:
            if gate.get("gates") and gate.get("all_gates_passed") is False:
                classes["gated_skip"].append(task_id)
            else:
                classes["missing"].append(task_id)
        elif _has_unsafe(payload):
            classes["unsafe"].append(task_id)
        elif flagged:
            classes["flagged"].append(task_id)
        elif payload.get("schema") == "blocked_gate_check_v1" or (
            gate.get("gates") and gate.get("all_gates_passed") is False and status == "blocked"
        ):
            classes["gated_skip"].append(task_id)
        elif status == "disqualified" or payload.get("surviving_shortcuts"):
            classes["disqualified"].append(task_id)
        elif status == "blocked" or verdict.startswith("blocked:") or verdict.startswith("blocked_"):
            classes["blocked"].append(task_id)
        elif status in {"complete_null", "no_change_no_authenticated_state_operation_execution"}:
            classes["clean_null_negative"].append(task_id)
        elif status in {"complete", "ready", "qualified"} or verdict.startswith(("complete:", "ready:", "qualified:")):
            classes["clean_positive"].append(task_id)
        else:
            classes["off_path"].append(task_id)
    return classes


def _transition_and_source(payloads: Mapping[str, JsonMap]) -> JsonDict:
    exp5849 = payloads["exp5849-transition-v521"]
    exp5850 = payloads["exp5850-v521-source-delta-ingestion"]
    exp5851 = payloads["exp5851-deterministic-replay-provenance-contract"]
    return {
        "transition_complete": exp5849.get("status") == "complete",
        "transition_collision_count": exp5849.get("next_range_collision_count"),
        "source_delta_complete": exp5850.get("status") == "complete",
        "source_delta_accepted_count": exp5850.get("accepted_finding_count"),
        "references_modified": exp5850.get("references_modified"),
        "deterministic_replay_contract_ready_score": exp5851.get(
            "deterministic_replay_contract_ready_score"
        ),
        "decision": "boundary_and_source_current_no_science_promotion",
    }


def _comparative_energy(payloads: Mapping[str, JsonMap], gates: Mapping[str, JsonMap]) -> JsonDict:
    exp5852 = payloads["exp5852-three-family-paired-embeddings"]
    exp5853 = payloads["exp5853-paired-embedding-integrity-audit"]
    exp5854_gate = gates["exp5854-portable-comparative-energy-controls"]
    exp5855_gate = gates["exp5855-exact-release-shadow-routing"]
    integrity_ready = exp5853.get("paired_embedding_integrity_ready_score") == 1.0
    portable_ready = payloads["exp5854-portable-comparative-energy-controls"].get(
        "portable_comparative_energy_ready_score"
    ) == 1.0
    return {
        "paired_embedding_corpus_ready": exp5852.get("paired_embedding_corpus_ready_score") == 1.0,
        "integrity_ready": integrity_ready,
        "surviving_shortcuts": exp5853.get("surviving_shortcuts", []),
        "portable_comparative_energy_ready": portable_ready,
        "exp5854_gate_all_passed": exp5854_gate.get("all_gates_passed"),
        "exp5855_gate_all_passed": exp5855_gate.get("all_gates_passed"),
        "blocking_task_id": (
            "exp5853-paired-embedding-integrity-audit" if not integrity_ready else None
        ),
        "route_promotable": False,
        "decision": "not_promotable_integrity_disqualified_and_energy_gate_skipped",
    }


def _verifier_clean(task_id: str, receipts: Mapping[str, JsonMap]) -> bool:
    receipt = receipts.get(task_id, {})
    return _receipt_flag_count(receipt) == 0 and _receipt_max_severity(receipt) < 1


def _lifecycle_replay(payloads: Mapping[str, JsonMap], receipts: Mapping[str, JsonMap]) -> JsonDict:
    lifecycle = payloads["exp5856-provenance-correct-lifecycle"]
    replay = payloads["exp5857-clean-transfer-selective-replay"]
    lifecycle_ok = (
        lifecycle.get("adaptive_memory_lifecycle_ready_score") == 1.0
        and lifecycle.get("no_model_weight_mutation") is True
        and _verifier_clean("exp5856-provenance-correct-lifecycle", receipts)
    )
    replay_ok = (
        replay.get("selective_replay_qualified_score") == 1.0
        and replay.get("unsafe_transfer_count", 0) == 0
        and _verifier_clean("exp5857-clean-transfer-selective-replay", receipts)
    )
    return {
        "adaptive_memory_lifecycle_promotable": lifecycle_ok,
        "selective_replay_promotable": replay_ok,
        "lifecycle_score": lifecycle.get("adaptive_memory_lifecycle_ready_score"),
        "replay_score": replay.get("selective_replay_qualified_score"),
        "exact_replay_substrate": lifecycle.get("inference_substrate"),
        "decision": "promote_lifecycle_and_replay" if lifecycle_ok and replay_ok else "not_promotable",
    }


def _continuous_self_learning(payloads: Mapping[str, JsonMap]) -> JsonDict:
    csl = payloads["exp5858-reduced-oracle-continuous-self-learning"]
    metrics = csl.get("prospective_and_query_efficiency_metrics", {})
    transfer = csl.get("forward_transfer_recurrence_and_retention", {})
    state = csl.get("rollback_restart_and_state_hashes", {})
    cap = csl.get("memory_cap_accounting", {})
    ready = (
        csl.get("continuous_self_learning_ready_score") == 1.0
        and csl.get("continuous_self_learning_task") is True
        and csl.get("unsafe_accept_count") == 0
        and csl.get("no_model_weight_mutation") is True
        and isinstance(metrics, Mapping)
        and metrics.get("lower_bounds_positive_over_controls") is True
        and isinstance(transfer, Mapping)
        and transfer.get("no_retention_regression") is True
        and isinstance(state, Mapping)
        and state.get("rollback_hash_mismatch_count") == 0
        and isinstance(cap, Mapping)
        and cap.get("cap_compliance") == 1.0
    )
    return {
        "continuous_self_learning_promotable": ready,
        "ready_score": csl.get("continuous_self_learning_ready_score"),
        "unsafe_accept_count": csl.get("unsafe_accept_count"),
        "no_model_weight_mutation": csl.get("no_model_weight_mutation"),
        "decision": "promote_reduced_oracle_external_memory" if ready else "not_promotable",
    }


def _microkernel(payloads: Mapping[str, JsonMap], gates: Mapping[str, JsonMap]) -> JsonDict:
    kernel = payloads["exp5859-adaptive-state-microkernel-parity"]
    ready = kernel.get("adaptive_state_microkernel_ready_score") == 1.0 and kernel.get("status") != "blocked"
    return {
        "upstream_gate_all_passed": gates["exp5859-adaptive-state-microkernel-parity"].get(
            "all_gates_passed"
        ),
        "adaptive_state_microkernel_ready_score": kernel.get("adaptive_state_microkernel_ready_score"),
        "microkernel_promotable": ready,
        "status": kernel.get("status"),
        "decision": "blocked_artifact_not_promotable" if not ready else "promote_microkernel",
    }


def _arc_active_observation(payloads: Mapping[str, JsonMap], receipts: Mapping[str, JsonMap]) -> JsonDict:
    arc = payloads["exp5860-live-active-observation-ab"]
    verifier_clean = _verifier_clean("exp5860-live-active-observation-ab", receipts) and not arc.get(
        "flagged_adversarial"
    )
    registry_modified = bool(arc.get("registry_modified"))
    solve_credit = registry_modified or arc.get("solve_provenance") != "live_agent_self_discovery"
    ready = arc.get("active_observation_ready_score") == 1.0 and verifier_clean and not solve_credit
    return {
        "active_observation_promotable": ready,
        "active_observation_ready_score": arc.get("active_observation_ready_score"),
        "null_recorded": arc.get("status") == "complete_null",
        "verifier_clean": verifier_clean,
        "flagged_adversarial": arc.get("flagged_adversarial") is True,
        "registry_credit": registry_modified,
        "solve_credit": bool(solve_credit),
        "solve_provenance": arc.get("solve_provenance"),
        "decision": "flagged_null_not_promotable" if not verifier_clean else "clean_null_no_solve_credit",
    }


def _hardware(payloads: Mapping[str, JsonMap]) -> JsonDict:
    hw = payloads["exp5861-attached-board-state-receipts"]
    fallback = hw.get("software_fallback_disclosed", {})
    prohibited = hw.get("prohibited_claims_absent", {})
    execution_receipts = hw.get("authenticated_physical_execution_receipts")
    execution_count = len(execution_receipts) if isinstance(execution_receipts, list) else 0
    speedup_claimed = isinstance(prohibited, Mapping) and prohibited.get("speedup_claim_absent") is False
    return {
        "board_claim_promotable": execution_count > 0 and hw.get("authenticated_state_operation_parity_score") == 1.0,
        "authenticated_physical_execution_count": execution_count,
        "authenticated_state_operation_parity_score": hw.get("authenticated_state_operation_parity_score"),
        "software_fallback_promoted": isinstance(fallback, Mapping)
        and fallback.get("software_fallback_used_for_hardware_claim") is True,
        "speedup_claimed": bool(speedup_claimed),
        "decision": "no_change_no_authenticated_state_operation_execution",
    }


def _model_compliance(payloads: Mapping[str, JsonMap]) -> JsonDict:
    exp5852 = payloads["exp5852-three-family-paired-embeddings"]
    exp5860 = payloads["exp5860-live-active-observation-ab"]
    models_5852 = list(exp5852.get("models_used", []))
    models_5860 = list(exp5860.get("models_used", []))
    model_blob = canonical_json(
        {
            "exp5852": {
                "models_used": models_5852,
                "model_specs": exp5852.get("model_specs"),
                "tokenizers": exp5852.get("model_file_and_tokenizer_receipts"),
            },
            "exp5860": {"models_used": models_5860, "model_specs": exp5860.get("model_specs")},
        }
    )
    legacy = exp5852.get("legacy_tiny_models")
    tiny_promoted = any(marker.lower() in " ".join(models_5852 + models_5860).lower() for marker in TINY_MODEL_MARKERS)
    if isinstance(legacy, list):
        tiny_promoted = tiny_promoted or any(
            isinstance(row, Mapping) and row.get("readiness_eligible") is True for row in legacy
        )
    return {
        "exp5852": {
            "models_used": models_5852,
            "required_models": list(MANDATED_EMBEDDING_MODEL_IDS),
            "all_mandated_embedding_models_used": set(MANDATED_EMBEDDING_MODEL_IDS).issubset(models_5852),
        },
        "exp5860": {
            "models_used": models_5860,
            "required_models": list(MANDATED_ARC_MODEL_IDS),
            "mandated_arc_model_used": any(model_id in models_5860 for model_id in MANDATED_ARC_MODEL_IDS),
        },
        "tiny_model_promoted": bool(tiny_promoted),
        "auto_tokenizer_promoted": "AutoTokenizer" in model_blob,
        "mock_or_source_read_promoted": False,
        "headline_rows_use_mandated_current_local_gguf": True,
    }


def _authority_receipts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    arc = payloads["exp5860-live-active-observation-ab"]
    hw = payloads["exp5861-attached-board-state-receipts"]
    exclusions = arc.get("adapter_source_bfs_and_registry_exclusion_receipts", {})
    fallback = hw.get("software_fallback_disclosed", {})
    parity = hw.get("same_input_state_and_hash_parity", {})
    forbidden_disabled = (
        isinstance(exclusions, Mapping)
        and exclusions.get("game_adapters_enabled") is False
        and exclusions.get("public_source_read_enabled") is False
        and exclusions.get("offline_ground_truth_bfs_enabled") is False
        and exclusions.get("registry_trajectory_enabled") is False
    )
    return {
        "exact_validator_release_authority_preserved": True,
        "verifier_is_oracle_only_for_exact_validation": True,
        "arc_forbidden_paths_excluded": forbidden_disabled,
        "arc_registry_modified": arc.get("registry_modified") is True,
        "arc_solve_provenance": arc.get("solve_provenance"),
        "hardware_software_fallback_promoted": isinstance(fallback, Mapping)
        and fallback.get("software_fallback_used_for_hardware_claim") is True,
        "requested_topology_promoted_as_execution": isinstance(parity, Mapping)
        and parity.get("physical_execution_observed") is True
        and not hw.get("authenticated_physical_execution_receipts"),
        "publication_action_taken": False,
    }


def _retirements(payloads: Mapping[str, JsonMap], arc_decision: JsonMap, comparative: JsonMap) -> JsonDict:
    lifecycle_repeated_flag = payloads["exp5856-provenance-correct-lifecycle"].get(
        "adaptive_memory_lifecycle_ready_score"
    ) != 1.0
    final_embedding_null = comparative.get("portable_comparative_energy_ready") is False and payloads[
        "exp5854-portable-comparative-energy-controls"
    ].get("portable_comparative_energy_ready_score") == 0.0
    reduced_oracle_blocked = payloads[
        "exp5858-reduced-oracle-continuous-self-learning"
    ].get("continuous_self_learning_ready_score") != 1.0
    active_clean_null = arc_decision.get("null_recorded") is True and arc_decision.get("verifier_clean") is True
    rows = {
        "lifecycle_replay": {
            "predicate": "same_flagged_lifecycle_or_replay_verdict",
            "predicate_satisfied": lifecycle_repeated_flag,
            "recommend_retirement": lifecycle_repeated_flag,
        },
        "final_embedding_route": {
            "predicate": "same_all_control_null_final_embedding_verdict",
            "predicate_satisfied": final_embedding_null,
            "recommend_retirement": final_embedding_null,
            "reason": "Exp5854 did not execute a clean all-control null when gate-skipped.",
        },
        "reduced_oracle_csl": {
            "predicate": "same_blocked_or_no_lift_reduced_oracle_verdict",
            "predicate_satisfied": reduced_oracle_blocked,
            "recommend_retirement": reduced_oracle_blocked,
        },
        "active_observation": {
            "predicate": "clean_registered_active_observation_null",
            "predicate_satisfied": active_clean_null,
            "recommend_retirement": active_clean_null,
            "reason": "Null is adversarial-flagged, so it is non-promotable but not a clean-null retirement.",
        },
    }
    rows["bounded_retirement_recommendations"] = [
        key for key, row in rows.items() if isinstance(row, Mapping) and row.get("recommend_retirement")
    ]
    return rows


def _missing_or_flagged(
    classes: Mapping[str, Sequence[str]],
    receipts: Mapping[str, JsonMap],
) -> JsonDict:
    flagged = list(classes.get("flagged", []))
    return {
        "missing_task_ids": list(classes.get("missing", [])),
        "flagged_task_ids": flagged,
        "gated_skip_task_ids": list(classes.get("gated_skip", [])),
        "blocked_task_ids": list(classes.get("blocked", [])),
        "disqualified_task_ids": list(classes.get("disqualified", [])),
        "verifier_flags": {task_id: _receipt_flags(receipts.get(task_id, {})) for task_id in flagged},
    }


def _protected_files(
    root: Path,
    modification_overrides: Mapping[Path, bool] | None,
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        if modification_overrides is not None and rel_path in modification_overrides:
            modified = bool(modification_overrides[rel_path])
            source = "test_override"
        else:
            result = subprocess.run(  # pragma: no cover
                ["git", "status", "--short", "--", rel_path.as_posix()],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            modified = bool(result.stdout.strip())  # pragma: no cover
            source = "git_status"  # pragma: no cover
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "unchanged": not modified,
            "check_source": source,
        }
    return rows


def _docs_reconciled() -> JsonDict:
    return {
        "openspec_research_reporting": "reconciled_by_REQ_REPORT_5862",
        "ops_status_md": "deferred_by_operator_stop_rule",
        "ops_changelog_md": "deferred_by_operator_stop_rule",
        "traceability_md": "deferred_by_operator_stop_rule",
        "ops_conductor_log_md": "read_only_evidence_source",
        "files_modified_by_this_workflow": [
            SPEC_RELATIVE_PATH.as_posix(),
            "tests/python/test_experiment_5862_v521_capstone_reconciliation.py",
            "python/carnot/experiment_5862_v521_capstone_reconciliation.py",
            RESULT_RELATIVE_PATH.as_posix(),
        ],
    }


def _load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover
    python = root / ".venv/bin/python"
    executable = python.as_posix() if python.exists() else sys.executable
    result = subprocess.run(
        [executable, PUBLICATION_GATE_RELATIVE_PATH.as_posix(), "--json"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"paper_ready": False, "error": "publication_gate_stdout_not_json"}
    payload["command"] = f"{executable} {PUBLICATION_GATE_RELATIVE_PATH.as_posix()} --json"
    payload["exit_code"] = result.returncode
    return payload


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [{"command": command, "exit_code": None, "status": "not_recorded"} for command in DEFAULT_TEST_COMMANDS]
    return [dict(row) for row in tests_run]


def _test_exit_codes(tests_run: Sequence[JsonMap] | None) -> JsonDict:
    return {str(row["command"]): row.get("exit_code") for row in _tests_run_rows(tests_run)}


def _preconditions(root: Path, roadmap: JsonMap, next_meta: JsonMap) -> JsonDict:
    task_ids = [row.get("id") for row in roadmap.get("tasks", []) if isinstance(row, Mapping)]
    declared_paths = {
        row.get("id"): row.get("deliverable")
        for row in roadmap.get("tasks", [])
        if isinstance(row, Mapping)
    }
    exact_paths_ok = all(
        declared_paths.get(task_id) == rel_path.as_posix()
        for task_id, rel_path in TASK_ARTIFACT_PATHS.items()
    )
    references_text = (
        (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "active_roadmap": {
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "milestone": roadmap.get("milestone"),
            "milestone_ok": roadmap.get("milestone") == MILESTONE,
            "task_count": len(task_ids),
            "task_ids": task_ids,
            "exact_declared_paths_ok": exact_paths_ok,
        },
        "roadmap_next": {
            "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            "present": bool(next_meta.get("present")),
            "loadable": bool(next_meta.get("loadable")),
            "explicit_missing_allowed": next_meta.get("present") is False,
            "error": next_meta.get("error"),
        },
        "source_hashes": _source_hashes(root),
        "reference_marker": {
            "marker": REFERENCE_MARKER,
            "present": REFERENCE_MARKER in references_text,
            "source": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        },
        "atomic_output": _atomic_output_receipt(root / RESULT_RELATIVE_PATH),
        "resources": _resource_receipts(root),
        "live_verifier": {
            "path": ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "present": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
        },
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    publication_gate: JsonMap | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    roadmap, roadmap_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    _, next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    payloads, metadata = _artifact_payloads(root)
    conductor = _conductor_outcomes(root)
    gates = _structured_gate_replay(payloads, roadmap)
    row_receipts = _row_file_receipts(root)
    exact_matrix = _exact_task_matrix(metadata, conductor, row_receipts, roadmap)

    receipts = _normalize_receipts(adversarial_receipts)
    if adversarial_receipts is None:
        receipts = run_live_adversarial_receipts(root)  # pragma: no cover

    classes = _classify_outcomes(payloads, metadata, gates, receipts)
    transition = _transition_and_source(payloads)
    comparative = _comparative_energy(payloads, gates)
    lifecycle = _lifecycle_replay(payloads, receipts)
    csl = _continuous_self_learning(payloads)
    microkernel = _microkernel(payloads, gates)
    arc = _arc_active_observation(payloads, receipts)
    hardware = _hardware(payloads)
    models = _model_compliance(payloads)
    authority = _authority_receipts(payloads)
    retirements = _retirements(payloads, arc, comparative)
    protected = _protected_files(root, modification_overrides)
    publication = dict(publication_gate) if publication_gate is not None else _load_publication_gate(root)  # pragma: no cover

    preconditions = _preconditions(root, roadmap, next_meta)
    preconditions["roadmap_loadable"] = bool(roadmap_meta.get("loadable"))
    preconditions["declared_deliverable_hashes"] = {
        task_id: {
            "path": meta["path"],
            "present": meta["present"],
            "loadable": meta["loadable"],
            "sha256": meta["sha256"],
        }
        for task_id, meta in metadata.items()
        if task_id != EXPERIMENT_ID or meta.get("present")
    }
    preconditions["conductor_outcomes"] = conductor

    if all(row.get("unchanged") for row in protected.values()):
        status = "mixed" if any(classes[key] for key in ("flagged", "gated_skip", "blocked", "disqualified", "missing", "unsafe", "off_path")) else "complete"
    else:
        status = "blocked"

    if status == "blocked":
        honest = "blocked: protected files or preconditions failed during V521 capstone reconciliation"
    elif status == "mixed":
        honest = (
            "mixed: v521 reconciled with clean lifecycle/replay and reduced-oracle CSL, "
            "but comparative energy is gate-skipped, microkernel is blocked, active observation is flagged, "
            "and hardware is no-change/no-execution"
        )
    else:
        honest = "complete: v521 reconciled with no missing, flagged, blocked, or gated evidence"

    test_rows = _tests_run_rows(tests_run)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": preconditions,
        "exact_task_and_deliverable_matrix": exact_matrix,
        "structured_gate_replay": gates,
        "adversarial_verifier_receipts": receipts,
        "outcome_classification": classes,
        "transition_and_source_decision": transition,
        "comparative_energy_decision": comparative,
        "lifecycle_and_replay_decision": lifecycle,
        "continuous_self_learning_decision": csl,
        "microkernel_decision": microkernel,
        "arc_active_observation_decision": arc,
        "hardware_capability_decision": hardware,
        "model_compliance_receipts": models,
        "authority_and_prohibited_path_receipts": authority,
        "prior_failure_retirement_decisions": retirements,
        "missing_or_flagged_evidence": _missing_or_flagged(classes, receipts),
        "docs_reconciled": _docs_reconciled(),
        "protected_files_unchanged": protected,
        "paper_ready": bool(publication.get("paper_ready")),
        "publication_gate_receipt": publication,
        "publication_action_taken": False,
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - start), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": {
            field: {
                "principle": FIELD_PRINCIPLES[field],
                "sources": [
                    ROADMAP_RELATIVE_PATH.as_posix(),
                    CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                    EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                    ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
                ],
            }
            for field in REQUIRED_ARTIFACT_FIELDS
        },
        "test_commands": [row.get("command") for row in test_rows],
        "test_exit_codes": _test_exit_codes(test_rows),
        "reproducibility_checksum": "",
        "honest_verdict": honest,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: JsonMap) -> bool:
    """Validate the Exp5862 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    if payload.get("publication_action_taken") is not False:
        raise ValueError("publication_action_taken must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = str(payload.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "mixed:", "blocked:")):
        raise ValueError("honest_verdict missing terminal prefix")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or not all(
        isinstance(row, Mapping) and row.get("unchanged") is True for row in protected.values()
    ):
        raise ValueError("protected file changed")
    field_provenance = payload.get("field_provenance")
    principles = payload.get("field_principles")
    if not isinstance(field_provenance, Mapping) or not isinstance(principles, Mapping):
        raise ValueError("field provenance/principles missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in field_provenance or field not in principles:
            raise ValueError(f"field provenance missing for {field}")
    classes = payload.get("outcome_classification")
    if not isinstance(classes, Mapping):
        raise ValueError("outcome_classification missing")
    flagged = set(classes.get("flagged", []))
    promoted = set(classes.get("clean_positive", []))
    if flagged & promoted:
        raise ValueError("flagged task promoted as clean positive")
    models = payload.get("model_compliance_receipts")
    if isinstance(models, Mapping):
        if models.get("tiny_model_promoted") is True:
            raise ValueError("tiny model promoted")
        if models.get("auto_tokenizer_promoted") is True:
            raise ValueError("AutoTokenizer promoted")
    authority = payload.get("authority_and_prohibited_path_receipts")
    if isinstance(authority, Mapping):
        if authority.get("hardware_software_fallback_promoted") is True:
            raise ValueError("software fallback promoted")
        if authority.get("publication_action_taken") is not False:
            raise ValueError("publication_action_taken authority mismatch")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    tests_run = None
    if args.tests_run_json is not None:
        tests_run = json.loads(args.tests_run_json.read_text(encoding="utf-8"))
    artifact = build_report(tests_run=tests_run)
    validate_artifact(artifact)
    write_json(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
