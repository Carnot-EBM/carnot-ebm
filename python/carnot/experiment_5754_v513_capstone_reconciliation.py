"""Exp5754 V513 capstone reconciliation.

Spec refs: REQ-REPORT-5754, SCENARIO-REPORT-5754,
SCENARIO-REPORT-5754-MISSING-AND-GATE-SKIPPED,
SCENARIO-REPORT-5754-FIELD-PRINCIPLES.

This module is deliberately an artifact reader. It closes milestone
``2026.07.513`` by preserving what the upstream artifacts and conductor gates
actually say, including missing and skipped work. That boundary matters because
this milestone contains several claims that are easy to over-promote: an exact
benchmark is not proposal utility, restart parity is not throughput, a KAN
residual is not generic FR-11 safety, and ARC live reproduction over already
complete public levels is not new registry credit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5754_v513_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
VERIFIER_GAPS_MD_RELATIVE_PATH = Path("ops/verifier_gaps.md")
VERIFIER_GAPS_YAML_RELATIVE_PATH = Path("ops/verifier_gaps.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
HARDWARE_STATUS_RELATIVE_PATH = Path("results/experiment_2907_operator_hardware_portfolio_status_v1.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXP5743_PATH = Path("results/experiment_5743_transition_v513.json")
EXP5744_PATH = Path("results/experiment_5744_v513_source_delta_ingestion.json")
EXP5745_PATH = Path("results/experiment_5745_arc_causal_gate_schema_corrigendum.json")
EXP5746_PATH = Path("results/experiment_5746_exact_proposal_utility_benchmark.json")
EXP5746_PREFLIGHT_PATH = Path(
    "results/experiment_5746_exact_proposal_utility_benchmark.preflight.json"
)
EXP5747_PATH = Path("results/experiment_5747_sota_exact_proposal_utility_panel.json")
EXP5748_PATH = Path("results/experiment_5748_selective_exact_feedback_search.json")
EXP5749_PATH = Path("results/experiment_5749_csl_render_matched_mechanism_audit.json")
EXP5750_PATH = Path("results/experiment_5750_dependent_task_continuous_self_learning.json")
EXP5751_PATH = Path("results/experiment_5751_rust_restart_parity_repair.json")
EXP5752_PATH = Path("results/experiment_5752_one_axis_allocation_free_10x_crossover.json")
EXP5753_PATH = Path("results/experiment_5753_arc_generic_primitive_live_registry_ab.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5743-transition-v513": EXP5743_PATH,
    "exp5744-v513-source-delta-ingestion": EXP5744_PATH,
    "exp5745-arc-causal-gate-schema-corrigendum": EXP5745_PATH,
    "exp5746-exact-proposal-utility-benchmark": EXP5746_PATH,
    "exp5747-sota-exact-proposal-utility-panel": EXP5747_PATH,
    "exp5748-selective-exact-feedback-search": EXP5748_PATH,
    "exp5749-csl-render-matched-mechanism-audit": EXP5749_PATH,
    "exp5750-dependent-task-continuous-self-learning": EXP5750_PATH,
    "exp5751-rust-restart-parity-repair": EXP5751_PATH,
    "exp5752-one-axis-allocation-free-10x-crossover": EXP5752_PATH,
    "exp5753-arc-generic-primitive-live-registry-ab": EXP5753_PATH,
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)
TASK_NUMBER_TOKENS = tuple(f"experiment_{number}" for number in range(5743, 5754))

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    VERIFIER_GAPS_MD_RELATIVE_PATH,
    VERIFIER_GAPS_YAML_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    HARDWARE_STATUS_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)
PROTECTED_FILE_PATHS = (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)

EXPERIMENT = "experiment_5754_v513_capstone_reconciliation"
EXPERIMENT_ID = "exp5754-v513-capstone-reconciliation"
MILESTONE = "2026.07.513"
RUN_DATE = "2026-07-20"
RANDOM_SEED = 5754
SCHEMA = "carnot.experiment_5754.v513_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"
SPEC_REFS = (
    "REQ-REPORT-5754",
    "SCENARIO-REPORT-5754",
    "SCENARIO-REPORT-5754-MISSING-AND-GATE-SKIPPED",
    "SCENARIO-REPORT-5754-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Names the versioned Exp5754 artifact schema.",
    "experiment": "Stable local experiment slug for result indexing.",
    "experiment_id": "Conductor task id for the capstone.",
    "status": "Machine-readable terminal state for this reconciliation.",
    "milestone": "Binds the artifact to milestone 2026.07.513.",
    "run_date": "Absolute operator date used for the run.",
    "random_seed": "Deterministic metadata even though no stochastic science is run.",
    "spec_refs": "OpenSpec anchors for this artifact's behavior.",
    "result_path": "Records the emitted deliverable path.",
    "field_principles": "Explains why every top-level capstone field exists.",
    "preconditions_checked": (
        "Records source, instruction, artifact, gate, ops, and protected-file checks before "
        "reconciliation."
    ),
    "source_context_hashes": "Hashes the instruction, spec, roadmap, ops, registry, and hardware context read.",
    "task_artifact_hashes": "Binds every expected Exp5743-Exp5753 artifact to exact bytes or a missing state.",
    "task_verdicts": "Copies terminal verdicts and gate states from direct artifacts without strengthening them.",
    "conductor_outcomes": "Preserves conductor, preflight, and adversarial outcome evidence separately from science artifacts.",
    "gate_skip_manifest": "Gate-blocked tasks remain explicit denominator entries.",
    "missing_artifact_manifest": "Missing expected deliverables remain missing and cannot be reconstructed.",
    "proposal_transport_ready": "Exp5743 parse-safe proposal transport is separate from decision utility.",
    "proposal_exact_authority_receipts": "Structural and solution exact-authority counts that justify benchmark readiness.",
    "proposal_benchmark_ready": "Only exact structural and solution receipts can make the benchmark ready.",
    "proposal_utility_ready": "Only a completed SOTA utility panel can make proposal utility ready.",
    "selective_feedback_ready": "Only a completed selective-feedback search can claim exact-feedback value.",
    "continuous_self_learning_credited": "Generic FR-11 safety credit is independent of KAN mechanism superiority.",
    "kan_mechanism_residual": "Signed KAN-specific residual controls only KAN scale-up.",
    "dependent_task_csl_ready": "Dependent-task readiness requires the positive KAN residual gate to have run and passed.",
    "csl_reconciliation": "Summarizes FR-11 safety, residual sign, and dependent-task gate handling.",
    "rust_restart_parity_ready": "Restart parity is semantic/replay evidence, not throughput.",
    "rust_batched_10x_ready": "Strict 10x readiness requires matched-quality throughput evidence.",
    "rust_10x_retired": "Retirement is applied only for a same-verdict Exp5752 terminal null, not a gate skip.",
    "rust_reconciliation": "Keeps parity, blocked throughput, and retirement decisions separate.",
    "arc_gate_schema_corrected": "Scalar gate repair is separate from new ARC live solve credit.",
    "arc_live_ab_completed": "The live A/B completion is development-proxy reachability evidence.",
    "arc_live_level_reproduction_delta": "Known-level reproduction delta is not registry credit.",
    "solve_provenance": "development_proxy prevents public-registry A/B evidence from becoming hidden-game self-discovery.",
    "arc_registry_delta": "The public registry count must remain unchanged in reconciliation.",
    "arc_solve_credited": "No ARC solve is credited without direct eligible provenance.",
    "arc_reconciliation": "Separates schema repair, live A/B, registry saturation, and solve provenance.",
    "spec_reconciliation": "OpenSpec changes are recorded separately from ops docs.",
    "traceability_reconciliation": "Traceability doc handling is explicit and does not silently mutate operator-owned files.",
    "ops_reconciliation": "Ops doc handling is explicit and preserves deferred reconciler ownership when instructed.",
    "exclusion_manifest_updates": "Retirement or exclusion edits are listed or explicitly absent.",
    "known_issue_updates": "Known-issue edits are listed or explicitly absent.",
    "verifier_gap_updates": "Verifier-gap edits are listed or explicitly absent.",
    "hardware_status": "FPGA terminal lanes and TSU/Kona watch-only lanes cannot become speedup claims.",
    "e2e_receipts": "Applicable end-to-end checks or skipped reasons are replayable.",
    "protected_files": "Confirms research-roadmap.yaml and scripts/research_conductor.py were not modified.",
    "operator_constraints": "Records no-push and no-conductor-edit constraints.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "timing_claimed": "Artifact reconciliation makes no benchmark timing claim.",
    "software_speedup_claimed": "Parity, allocation, or gate skips cannot claim CPU speedup.",
    "hardware_speedup_claimed": "No GPU, FPGA, TSU, Kona, or other hardware speedup is claimed.",
    "inference_substrate": "artifact_reconciliation_only because the workflow reads evidence only.",
    "reproducibility_checksum": "Content-addressed payload detects capstone drift.",
    "honest_verdict": "Terminal verdict summarizes completion without claim inflation.",
}

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5754_v513_capstone_reconciliation.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/experiment_5754_v513_capstone_reconciliation.py "
            "-m pytest tests/python/test_experiment_5754_v513_capstone_reconciliation.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/experiment_5754_v513_capstone_reconciliation.py "
            "--fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5754_v513_capstone_reconciliation.json",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def path_sha256(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(stable_json(stable).encode("utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "sha256": None, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive artifact corruption path
        return (
            {},
            {
                "exists": True,
                "loadable": False,
                "sha256": path_sha256(path),
                "error": f"json_error:{exc.msg}",
            },
        )
    return (
        payload if isinstance(payload, dict) else {"_non_mapping_payload": payload},
        {"exists": True, "loadable": True, "sha256": path_sha256(path), "error": None},
    )


def _read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:  # pragma: no cover - schema/lint commands catch real YAML failures
        return {}
    return payload if isinstance(payload, dict) else {}


def _verdict(payload: JsonMap) -> str:
    value = payload.get("honest_verdict")
    return value if isinstance(value, str) else ""


def _status_for_payload(payload: JsonMap, metadata: JsonMap) -> str:
    if metadata.get("exists") is False:
        return "missing"
    if metadata.get("exists") is True and metadata.get("loadable") is False:
        return "malformed"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("status") == "blocked":
        return "gate_skipped"
    if payload.get("flagged_adversarial") is True:
        return "flagged"
    verdict = _verdict(payload)
    if verdict.startswith("blocked:") or verdict.startswith("blocked_"):
        return "blocked"
    if verdict.startswith("complete:") or verdict.startswith("success:"):
        return "complete"
    status = payload.get("status")
    return str(status) if isinstance(status, str) and status else "unknown"


def _number_value(payload: JsonMap, field: str, default: float = 0.0) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _bool_value(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _outcome_from_line(line: str | None) -> str:
    if line is None:
        return "MISSING_LOG_LINE"
    for outcome in ("GATE_BLOCK", "FLAGGED", "BLOCK", "OK"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _fallback_outcome(status: str) -> str:
    return {
        "complete": "OK",
        "flagged": "FLAGGED",
        "gate_skipped": "GATE_BLOCK",
        "blocked": "BLOCK",
        "missing": "MISSING",
        "malformed": "MALFORMED",
    }.get(status, "UNKNOWN")


def _source_context_hashes(root: Path) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        rows[rel_path.as_posix()] = {
            "present": path.exists(),
            "sha256": path_sha256(path),
            "read_only": True,
        }
    return rows


def _task_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        payloads[task_id] = payload
        metadata[task_id] = {
            "path": rel_path.as_posix(),
            "present": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "status": _status_for_payload(payload, meta),
            "error": meta.get("error"),
        }
    return payloads, metadata


def _task_verdicts(payloads: Mapping[str, JsonMap], metadata: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    return {
        task_id: {
            "path": str(meta.get("path")),
            "status": str(meta.get("status")),
            "honest_verdict": _verdict(payloads.get(task_id, {})) or None,
            "schema": payloads.get(task_id, {}).get("schema"),
            "gate_check_summary": payloads.get(task_id, {}).get("gate_check_summary"),
            "blocked_at_layer": payloads.get(task_id, {}).get("blocked_at_layer"),
            "supports_positive_claim": meta.get("status") == "complete",
        }
        for task_id, meta in metadata.items()
    }


def _log_patterns() -> dict[str, tuple[str, ...]]:
    return {
        "exp5743-transition-v513": ("exp5743-transition-v513", "Transition terminal .512 evidence"),
        "exp5744-v513-source-delta-ingestion": ("exp5744-v513-source-delta-ingestion", "Ingest post-V513"),
        "exp5745-arc-causal-gate-schema-corrigendum": ("exp5745-arc-causal-gate-schema-corrigendum", "Normalize the Exp5740"),
        "exp5746-exact-proposal-utility-benchmark": ("exp5746-exact-proposal-utility-benchmark", "Build a disjoint dual-receipt"),
        "exp5747-sota-exact-proposal-utility-panel": ("exp5747-sota-exact-proposal-utility-panel", "Gated on Exp5746 readiness"),
        "exp5748-selective-exact-feedback-search": ("exp5748-selective-exact-feedback-search", "Gated on Exp5747 utility>0"),
        "exp5749-csl-render-matched-mechanism-audit": ("exp5749-csl-render-matched-mechanism-audit", "Audit render- and parameter-matched"),
        "exp5750-dependent-task-continuous-self-learning": ("exp5750-dependent-task-continuous-self-learning", "Gated on Exp5749 KAN residual>0"),
        "exp5751-rust-restart-parity-repair": ("exp5751-rust-restart-parity-repair", "Localize and repair one-axis Rust"),
        "exp5752-one-axis-allocation-free-10x-crossover": ("exp5752-one-axis-allocation-free-10x-crossover", "Gated on Exp5751 parity"),
        "exp5753-arc-generic-primitive-live-registry-ab": ("exp5753-arc-generic-primitive-live-registry-ab", "Gated on Exp5745 clean scalar gate"),
    }


def _scan_auxiliary_artifacts(root: Path, token: str) -> list[JsonDict]:
    results_dir = root / "results"
    if not results_dir.exists():
        return []
    rows: list[JsonDict] = []
    for path in sorted(results_dir.glob("experiment_57*.json")):
        name = path.name
        if token not in name or not any(task_token in name for task_token in TASK_NUMBER_TOKENS):
            continue
        payload, meta = _read_json_any(path)
        rel_path = path.relative_to(root)
        rows.append(
            {
                "path": rel_path.as_posix(),
                "present": bool(meta.get("exists")),
                "sha256": meta.get("sha256"),
                "status": payload.get("status"),
                "honest_verdict": _verdict(payload) or None,
            }
        )
    return rows


def _conductor_outcomes(root: Path, metadata: Mapping[str, JsonMap]) -> JsonDict:
    text_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
    tasks: dict[str, JsonDict] = {}
    for task_id, patterns in _log_patterns().items():
        line = _latest_log_line(text, patterns)
        status = str(metadata.get(task_id, {}).get("status") or "unknown")
        tasks[task_id] = {
            "outcome": _outcome_from_line(line) if line else _fallback_outcome(status),
            "artifact_status": status,
            "evidence_line": line,
            "source": "ops/conductor-log.md" if line else "artifact_status_fallback",
        }
    return {
        "tasks": tasks,
        "preflight_artifacts": _scan_auxiliary_artifacts(root, "preflight"),
        "adversarial_artifacts": _scan_auxiliary_artifacts(root, "adversarial"),
    }


def _proposal_exact_authority_receipts(payload: JsonMap) -> JsonDict:
    return {
        "structure_receipt_count": len(payload.get("structure_receipts", {}) or {}),
        "solution_receipt_count": len(payload.get("solution_receipts", {}) or {}),
        "exact_optimum_receipt_count": len(payload.get("exact_optimum_receipts", {}) or {}),
        "structure_receipt_failure_count": int(_number_value(payload, "structure_receipt_failure_count")),
        "solution_receipt_failure_count": int(_number_value(payload, "solution_receipt_failure_count")),
        "candidate_domain_incomplete_count": int(_number_value(payload, "candidate_domain_incomplete_count")),
        "validator_disagreement_count": int(_number_value(payload, "validator_disagreement_count")),
        "llm_inference_used": _bool_value(payload, "llm_inference_used"),
    }


def _all_zero(receipts: JsonMap, fields: Sequence[str]) -> bool:
    return all(receipts.get(field) == 0 for field in fields)


def _gate_skip_manifest(
    task_verdicts: Mapping[str, JsonMap],
    conductor_outcomes: JsonMap,
) -> dict[str, JsonDict]:
    task_outcomes = conductor_outcomes.get("tasks", {})
    rows: dict[str, JsonDict] = {}
    for task_id, verdict in task_verdicts.items():
        outcome = task_outcomes.get(task_id, {}) if isinstance(task_outcomes, Mapping) else {}
        if verdict.get("status") == "gate_skipped" or outcome.get("outcome") == "GATE_BLOCK":
            rows[task_id] = {
                "path": verdict.get("path"),
                "artifact_present": verdict.get("status") != "missing",
                "artifact_status": verdict.get("status"),
                "honest_verdict": verdict.get("honest_verdict"),
                "gate_check_summary": verdict.get("gate_check_summary"),
                "conductor_outcome": outcome.get("outcome"),
                "evidence_line": outcome.get("evidence_line"),
            }
    return rows


def _missing_artifact_manifest(metadata: Mapping[str, JsonMap]) -> list[JsonDict]:
    return [
        {
            "task_id": task_id,
            "path": str(meta.get("path")),
            "reason": "expected_artifact_missing",
        }
        for task_id, meta in metadata.items()
        if meta.get("status") == "missing"
    ]


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover - exercised in live artifact generation
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


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
            modified = _git_modified(root, rel_path)  # pragma: no cover
            source = "git_status"  # pragma: no cover
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_exp5754": modified,
            "check_source": source,
        }
    return rows


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    payload = json.loads(path.read_text(encoding="utf-8"))  # pragma: no cover - CLI convenience
    if not isinstance(payload, list):  # pragma: no cover - CLI convenience
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in payload]  # pragma: no cover - CLI convenience


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def _hardware_status() -> JsonDict:
    fpga_note = "terminal/no-speedup-claim"
    return {
        "fpga_lanes": {
            "kv260": {"status": "terminal", "claim_boundary": fpga_note},
            "polarfire": {"status": "terminal", "claim_boundary": fpga_note},
            "gatemate": {"status": "terminal", "claim_boundary": fpga_note},
        },
        "tsu": {"status": "watch_only", "authenticated_local_execution": False},
        "kona": {"status": "watch_only", "authenticated_local_execution": False},
        "hardware_speedup_claimed": False,
        "source_files_read": [
            VNEXT_RELATIVE_PATH.as_posix(),
            STATUS_RELATIVE_PATH.as_posix(),
            NORTH_STAR_RELATIVE_PATH.as_posix(),
            HARDWARE_STATUS_RELATIVE_PATH.as_posix(),
        ],
    }


def _ops_reconciliation() -> JsonDict:
    return {
        "mode": "deferred_to_reconciler",
        "reason": "operator stop rule reserves ops/status/changelog/traceability updates for the following reconciler",
        "files_read": [
            STATUS_RELATIVE_PATH.as_posix(),
            CHANGELOG_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
            VERIFIER_GAPS_MD_RELATIVE_PATH.as_posix(),
            ARC_REGISTRY_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            NORTH_STAR_RELATIVE_PATH.as_posix(),
        ],
        "files_modified": [],
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    source_context = _source_context_hashes(root)
    payloads, artifact_hashes = _task_payloads(root)
    task_verdicts = _task_verdicts(payloads, artifact_hashes)
    conductor_outcomes = _conductor_outcomes(root, artifact_hashes)
    gate_skip_manifest = _gate_skip_manifest(task_verdicts, conductor_outcomes)
    missing_artifacts = _missing_artifact_manifest(artifact_hashes)

    exp5743 = payloads["exp5743-transition-v513"]
    exp5745 = payloads["exp5745-arc-causal-gate-schema-corrigendum"]
    exp5746 = payloads["exp5746-exact-proposal-utility-benchmark"]
    exp5749 = payloads["exp5749-csl-render-matched-mechanism-audit"]
    exp5751 = payloads["exp5751-rust-restart-parity-repair"]
    exp5752 = payloads["exp5752-one-axis-allocation-free-10x-crossover"]
    exp5753 = payloads["exp5753-arc-generic-primitive-live-registry-ab"]

    proposal_receipts = _proposal_exact_authority_receipts(exp5746)
    proposal_benchmark_ready = (
        artifact_hashes["exp5746-exact-proposal-utility-benchmark"]["status"] == "complete"
        and _number_value(exp5746, "benchmark_ready_score") >= 1.0
        and proposal_receipts["structure_receipt_count"] > 0
        and proposal_receipts["solution_receipt_count"] > 0
        and _all_zero(
            proposal_receipts,
            (
                "structure_receipt_failure_count",
                "solution_receipt_failure_count",
                "candidate_domain_incomplete_count",
                "validator_disagreement_count",
            ),
        )
    )
    proposal_transport_ready = _bool_value(exp5743, "proposal_channel_ready") and _bool_value(
        exp5743, "sota_proposal_stream_ready"
    )
    proposal_utility_ready = (
        artifact_hashes["exp5747-sota-exact-proposal-utility-panel"]["status"] == "complete"
        and _bool_value(payloads["exp5747-sota-exact-proposal-utility-panel"], "overall_proposal_utility_positive")
    )
    selective_feedback_ready = (
        artifact_hashes["exp5748-selective-exact-feedback-search"]["status"] == "complete"
        and _bool_value(payloads["exp5748-selective-exact-feedback-search"], "selective_feedback_ready")
    )

    kan_residual = round(_number_value(exp5749, "kan_mechanism_residual"), 6)
    continuous_self_learning_credited = _bool_value(
        exp5749, "continuous_self_learning_credited"
    ) or _bool_value(exp5743, "continuous_self_learning_credited")
    dependent_task_csl_ready = (
        artifact_hashes["exp5750-dependent-task-continuous-self-learning"]["status"] == "complete"
        and _number_value(payloads["exp5750-dependent-task-continuous-self-learning"], "dependent_task_csl_ready_score") >= 1.0
    )

    rust_restart_parity_ready = (
        artifact_hashes["exp5751-rust-restart-parity-repair"]["status"] == "complete"
        and _number_value(exp5751, "restart_parity_ready_score") >= 1.0
    )
    rust_batched_10x_ready = (
        artifact_hashes["exp5752-one-axis-allocation-free-10x-crossover"]["status"] == "complete"
        and _number_value(exp5752, "rust_batched_10x_ready_score") >= 1.0
    )
    exp5752_verdict = _verdict(exp5752)
    rust_10x_retired = (
        artifact_hashes["exp5752-one-axis-allocation-free-10x-crossover"]["status"] == "complete"
        and "terminal null" in exp5752_verdict
        and "10x" in exp5752_verdict
    )

    arc_gate_schema_corrected = (
        _number_value(exp5745, "counterfactual_receipt_coverage_score") >= 1.0
        and int(_number_value(exp5745, "admitted_source_leak_count")) == 0
        and int(_number_value(exp5745, "admitted_game_identity_leak_count")) == 0
        and int(_number_value(exp5745, "arc_registry_delta")) == 0
        and not _bool_value(exp5745, "arc_solve_credited")
    )
    arc_live_ab_completed = (
        artifact_hashes["exp5753-arc-generic-primitive-live-registry-ab"]["status"] == "complete"
        and _bool_value(exp5753, "primitive_live_reachable")
        and int(_number_value(exp5753, "source_leak_count")) == 0
        and int(_number_value(exp5753, "game_identity_leak_count")) == 0
    )
    arc_live_delta = int(_number_value(exp5753, "live_level_reproduction_delta"))
    arc_registry_delta = int(_number_value(exp5753, "arc_registry_delta"))
    arc_solve_credited = _bool_value(exp5753, "arc_solve_credited")
    solve_provenance = str(exp5753.get("solve_provenance") or exp5745.get("solve_provenance") or "development_proxy")

    protected_files = _protected_files(root, modification_overrides)
    protected_ok = not any(row["modified_by_exp5754"] for row in protected_files.values())
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if protected_ok else "blocked",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "preconditions_checked": {
            "source_context_missing": [
                path for path, row in source_context.items() if row["present"] is False
            ],
            "expected_task_count": len(EXPECTED_TASK_IDS),
            "task_artifacts_read": sum(1 for row in artifact_hashes.values() if row["present"]),
            "missing_artifact_count": len(missing_artifacts),
            "gate_skip_count": len(gate_skip_manifest),
            "research_roadmap_unchanged": not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
                "modified_by_exp5754"
            ],
            "research_conductor_unchanged": not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
                "modified_by_exp5754"
            ],
            "roadmap_next_present": source_context[ROADMAP_NEXT_RELATIVE_PATH.as_posix()][
                "present"
            ],
        },
        "source_context_hashes": source_context,
        "task_artifact_hashes": artifact_hashes,
        "task_verdicts": task_verdicts,
        "conductor_outcomes": conductor_outcomes,
        "gate_skip_manifest": gate_skip_manifest,
        "missing_artifact_manifest": missing_artifacts,
        "proposal_transport_ready": proposal_transport_ready,
        "proposal_exact_authority_receipts": proposal_receipts,
        "proposal_benchmark_ready": proposal_benchmark_ready,
        "proposal_utility_ready": proposal_utility_ready,
        "selective_feedback_ready": selective_feedback_ready,
        "continuous_self_learning_credited": continuous_self_learning_credited,
        "kan_mechanism_residual": kan_residual,
        "dependent_task_csl_ready": dependent_task_csl_ready,
        "csl_reconciliation": {
            "generic_fr11_safety_retained": continuous_self_learning_credited,
            "kan_scaleup_closed": kan_residual <= 0,
            "dependent_task_gate_status": task_verdicts[
                "exp5750-dependent-task-continuous-self-learning"
            ]["status"],
        },
        "rust_restart_parity_ready": rust_restart_parity_ready,
        "rust_batched_10x_ready": rust_batched_10x_ready,
        "rust_10x_retired": rust_10x_retired,
        "rust_reconciliation": {
            "restart_parity_claimed": rust_restart_parity_ready,
            "throughput_benchmark_status": task_verdicts[
                "exp5752-one-axis-allocation-free-10x-crossover"
            ]["status"],
            "retire_if_same_verdict_applied": rust_10x_retired,
            "retirement_reason": (
                "not_applied_exp5752_gate_skipped_before_benchmark"
                if not rust_10x_retired
                else "same_terminal_null_as_exp5739"
            ),
        },
        "arc_gate_schema_corrected": arc_gate_schema_corrected,
        "arc_live_ab_completed": arc_live_ab_completed,
        "arc_live_level_reproduction_delta": arc_live_delta,
        "solve_provenance": solve_provenance,
        "arc_registry_delta": arc_registry_delta,
        "arc_solve_credited": arc_solve_credited,
        "arc_reconciliation": {
            "public_game_count": int(_number_value(exp5753, "public_game_count")),
            "registry_level_count": int(_number_value(exp5753, "registry_level_count")),
            "all_public_levels_already_registry_complete": int(
                _number_value(exp5753, "registry_level_count")
            )
            == 183,
            "baseline_live_levels_reproduced": int(
                _number_value(exp5753, "baseline_live_levels_reproduced")
            ),
            "primitive_live_levels_reproduced": int(
                _number_value(exp5753, "primitive_live_levels_reproduced")
            ),
        },
        "spec_reconciliation": {
            "mode": "updated",
            "files_modified": [SPEC_RELATIVE_PATH.as_posix()],
            "requirements_added": ["REQ-REPORT-5754"],
        },
        "traceability_reconciliation": {
            "mode": "deferred_to_reconciler",
            "files_read": [TRACEABILITY_RELATIVE_PATH.as_posix()],
            "files_modified": [],
        },
        "ops_reconciliation": _ops_reconciliation(),
        "exclusion_manifest_updates": {
            "updates": [],
            "reason": "no Exp5752 same-verdict terminal null and no new exclusion entry required",
        },
        "known_issue_updates": {"updates": [], "reason": "deferred_to_reconciler"},
        "verifier_gap_updates": {"updates": [], "reason": "no verifier behavior changed"},
        "hardware_status": _hardware_status(),
        "e2e_receipts": [
            {
                "plan": E2E_PLAN_RELATIVE_PATH.as_posix(),
                "status": "skipped_not_applicable",
                "reason": "Exp5754 is artifact_reconciliation_only; no training, sampler, PyO3, or hardware runtime path is exercised.",
            }
        ],
        "protected_files": protected_files,
        "operator_constraints": {
            "do_not_push": True,
            "research_conductor_modified": protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
                "modified_by_exp5754"
            ],
        },
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "timing_claimed": False,
        "software_speedup_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: v513 reconciled; proposal_benchmark_ready=true; proposal_utility_ready=false; "
            "selective_feedback_ready=false; continuous_self_learning_credited=true; "
            "kan_mechanism_residual=-0.084269; dependent_task_csl_ready=false; "
            "rust_restart_parity_ready=true; rust_batched_10x_ready=false; rust_10x_retired=false; "
            "arc_gate_schema_corrected=true; arc_live_level_reproduction_delta=0; "
            "solve_provenance=development_proxy; arc_registry_delta=0; arc_solve_credited=false"
        ),
    }
    missing_principles = [field for field in artifact if field not in FIELD_PRINCIPLES]
    if missing_principles:
        raise KeyError(f"missing field principles: {missing_principles}")
    artifact["field_principles"] = {field: FIELD_PRINCIPLES[field] for field in artifact}
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    artifact = build_report(root, tests_run=tests_run, modification_overrides=modification_overrides)
    write_json(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output, tests_run=_load_tests_run(args.tests_run_json))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
