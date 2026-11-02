"""Exp6142 transition receipt from terminal milestone .532 into .533.

Spec refs: REQ-REPORT-6142,
SCENARIO-REPORT-6142-ACTIVATED-MATRIX,
SCENARIO-REPORT-6142-TERMINAL-CLASSES,
SCENARIO-REPORT-6142-DUPLICATE-ACTIVATION,
SCENARIO-REPORT-6142-RANGE-COLLISION,
SCENARIO-REPORT-6142-SCHEMA.

The transition is a ledger task. It archives only the activated identities and
declared deliverable paths, then records the next active roadmap boundary
without turning a retired scientific line into a downstream run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6142_transition_v533.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EVIDENCE_INDEX_RELATIVE_PATH = Path("scripts/evidence_index_collision_preflight.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_6142_transition_v533"
EXPERIMENT_ID = "exp6142-transition-v533"
MILESTONE_FROM = "2026.08.532"
MILESTONE_TO = "2026.08.533"
MILESTONE_FROM_TITLE = (
    "Empirically Calibrated Phase-D Verification, Certified Strategy Memory, "
    "and ARC Change Fidelity"
)
MILESTONE_TO_TITLE = (
    "Task-Aware Energy Calibration, Certified Continuous Learning, "
    "and Stochastic Program Compilation"
)
RUN_DATE = "20260805"
RANDOM_SEED = 6142
SCHEMA = "carnot.experiment_6142.transition_v533.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-6142",
    "SCENARIO-REPORT-6142-ACTIVATED-MATRIX",
    "SCENARIO-REPORT-6142-TERMINAL-CLASSES",
    "SCENARIO-REPORT-6142-DUPLICATE-ACTIVATION",
    "SCENARIO-REPORT-6142-RANGE-COLLISION",
    "SCENARIO-REPORT-6142-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp6138-transition-v532": Path("results/experiment_6138_transition_v532.json"),
    "exp6139-v532-source-delta-ingestion": Path(
        "results/experiment_6139_v532_source_delta_ingestion.json"
    ),
    "exp6140-phase-d-exp6128-option-psychometrics": Path(
        "results/experiment_6140_phase_d_exp6128_option_psychometrics.json"
    ),
    "exp6141-phase-d-empirical-item-bank": Path(
        "results/experiment_6141_phase_d_empirical_item_bank.json"
    ),
}

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp6138-transition-v532": "Exact terminal-boundary handoff from .531 into .532",
    "exp6139-v532-source-delta-ingestion": (
        "Reliable dated evidence refresh after the V532 planner marker"
    ),
    "exp6140-phase-d-exp6128-option-psychometrics": (
        "Frozen Exp6128 option-aware psychometrics and family-mixture diagnosis"
    ),
    "exp6141-phase-d-empirical-item-bank": (
        "Gated on Exp6140 design readiness: exact empirical Phase-D item bank"
    ),
}

CONDUCTOR_MATCH_MARKERS: dict[str, str] = {
    "exp6138-transition-v532": "Exact terminal-boundary handoff from .531",
    "exp6139-v532-source-delta-ingestion": "Reliable dated evidence refresh after the V532",
    "exp6140-phase-d-exp6128-option-psychometrics": "Frozen Exp6128 option-aware psychometrics",
    "exp6141-phase-d-empirical-item-bank": "Gated on Exp6140 design readiness",
}

EXPECTED_TERMINAL_CLASSES: dict[str, str] = {
    "exp6138-transition-v532": "complete",
    "exp6139-v532-source-delta-ingestion": "complete-null",
    "exp6140-phase-d-exp6128-option-psychometrics": "retired",
    "exp6141-phase-d-empirical-item-bank": "structured-gate-skip",
}

SCIENTIFIC_RETIREMENT_TASK_ID = "exp6140-phase-d-exp6128-option-psychometrics"
STRUCTURED_GATE_SKIP_TASK_ID = "exp6141-phase-d-empirical-item-bank"
FORMER_PROPOSAL_ONLY_TASK_IDS = tuple(f"exp{number}" for number in range(6142, 6152))
NEXT_RANGE_NUMBERS = range(6142, 6156)

PROTECTED_FILE_PATHS = (
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

SOURCE_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_6142_transition_v533.py"),
    Path("tests/python/test_experiment_6142_transition_v533.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)

CANONICAL_PLAN_REFERENCE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
)

HISTORICAL_PROPOSAL_REFERENCE_PATHS = (
    Path("python/carnot/experiment_6138_transition_v532.py"),
    Path("tests/python/test_experiment_6138_transition_v532.py"),
    Path("results/experiment_6138_transition_v532.json"),
    Path("python/carnot/experiment_6139_v532_source_delta_ingestion.py"),
    Path("tests/python/test_experiment_6139_v532_source_delta_ingestion.py"),
    Path("results/experiment_6139_v532_source_delta_ingestion.json"),
)

REQUIRED_TASK_OWNED_GATE_KINDS = (
    "unit",
    "coverage",
    "spec_coverage",
    "yaml_parse",
    "exact_path",
    "retirement",
    "gate_skip",
    "duplicate_history",
    "activation",
    "exclusion_manifest",
    "range_collision",
    "adversarial_verifier",
    "protected_file",
    "applicable_e2e",
    "no_new_root_clutter",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "exact_terminal_classification",
    "scientific_retirement_receipt",
    "structured_gate_skip_receipt",
    "proposal_only_identities_excluded",
    "adversarial_verifier_receipts",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "staged_roadmap_activation_receipt",
    "next_task_range",
    "next_range_collision_count",
    "docs_reconciled",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "terminal transition state follows activated identity, exact-path, retirement, "
        "gate-skip, activation, and collision receipts."
    ),
    "preconditions_checked": (
        "activated and staged roadmaps, receipts, history, exclusions, dirty worktree, "
        "root clutter, and protected hashes are parsed before mutation."
    ),
    "milestone_transition": (
        "only the four activated task identities and declared paths define `.532`."
    ),
    "activated_task_and_deliverable_matrix": (
        "only the four activated task identities and declared paths define `.532`."
    ),
    "exact_terminal_classification": (
        "complete, complete-null, retired, and structured gate-skip states remain "
        "separate terminal evidence classes."
    ),
    "scientific_retirement_receipt": (
        "a retired scientific line and a downstream skip remain distinct terminal evidence."
    ),
    "structured_gate_skip_receipt": (
        "a retired scientific line and a downstream skip remain distinct terminal evidence."
    ),
    "proposal_only_identities_excluded": (
        "old proposal prose is neither an experiment nor a collision once replaced by "
        "this canonical active range."
    ),
    "adversarial_verifier_receipts": (
        "present exact artifacts are freshly checked; the absent structured skip is not fabricated."
    ),
    "research_complete_append_count": "append `.532` at most once and amplify no history.",
    "duplicate_history_amplification_count": ("append `.532` at most once and amplify no history."),
    "staged_roadmap_activation_receipt": (
        "activation is exact when staged YAML exists and already-active when the conductor "
        "has consumed it into `research-roadmap.yaml`."
    ),
    "next_task_range": "bare zero collisions authorize exactly Exp6142-Exp6155.",
    "next_range_collision_count": ("bare zero collisions authorize exactly Exp6142-Exp6155."),
    "docs_reconciled": (
        "transition-owned spec updates are recorded while conductor-owned ops "
        "reconciliation may be deferred."
    ),
    "protected_files_unchanged": (
        "historical artifacts, conductor, exclusions, and unrelated dirty files remain "
        "byte-identical except for intentional ledger/result writes."
    ),
    "duration_s": "use measured `aggregation_from_upstream_artifacts`.",
    "inference_substrate": (
        "set `aggregation_from_upstream_artifacts`; this task invokes no research LLM."
    ),
    "field_provenance": (
        "every required field traces to exact local receipts instead of broad glob inference."
    ),
    "test_commands": (
        "commands document focused unit/spec coverage, YAML parse, exact-path, retirement, "
        "gate-skip, duplicate-history, activation, exclusion, collision, "
        "adversarial-verifier, protected-file, applicable E2E, full-suite, and "
        "root-clutter checks."
    ),
    "test_exit_codes": "exit codes prevent failed checks from becoming success.",
    "reproducibility_checksum": (
        "a checksum detects later transition, activation, history, collision, or evidence drift."
    ),
    "honest_verdict": (
        "use a terminal `complete:` or `blocked:` prefix and state whether `.533` was activated."
    ),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6142_transition_v533.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include="
    "python/carnot/experiment_6142_transition_v533.py -m pytest "
    "tests/python/test_experiment_6142_transition_v533.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include="
    "python/carnot/experiment_6142_transition_v533.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present .532 declared deliverables>",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: JsonMap) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


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
        meta["error"] = "json_not_mapping"
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


def _load_yaml_any(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _history_blocks(root: Path) -> list[JsonMap]:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    try:
        payload = _load_yaml_any(path)
    except yaml.YAMLError:
        return []
    blocks = payload.get("milestones") if isinstance(payload, Mapping) else payload
    return (
        [block for block in blocks if isinstance(block, Mapping)]
        if isinstance(blocks, list)
        else []
    )


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    return tuple(
        (str(row.get("id")), str(row.get("deliverable") or ""))
        for row in tasks
        if isinstance(row, Mapping)
    )


def _duplicate_history_block_count(blocks: Sequence[JsonMap]) -> int:
    grouped: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
    for block in blocks:
        grouped[(str(block.get("id")), _task_signature(block))] += 1
    return sum(count - 1 for count in grouped.values() if count > 1)


def _completion_block_data() -> JsonDict:
    return {
        "id": MILESTONE_FROM,
        "title": MILESTONE_FROM_TITLE,
        "doc": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-05",
        "finding": "Terminal outcomes preserved by transition artifact.",
        "tasks": [
            {
                "id": task_id,
                "title": ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": EXPECTED_TERMINAL_CLASSES[task_id],
            }
            for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _write_history_blocks(path: Path, original: Any, blocks: list[JsonMap]) -> None:
    if isinstance(original, Mapping):
        updated = dict(original)
        updated["milestones"] = blocks
        data = yaml.safe_dump(updated, sort_keys=False).encode("utf-8")
    else:
        data = yaml.safe_dump(blocks, sort_keys=False).encode("utf-8")
    _atomic_write_bytes(path, data)


def _append_completion_if_absent(root: Path, terminal: bool) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    canonical_signature = _task_signature(_completion_block_data())
    before_signatures = {
        _task_signature(block) for block in before_blocks if block.get("id") == MILESTONE_FROM
    }
    before_milestone_count = sum(1 for block in before_blocks if block.get("id") == MILESTONE_FROM)
    base = {
        "before_sha256": before_meta["sha256"],
        "before_duplicate_history_count": before_duplicate_count,
        "before_milestone_block_count": before_milestone_count,
        "before_canonical_signature_count": len(before_signatures),
    }
    if not terminal:
        return {
            **base,
            "append_count": 0,
            "appended": False,
            "reason": "nonterminal_identity_present",
            "after_sha256": before_meta["sha256"],
            "after_duplicate_history_count": before_duplicate_count,
            "after_milestone_block_count": before_milestone_count,
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    if canonical_signature in before_signatures:
        return {
            **base,
            "append_count": 0,
            "appended": False,
            "reason": "exact_milestone_block_present",
            "after_sha256": before_meta["sha256"],
            "after_duplicate_history_count": before_duplicate_count,
            "after_milestone_block_count": before_milestone_count,
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    try:
        original = _load_yaml_any(path)
    except yaml.YAMLError:
        original = {}
    after_blocks = list(before_blocks)
    after_blocks.append(_completion_block_data())
    _write_history_blocks(path, original, after_blocks)
    written_blocks = _history_blocks(root)
    after_duplicate_count = _duplicate_history_block_count(written_blocks)
    after_signatures = {
        _task_signature(block) for block in written_blocks if block.get("id") == MILESTONE_FROM
    }
    return {
        **base,
        "append_count": 1,
        "appended": True,
        "reason": "exact_milestone_block_absent",
        "after_sha256": path_sha256(path),
        "after_duplicate_history_count": after_duplicate_count,
        "after_milestone_block_count": sum(
            1 for block in written_blocks if block.get("id") == MILESTONE_FROM
        ),
        "after_canonical_signature_count": len(after_signatures),
        "duplicate_history_amplification_count": max(
            0, after_duplicate_count - before_duplicate_count
        ),
    }


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        flags = reports[0].get("flags")
        return (
            [dict(flag) for flag in flags if isinstance(flag, Mapping)]
            if isinstance(flags, list)
            else []
        )
    return []


def _receipt_flag_count(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("flag_count") or 0)
        return int(stdout_json.get("flagged_count") or 0)
    return int(receipt.get("flag_count") or 0)


def _receipt_max_severity(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            raw = reports[0].get("max_severity")
            return int(raw) if raw is not None else -1
    raw = receipt.get("max_severity")
    return int(raw) if raw is not None else -1


def _complete_receipt(row: JsonMap) -> JsonDict:
    receipt = dict(row)
    receipt["flag_count"] = _receipt_flag_count(receipt)
    receipt["max_severity"] = _receipt_max_severity(receipt)
    receipt["flags"] = _receipt_flags(receipt)
    receipt.setdefault("receipt_hash", sha256_json(receipt.get("stdout_json", {})))
    return receipt


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[Any] | None,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    source = receipts.values() if isinstance(receipts, Mapping) else receipts
    rows: dict[str, JsonDict] = {}
    for row in source:
        if isinstance(row, Mapping) and row.get("task_id"):
            task_id = str(row["task_id"])
            if metadata.get(task_id, {}).get("present"):
                rows[task_id] = _complete_receipt(row)
    return rows


def run_live_adversarial_receipts(
    root: Path, metadata: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:  # pragma: no cover
    executable = (
        (root / ".venv/bin/python").as_posix()
        if (root / ".venv/bin/python").exists()
        else sys.executable
    )
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [
            executable,
            ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = _complete_receipt(
            {
                "task_id": task_id,
                "artifact_path": rel_path.as_posix(),
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_json": stdout_json,
                "stderr": result.stderr,
                "receipt_hash": sha256_json(stdout_json),
            }
        )
    return receipts


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in sorted(set(SOURCE_HASH_PATHS), key=lambda value: value.as_posix())
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    probe.write_text("atomic-probe\n", encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    probe.unlink()
    return {"declared_path": path.as_posix(), "atomic_probe_write_ok": ok, "ok": ok}


def _protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {rel_path.as_posix(): path_sha256(root / rel_path) for rel_path in PROTECTED_FILE_PATHS}


def _protected_files_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_file_hashes(root)
    files = {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256_before": before.get(rel_path.as_posix()),
            "sha256_after": after.get(rel_path.as_posix()),
            "unchanged": before.get(rel_path.as_posix()) == after.get(rel_path.as_posix()),
        }
        for rel_path in PROTECTED_FILE_PATHS
    }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _conductor_status_from_line(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    return parts[3] if len(parts) > 3 else ""


def _queued_count(line: str) -> int | None:
    match = re.search(r"(\d+)\s+tasks queued", line)
    return int(match.group(1)) if match else None


def _conductor_receipts(root: Path) -> JsonDict:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    by_task: dict[str, JsonDict] = {}
    for task_id, marker in CONDUCTOR_MATCH_MARKERS.items():
        matches = [line for line in lines if marker in line or task_id in line]
        latest = matches[-1] if matches else ""
        by_task[task_id] = {
            "attempt_count": len(matches),
            "latest_line": latest,
            "latest_status": _conductor_status_from_line(latest) if latest else "",
        }
    plan_532 = [line for line in lines if "Plan milestone 2026.08.532" in line]
    activation_532 = [line for line in lines if "Milestone 2026.08.532 activated" in line]
    plan_533 = [line for line in lines if "Plan milestone 2026.08.533" in line]
    activation_533 = [line for line in lines if "Milestone 2026.08.533 activated" in line]
    return {
        "source_plan_line": plan_532[-1] if plan_532 else "",
        "source_plan_status": _conductor_status_from_line(plan_532[-1]) if plan_532 else "",
        "source_activation_line": activation_532[-1] if activation_532 else "",
        "source_activation_status": _conductor_status_from_line(activation_532[-1])
        if activation_532
        else "",
        "source_activated_task_count_claim": _queued_count(activation_532[-1])
        if activation_532
        else None,
        "destination_plan_line": plan_533[-1] if plan_533 else "",
        "destination_plan_status": _conductor_status_from_line(plan_533[-1]) if plan_533 else "",
        "destination_activation_line": activation_533[-1] if activation_533 else "",
        "destination_activation_status": _conductor_status_from_line(activation_533[-1])
        if activation_533
        else "",
        "destination_activated_task_count_claim": _queued_count(activation_533[-1])
        if activation_533
        else None,
        "by_task": by_task,
    }


def _artifact_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _classify_task(
    task_id: str,
    payload: JsonMap,
    meta: JsonMap,
    conductor_row: JsonMap,
) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if not meta.get("present"):
        latest = str(conductor_row.get("latest_line") or "")
        if (
            task_id == STRUCTURED_GATE_SKIP_TASK_ID
            and conductor_row.get("latest_status") == "GATE_BLOCK"
            and "Pre-emptive skip" in latest
        ):
            return "structured-gate-skip"
        return "missing"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired"
    if status == "complete_null" or verdict.startswith("complete_null:"):
        return "complete-null"
    if status.startswith("complete") or verdict.startswith("complete:"):
        return "complete"
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        return "blocked"
    return "missing"


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor_by_task: Mapping[str, JsonMap],
) -> JsonDict:
    by_task: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _classify_task(
            task_id,
            payloads[task_id],
            metadata[task_id],
            conductor_by_task.get(task_id, {}),
        )
        by_task[task_id] = terminal
        by_class.setdefault(terminal, []).append(task_id)
    nonterminal = [task_id for task_id, terminal in by_task.items() if terminal == "missing"]
    return {
        "terminal_class_by_task_id": by_task,
        "task_ids_by_terminal_class": by_class,
        "expected_terminal_class_by_task_id": dict(EXPECTED_TERMINAL_CLASSES),
        "all_activated_terminal": not nonterminal and by_task == EXPECTED_TERMINAL_CLASSES,
        "nonterminal_task_ids": nonterminal,
        "classification_source": "exact_declared_deliverables_plus_conductor_receipts",
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _task_number(task_id: str) -> int | None:
    match = re.search(r"exp(\d{4})", task_id, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _same_number_aliases(root: Path, task_id: str, declared_path: Path) -> list[str]:
    number = _task_number(task_id)
    results = root / "results"
    if number is None or not results.exists():
        return []
    aliases: list[str] = []
    for path in results.glob(f"*{number}*"):
        rel_path = path.relative_to(root)
        if path.is_file() and rel_path != declared_path:
            aliases.append(rel_path.as_posix())
    return sorted(aliases)


def _activated_matrix(
    root: Path,
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    classes: Mapping[str, str],
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        payload = payloads[task_id]
        meta = metadata[task_id]
        terminal = classes[task_id]
        evidence_source = (
            "conductor_log_structured_skip_without_artifact"
            if terminal == "structured-gate-skip"
            else "declared_deliverable_path"
        )
        matrix[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "title": ACTIVATED_TASK_TITLES[task_id],
            "declared_deliverable": rel_path.as_posix(),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "activated": True,
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "retirement_triggered": payload.get("retirement_triggered") is True,
            "same_number_aliases_ignored": _same_number_aliases(root, task_id, rel_path),
            "conductor": conductor.get(task_id, {}),
            "terminal_class": terminal,
            "terminal_evidence_source": evidence_source,
        }
    return matrix


def _scientific_retirement_receipt(
    payloads: Mapping[str, JsonMap], matrix: Mapping[str, JsonMap]
) -> JsonDict:
    payload = payloads.get(SCIENTIFIC_RETIREMENT_TASK_ID, {})
    row = matrix.get(SCIENTIFIC_RETIREMENT_TASK_ID, {})
    skip_row = matrix.get(STRUCTURED_GATE_SKIP_TASK_ID, {})
    verdict = str(payload.get("honest_verdict") or "")
    source_domain_recovery_retired = "Exp6128" in verdict and "source-domain recovery" in verdict
    return {
        "task_id": SCIENTIFIC_RETIREMENT_TASK_ID,
        "declared_deliverable": ACTIVATED_TASK_ARTIFACT_PATHS[
            SCIENTIFIC_RETIREMENT_TASK_ID
        ].as_posix(),
        "terminal_class": row.get("terminal_class"),
        "status": payload.get("status"),
        "honest_verdict": verdict,
        "retirement_triggered": payload.get("retirement_triggered") is True,
        "empirical_item_bank_design_ready_score": payload.get(
            "empirical_item_bank_design_ready_score"
        ),
        "source_domain_recovery_retired": source_domain_recovery_retired,
        "distinct_from_structured_gate_skip": (
            row.get("terminal_class") == "retired"
            and skip_row.get("terminal_class") == "structured-gate-skip"
        ),
        "downstream_skip_task_id": STRUCTURED_GATE_SKIP_TASK_ID,
        "principle": FIELD_PRINCIPLES["scientific_retirement_receipt"],
    }


def _structured_gate_skip_receipt(matrix: Mapping[str, JsonMap]) -> JsonDict:
    row = matrix.get(STRUCTURED_GATE_SKIP_TASK_ID, {})
    conductor = row.get("conductor", {}) if isinstance(row.get("conductor"), Mapping) else {}
    latest_line = str(conductor.get("latest_line") or "")
    return {
        "task_id": STRUCTURED_GATE_SKIP_TASK_ID,
        "declared_deliverable": ACTIVATED_TASK_ARTIFACT_PATHS[
            STRUCTURED_GATE_SKIP_TASK_ID
        ].as_posix(),
        "terminal_class": row.get("terminal_class"),
        "declared_artifact_present": bool(row.get("present")),
        "conductor_latest_status": conductor.get("latest_status"),
        "conductor_latest_line": latest_line,
        "upstream_retired_task_id": SCIENTIFIC_RETIREMENT_TASK_ID,
        "upstream_retired_mentioned": SCIENTIFIC_RETIREMENT_TASK_ID in latest_line,
        "reported_as_run": False,
        "artifact_invented": bool(row.get("present")),
        "principle": FIELD_PRINCIPLES["structured_gate_skip_receipt"],
    }


def _active_range_task_ids(active_roadmap: JsonMap) -> list[str]:
    tasks = active_roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    ids: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        raw_id = str(task.get("id") or "")
        number = _task_number(raw_id)
        if number in NEXT_RANGE_NUMBERS:
            ids.append(f"exp{number}")
    return ids


def _proposal_only_identities_excluded(
    active_roadmap: JsonMap, history_blocks: Sequence[JsonMap], matrix: Mapping[str, JsonMap]
) -> JsonDict:
    source_matrix_ids = set(matrix)
    source_completion_task_ids = {
        str(task.get("id"))
        for block in history_blocks
        if block.get("id") == MILESTONE_FROM and isinstance(block.get("tasks"), list)
        for task in block["tasks"]
        if isinstance(task, Mapping)
    }
    former_ids = list(FORMER_PROPOSAL_ONLY_TASK_IDS)
    canonical_ids = _active_range_task_ids(active_roadmap)
    completion_overlap = sorted(set(former_ids) & source_completion_task_ids)
    matrix_overlap = sorted(set(former_ids) & source_matrix_ids)
    expected_canonical = [f"exp{number}" for number in NEXT_RANGE_NUMBERS]
    return {
        "former_proposal_only_task_ids": former_ids,
        "v532_activated_matrix_task_ids": list(matrix),
        "former_proposal_ids_in_v532_matrix": matrix_overlap,
        "former_proposal_ids_in_v532_completion": completion_overlap,
        "old_proposal_carries_completion_credit": bool(completion_overlap or matrix_overlap),
        "all_excluded_from_v532_archive": not completion_overlap and not matrix_overlap,
        "canonical_active_range_task_ids": canonical_ids,
        "canonical_range_replaces_old_proposal": canonical_ids == expected_canonical,
        "principle": FIELD_PRINCIPLES["proposal_only_identities_excluded"],
    }


def _adversarial_receipts_group(
    receipts: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
) -> JsonDict:
    reports: list[JsonDict] = []
    failed_receipt_task_ids: list[str] = []
    warning_receipt_task_ids: list[str] = []
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        row = matrix[task_id]
        if not row["present"]:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            continue
        flag_count = _receipt_flag_count(receipt)
        max_severity = _receipt_max_severity(receipt)
        reports.append(
            {
                "task_id": task_id,
                "artifact": row["declared_deliverable"],
                "command": str(receipt.get("command") or ""),
                "exit_code": receipt.get("exit_code"),
                "loaded": True,
                "flag_count": flag_count,
                "max_severity": max_severity,
                "flags": _receipt_flags(receipt),
                "receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
        )
        if receipt.get("exit_code") != 0 and max_severity >= 2:
            failed_receipt_task_ids.append(task_id)
        elif flag_count > 0:
            warning_receipt_task_ids.append(task_id)
    return {
        "reports": reports,
        "verified_present_declared_deliverable_count": len(reports),
        "missing_declared_deliverables_not_verified": [
            row["declared_deliverable"] for row in matrix.values() if not row["present"]
        ],
        "failed_receipt_task_ids": failed_receipt_task_ids,
        "warning_receipt_task_ids": warning_receipt_task_ids,
        "flagged_count": sum(int(row["flag_count"]) for row in reports),
        "principle": FIELD_PRINCIPLES["adversarial_verifier_receipts"],
    }


def _root_clutter_inventory(root: Path) -> list[str]:
    allowed = {
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "GEMINI.md",
        "OPENCODE.md",
        "README.md",
        "LICENSE",
        "NOTICE",
        "Cargo.toml",
        "Cargo.lock",
        "Dockerfile.sandbox",
        "Makefile",
        "MANIFEST.in",
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "research-complete.yaml",
        "research-roadmap.yaml",
        "research-roadmap-next.yaml",
        "research-program.md",
        "research-studying.md",
        "research-references.md",
        "research-hardware-wishlist.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "RELEASES.md",
        "RELEASE_NOTES.md",
        "SECURITY.md",
        "rustfmt.toml",
        "docker-compose.yml",
    }
    if not root.exists():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and not entry.name.startswith(".") and entry.name not in allowed
    )


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            {"command": command, "exit_code": None, "status": "not_recorded"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    return [dict(row) for row in tests_run]


def _suite_kinds(row: JsonMap) -> set[str]:
    values: set[str] = set()
    raw = row.get("suite_kind")
    if raw:
        values.add(str(raw))
    raw_many = row.get("suite_kinds")
    if isinstance(raw_many, list):
        values.update(str(value) for value in raw_many)
    return values


def _task_owned_gate_receipts(rows: Sequence[JsonMap]) -> JsonDict:
    task_owned = [dict(row) for row in rows if row.get("ownership_class") == "task_owned"]
    kinds: set[str] = set()
    for row in task_owned:
        kinds.update(_suite_kinds(row))
    failures = [
        row
        for row in task_owned
        if not isinstance(row.get("exit_code"), int) or int(row["exit_code"]) != 0
    ]
    missing = [kind for kind in REQUIRED_TASK_OWNED_GATE_KINDS if kind not in kinds]
    return {
        "required_gate_kinds": list(REQUIRED_TASK_OWNED_GATE_KINDS),
        "observed_gate_kinds": sorted(kinds),
        "all_required_gate_kinds_present": not missing,
        "missing_required_gate_kinds": missing,
        "task_owned_failures": failures,
        "receipts": task_owned,
        "principle": FIELD_PRINCIPLES["test_commands"],
    }


def _root_clutter_delta(rows: Sequence[JsonMap]) -> int:
    before: set[str] = set()
    after: set[str] = set()
    before_seen = False
    after_seen = False
    for row in rows:
        if row.get("ownership_class") != "root_clutter":
            continue
        raw = row.get("root_clutter_paths")
        paths = {str(value) for value in raw} if isinstance(raw, list) else set()
        if row.get("phase") == "before":
            before_seen = True
            before.update(paths)
        if row.get("phase") == "after":
            after_seen = True
            after.update(paths)
    if not before_seen and after_seen:
        before = set(after)
    if not after_seen and before_seen:
        after = set(before)
    return len(after - before)


def _range_number_mentions(text: str) -> set[int]:
    lowered = text.lower()
    if "614" not in lowered and "615" not in lowered:
        return set()
    numbers: set[int] = set()
    for number in NEXT_RANGE_NUMBERS:
        if re.search(rf"(?<![a-z0-9_])exp{number}(?![a-z0-9])", lowered) or re.search(
            rf"(?<![a-z0-9_])experiment_{number}(?![a-z0-9])", lowered
        ):
            numbers.add(number)
    return numbers


def _scan_candidate_paths(root: Path) -> list[Path]:
    candidates = [
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    for folder in (
        "python",
        "tests",
        "scripts",
        "openspec/change-proposals",
        "openspec/capabilities",
        "ops",
    ):
        base = root / folder
        if base.exists():
            candidates.extend(
                path.relative_to(root)
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".test_suite_mutation_runs" not in path.parts
                and path.suffix != ".pyc"
            )
    results = root / "results"
    if results.exists():
        candidates.extend(path.relative_to(root) for path in results.iterdir() if path.is_file())
    return sorted(set(candidates), key=lambda value: value.as_posix())


def _allowed_range_reference_kind(rel_path: Path, numbers: set[int]) -> str | None:
    if rel_path in OWNED_REFERENCE_PATHS:
        return "transition_owned_reference"
    if rel_path in CANONICAL_PLAN_REFERENCE_PATHS:
        return "canonical_v533_plan_reference"
    if rel_path in HISTORICAL_PROPOSAL_REFERENCE_PATHS and numbers <= set(range(6142, 6152)):
        return "replaced_v532_proposal_only_reference"
    return None


def _range_collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed: list[JsonDict] = []
    for rel_path in _scan_candidate_paths(root):
        path = root / rel_path
        text = rel_path.as_posix()
        if path.exists() and path.stat().st_size <= 5_000_000:
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
        numbers = _range_number_mentions(text)
        if not numbers:
            continue
        kind = _allowed_range_reference_kind(rel_path, numbers)
        row = {
            "path": rel_path.as_posix(),
            "kind": kind or "unexpected_next_range_reference",
            "numbers": sorted(numbers),
        }
        if kind:
            allowed.append(row)
        else:
            collisions.append(row)
    return {
        "range": {"start": 6142, "end": 6155},
        "collision_count": len(collisions),
        "collisions": collisions,
        "allowed_references": allowed,
        "principle": FIELD_PRINCIPLES["next_range_collision_count"],
    }


def _dirty_worktree_receipt(root: Path) -> JsonDict:
    if not (root / ".git").exists():
        return {"git_present": False, "dirty_paths": [], "command_exit_code": None}
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    paths = [line[3:] for line in result.stdout.splitlines() if len(line) > 3]
    return {
        "git_present": True,
        "dirty_paths": sorted(paths),
        "command_exit_code": result.returncode,
    }


def _activate_staged_roadmap(root: Path) -> JsonDict:
    active_path = root / ROADMAP_RELATIVE_PATH
    staged_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_before, active_meta_before = _read_yaml_mapping(active_path)
    staged_payload, staged_meta = _read_yaml_mapping(staged_path)
    before_active_sha = active_meta_before["sha256"]
    if staged_meta["present"] and not staged_meta["loadable"]:
        return {
            "mode": "staged_unloadable",
            "activated": False,
            "staged_present": True,
            "staged_loadable": False,
            "active_before_sha256": before_active_sha,
            "active_after_sha256": path_sha256(active_path),
            "copied_exactly": False,
            "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
        }
    if staged_meta["loadable"]:
        if staged_payload.get("milestone") != MILESTONE_TO:
            return {
                "mode": "staged_milestone_mismatch",
                "activated": False,
                "staged_present": True,
                "staged_loadable": True,
                "staged_milestone": staged_payload.get("milestone"),
                "active_before_sha256": before_active_sha,
                "active_after_sha256": path_sha256(active_path),
                "copied_exactly": False,
                "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
            }
        data = staged_path.read_bytes()
        _atomic_write_bytes(active_path, data)
        copied_exactly = active_path.read_bytes() == data
        active_after, active_meta_after = _read_yaml_mapping(active_path)
        return {
            "mode": "copied_staged_roadmap",
            "activated": copied_exactly and active_after.get("milestone") == MILESTONE_TO,
            "staged_present": True,
            "staged_loadable": True,
            "staged_sha256": staged_meta["sha256"],
            "active_before_sha256": before_active_sha,
            "active_after_sha256": active_meta_after["sha256"],
            "active_milestone_after": active_after.get("milestone"),
            "active_roadmap_task_count": len(active_after.get("tasks") or []),
            "copied_exactly": copied_exactly,
            "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
        }
    already_active = (
        active_meta_before["loadable"] and active_before.get("milestone") == MILESTONE_TO
    )
    return {
        "mode": "already_active" if already_active else "missing_staged_and_inactive",
        "activated": bool(already_active),
        "staged_present": False,
        "staged_loadable": False,
        "active_before_sha256": before_active_sha,
        "active_after_sha256": path_sha256(active_path),
        "active_milestone_after": active_before.get("milestone")
        if active_meta_before["loadable"]
        else None,
        "active_roadmap_task_count": len(active_before.get("tasks") or [])
        if active_meta_before["loadable"] and isinstance(active_before.get("tasks"), list)
        else 0,
        "copied_exactly": False,
        "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
    }


def _docs_reconciled(root: Path) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_6142_present": "REQ-REPORT-6142" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        *[path.as_posix() for path in ACTIVATED_TASK_ARTIFACT_PATHS.values()],
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": base_sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(
    failed_preconditions: Sequence[str], activation: JsonMap
) -> tuple[str, str]:
    if failed_preconditions:
        reason = ",".join(failed_preconditions[:3])
        return "blocked", f"blocked: Exp6142 transition preconditions failed ({reason})"
    mode = activation.get("mode")
    return (
        "complete_with_terminal_receipts",
        "complete: archived exactly four terminal .532 identities into .533; "
        "Exp6140 scientific retirement, Exp6141 structured skip, and "
        f"next_range_collision_count=0 preserved; .533 activation mode={mode}",
    )


def _active_task_ids_and_deliverables(active_roadmap: JsonMap) -> tuple[list[str], list[str]]:
    tasks = active_roadmap.get("tasks")
    if not isinstance(tasks, list):
        return [], []
    ids = [str(row.get("id")) for row in tasks if isinstance(row, Mapping) and row.get("id")]
    deliverables = [
        str(row.get("deliverable"))
        for row in tasks
        if isinstance(row, Mapping) and row.get("deliverable")
    ]
    return ids, deliverables


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    root = root.resolve()
    protected_before = _protected_file_hashes(root)
    activation = _activate_staged_roadmap(root)
    active_roadmap, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    roadmap_doc_text = (
        (root / ROADMAP_DOC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / ROADMAP_DOC_RELATIVE_PATH).exists()
        else ""
    )
    roadmap_doc_present = bool(roadmap_doc_text)
    complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    exclusion_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)[1]
    history_blocks = _history_blocks(root)
    payloads, metadata = _artifact_payloads(root)
    conductor = _conductor_receipts(root)
    classes = _exact_terminal_classification(payloads, metadata, conductor["by_task"])
    matrix = _activated_matrix(
        root,
        metadata,
        payloads,
        conductor["by_task"],
        classes["terminal_class_by_task_id"],
    )
    retirement = _scientific_retirement_receipt(payloads, matrix)
    skip = _structured_gate_skip_receipt(matrix)
    append_receipt = _append_completion_if_absent(root, bool(classes["all_activated_terminal"]))
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    verifier_group = _adversarial_receipts_group(receipts, matrix)
    range_scan = _range_collision_scan(root)
    protected = _protected_files_unchanged(root, protected_before)
    test_rows = _tests_run_rows(tests_run)
    task_gate = _task_owned_gate_receipts(test_rows)
    root_clutter_new_count = _root_clutter_delta(test_rows)
    atomic = _atomic_output_receipt(root / RESULT_RELATIVE_PATH)
    docs = _docs_reconciled(root)
    active_task_ids, active_deliverables = _active_task_ids_and_deliverables(active_roadmap)
    proposal_only = _proposal_only_identities_excluded(active_roadmap, history_blocks, matrix)
    present_task_ids = [task_id for task_id, row in matrix.items() if row["present"]]
    receipt_task_ids = {row["task_id"] for row in verifier_group["reports"]}
    missing_receipts = [task_id for task_id in present_task_ids if task_id not in receipt_task_ids]
    failed_preconditions: list[str] = []
    if active_meta["present"] and not active_meta["loadable"]:
        failed_preconditions.append("active_roadmap_unloadable")
    if active_meta["loadable"] and active_roadmap.get("milestone") != MILESTONE_TO:
        failed_preconditions.append("active_roadmap_milestone_mismatch")
    if not activation["activated"]:
        failed_preconditions.append("staged_roadmap_activation_failed")
    if roadmap_next_meta["present"] and not roadmap_next_meta["loadable"]:
        failed_preconditions.append("roadmap_next_unloadable")
    if roadmap_next_meta["loadable"] and roadmap_next.get("milestone") != MILESTONE_TO:
        failed_preconditions.append("roadmap_next_milestone_mismatch")
    if not roadmap_doc_present:
        failed_preconditions.append("vnext_proposal_missing")
    if roadmap_doc_present and (
        "2026.08.533" not in roadmap_doc_text or "Exp6142-Exp6155" not in roadmap_doc_text
    ):
        failed_preconditions.append("vnext_proposal_range_mismatch")
    if complete_meta["present"] and not complete_meta["loadable"]:
        failed_preconditions.append("research_complete_unparseable")
    if exclusion_meta["present"] and not exclusion_meta["loadable"]:
        failed_preconditions.append("exclusion_manifest_unparseable")
    if (
        conductor["source_activation_status"] != "OK"
        or conductor["source_activated_task_count_claim"] != 4
    ):
        failed_preconditions.append("v532_activation_line_missing_or_not_four")
    if (
        conductor["destination_activation_status"] != "OK"
        or conductor["destination_activated_task_count_claim"] != 14
    ):
        failed_preconditions.append("v533_activation_line_missing_or_not_fourteen")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if classes["terminal_class_by_task_id"] != EXPECTED_TERMINAL_CLASSES:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if not (
        retirement["retirement_triggered"]
        and retirement["source_domain_recovery_retired"]
        and retirement["distinct_from_structured_gate_skip"]
    ):
        failed_preconditions.append("scientific_retirement_not_preserved")
    if not (
        skip["terminal_class"] == "structured-gate-skip"
        and skip["declared_artifact_present"] is False
        and skip["reported_as_run"] is False
        and skip["upstream_retired_mentioned"] is True
    ):
        failed_preconditions.append("structured_gate_skip_not_preserved")
    if not (
        proposal_only["all_excluded_from_v532_archive"]
        and proposal_only["canonical_range_replaces_old_proposal"]
    ):
        failed_preconditions.append("proposal_only_identity_included")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if verifier_group["failed_receipt_task_ids"]:
        failed_preconditions.append("adversarial_verifier_failed")
    if task_gate["task_owned_failures"]:
        failed_preconditions.append("task_owned_gate_failed")
    if not task_gate["all_required_gate_kinds_present"]:
        failed_preconditions.append("task_owned_gate_missing")
    if root_clutter_new_count > 0:
        failed_preconditions.append("root_clutter_debt_amplified")
    if append_receipt["duplicate_history_amplification_count"] != 0:
        failed_preconditions.append("duplicate_history_amplified")
    if range_scan["collision_count"] != 0:
        failed_preconditions.append("next_range_collision")
    if not docs["openspec_research_reporting_req_6142_present"]:
        failed_preconditions.append("openspec_req_6142_missing")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    status, verdict = _status_and_verdict(failed_preconditions, activation)
    result_duration = duration_s if duration_s is not None else round(time.monotonic() - start, 6)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE_TO,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": {
            "active_roadmap": {
                **active_meta,
                "milestone": active_roadmap.get("milestone") if active_meta["loadable"] else None,
                "task_ids": active_task_ids,
                "deliverables": active_deliverables,
            },
            "roadmap_next": {
                **roadmap_next_meta,
                "milestone": roadmap_next.get("milestone")
                if roadmap_next_meta["loadable"]
                else None,
                "absence_is_failure_when_active_roadmap_is_v533": False,
            },
            "vnext_proposal": {
                "path": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "present": roadmap_doc_present,
                "sha256": path_sha256(root / ROADMAP_DOC_RELATIVE_PATH),
                "names_milestone_2026_08_533": "2026.08.533" in roadmap_doc_text,
                "names_range_exp6142_exp6155": "Exp6142-Exp6155" in roadmap_doc_text,
            },
            "research_complete": complete_meta,
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
                **conductor,
            },
            "exclusion_manifest": exclusion_meta,
            "known_issues": {
                "path": KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
                "present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
            },
            "source_hashes": _source_hashes(root),
            "root_clutter_inventory": _root_clutter_inventory(root),
            "root_clutter_new_count_from_test_receipts": root_clutter_new_count,
            "dirty_worktree": _dirty_worktree_receipt(root),
            "declared_present_deliverable_hashes": {
                task_id: row["sha256"] for task_id, row in matrix.items() if row["present"]
            },
            "atomic_output": atomic,
            "adversarial_verifier_available": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "range_collision_scan": range_scan,
            "protected_file_hashes_before": protected_before,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "scientific_retirement_receipt": retirement,
        "structured_gate_skip_receipt": skip,
        "proposal_only_identities_excluded": proposal_only,
        "adversarial_verifier_receipts": verifier_group,
        "task_owned_gate_receipts": task_gate,
        "research_complete_append_count": append_receipt["append_count"],
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_receipt": append_receipt,
        "staged_roadmap_activation_receipt": activation,
        "next_task_range": {
            "start": "exp6142",
            "end": "exp6155",
            "reserved_count": len(list(NEXT_RANGE_NUMBERS)),
            "active_roadmap_task_count": len(active_task_ids),
            "active_roadmap_task_ids": active_task_ids,
            "active_roadmap_deliverables": active_deliverables,
            "proposal_reservation_source": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        },
        "next_range_collision_count": range_scan["collision_count"],
        "docs_reconciled": docs,
        "protected_files_unchanged": protected,
        "duration_s": result_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": [str(row.get("command", "")) for row in test_rows],
        "test_exit_codes": {str(row.get("command", "")): row.get("exit_code") for row in test_rows},
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required field: {missing[0]}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if not isinstance(payload.get("next_range_collision_count"), int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if payload["next_range_collision_count"] != 0 and payload.get("status") != "blocked":
        raise ValueError("next_range_collision_count must be zero unless status is blocked")
    if payload.get("research_complete_append_count") not in {0, 1}:
        raise ValueError("research_complete_append_count must be zero or one")
    if payload.get("duplicate_history_amplification_count") != 0:
        raise ValueError("duplicate_history_amplification_count must be zero")
    matrix = payload.get("activated_task_and_deliverable_matrix")
    if not isinstance(matrix, Mapping) or len(matrix) != 4:
        raise ValueError("activated matrix must contain exactly four .532 identities")
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = matrix.get(task_id)
        if not isinstance(row, Mapping):
            raise ValueError("activated matrix must contain exactly four .532 identities")
        if row.get("identity") != [MILESTONE_FROM, task_id, rel_path.as_posix()]:
            raise ValueError("activated identity mismatch")
    classes = payload.get("exact_terminal_classification")
    if (
        not isinstance(classes, Mapping)
        or classes.get("terminal_class_by_task_id") != EXPECTED_TERMINAL_CLASSES
    ):
        raise ValueError("terminal classes do not preserve .532 outcomes")
    retirement = payload.get("scientific_retirement_receipt")
    if (
        not isinstance(retirement, Mapping)
        or retirement.get("task_id") != SCIENTIFIC_RETIREMENT_TASK_ID
        or retirement.get("retirement_triggered") is not True
        or retirement.get("source_domain_recovery_retired") is not True
        or retirement.get("distinct_from_structured_gate_skip") is not True
    ):
        raise ValueError("scientific retirement receipt missing")
    skip = payload.get("structured_gate_skip_receipt")
    if (
        not isinstance(skip, Mapping)
        or skip.get("task_id") != STRUCTURED_GATE_SKIP_TASK_ID
        or skip.get("declared_artifact_present") is not False
        or skip.get("reported_as_run") is not False
        or skip.get("terminal_class") != "structured-gate-skip"
    ):
        raise ValueError("structured gate skip receipt missing")
    proposal = payload.get("proposal_only_identities_excluded")
    if (
        not isinstance(proposal, Mapping)
        or proposal.get("former_proposal_only_task_ids") != list(FORMER_PROPOSAL_ONLY_TASK_IDS)
        or proposal.get("old_proposal_carries_completion_credit") is not False
        or proposal.get("all_excluded_from_v532_archive") is not True
        or proposal.get("canonical_range_replaces_old_proposal") is not True
    ):
        raise ValueError("proposal-only identities were not excluded")
    verifier = payload.get("adversarial_verifier_receipts")
    if not isinstance(verifier, Mapping):
        raise ValueError("adversarial verifier receipts missing")
    if verifier.get("verified_present_declared_deliverable_count") != 3:
        raise ValueError("adversarial verifier receipts must cover three present .532 artifacts")
    for report in verifier.get("reports", []):
        if not isinstance(report, Mapping):
            raise ValueError("adversarial verifier receipt malformed")
        if "scripts/adversarial_verify.py --json" not in str(report.get("command") or ""):
            raise ValueError("adversarial verifier receipt command missing")
        if not report.get("receipt_hash"):
            raise ValueError("adversarial verifier receipt hash missing")
    activation = payload.get("staged_roadmap_activation_receipt")
    if not isinstance(activation, Mapping) or activation.get("activated") is not True:
        raise ValueError("activation receipt missing")
    task_gate = payload.get("task_owned_gate_receipts")
    if (
        isinstance(task_gate, Mapping)
        and task_gate.get("all_required_gate_kinds_present") is not True
    ):
        raise ValueError("task-owned gate receipt missing")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file receipt missing")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field provenance missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        row = provenance.get(field)
        if not isinstance(row, Mapping) or row.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field provenance missing principle for {field}")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("checksum mismatch")


def emit_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
) -> JsonDict:  # pragma: no cover
    report = build_report(root, tests_run=tests_run, adversarial_receipts=adversarial_receipts)
    validate_artifact(report)
    write_json(root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.root)
    validate_artifact(report)
    output = args.output or (args.root / RESULT_RELATIVE_PATH)
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
