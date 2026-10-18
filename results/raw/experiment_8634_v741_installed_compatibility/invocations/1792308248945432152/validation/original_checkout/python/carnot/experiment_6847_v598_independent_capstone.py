"""Build Exp6847 V598 independent capstone.

Spec refs: REQ-RESEARCH-6847,
SCENARIO-RESEARCH-6847-MISSING-ARTIFACTS,
SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY,
SCENARIO-RESEARCH-6847-CLOSED-VERDICTS,
SCENARIO-RESEARCH-6847-BRANCH-INDEPENDENCE, and
SCENARIO-RESEARCH-6847-RETIREMENT.

The reducer reads the active V598 roadmap and all available terminal artifacts.
It does not call an LLM and does not import producer acceptance decisions. Every
acceptance row is recomputed from artifact presence, per-unit rows, source
hashes, and receipts, while missing or failed upstream artifacts stay explicit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.598"
EXPERIMENT_ID = "exp6847-v598-independent-capstone"
CAPSTONE_TASK_ID = "exp6847"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_6847_v598_independent_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_6847_v598_independent_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6847_v598_independent_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_6847_v598_independent_capstone.py")

FULL_TASK_IDS = (
    "exp6835-v598-terminal-evidence-freeze",
    "exp6836-typed-obligation-program-fixture",
    "exp6837-three-family-output-free-compatibility",
    "exp6838-compatibility-shortcut-identifiability-audit",
    "exp6839-bounded-residual-memory-kernel",
    "exp6840-residual-memory-chronological-shard-a",
    "exp6841-residual-memory-delayed-correction-shard-b",
    "exp6842-sealed-memory-pathway-portability-audit",
    "exp6843-live-arc-evidence-stratum-freeze",
    "exp6844-supervisor-action-outcome-credit-audit",
    "exp6845-tool-gap-causal-support-audit",
    "exp6846-typed-arc-shadow-monitor",
    "exp6847-v598-independent-capstone",
)
EXPECTED_TASK_IDS = tuple(f"exp{number}" for number in range(6835, 6848))

TASK_PATHS = {
    "exp6835": "results/experiment_6835_v598_terminal_evidence_freeze.json",
    "exp6836": "results/experiment_6836_typed_obligation_program_fixture.json",
    "exp6837": "results/experiment_6837_three_family_output_free_compatibility.json",
    "exp6838": "results/experiment_6838_compatibility_shortcut_identifiability_audit.json",
    "exp6839": "results/experiment_6839_bounded_residual_memory_kernel.json",
    "exp6840": "results/experiment_6840_residual_memory_chronological_shard_a.json",
    "exp6841": "results/experiment_6841_residual_memory_delayed_correction_shard_b.json",
    "exp6842": "results/experiment_6842_sealed_memory_pathway_portability_audit.json",
    "exp6843": "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
    "exp6844": "results/experiment_6844_supervisor_action_outcome_credit_audit.json",
    "exp6845": "results/experiment_6845_tool_gap_causal_support_audit.json",
    "exp6846": "results/experiment_6846_typed_arc_shadow_monitor.json",
    "exp6847": RESULT_PATH.as_posix(),
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "artifact_inventory",
    "rows",
    "typed_compatibility_disposition",
    "learning_kernel_disposition",
    "continuous_self_learning_disposition",
    "supervisor_credit_disposition",
    "tool_gap_disposition",
    "arc_shadow_monitor_disposition",
    "prior_failure_retirement_decisions",
    "allowed_claims",
    "forbidden_claims",
    "next_milestone_recommendation",
    "v598_disposition_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

BRANCH_DISPOSITION_FIELDS = (
    "typed_compatibility_disposition",
    "learning_kernel_disposition",
    "continuous_self_learning_disposition",
    "supervisor_credit_disposition",
    "tool_gap_disposition",
    "arc_shadow_monitor_disposition",
)

FIELD_PRINCIPLES = {
    "schema": "The schema pins the REQ-RESEARCH-6847 terminal artifact contract.",
    "experiment_id": "The capstone has its own experiment identity and imports no upstream verdict.",
    "milestone": "The artifact is scoped only to the V598 roadmap milestone.",
    "run_date": "The execution date is recorded as supplied by the operator.",
    "random_seed": "The fixed seed records deterministic reduction even though no sampling occurs.",
    "status": "The capstone is terminal even when upstream branches are blocked or missing.",
    "result_path": "The output path is fixed by the V598 roadmap and spec.",
    "spec_refs": "Spec anchors make each test and output requirement auditable.",
    "field_principles": "Every top-level field explains why it is present.",
    "preconditions_checked": "Roadmap, milestone, and planned artifact checks run before reduction.",
    "inference_substrate": (
        "The reviewed aggregation/no-LLM substrate covers deterministic CPU synthesis."
    ),
    "duration_s": "Wall-clock duration covers roadmap parsing, artifact loading, and reduction.",
    "source_artifact_hashes": "Every input artifact and planning document is hash-recorded.",
    "reproducibility_checksum": "A stable content hash protects the final capstone record.",
    "artifact_inventory": "Every planned V598 artifact keeps its terminal, missing, or current state.",
    "rows": "Each acceptance criterion records evidence, expected value, observed value, and status.",
    "typed_compatibility_disposition": (
        "Typed compatibility is judged only from fixture readiness, live scoring, and audit evidence."
    ),
    "learning_kernel_disposition": (
        "The memory kernel branch is judged only from Exp6839 execution receipts."
    ),
    "continuous_self_learning_disposition": (
        "Held-future learning is judged from chronological shards and the sealed audit only."
    ),
    "supervisor_credit_disposition": (
        "Supervisor credit is judged only from matched supervisor outcome cells."
    ),
    "tool_gap_disposition": "Tool-gap support is judged only from tool-gap obligations and outcomes.",
    "arc_shadow_monitor_disposition": (
        "The ARC shadow monitor is judged only from default-off reachability and replay rows."
    ),
    "prior_failure_retirement_decisions": (
        "Prior failures are checked without letting older evidence rescue a current branch."
    ),
    "allowed_claims": "Claims are bounded to recomputed evidence and terminal branch state.",
    "forbidden_claims": "Unsupported solve, benefit, policy, or cross-mechanism claims stay barred.",
    "next_milestone_recommendation": (
        "Each branch receives exactly one evidence-bounded next action without editing the roadmap."
    ),
    "v598_disposition_complete_score": (
        "Completeness measures branch coverage, not scientific or operational positivity."
    ),
    "gate_check_summary": "Failed rows and observed values remain visible for blocked dispositions.",
    "verifier_is_oracle": "The capstone audits evidence and never acts as its own oracle.",
    "verdict_class": "The top-level verdict uses the closed enum and preserves mixed outcomes.",
    "honest_verdict": "The terminal verdict is prefixed complete_ and states evidence preservation.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str | None:
    """Hash a file as ``sha256:<hex>`` or return ``None`` when absent."""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after removing its own checksum field."""

    unsigned = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return f"sha256:{hashlib.sha256(canonical_json(unsigned)).hexdigest()}"


def spec_anchors(text: str) -> list[str]:
    """Return REQ-* and SCENARIO-* anchors from a spec section."""

    return re.findall(r"\b(?:REQ|SCENARIO)-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", text)


def short_task_id(full_id: str) -> str:
    """Return ``expNNNN`` for an expected V598 task id."""

    match = re.fullmatch(r"(exp\d{4})(?:-[a-z0-9-]+)?", full_id)
    if not match:
        raise ValueError(f"invalid V598 task id: {full_id}")
    task_id = match.group(1)
    if task_id not in TASK_PATHS:
        raise ValueError(f"invalid V598 task id: {full_id}")
    return task_id


def _next_deliverable(lines: Sequence[str], start_index: int) -> str:
    """Return the next deliverable path before the following experiment heading."""

    for line in lines[start_index + 1 :]:
        if line.startswith("### Exp "):
            break
        match = re.search(r"\*\*Deliverable:\*\*\s+`([^`]+)`", line)
        if match:
            return match.group(1)
    raise ValueError("deliverable missing for design task")


def parse_design_tasks(text: str) -> tuple[str, list[JsonDict]]:
    """Parse V598 experiment headings and optional deliverables from the design doc."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s+(?:`([^`]+)`|([^\s`]+))", text)
    if not milestone_match:
        raise ValueError("milestone missing from V598 design")
    milestone = milestone_match.group(1) or milestone_match.group(2)
    tasks: list[JsonDict] = []
    heading_matches = list(re.finditer(r"^### Exp\s*(\d{4}):\s*(.+)$", text, re.MULTILINE))
    for index, match in enumerate(heading_matches):
        next_start = (
            heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(text)
        )
        block = text[match.end() : next_start]
        deliverable_match = re.search(r"\*\*Deliverable:\*\*\s+`([^`]+)`", block)
        task_id = f"exp{match.group(1)}"
        tasks.append(
            {
                "task_id": task_id,
                "title": match.group(2).strip(),
                "deliverable": deliverable_match.group(1) if deliverable_match else None,
            }
        )
    return milestone, tasks


def _milestone_inputs(repo_root: Path) -> tuple[Any, str]:
    """The roadmap manifest and design text AS THEY WERE for this capstone's milestone.

    WHY THIS IS NOT JUST A FILE READ (2026-09-05). This module froze MILESTONE and
    compared it against the LIVE research-roadmap.yaml, which advances every
    milestone. So the capstone and its tests worked only while V598 was active:
    once the roadmap moved on, four tests errored at fixture setup. The V576 and
    V580 capstones fixed the same defect by recovering the archived roadmap from
    git history. This helper does the same, and recovers the design document from
    the SAME commit, because `load_planned_tasks` validates the two against each
    other. See commit history for the incident.

    The live files are used unchanged while the roadmap still holds this
    milestone, so behaviour during the capstone's own milestone is bit-identical.
    A live manifest that is not a mapping is returned as-is, so the caller reports
    that shape error rather than a milestone error.
    """

    roadmap_path = repo_root / ACTIVE_ROADMAP_PATH
    with roadmap_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, Mapping) or manifest.get("milestone") == MILESTONE:
        return manifest, (repo_root / DESIGN_PATH).read_text(encoding="utf-8")

    rel_roadmap = ACTIVE_ROADMAP_PATH.as_posix()
    rel_design = DESIGN_PATH.as_posix()
    git = ["git", "-C", str(repo_root)]
    log = subprocess.run(
        [*git, "log", "--format=%H", "-n", "400", "--", rel_roadmap],
        capture_output=True,
        text=True,
        check=False,
    )
    if log.returncode == 0:
        for commit in log.stdout.split():
            blob = subprocess.run(
                [*git, "show", f"{commit}:{rel_roadmap}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if blob.returncode != 0:
                continue
            try:
                archived = yaml.safe_load(blob.stdout)
            except yaml.YAMLError:
                continue
            if not isinstance(archived, Mapping) or archived.get("milestone") != MILESTONE:
                continue
            design = subprocess.run(
                [*git, "show", f"{commit}:{rel_design}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if design.returncode != 0:
                continue
            return archived, design.stdout

    raise ValueError(
        f"expected V598 roadmap milestone {MILESTONE}; the live roadmap has moved on and "
        f"no commit in the last 400 touching {rel_roadmap} still holds it together with "
        f"{rel_design}"
    )


def load_planned_tasks(repo_root: Path) -> list[JsonDict]:
    """Load and validate the V598 roadmap against the milestone document."""

    manifest, design_text = _milestone_inputs(repo_root)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("V598 roadmap must be a mapping with tasks")

    milestone_doc = manifest.get("milestone_doc")
    if milestone_doc != DESIGN_PATH.as_posix():
        raise ValueError("expected V598 design milestone document path")
    design_milestone, design_tasks = parse_design_tasks(design_text)
    if design_milestone != MILESTONE:
        raise ValueError(f"expected V598 design milestone {MILESTONE}")

    design_ids = [task["task_id"] for task in design_tasks]
    if design_ids != list(EXPECTED_TASK_IDS):
        raise ValueError("V598 design must contain Exp6835 through Exp6847")

    tasks = manifest["tasks"]
    if len(tasks) != len(EXPECTED_TASK_IDS):
        raise ValueError("V598 roadmap must contain exact 13 tasks")

    design_by_id = {task["task_id"]: task for task in design_tasks}
    planned: list[JsonDict] = []
    for index, raw in enumerate(tasks):
        if not isinstance(raw, Mapping):
            raise ValueError("V598 roadmap task must be a mapping")
        full_id = str(raw.get("id", ""))
        task_id = short_task_id(full_id)
        if task_id != EXPECTED_TASK_IDS[index]:
            raise ValueError("V598 roadmap tasks must be ordered Exp6835 through Exp6847")
        deliverable = str(raw.get("deliverable", ""))
        expected_deliverable = TASK_PATHS[task_id]
        if deliverable != expected_deliverable:
            raise ValueError(f"deliverable mismatch for {task_id}: {deliverable}")
        design_deliverable = design_by_id[task_id].get("deliverable")
        if design_deliverable is not None and design_deliverable != deliverable:
            raise ValueError(f"deliverable mismatch for {task_id}: {design_deliverable}")
        planned.append(
            {
                "task_id": task_id,
                "full_id": full_id,
                "title": str(raw.get("title", "")),
                "path": deliverable,
                "requires_gpu": bool(raw.get("requires_gpu", False)),
                "gated_on": list(raw.get("gated_on") or []),
                "prior_failures": list(raw.get("prior_failures") or []),
                "design_title": design_by_id[task_id]["title"],
                "design_deliverable": design_deliverable,
            }
        )
    return planned


def _is_json_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def load_source_artifacts(
    repo_root: Path, planned: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Load available planned artifacts while preserving missing and invalid states."""

    sources: dict[str, JsonDict] = {}
    for task in planned:
        task_id = str(task["task_id"])
        relative_path = Path(str(task["path"]))
        absolute_path = repo_root / relative_path
        if task_id == CAPSTONE_TASK_ID:
            sources[task_id] = {
                "task_id": task_id,
                "path": relative_path.as_posix(),
                "artifact_state": "current_synthesis",
                "payload": None,
                "sha256": sha256_file(absolute_path),
                "error": None,
            }
            continue
        if not absolute_path.exists():
            sources[task_id] = {
                "task_id": task_id,
                "path": relative_path.as_posix(),
                "artifact_state": "missing",
                "payload": None,
                "sha256": None,
                "error": "artifact missing",
            }
            continue
        artifact_hash = sha256_file(absolute_path)
        try:
            payload = json.loads(absolute_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sources[task_id] = {
                "task_id": task_id,
                "path": relative_path.as_posix(),
                "artifact_state": "invalid",
                "payload": None,
                "sha256": artifact_hash,
                "error": f"json decode failed: {exc}",
            }
            continue
        if not _is_json_mapping(payload):
            sources[task_id] = {
                "task_id": task_id,
                "path": relative_path.as_posix(),
                "artifact_state": "invalid",
                "payload": None,
                "sha256": artifact_hash,
                "error": "artifact JSON is not an object",
            }
            continue
        sources[task_id] = {
            "task_id": task_id,
            "path": relative_path.as_posix(),
            "artifact_state": "present",
            "payload": payload,
            "sha256": artifact_hash,
            "error": None,
        }
    return sources


def classify_source_record(record: Mapping[str, Any]) -> str:
    """Classify one artifact state with the closed V598 verdict enum."""

    state = record.get("artifact_state")
    payload = record.get("payload")
    if state == "missing":
        return "blocked"
    if state == "invalid":
        return "disqualified"
    if state == "current_synthesis":
        return "partial"
    if not isinstance(payload, Mapping):
        return "disqualified"

    declared = payload.get("verdict_class")
    if payload.get("verifier_is_oracle") is True and declared == "positive":
        return "circular_positive"
    if declared in CLOSED_VERDICT_CLASSES:
        return str(declared)

    text = " ".join(str(payload.get(key, "")).lower() for key in ("status", "honest_verdict"))
    if "circular" in text and "positive" in text:
        return "circular_positive"
    if "disqualified" in text:
        return "disqualified"
    if "blocked" in text or "gate_check_failed" in text:
        return "blocked"
    if "partial" in text:
        return "partial"
    if "positive" in text or "success" in text:
        return "positive"
    return "null"


def build_artifact_inventory(
    repo_root: Path, planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Return one inventory row per planned V598 task."""

    rows: list[JsonDict] = []
    for task in planned:
        task_id = str(task["task_id"])
        record = sources.get(task_id, {})
        payload = record.get("payload")
        rows.append(
            {
                "task_id": task_id,
                "full_id": task["full_id"],
                "title": task["title"],
                "path": task["path"],
                "path_exists": (repo_root / str(task["path"])).exists(),
                "artifact_state": record.get("artifact_state", "missing"),
                "file_sha256": record.get("sha256"),
                "declared_status": payload.get("status") if isinstance(payload, Mapping) else None,
                "declared_honest_verdict": (
                    payload.get("honest_verdict") if isinstance(payload, Mapping) else None
                ),
                "declared_verdict_class": (
                    payload.get("verdict_class") if isinstance(payload, Mapping) else None
                ),
                "verdict_class": classify_source_record(record),
                "verifier_is_oracle": (
                    payload.get("verifier_is_oracle") if isinstance(payload, Mapping) else None
                ),
                "prior_failure_count": len(task.get("prior_failures") or []),
                "error": record.get("error"),
            }
        )
    return rows


def _normalize_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _source_hash_receipts(payload: Mapping[str, Any]) -> list[JsonDict]:
    receipts = payload.get("source_artifact_hashes")
    if not isinstance(receipts, Mapping):
        return []
    rows: list[JsonDict] = []
    for source_id, receipt in sorted(receipts.items()):
        if isinstance(receipt, Mapping):
            declared = (
                receipt.get("file_sha256")
                or receipt.get("sha256")
                or receipt.get("artifact_sha256")
                or receipt.get("hash")
            )
            path = receipt.get("path") or receipt.get("artifact_path")
        else:
            declared = receipt
            path = None
        rows.append(
            {
                "source_id": str(source_id),
                "path": str(path) if path else TASK_PATHS.get(str(source_id)),
                "declared_sha256": _normalize_hash(declared),
            }
        )
    return rows


def _source_hash_rows(repo_root: Path, task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for receipt in _source_hash_receipts(payload):
        path = receipt["path"]
        declared = receipt["declared_sha256"]
        observed = sha256_file(repo_root / path) if path else None
        passed = bool(declared and observed and declared == observed)
        rows.append(
            _criterion_row(
                task_id,
                f"source_hash.{receipt['source_id']}",
                evidence={"path": path, "source_id": receipt["source_id"]},
                expected_value=declared,
                observed_value=observed,
                passed=passed,
                verdict_class="null" if passed else "disqualified",
                producer_value=None,
                allowed_claim=(
                    f"{task_id} may cite {receipt['source_id']} only when this source hash matches."
                ),
            )
        )
    return rows


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _criterion_row(
    task_id: str,
    criterion: str,
    *,
    evidence: Mapping[str, Any],
    expected_value: Any,
    observed_value: Any,
    passed: bool,
    verdict_class: str,
    producer_value: Any = None,
    allowed_claim: str,
) -> JsonDict:
    producer_agrees = True
    status = "passed" if passed else "failed"
    row_verdict = verdict_class
    if producer_value is not None and not _values_equal(producer_value, observed_value):
        producer_agrees = False
        status = "producer_disagreement"
        row_verdict = "partial"
    return {
        "experiment_id": task_id,
        "criterion_id": f"{task_id}.{criterion}",
        "criterion": criterion,
        "evidence": dict(evidence),
        "expected_value": expected_value,
        "observed_value": observed_value,
        "producer_value": producer_value,
        "producer_agrees": producer_agrees,
        "status": status,
        "verdict_class": row_verdict,
        "allowed_claim": allowed_claim,
    }


def _path_get(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare_gate(observed: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return _values_equal(observed, expected)
    if op == ">=":
        return isinstance(observed, int | float) and float(observed) >= float(expected)
    if op == "<=":
        return isinstance(observed, int | float) and float(observed) <= float(expected)
    raise ValueError(f"unsupported roadmap gate op: {op}")


def _roadmap_gate_rows(
    planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in planned:
        task_id = str(task["task_id"])
        for gate in task.get("gated_on") or []:
            upstream = short_task_id(str(gate["upstream"]))
            artifact_field = str(gate["artifact_field"])
            expected = gate["value"]
            op = str(gate["op"])
            payload = sources.get(upstream, {}).get("payload")
            observed = _path_get(payload, artifact_field)
            passed = _compare_gate(observed, op, expected)
            rows.append(
                _criterion_row(
                    task_id,
                    f"roadmap_gate.{upstream}.{artifact_field}",
                    evidence={
                        "upstream": upstream,
                        "artifact_field": artifact_field,
                        "op": op,
                    },
                    expected_value=expected,
                    observed_value=observed,
                    passed=passed,
                    verdict_class="null" if passed else "blocked",
                    producer_value=None,
                    allowed_claim=(
                        f"{task_id} may run only if {upstream}.{artifact_field} {op} {expected}."
                    ),
                )
            )
    return rows


def _producer_score(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    return value


def _int_score(value: bool) -> int:
    return 1 if value else 0


def _rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("rows")
    return list(value) if isinstance(value, list) else []


def _per_game_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("per_game_results")
    return list(value) if isinstance(value, list) else []


def _unique_count(rows: Iterable[Mapping[str, Any]], *keys: str) -> int:
    identities = []
    for row in rows:
        for key in keys:
            if row.get(key) is not None:
                identities.append(row[key])
                break
    return len(set(identities))


def _bool_path(row: Mapping[str, Any], *keys: str) -> bool:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return False
        current = current.get(key)
    return current is True


def _gate_summary_rows(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    summary = payload.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        return []
    checks = list(summary.get("checks") or [])
    rows: list[JsonDict] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("check", "unnamed_gate"))
        observed = check.get("observed")
        expected = check.get("expected", True)
        passed = check.get("passed") is True
        rows.append(
            _criterion_row(
                task_id,
                f"gate.{name}",
                evidence={"source": "gate_check_summary.checks", "check": name},
                expected_value=expected,
                observed_value=observed,
                passed=passed,
                verdict_class="null" if passed else "blocked",
                producer_value=observed if passed else None,
                allowed_claim=f"{task_id} preserves upstream gate {name} without changing it.",
            )
        )
    readiness = summary.get("readiness_gates")
    if isinstance(readiness, Mapping):
        for name, check in sorted(readiness.items()):
            if not isinstance(check, Mapping):
                continue
            observed = check.get("observed")
            passed = check.get("passed") is True
            rows.append(
                _criterion_row(
                    task_id,
                    f"readiness_gate.{name}",
                    evidence={"source": "gate_check_summary.readiness_gates", "check": name},
                    expected_value=True,
                    observed_value=observed,
                    passed=passed,
                    verdict_class="null" if passed else "blocked",
                    producer_value=None,
                    allowed_claim=f"{task_id} readiness gate {name} is preserved as measured.",
                )
            )
    return rows


def _recompute_6835(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    observed_cells = {
        cell
        for row in raw_rows
        for cell in row.get("observed_target_cell_ids", [])
        if isinstance(row, Mapping)
    }
    missing_cells = {
        cell
        for row in raw_rows
        for cell in row.get("missing_target_cell_ids", [])
        if isinstance(row, Mapping)
    }
    unique_rows = _unique_count(raw_rows, "row_id", "row_sha256")
    source_null_preserved = _path_get(payload, "source_null_preserved.preserved") is True
    row_count_ok = len(raw_rows) == 3780
    unique_ok = unique_rows == len(raw_rows)
    observed_expected = _path_get(payload, "observed_target_cells.count")
    missing_expected = _path_get(payload, "missing_target_cells.count")
    ready = _int_score(
        row_count_ok
        and unique_ok
        and observed_expected == len(observed_cells)
        and missing_expected == len(missing_cells)
        and source_null_preserved
    )
    return [
        _criterion_row(
            task_id,
            "row_count",
            evidence={"row_source": "rows"},
            expected_value=3780,
            observed_value=len(raw_rows),
            passed=row_count_ok,
            verdict_class="null" if row_count_ok else "blocked",
            producer_value=None,
            allowed_claim="Exp6835 has the expected source-unit and obligation-cell row count.",
        ),
        _criterion_row(
            task_id,
            "unique_row_id_count",
            evidence={"row_source": "rows.row_id"},
            expected_value=len(raw_rows),
            observed_value=unique_rows,
            passed=unique_ok,
            verdict_class="null" if unique_ok else "disqualified",
            producer_value=None,
            allowed_claim="Exp6835 row identities are unique.",
        ),
        _criterion_row(
            task_id,
            "observed_target_cells",
            evidence={"row_source": "rows.observed_target_cell_ids"},
            expected_value=observed_expected,
            observed_value=len(observed_cells),
            passed=observed_expected == len(observed_cells),
            verdict_class="null" if observed_expected == len(observed_cells) else "partial",
            producer_value=observed_expected,
            allowed_claim="Observed V598 target cells are recomputed from per-row receipts.",
        ),
        _criterion_row(
            task_id,
            "missing_target_cells",
            evidence={"row_source": "rows.missing_target_cell_ids"},
            expected_value=missing_expected,
            observed_value=len(missing_cells),
            passed=missing_expected == len(missing_cells),
            verdict_class="null" if missing_expected == len(missing_cells) else "partial",
            producer_value=missing_expected,
            allowed_claim="Missing V598 target cells remain explicit.",
        ),
        _criterion_row(
            task_id,
            "source_null_preserved",
            evidence={"field": "source_null_preserved.preserved"},
            expected_value=True,
            observed_value=source_null_preserved,
            passed=source_null_preserved,
            verdict_class="null" if source_null_preserved else "blocked",
            producer_value=source_null_preserved,
            allowed_claim="The Exp6834 null is preserved and not repaired by Exp6835.",
        ),
        _criterion_row(
            task_id,
            "v598_evidence_root_ready_score",
            evidence={"row_source": "rows", "field": "v598_evidence_root_ready_score"},
            expected_value=1,
            observed_value=ready,
            passed=ready == 1,
            verdict_class="null" if ready == 1 else "blocked",
            producer_value=_producer_score(payload, "v598_evidence_root_ready_score"),
            allowed_claim="The V598 evidence root is complete enough for downstream fixture work.",
        ),
    ]


def _recompute_6836(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    unique_rows = _unique_count(raw_rows, "row_id", "row_hash")
    pair_rows_ok = len(raw_rows) == 8 and unique_rows == len(raw_rows)
    pair_lengths_ok = all(row.get("pair_token_length_equal") is True for row in raw_rows)
    candidate_counts_ok = all(len(row.get("candidates", [])) == 2 for row in raw_rows)
    compile_results = payload.get("compile_parity_results")
    per_candidate = []
    if isinstance(compile_results, Mapping):
        per_candidate = list(compile_results.get("per_candidate") or [])
    compile_flags_ok = bool(compile_results) and all(
        compile_results.get(key) is True
        for key in (
            "all_views_share_atom_identities",
            "energy_zero_iff_satisfied",
            "guard_parity",
            "diagnostics_match",
        )
    )
    compile_rows_ok = bool(per_candidate) and all(
        row.get("energy_zero_iff_satisfied") is True
        and row.get("memory_guard_matches_satisfaction") is True
        and row.get("arc_guard_matches_satisfaction") is True
        and row.get("diagnostics_match") is True
        and row.get("all_views_equal") is True
        for row in per_candidate
        if isinstance(row, Mapping)
    )
    mutation_rows = list(payload.get("checker_mutation_results") or [])
    mutation_ok = bool(mutation_rows) and all(
        row.get("expected_rejected") is True and row.get("observed_rejected") is True
        for row in mutation_rows
        if isinstance(row, Mapping)
    )
    shortcut_manifest = payload.get("shortcut_control_manifest")
    no_model_scores = (
        isinstance(shortcut_manifest, Mapping)
        and _path_get(shortcut_manifest, "fixture_integrity.passed") is True
        and shortcut_manifest.get("model_scores_present") is False
        and shortcut_manifest.get("generated_answers_present") is False
    )
    pair_ready = _int_score(
        pair_rows_ok and pair_lengths_ok and candidate_counts_ok and no_model_scores
    )
    program_ready = _int_score(
        pair_ready == 1 and compile_flags_ok and compile_rows_ok and mutation_ok
    )
    return [
        _criterion_row(
            task_id,
            "candidate_pair_rows",
            evidence={"row_source": "rows", "checks": ["row_count", "unique_row_id"]},
            expected_value=8,
            observed_value=len(raw_rows),
            passed=pair_rows_ok,
            verdict_class="null" if pair_rows_ok else "blocked",
            producer_value=None,
            allowed_claim="Exp6836 emitted eight fixed candidate-pair fixture rows.",
        ),
        _criterion_row(
            task_id,
            "equal_token_pairing",
            evidence={"row_source": "rows.pair_token_length_equal"},
            expected_value=True,
            observed_value=pair_lengths_ok,
            passed=pair_lengths_ok,
            verdict_class="null" if pair_lengths_ok else "disqualified",
            producer_value=None,
            allowed_claim="Candidate pairs have equal token lengths for output-free scoring.",
        ),
        _criterion_row(
            task_id,
            "compile_parity",
            evidence={"row_source": "compile_parity_results.per_candidate"},
            expected_value=True,
            observed_value=compile_flags_ok and compile_rows_ok,
            passed=compile_flags_ok and compile_rows_ok,
            verdict_class="null" if compile_flags_ok and compile_rows_ok else "disqualified",
            producer_value=None,
            allowed_claim="Compiled energy, predicates, guards, and diagnostics share identities.",
        ),
        _criterion_row(
            task_id,
            "checker_mutation_rejection",
            evidence={"row_source": "checker_mutation_results"},
            expected_value=True,
            observed_value=mutation_ok,
            passed=mutation_ok,
            verdict_class="null" if mutation_ok else "disqualified",
            producer_value=None,
            allowed_claim="Mutated fixtures are rejected by exact checkers.",
        ),
        _criterion_row(
            task_id,
            "obligation_pair_fixture_ready_score",
            evidence={"row_source": "rows", "field": "obligation_pair_fixture_ready_score"},
            expected_value=1,
            observed_value=pair_ready,
            passed=pair_ready == 1,
            verdict_class="null" if pair_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "obligation_pair_fixture_ready_score"),
            allowed_claim="The fixed candidate-pair fixture is ready for later local scoring.",
        ),
        _criterion_row(
            task_id,
            "typed_obligation_program_ready_score",
            evidence={"row_source": "rows", "field": "typed_obligation_program_ready_score"},
            expected_value=1,
            observed_value=program_ready,
            passed=program_ready == 1,
            verdict_class="null" if program_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "typed_obligation_program_ready_score"),
            allowed_claim="The typed obligation program is internally ready and deterministic.",
        ),
    ]


def _recompute_6837(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    model_specs = payload.get("model_specs") or payload.get("MODEL_SPECS") or []
    model_count = len(model_specs) if isinstance(model_specs, list) else 0
    stream_ready = _int_score(bool(raw_rows) and model_count == 3)
    rows = [
        _criterion_row(
            task_id,
            "model_specs_present",
            evidence={"field": "model_specs"},
            expected_value=3,
            observed_value=model_count,
            passed=model_count == 3,
            verdict_class="null" if model_count == 3 else "blocked",
            producer_value=None,
            allowed_claim="Exp6837 names the three required model families.",
        ),
        _criterion_row(
            task_id,
            "forced_sequence_rows_present",
            evidence={"row_source": "rows"},
            expected_value=">0",
            observed_value=len(raw_rows),
            passed=bool(raw_rows),
            verdict_class="null" if raw_rows else "blocked",
            producer_value=None,
            allowed_claim="Output-free compatibility needs scored forced-sequence rows.",
        ),
        _criterion_row(
            task_id,
            "obligation_compatibility_stream_ready_score",
            evidence={"row_source": "rows", "field": "obligation_compatibility_stream_ready_score"},
            expected_value=1,
            observed_value=stream_ready,
            passed=stream_ready == 1,
            verdict_class="null" if stream_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "obligation_compatibility_stream_ready_score"),
            allowed_claim="Typed compatibility margins are ready only after live rows exist.",
        ),
    ]
    return rows + _gate_summary_rows(task_id, payload)


def _recompute_6839(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    unique_rows = _unique_count(raw_rows, "row_id", "row_sha256")
    exact_rows = list(payload.get("exact_outcome_credit_rows") or [])
    admission_rows = list(payload.get("admission_rows") or [])
    restart = (
        payload.get("restart_results")
        or payload.get("restart_durability_results")
        or payload.get("restart_receipt")
        or {}
    )
    rollback = payload.get("rollback_results") or payload.get("rollback_receipt") or {}
    restart_ok = isinstance(restart, Mapping) and all(
        restart.get(key) is True
        for key in ("matches_clean_replay", "bytes_identity", "crash_recovery")
        if key in restart
    )
    if isinstance(restart, Mapping) and "crash_recovery_ignored_partial_checkpoint" in restart:
        restart_ok = restart_ok and restart.get("crash_recovery_ignored_partial_checkpoint") is True
    rollback_ok = isinstance(rollback, Mapping) and (
        rollback.get("rolled_back") is True
        and (
            rollback.get("restored_parent_bytes") is True
            or rollback.get("restored_parent_hash") is True
        )
    )
    exact_ok = len(exact_rows) == len(raw_rows) and bool(exact_rows)
    admission_ok = len(admission_rows) == len(raw_rows) and bool(admission_rows)
    ready = _int_score(
        bool(raw_rows)
        and unique_rows == len(raw_rows)
        and exact_ok
        and admission_ok
        and restart_ok
        and rollback_ok
    )
    return [
        _criterion_row(
            task_id,
            "kernel_rows",
            evidence={"row_source": "rows"},
            expected_value=384,
            observed_value=len(raw_rows),
            passed=len(raw_rows) == 384 and unique_rows == len(raw_rows),
            verdict_class="null"
            if len(raw_rows) == 384 and unique_rows == len(raw_rows)
            else "blocked",
            producer_value=None,
            allowed_claim="Exp6839 emitted unique residual-memory kernel rows.",
        ),
        _criterion_row(
            task_id,
            "exact_outcome_credit_rows",
            evidence={"row_source": "exact_outcome_credit_rows"},
            expected_value=len(raw_rows),
            observed_value=len(exact_rows),
            passed=exact_ok,
            verdict_class="null" if exact_ok else "blocked",
            producer_value=None,
            allowed_claim="Every kernel row has exact outcome-credit evidence.",
        ),
        _criterion_row(
            task_id,
            "admission_rows",
            evidence={"row_source": "admission_rows"},
            expected_value=len(raw_rows),
            observed_value=len(admission_rows),
            passed=admission_ok,
            verdict_class="null" if admission_ok else "blocked",
            producer_value=None,
            allowed_claim="Every kernel row has an admission decision row.",
        ),
        _criterion_row(
            task_id,
            "restart_and_rollback",
            evidence={"fields": ["restart_results", "rollback_results"]},
            expected_value=True,
            observed_value={"restart": restart_ok, "rollback": rollback_ok},
            passed=restart_ok and rollback_ok,
            verdict_class="null" if restart_ok and rollback_ok else "blocked",
            producer_value=None,
            allowed_claim="The residual-memory kernel has restart and rollback receipts.",
        ),
        _criterion_row(
            task_id,
            "residual_memory_kernel_ready_score",
            evidence={"field": "residual_memory_kernel_ready_score"},
            expected_value=1,
            observed_value=ready,
            passed=ready == 1,
            verdict_class="null" if ready == 1 else "blocked",
            producer_value=_producer_score(payload, "residual_memory_kernel_ready_score"),
            allowed_claim="The bounded kernel is executable; this is not a future-benefit claim.",
        ),
        _criterion_row(
            task_id,
            "csl_kernel_execution_complete_score",
            evidence={"field": "csl_kernel_execution_complete_score"},
            expected_value=1,
            observed_value=ready,
            passed=ready == 1,
            verdict_class="null" if ready == 1 else "blocked",
            producer_value=_producer_score(payload, "csl_kernel_execution_complete_score"),
            allowed_claim="The CSL kernel execution surface is complete.",
        ),
    ]


def _held_future_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    by_arm: dict[str, JsonDict] = defaultdict(
        lambda: {"held_future_rows": 0, "decision_rows": 0, "correct_count": 0}
    )
    for row in rows:
        if not _bool_path(row, "held_future_metric", "is_held_future"):
            continue
        arm = str(row.get("arm"))
        stats = by_arm[arm]
        stats["held_future_rows"] += 1
        if row.get("abstained") is not True:
            stats["decision_rows"] += 1
        if row.get("decision_correct") is True:
            stats["correct_count"] += 1
    for stats in by_arm.values():
        decision_rows = stats["decision_rows"]
        held_future_rows = stats["held_future_rows"]
        stats["decision_accuracy"] = (
            round(stats["correct_count"] / decision_rows, 6) if decision_rows else None
        )
        stats["overall_exact_match_rate"] = (
            round(stats["correct_count"] / held_future_rows, 6) if held_future_rows else None
        )
    return dict(by_arm)


def _recompute_csl_shard(
    task_id: str, payload: Mapping[str, Any], complete_field: str
) -> list[JsonDict]:
    raw_rows = _rows(payload)
    unique_rows = _unique_count(raw_rows, "row_id", "row_sha256")
    frozen = all(row.get("decision_frozen_before_outcome_reveal") is True for row in raw_rows)
    revealed = all(row.get("outcome_revealed_after_decision") is True for row in raw_rows)
    by_arm = _held_future_by_arm(raw_rows)
    no_memory = by_arm.get("no_memory", {})
    verified = by_arm.get("verified_residual_memory", {})
    delta = None
    if (
        verified.get("decision_accuracy") is not None
        and no_memory.get("decision_accuracy") is not None
    ):
        delta = round(verified["decision_accuracy"] - no_memory["decision_accuracy"], 6)
    complete = _int_score(
        bool(raw_rows) and unique_rows == len(raw_rows) and frozen and revealed and bool(by_arm)
    )
    return [
        _criterion_row(
            task_id,
            "chronological_rows",
            evidence={"row_source": "rows", "checks": ["decision_frozen", "outcome_revealed"]},
            expected_value={"rows": len(raw_rows), "frozen": True, "revealed": True},
            observed_value={
                "rows": len(raw_rows),
                "unique_rows": unique_rows,
                "frozen": frozen,
                "revealed": revealed,
            },
            passed=complete == 1,
            verdict_class="null" if complete == 1 else "blocked",
            producer_value=None,
            allowed_claim=f"{task_id} rows preserve chronological decision/outcome separation.",
        ),
        _criterion_row(
            task_id,
            "held_future_arm_metrics",
            evidence={"row_source": "rows.held_future_metric"},
            expected_value="reported",
            observed_value={
                "no_memory": no_memory,
                "verified_residual_memory": verified,
                "decision_accuracy_delta": delta,
            },
            passed=bool(no_memory) and bool(verified),
            verdict_class="null" if no_memory and verified else "blocked",
            producer_value=None,
            allowed_claim=f"{task_id} held-future metrics are diagnostic, not a benefit claim.",
        ),
        _criterion_row(
            task_id,
            complete_field,
            evidence={"row_source": "rows", "field": complete_field},
            expected_value=1,
            observed_value=complete,
            passed=complete == 1,
            verdict_class="null" if complete == 1 else "blocked",
            producer_value=_producer_score(payload, complete_field),
            allowed_claim=f"{task_id} completed its chronological CSL shard.",
        ),
    ]


def _recompute_6842(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    attacks = sorted({str(row.get("attack")) for row in raw_rows})
    imported_count = sum(1 for row in raw_rows if row.get("producer_aggregate_imported") is True)
    unique_rows = _unique_count(raw_rows, "row_id", "row_sha256")
    expected_attacks = {
        "capacity",
        "deletion",
        "fresh_reduction",
        "latent_error",
        "poison",
        "reorder",
        "restart",
        "rollback",
        "stale_credit",
        "substitution",
    }
    fresh_rows = [
        row
        for row in raw_rows
        if row.get("attack") == "fresh_reduction"
        and row.get("arm") == "verified_residual_memory"
        and row.get("split") == "held_future"
        and isinstance(row.get("effect_vs_no_memory"), int | float)
    ]
    effects = [float(row["effect_vs_no_memory"]) for row in fresh_rows]
    wins = sum(1 for effect in effects if effect > 0)
    losses = sum(1 for effect in effects if effect < 0)
    mean_effect = round(sum(effects) / len(effects), 6) if effects else None
    held_future_benefit = (
        bool(effects) and mean_effect is not None and mean_effect > 0 and wins > losses
    )
    poison_accepted = sum(
        1 for row in raw_rows if row.get("attack") == "poison" and row.get("poison_accepted")
    )
    stale_accepted = sum(
        1
        for row in raw_rows
        if row.get("attack") == "stale_credit" and row.get("stale_credit_accepted")
    )
    leakage = poison_accepted == 0 and stale_accepted == 0
    restart = [row for row in raw_rows if row.get("attack") == "restart"]
    rollback = [row for row in raw_rows if row.get("attack") == "rollback"]
    capacity = [row for row in raw_rows if row.get("attack") == "capacity"]
    durability = (
        bool(restart)
        and all(row.get("restart_clean_replay_match") is True for row in restart)
        and bool(rollback)
        and all(row.get("rollback_restored_parent") is True for row in rollback)
        and bool(capacity)
        and all(row.get("capacity_within_bound") is True for row in capacity)
    )
    family_effects: dict[str, list[float]] = defaultdict(list)
    for row in fresh_rows:
        family_effects[str(row.get("family"))].append(float(row["effect_vs_no_memory"]))
    family_means = {
        family: round(sum(values) / len(values), 6)
        for family, values in sorted(family_effects.items())
    }
    portability = bool(family_means) and all(mean > 0 for mean in family_means.values())
    dose_on_wins = [
        abs(float(row.get("memory_dose", 0)))
        for row in fresh_rows
        if row["effect_vs_no_memory"] > 0
    ]
    dose_on_losses = [
        abs(float(row.get("memory_dose", 0)))
        for row in fresh_rows
        if row["effect_vs_no_memory"] < 0
    ]
    mean_win_dose = round(sum(dose_on_wins) / len(dose_on_wins), 6) if dose_on_wins else None
    mean_loss_dose = round(sum(dose_on_losses) / len(dose_on_losses), 6) if dose_on_losses else None
    calibrated_dose = bool(
        held_future_benefit
        and mean_win_dose is not None
        and mean_loss_dose is not None
        and mean_win_dose > mean_loss_dose
    )
    complete = _int_score(
        bool(raw_rows)
        and unique_rows == len(raw_rows)
        and set(attacks) == expected_attacks
        and imported_count == 0
    )
    ready = _int_score(
        complete == 1
        and held_future_benefit
        and durability
        and leakage
        and portability
        and calibrated_dose
    )
    return [
        _criterion_row(
            task_id,
            "sealed_csl_audit_complete_score",
            evidence={"row_source": "rows", "checks": ["attack_coverage", "unique_rows"]},
            expected_value=1,
            observed_value=complete,
            passed=complete == 1,
            verdict_class="null" if complete == 1 else "blocked",
            producer_value=_producer_score(payload, "sealed_csl_audit_complete_score"),
            allowed_claim="Exp6842 fully replayed the sealed CSL audit rows.",
        ),
        _criterion_row(
            task_id,
            "producer_aggregate_imported",
            evidence={"row_source": "rows.producer_aggregate_imported"},
            expected_value=0,
            observed_value=imported_count,
            passed=imported_count == 0,
            verdict_class="null" if imported_count == 0 else "disqualified",
            producer_value=None,
            allowed_claim="The sealed CSL audit rows do not import upstream aggregate decisions.",
        ),
        _criterion_row(
            task_id,
            "held_future_benefit_gate",
            evidence={"row_source": "rows[attack=fresh_reduction,split=held_future]"},
            expected_value={"mean_effect_vs_no_memory": ">0", "wins_gt_losses": True},
            observed_value={
                "rows": len(fresh_rows),
                "mean_effect_vs_no_memory": mean_effect,
                "wins": wins,
                "losses": losses,
            },
            passed=held_future_benefit,
            verdict_class="null" if held_future_benefit else "blocked",
            producer_value=None,
            allowed_claim="Held-future benefit is required before any continuous-learning upgrade.",
        ),
        _criterion_row(
            task_id,
            "durability_gate",
            evidence={"row_source": "rows[attack in restart,rollback,capacity]"},
            expected_value=True,
            observed_value={
                "restart_rows": len(restart),
                "rollback_rows": len(rollback),
                "capacity_rows": len(capacity),
                "durability": durability,
            },
            passed=durability,
            verdict_class="null" if durability else "blocked",
            producer_value=None,
            allowed_claim="Memory durability receipts pass restart, rollback, and capacity checks.",
        ),
        _criterion_row(
            task_id,
            "leakage_gate",
            evidence={"row_source": "rows[attack in poison,stale_credit]"},
            expected_value={"poison_accepted": 0, "stale_credit_accepted": 0},
            observed_value={
                "poison_accepted_count": poison_accepted,
                "stale_credit_accepted_count": stale_accepted,
            },
            passed=leakage,
            verdict_class="null" if leakage else "blocked",
            producer_value=None,
            allowed_claim="The sealed audit did not accept poison or stale credit.",
        ),
        _criterion_row(
            task_id,
            "portability_gate",
            evidence={"row_source": "rows[attack=fresh_reduction].family"},
            expected_value="all family means > 0",
            observed_value=family_means,
            passed=portability,
            verdict_class="null" if portability else "blocked",
            producer_value=None,
            allowed_claim="A CSL claim requires held-future benefit across families.",
        ),
        _criterion_row(
            task_id,
            "calibrated_dose_gate",
            evidence={"row_source": "rows.memory_dose"},
            expected_value="benefit and win dose > loss dose",
            observed_value={
                "mean_abs_dose_on_wins": mean_win_dose,
                "mean_abs_dose_on_losses": mean_loss_dose,
                "held_future_benefit_gate": held_future_benefit,
            },
            passed=calibrated_dose,
            verdict_class="null" if calibrated_dose else "blocked",
            producer_value=None,
            allowed_claim="A CSL claim requires calibrated dose coupled to positive benefit.",
        ),
        _criterion_row(
            task_id,
            "continuous_self_learning_ready_score",
            evidence={"row_source": "rows", "field": "continuous_self_learning_ready_score"},
            expected_value=1,
            observed_value=ready,
            passed=ready == 1,
            verdict_class="null" if ready == 1 else "blocked",
            producer_value=_producer_score(payload, "continuous_self_learning_ready_score"),
            allowed_claim="Continuous self-learning remains null unless all sealed gates pass.",
        ),
    ]


def _recompute_6843(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _rows(payload)
    unique_rows = _unique_count(raw_rows, "row_id", "row_sha256", "stratum_identity")
    supervisor_rows = sum(1 for row in raw_rows if row.get("supervisor_receipt_complete") is True)
    tool_gap_rows = sum(1 for row in raw_rows if row.get("tool_gap_receipt_complete") is True)
    solve_claim = payload.get("solve_claim") is True
    inventory_ready = _int_score(
        bool(raw_rows) and unique_rows == len(raw_rows) and not solve_claim
    )
    supervisor_ready = _int_score(supervisor_rows > 0)
    tool_gap_ready = _int_score(tool_gap_rows > 0)
    return [
        _criterion_row(
            task_id,
            "arc_inventory_complete_score",
            evidence={"row_source": "rows", "field": "arc_inventory_complete_score"},
            expected_value=1,
            observed_value=inventory_ready,
            passed=inventory_ready == 1,
            verdict_class="null" if inventory_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "arc_inventory_complete_score"),
            allowed_claim="The live ARC inventory is terminal and solve-free.",
        ),
        _criterion_row(
            task_id,
            "supervisor_cells_ready_score",
            evidence={"row_source": "rows.supervisor_receipt_complete"},
            expected_value=1,
            observed_value=supervisor_ready,
            passed=supervisor_ready == 1,
            verdict_class="null" if supervisor_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "supervisor_cells_ready_score"),
            allowed_claim="Supervisor cells are available for a separate credit audit.",
        ),
        _criterion_row(
            task_id,
            "tool_gap_cells_ready_score",
            evidence={"row_source": "rows.tool_gap_receipt_complete"},
            expected_value=1,
            observed_value=tool_gap_ready,
            passed=tool_gap_ready == 1,
            verdict_class="null" if tool_gap_ready == 1 else "blocked",
            producer_value=_producer_score(payload, "tool_gap_cells_ready_score"),
            allowed_claim="Tool-gap cells are available only as receipt inventory.",
        ),
    ]


def _recompute_6844(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _per_game_rows(payload)
    kinds = Counter(str(row.get("row_kind")) for row in raw_rows)
    joined = sum(1 for row in raw_rows if row.get("exact_outcome_joined") is True)
    temporal_ok = all(row.get("temporal_order_verified") is True for row in raw_rows)
    headroom_nonzero = any(_path_get(row, "headroom.nonzero_headroom") is True for row in raw_rows)
    effect_delta = [
        row.get("effect_delta")
        for row in payload.get("action_credit_results", [])
        if isinstance(row, Mapping) and isinstance(row.get("effect_delta"), int | float)
    ]
    effect_eligible = _int_score(
        bool(raw_rows)
        and joined == len(raw_rows)
        and temporal_ok
        and kinds.get("eligible_redirect", 0) > 0
        and kinds.get("matched_control", 0) > 0
        and headroom_nonzero
        and any(float(value) != 0.0 for value in effect_delta)
    )
    complete = _int_score(bool(raw_rows) and joined == len(raw_rows) and temporal_ok)
    return [
        _criterion_row(
            task_id,
            "supervisor_causal_audit_complete_score",
            evidence={"row_source": "per_game_results"},
            expected_value=1,
            observed_value=complete,
            passed=complete == 1,
            verdict_class="null" if complete == 1 else "blocked",
            producer_value=_producer_score(payload, "supervisor_causal_audit_complete_score"),
            allowed_claim="The supervisor audit joined exact later outcomes.",
        ),
        _criterion_row(
            task_id,
            "headroom_nonzero",
            evidence={"row_source": "per_game_results.headroom"},
            expected_value=True,
            observed_value=headroom_nonzero,
            passed=headroom_nonzero,
            verdict_class="null" if headroom_nonzero else "blocked",
            producer_value=None,
            allowed_claim="Supervisor outcome credit requires nonzero matched headroom.",
        ),
        _criterion_row(
            task_id,
            "supervisor_effect_eligible_score",
            evidence={
                "row_source": "per_game_results",
                "field": "supervisor_effect_eligible_score",
            },
            expected_value=1,
            observed_value=effect_eligible,
            passed=effect_eligible == 1,
            verdict_class="null" if effect_eligible == 1 else "blocked",
            producer_value=_producer_score(payload, "supervisor_effect_eligible_score"),
            allowed_claim="Supervisor effect remains blocked unless headroom and nonzero effect exist.",
        ),
    ]


def _recompute_6845(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    ledger = payload.get("obligation_ledger")
    obligation_rows = []
    if isinstance(ledger, Mapping):
        obligation_rows = list(ledger.get("rows") or [])
        obligation_count = int(ledger.get("row_count") or len(obligation_rows))
    else:
        obligation_count = 0
    joins = payload.get("request_receipt_joins")
    joined_count = joins.get("joined_count", 0) if isinstance(joins, Mapping) else 0
    transport_ok = (
        isinstance(joins, Mapping)
        and joins.get("request_response_mismatch_count", 0) == 0
        and joins.get("missing_identity_count", 0) == 0
    )
    audit_complete = _int_score(transport_ok and joined_count == obligation_count)
    effect_eligible = _int_score(obligation_count > 0 and joined_count == obligation_count)
    return [
        _criterion_row(
            task_id,
            "tool_gap_audit_complete_score",
            evidence={
                "row_source": "obligation_ledger.rows",
                "field": "tool_gap_audit_complete_score",
            },
            expected_value=1,
            observed_value=audit_complete,
            passed=audit_complete == 1,
            verdict_class="null" if audit_complete == 1 else "blocked",
            producer_value=_producer_score(payload, "tool_gap_audit_complete_score"),
            allowed_claim="The tool-gap audit completed transport bookkeeping.",
        ),
        _criterion_row(
            task_id,
            "tool_gap_obligations",
            evidence={"row_source": "obligation_ledger.rows"},
            expected_value=">0",
            observed_value=obligation_count,
            passed=obligation_count > 0,
            verdict_class="null" if obligation_count > 0 else "blocked",
            producer_value=None,
            allowed_claim="Tool-gap support requires at least one observed tool-gap obligation.",
        ),
        _criterion_row(
            task_id,
            "tool_gap_effect_eligible_score",
            evidence={
                "row_source": "obligation_ledger.rows",
                "field": "tool_gap_effect_eligible_score",
            },
            expected_value=1,
            observed_value=effect_eligible,
            passed=effect_eligible == 1,
            verdict_class="null" if effect_eligible == 1 else "blocked",
            producer_value=_producer_score(payload, "tool_gap_effect_eligible_score"),
            allowed_claim="Tool-gap causal support remains blocked without obligations.",
        ),
    ]


def _recompute_6846(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = _per_game_rows(payload)
    agreement_count = sum(1 for row in raw_rows if row.get("agreement") is True)
    total = len(raw_rows)
    agreement_rate = round(agreement_count / total, 6) if total else 0.0
    false_interventions = sum(1 for row in raw_rows if row.get("false_intervention") is True)
    missed_violations = sum(1 for row in raw_rows if row.get("missed_violation") is True)
    byte_identity = bool(raw_rows) and all(
        row.get("action_byte_identity") is True for row in raw_rows
    )
    reachability = _path_get(payload, "canonical_reachability_receipt.reachable") is True
    default_off = payload.get("default_off_receipt")
    default_off_ok = (
        isinstance(default_off, Mapping)
        and default_off.get("default_enabled") is False
        and default_off.get("action_byte_identity_preserved") is True
    )
    latency_bound = _path_get(payload, "latency_results.bound_s")
    max_latency = max(
        (
            float(row.get("latency_s", 0.0))
            for row in raw_rows
            if isinstance(row.get("latency_s"), int | float)
        ),
        default=0.0,
    )
    latency_ok = isinstance(latency_bound, int | float) and max_latency <= float(latency_bound)
    ready = _int_score(
        bool(raw_rows)
        and reachability
        and default_off_ok
        and agreement_rate == 1.0
        and false_interventions == 0
        and missed_violations == 0
        and byte_identity
        and latency_ok
        and payload.get("solve_claim") is not True
    )
    producer_agreement = _path_get(payload, "exact_agreement_results.agreement_rate")
    return [
        _criterion_row(
            task_id,
            "canonical_reachability",
            evidence={"field": "canonical_reachability_receipt.reachable"},
            expected_value=True,
            observed_value=reachability,
            passed=reachability,
            verdict_class="null" if reachability else "blocked",
            producer_value=reachability,
            allowed_claim="The typed ARC shadow monitor has a reachable canonical seam.",
        ),
        _criterion_row(
            task_id,
            "default_off_action_identity",
            evidence={"field": "default_off_receipt"},
            expected_value=True,
            observed_value=default_off_ok,
            passed=default_off_ok,
            verdict_class="null" if default_off_ok else "blocked",
            producer_value=default_off_ok,
            allowed_claim="The monitor is default-off and preserves action identity.",
        ),
        _criterion_row(
            task_id,
            "exact_agreement_rate",
            evidence={"row_source": "per_game_results.agreement"},
            expected_value=1.0,
            observed_value=agreement_rate,
            passed=agreement_rate == 1.0,
            verdict_class="null" if agreement_rate == 1.0 else "blocked",
            producer_value=producer_agreement,
            allowed_claim="Shadow decisions exactly match replay truth labels.",
        ),
        _criterion_row(
            task_id,
            "false_intervention_rate",
            evidence={"row_source": "per_game_results.false_intervention"},
            expected_value=0.0,
            observed_value=round(false_interventions / total, 6) if total else None,
            passed=total > 0 and false_interventions == 0,
            verdict_class="null" if total > 0 and false_interventions == 0 else "blocked",
            producer_value=_path_get(payload, "false_intervention_results.rate"),
            allowed_claim="No false intervention occurred in the default-off replay rows.",
        ),
        _criterion_row(
            task_id,
            "missed_violation_rate",
            evidence={"row_source": "per_game_results.missed_violation"},
            expected_value=0.0,
            observed_value=round(missed_violations / total, 6) if total else None,
            passed=total > 0 and missed_violations == 0,
            verdict_class="null" if total > 0 and missed_violations == 0 else "blocked",
            producer_value=_path_get(payload, "missed_violation_results.rate"),
            allowed_claim="No missed violation occurred in the replay rows.",
        ),
        _criterion_row(
            task_id,
            "latency_bound",
            evidence={"row_source": "per_game_results.latency_s"},
            expected_value=latency_bound,
            observed_value=max_latency,
            passed=latency_ok,
            verdict_class="null" if latency_ok else "blocked",
            producer_value=_path_get(payload, "latency_results.max_latency_s"),
            allowed_claim="Monitor latency stayed within the declared bound.",
        ),
        _criterion_row(
            task_id,
            "action_byte_identity",
            evidence={"row_source": "per_game_results.action_byte_identity"},
            expected_value=True,
            observed_value=byte_identity,
            passed=byte_identity,
            verdict_class="null" if byte_identity else "blocked",
            producer_value=_path_get(payload, "action_byte_identity_results.all_identical"),
            allowed_claim="The shadow monitor did not alter returned actions.",
        ),
        _criterion_row(
            task_id,
            "typed_arc_shadow_monitor_ready_score",
            evidence={
                "row_source": "per_game_results",
                "field": "typed_arc_shadow_monitor_ready_score",
            },
            expected_value=1,
            observed_value=ready,
            passed=ready == 1,
            verdict_class="null" if ready == 1 else "blocked",
            producer_value=_producer_score(payload, "typed_arc_shadow_monitor_ready_score"),
            allowed_claim="The typed ARC shadow monitor is ready as a default-off monitor only.",
        ),
    ]


def _artifact_specific_rows(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    reducers = {
        "exp6835": _recompute_6835,
        "exp6836": _recompute_6836,
        "exp6837": _recompute_6837,
        "exp6839": _recompute_6839,
        "exp6840": lambda task, data: _recompute_csl_shard(
            task, data, "csl_shard_a_complete_score"
        ),
        "exp6841": lambda task, data: _recompute_csl_shard(
            task, data, "csl_shard_b_complete_score"
        ),
        "exp6842": _recompute_6842,
        "exp6843": _recompute_6843,
        "exp6844": _recompute_6844,
        "exp6845": _recompute_6845,
        "exp6846": _recompute_6846,
    }
    reducer = reducers.get(task_id)
    return reducer(task_id, payload) if reducer else []


def build_rows(
    repo_root: Path, planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Build one row per experiment and acceptance criterion."""

    rows: list[JsonDict] = []
    for task in planned:
        task_id = str(task["task_id"])
        record = sources.get(task_id, {})
        state = record.get("artifact_state", "missing")
        present = state in {"present", "current_synthesis"}
        rows.append(
            _criterion_row(
                task_id,
                "artifact_present",
                evidence={"path": task["path"], "artifact_state": state},
                expected_value=True,
                observed_value=present,
                passed=present,
                verdict_class="partial"
                if state == "current_synthesis"
                else ("null" if present else "blocked"),
                producer_value=None,
                allowed_claim=f"{task_id} artifact state is preserved for the V598 capstone.",
            )
        )
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            rows.extend(_source_hash_rows(repo_root, task_id, payload))
            rows.extend(_artifact_specific_rows(task_id, payload))
    rows.extend(_roadmap_gate_rows(planned, sources))
    return rows


def _row_by_criterion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["criterion_id"]): row for row in rows}


def _blocking(rows_by_id: Mapping[str, Mapping[str, Any]], criteria: Sequence[str]) -> list[str]:
    return [
        criterion
        for criterion in criteria
        if rows_by_id.get(criterion, {}).get("status") != "passed"
    ]


def _status_for_blocking(blocking: Sequence[str], *, ready_score: int) -> str:
    if blocking:
        return "blocked"
    return "null" if ready_score == 1 else "partial"


def _make_disposition(
    disposition_id: str,
    *,
    verdict_class: str,
    ready_score: int,
    evidence_criteria: Sequence[str],
    blocking_criteria: Sequence[str],
    allowed_claim: str,
    forbidden_claims: Sequence[str],
    next_action: str,
) -> JsonDict:
    return {
        "disposition_id": disposition_id,
        "verdict_class": verdict_class,
        "ready_score": ready_score,
        "evidence_criteria": list(evidence_criteria),
        "blocking_criteria": list(blocking_criteria),
        "allowed_claim": allowed_claim,
        "forbidden_claims": list(forbidden_claims),
        "next_action": next_action,
    }


def build_branch_dispositions(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Create six independent branch dispositions from local criterion rows."""

    by_id = _row_by_criterion(rows)
    typed_blocking = _blocking(
        by_id,
        (
            "exp6837.obligation_compatibility_stream_ready_score",
            "exp6838.artifact_present",
        ),
    )
    kernel_blocking = _blocking(
        by_id,
        (
            "exp6839.residual_memory_kernel_ready_score",
            "exp6839.csl_kernel_execution_complete_score",
        ),
    )
    learning_blocking = _blocking(
        by_id,
        (
            "exp6840.csl_shard_a_complete_score",
            "exp6841.csl_shard_b_complete_score",
            "exp6842.continuous_self_learning_ready_score",
        ),
    )
    supervisor_blocking = _blocking(
        by_id,
        (
            "exp6844.headroom_nonzero",
            "exp6844.supervisor_effect_eligible_score",
        ),
    )
    tool_blocking = _blocking(
        by_id,
        (
            "exp6845.tool_gap_obligations",
            "exp6845.tool_gap_effect_eligible_score",
        ),
    )
    monitor_blocking = _blocking(
        by_id,
        (
            "exp6846.canonical_reachability",
            "exp6846.default_off_action_identity",
            "exp6846.exact_agreement_rate",
            "exp6846.false_intervention_rate",
            "exp6846.missed_violation_rate",
            "exp6846.latency_bound",
            "exp6846.action_byte_identity",
            "exp6846.typed_arc_shadow_monitor_ready_score",
        ),
    )
    kernel_ready = _int_score(not kernel_blocking)
    monitor_ready = _int_score(not monitor_blocking)
    learning_ready = _int_score(not learning_blocking)
    return {
        "typed_compatibility_disposition": _make_disposition(
            "typed_compatibility",
            verdict_class="blocked" if typed_blocking else "null",
            ready_score=_int_score(not typed_blocking),
            evidence_criteria=[
                "exp6836.typed_obligation_program_ready_score",
                "exp6836.obligation_pair_fixture_ready_score",
                "exp6837.obligation_compatibility_stream_ready_score",
                "exp6838.artifact_present",
            ],
            blocking_criteria=typed_blocking,
            allowed_claim=(
                "Typed obligation fixtures are ready, but live margins and shortcut audit evidence "
                "are unavailable."
            ),
            forbidden_claims=[
                "No three-family output-free typed compatibility margin claim is supported.",
                "No shortcut-identifiability claim is supported.",
            ],
            next_action=(
                "Retire the missing Exp6838 audit from the current branch and rerun typed scoring "
                "only after exclusive GPU leases and live canaries are available."
            ),
        ),
        "learning_kernel_disposition": _make_disposition(
            "learning_kernel",
            verdict_class=_status_for_blocking(kernel_blocking, ready_score=kernel_ready),
            ready_score=kernel_ready,
            evidence_criteria=[
                "exp6839.residual_memory_kernel_ready_score",
                "exp6839.csl_kernel_execution_complete_score",
            ],
            blocking_criteria=kernel_blocking,
            allowed_claim=(
                "The bounded residual-memory kernel executed with exact credit, admission, "
                "restart, and rollback receipts."
            ),
            forbidden_claims=[
                "The kernel alone does not support held-future learning benefit.",
            ],
            next_action=(
                "Keep the kernel as an execution substrate and change only the learning rule or "
                "evaluation headroom in a later milestone."
            ),
        ),
        "continuous_self_learning_disposition": _make_disposition(
            "continuous_self_learning",
            verdict_class="null",
            ready_score=learning_ready,
            evidence_criteria=[
                "exp6840.csl_shard_a_complete_score",
                "exp6841.csl_shard_b_complete_score",
                "exp6842.continuous_self_learning_ready_score",
            ],
            blocking_criteria=learning_blocking,
            allowed_claim=(
                "Chronological shards and the sealed audit completed, but held-future benefit, "
                "dose, and portability gates failed."
            ),
            forbidden_claims=[
                "No continuous self-learning improvement claim is supported.",
                "No future-task transfer claim is supported.",
            ],
            next_action=(
                "Mechanically retire this residual-memory rule shape unless a new mechanism first "
                "passes a held-future benefit gate with nonzero headroom."
            ),
        ),
        "supervisor_credit_disposition": _make_disposition(
            "supervisor_credit",
            verdict_class="blocked" if supervisor_blocking else "null",
            ready_score=_int_score(not supervisor_blocking),
            evidence_criteria=[
                "exp6844.supervisor_causal_audit_complete_score",
                "exp6844.headroom_nonzero",
                "exp6844.supervisor_effect_eligible_score",
            ],
            blocking_criteria=supervisor_blocking,
            allowed_claim=(
                "Supervisor rows joined exact later outcomes, but all matched cells had zero "
                "headroom and zero progress effect."
            ),
            forbidden_claims=[
                "No supervisor causal credit or action-outcome benefit claim is supported.",
            ],
            next_action=(
                "Collect matched supervisor and control cells with nonzero headroom before any "
                "new supervisor-credit estimate."
            ),
        ),
        "tool_gap_disposition": _make_disposition(
            "tool_gap",
            verdict_class="blocked" if tool_blocking else "null",
            ready_score=_int_score(not tool_blocking),
            evidence_criteria=[
                "exp6845.tool_gap_audit_complete_score",
                "exp6845.tool_gap_obligations",
                "exp6845.tool_gap_effect_eligible_score",
            ],
            blocking_criteria=tool_blocking,
            allowed_claim=(
                "The tool-gap audit completed bookkeeping, but no tool-gap obligations were "
                "observed."
            ),
            forbidden_claims=[
                "No tool-gap causal support or utility lift claim is supported.",
            ],
            next_action=(
                "Instrument first-party tool-gap obligation receipts before running any utility "
                "or outcome comparison."
            ),
        ),
        "arc_shadow_monitor_disposition": _make_disposition(
            "arc_shadow_monitor",
            verdict_class=_status_for_blocking(monitor_blocking, ready_score=monitor_ready),
            ready_score=monitor_ready,
            evidence_criteria=[
                "exp6846.canonical_reachability",
                "exp6846.default_off_action_identity",
                "exp6846.exact_agreement_rate",
                "exp6846.typed_arc_shadow_monitor_ready_score",
            ],
            blocking_criteria=monitor_blocking,
            allowed_claim=(
                "The typed ARC shadow monitor is reachable, default-off, action-preserving, "
                "and exact in replay."
            ),
            forbidden_claims=[
                "No live ARC level solve claim is supported by the shadow monitor.",
                "No action-changing ARC intervention claim is supported.",
            ],
            next_action=(
                "Keep the monitor default-off and use it only as a replayable attribution guard "
                "until a separate intervention experiment is designed."
            ),
        ),
    }


def _prior_failure_matches(prior_verdict: str, current: Mapping[str, Any]) -> bool:
    normalized = prior_verdict.lower().replace(":", "_").strip()
    candidates = {
        str(current.get("declared_honest_verdict", "")).lower(),
        str(current.get("declared_status", "")).lower(),
        str(current.get("declared_verdict_class", "")).lower(),
        str(current.get("verdict_class", "")).lower(),
    }
    return normalized in candidates


def build_prior_failure_retirement_decisions(
    planned: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Check every roadmap prior_failures entry for repeated terminal verdicts."""

    current_by_task = {row["task_id"]: row for row in inventory}
    decisions: list[JsonDict] = []
    for task in planned:
        task_id = str(task["task_id"])
        current = current_by_task.get(task_id, {})
        for prior in task.get("prior_failures") or []:
            prior_verdict = str(prior.get("verdict", ""))
            verdict_recurs = _prior_failure_matches(prior_verdict, current)
            retirement_flag = bool(prior.get("retire_if_same_verdict", False) and verdict_recurs)
            decisions.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior_verdict,
                    "current_artifact_state": current.get("artifact_state"),
                    "current_verdict_class": current.get("verdict_class"),
                    "current_honest_verdict": current.get("declared_honest_verdict"),
                    "verdict_recurs": verdict_recurs,
                    "retirement_recommended": retirement_flag,
                    "recommendation": (
                        "mechanical_retirement"
                        if retirement_flag
                        else "do_not_retire_from_prior_failure_match"
                    ),
                }
            )
    return decisions


def _source_hash_manifest(repo_root: Path, planned: Sequence[Mapping[str, Any]]) -> JsonDict:
    docs = {
        "active_roadmap": ACTIVE_ROADMAP_PATH,
        "prompt_next_roadmap": NEXT_ROADMAP_PATH,
        "milestone_doc": DESIGN_PATH,
        "reporting_spec": REPORT_SPEC_PATH,
        "module": MODULE_PATH,
        "wrapper": WRAPPER_PATH,
        "tests": TEST_PATH,
    }
    return {
        "documents": {
            key: {"path": path.as_posix(), "file_sha256": sha256_file(repo_root / path)}
            for key, path in docs.items()
        },
        "artifacts": {
            str(task["task_id"]): {
                "path": task["path"],
                "file_sha256": sha256_file(repo_root / str(task["path"])),
            }
            for task in planned
        },
    }


def _preconditions(
    repo_root: Path, planned: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]
) -> JsonDict:
    missing = [row["task_id"] for row in inventory if row["artifact_state"] == "missing"]
    invalid = [row["task_id"] for row in inventory if row["artifact_state"] == "invalid"]
    return {
        "active_roadmap_present": (repo_root / ACTIVE_ROADMAP_PATH).exists(),
        "active_roadmap_milestone": MILESTONE,
        "milestone_document_present": (repo_root / DESIGN_PATH).exists(),
        "prompt_next_roadmap_present": (repo_root / NEXT_ROADMAP_PATH).exists(),
        "planned_artifact_count": len(planned),
        "planned_artifacts_match_exp6835_through_exp6847": [task["task_id"] for task in planned]
        == list(EXPECTED_TASK_IDS),
        "missing_artifacts_preserved": missing,
        "invalid_artifacts_preserved": invalid,
        "capstone_ungated": True,
        "active_roadmap_modified": False,
    }


def _gate_check_summary(
    rows: Sequence[Mapping[str, Any]], dispositions: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    failed = [
        {
            "criterion_id": row["criterion_id"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
            "status": row["status"],
            "verdict_class": row["verdict_class"],
        }
        for row in rows
        if row["status"] != "passed"
    ]
    blocked = {
        key: value["blocking_criteria"]
        for key, value in dispositions.items()
        if value.get("verdict_class") == "blocked"
    }
    return {
        "passed": not failed,
        "failed_checks": failed,
        "blocked_dispositions": blocked,
        "producer_disagreement_count": sum(
            1 for row in rows if row["status"] == "producer_disagreement"
        ),
    }


def _allowed_claims(dispositions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [str(disposition["allowed_claim"]) for disposition in dispositions.values()]


def _forbidden_claims(dispositions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    claims = [
        "No V598 artifact supports an ARC game or level solve claim.",
        "No branch may be rescued with evidence from another configuration or mechanism.",
    ]
    for disposition in dispositions.values():
        claims.extend(str(claim) for claim in disposition["forbidden_claims"])
    return claims


def _next_milestone_recommendation(dispositions: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "disposition_id": disposition["disposition_id"],
            "recommendation": disposition["next_action"],
            "evidence_bound": disposition["evidence_criteria"],
            "roadmap_mutation_performed": False,
        }
        for disposition in dispositions.values()
    ]


def _with_field_principles(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} is required by REQ-RESEARCH-6847.")
        for key in artifact
    }
    return artifact


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    planned: Sequence[Mapping[str, Any]] | None = None,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6847 artifact without writing it."""

    planned_tasks = list(planned) if planned is not None else load_planned_tasks(repo_root)
    source_records = (
        dict(sources) if sources is not None else load_source_artifacts(repo_root, planned_tasks)
    )
    inventory = build_artifact_inventory(repo_root, planned_tasks, source_records)
    rows = build_rows(repo_root, planned_tasks, source_records)
    dispositions = build_branch_dispositions(rows)
    complete_score = _int_score(
        len(inventory) == len(EXPECTED_TASK_IDS)
        and set(dispositions) == set(BRANCH_DISPOSITION_FIELDS)
        and all(disposition.get("disposition_id") for disposition in dispositions.values())
    )
    artifact: JsonDict = {
        "schema": "carnot.research.v598.independent_capstone.v1",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_terminal_null",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": [
            "REQ-RESEARCH-6847",
            "SCENARIO-RESEARCH-6847-MISSING-ARTIFACTS",
            "SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY",
            "SCENARIO-RESEARCH-6847-CLOSED-VERDICTS",
            "SCENARIO-RESEARCH-6847-BRANCH-INDEPENDENCE",
            "SCENARIO-RESEARCH-6847-RETIREMENT",
        ],
        "field_principles": {},
        "random_seed": 6847,
        "preconditions_checked": _preconditions(repo_root, planned_tasks, inventory),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": _source_hash_manifest(repo_root, planned_tasks),
        "reproducibility_checksum": "",
        "artifact_inventory": inventory,
        "rows": rows,
        **dispositions,
        "prior_failure_retirement_decisions": build_prior_failure_retirement_decisions(
            planned_tasks, inventory
        ),
        "allowed_claims": _allowed_claims(dispositions),
        "forbidden_claims": _forbidden_claims(dispositions),
        "next_milestone_recommendation": _next_milestone_recommendation(dispositions),
        "v598_disposition_complete_score": complete_score,
        "gate_check_summary": _gate_check_summary(rows, dispositions),
        "verifier_is_oracle": False,
        "verdict_class": "null",
        "honest_verdict": "complete_null_v598_independent_capstone_all_branch_states_preserved",
    }
    _with_field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors for an Exp6847 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not principles:
        errors.append("field_principles must be a nonempty mapping")
    elif set(principles) != set(artifact):
        errors.append("field_principles must cover every top-level field")

    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare deterministic CPU independent synthesis")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") != "null":
        errors.append("top-level verdict_class must be null")
    if str(artifact.get("honest_verdict", "")).startswith("complete_") is False:
        errors.append("honest_verdict must be terminal-prefixed complete_")
    if not artifact.get("rows"):
        errors.append("rows must be nonempty")
    if not artifact.get("artifact_inventory"):
        errors.append("artifact_inventory must be nonempty")
    if not artifact.get("source_artifact_hashes"):
        errors.append("source_artifact_hashes must be nonempty")
    if artifact.get("v598_disposition_complete_score") != 1:
        errors.append("v598_disposition_complete_score must equal 1")

    inventory = artifact.get("artifact_inventory", [])
    if isinstance(inventory, list):
        for row in inventory:
            verdict_class = row.get("verdict_class") if isinstance(row, Mapping) else None
            if verdict_class not in CLOSED_VERDICT_CLASSES:
                errors.append(f"inventory row has invalid verdict_class: {verdict_class}")
                break

    rows = artifact.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append("rows must contain mappings")
                break
            verdict_class = row.get("verdict_class")
            if verdict_class not in CLOSED_VERDICT_CLASSES:
                errors.append(f"criterion row has invalid verdict_class: {verdict_class}")
                break
            if (
                row.get("status") == "producer_disagreement"
                and row.get("producer_agrees") is not False
            ):
                errors.append("producer disagreement rows must set producer_agrees=false")
                break

    for field in BRANCH_DISPOSITION_FIELDS:
        disposition = artifact.get(field)
        if not isinstance(disposition, Mapping):
            errors.append(f"{field} must be a mapping")
            continue
        if disposition.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
            errors.append(f"{field} has invalid verdict_class")
        if disposition.get("verdict_class") == "blocked" and not disposition.get(
            "blocking_criteria"
        ):
            errors.append(f"{field} blocked without blocking_criteria")

    recommendations = artifact.get("next_milestone_recommendation")
    if not isinstance(recommendations, list) or len(recommendations) != len(
        BRANCH_DISPOSITION_FIELDS
    ):
        errors.append("next_milestone_recommendation must contain one row per branch")

    allowed_claims = artifact.get("allowed_claims", [])
    if any("policy benefit" in str(claim) for claim in allowed_claims):
        errors.append("allowed_claims must not contain policy benefit overclaims")
    forbidden_claims = artifact.get("forbidden_claims", [])
    if not any("level solve" in str(claim) for claim in forbidden_claims):
        errors.append("forbidden_claims must bar level solve claims")

    expected_checksum = reproducibility_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    return errors


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():  # pragma: no cover - normal replace removes the temp file.
            tmp.unlink()


def _build_and_write(repo_root: Path, output: Path, run_date: str) -> int:
    start = time.monotonic()
    planned = load_planned_tasks(repo_root)
    sources = load_source_artifacts(repo_root, planned)
    artifact = build_artifact(
        repo_root,
        run_date=run_date,
        duration_s=time.monotonic() - start,
        planned=planned,
        sources=sources,
    )
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(error)
        return 1
    _write_json(output, artifact)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260901")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    output = args.output if args.output is not None else args.repo_root / RESULT_PATH
    if args.validate:
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read artifact: {exc}")
            return 1
        errors = validate_artifact(payload)
        if errors:
            for error in errors:
                print(error)
            return 1
        return 0
    return _build_and_write(args.repo_root, output, str(args.date))


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper in tests.
    raise SystemExit(main())
