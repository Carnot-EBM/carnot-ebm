"""Build the append-only V602 evidence and manifest contract.

The reducer reads committed artifacts, roadmaps, and Git objects. It does not
run any scientific method. This keeps evidence repair separate from the old
measurements that the contract audits. See REQ-REPORT-6874.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
GitRunner = Callable[[Path, tuple[str, ...], bool], str | None]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6874_v602_evidence_substrate_manifest_contract.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
VERIFIER_PATH = Path("scripts/adversarial_verify.py")
ALIAS_LINT_PATH = Path("scripts/substrate_alias_evidence_lint.py")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
V601_MILESTONE = "2026.09.601"
V602_MILESTONE = "2026.09.602"
RANDOM_SEED = 6874
V601_ACTIVATION_COMMIT = "568474739f"
V602_STAGE_COMMIT = "3bd07b29b4"
V602_ACTIVATION_COMMIT = "7d0690f600"
V602_DELETION_COMMIT = "78bc93c6c8"

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "v601_terminal_task_rows",
    "v601_stored_vs_fresh_adversarial_rows",
    "deterministic_substrate_requalification_rows",
    "fixed_sequence_semantic_branch_closed",
    "reliability_update_branch_closed",
    "v601_design_yaml_mismatch_rows",
    "v601_unexecuted_design_task_rows",
    "v602_document_yaml_parity_rows",
    "v602_activation_copy_receipt",
    "v602_gate_contract_rows",
    "prior_failure_contract_rows",
    "retired_dependency_rows",
    "random_seed",
    "reproducibility_checksum",
    "v602_evidence_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
OWN_SOURCE_PATHS = {
    "module": Path("python/carnot/experiment_6874_v602_evidence_substrate_manifest_contract.py"),
    "wrapper": Path("scripts/experiments/experiment_6874_v602_evidence_substrate_manifest_contract.py"),
    "focused_tests": Path(
        "tests/python/test_experiment_6874_v602_evidence_substrate_manifest_contract.py"
    ),
    "spec": SPEC_PATH,
}
BASE_SOURCE_PATHS = {
    "active_v602_roadmap": ROADMAP_PATH,
    "v602_design": DESIGN_PATH,
    "conductor_log": CONDUCTOR_LOG_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "adversarial_verifier": VERIFIER_PATH,
    "substrate_alias_lint": ALIAS_LINT_PATH,
}

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents a downstream reader from guessing the record shape.",
    "experiment_id": "The fixed identity binds this report to the activated task.",
    "run_date": "The execution date distinguishes this audit from later evidence states.",
    "status": "A terminal status confirms that the reducer wrote a complete record.",
    "field_principles": "A reason for each field keeps the evidence contract understandable.",
    "preconditions_checked": "Explicit checks show why a blocked reduction could not proceed.",
    "inference_substrate": "The no-LLM substrate prevents this audit from becoming new science.",
    "duration_s": "Measured wall time makes the local aggregation operationally auditable.",
    "source_artifact_hashes": "Hashes bind each finding to the exact source bytes that were read.",
    "rows": "One row per V601 or V602 task prevents denominator loss.",
    "v601_terminal_task_rows": "Terminal rows keep artifact and conductor evidence separate.",
    "v601_stored_vs_fresh_adversarial_rows": "Separate flag sets preserve verifier drift.",
    "deterministic_substrate_requalification_rows": "Code and receipts, not prose, prove that no LLM ran.",
    "fixed_sequence_semantic_branch_closed": "The failed calibrated rule and correct block close this method.",
    "reliability_update_branch_closed": "No gain over read-only closes the V601 update method.",
    "v601_design_yaml_mismatch_rows": "Mismatch rows explain which proposed tasks never became executable.",
    "v601_unexecuted_design_task_rows": "Exact absence checks prevent design text from becoming run evidence.",
    "v602_document_yaml_parity_rows": "Per-task comparisons expose omissions and field drift.",
    "v602_activation_copy_receipt": "Git object identities prove the staged roadmap copy and later deletion.",
    "v602_gate_contract_rows": "Exact gate rows stop spelling drift from silently breaking dependencies.",
    "prior_failure_contract_rows": "Primary verdict checks stop an unchanged failed method from returning.",
    "retired_dependency_rows": "Retirement checks prevent an executable gate from naming a dead upstream.",
    "random_seed": "A fixed seed records that ordered reductions use one stable identity.",
    "reproducibility_checksum": "The checksum binds the deterministic report while excluding elapsed time.",
    "v602_evidence_contract_ready_score": "Exp6875 and Exp6881 consume this exact readiness field.",
    "gate_check_summary": "Expected and observed values make every blocked verdict diagnosable.",
    "verifier_is_oracle": "False prevents this evidence reducer from authorizing scientific truth.",
    "verdict_class": "The closed class states the evidence boundary without free-form promotion.",
    "honest_verdict": "The terminal prefix lets the conductor classify the completed audit safely.",
}

_VERIFY_CACHE: dict[tuple[str, str], JsonDict] = {}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for content identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 identity in the project format."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash a file while preserving a missing path as a null value."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def unwrap_principle(value: Any) -> Any:
    """Read only the explicit two-key principle wrapper."""

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one exact gate check."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every check and expose the first failure for conductor readers."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _experiment_number(value: Any) -> int | None:
    """Read an experiment number from either an integer or a named identity."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"(?:exp|experiment_)?(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _normalized_date(value: Any) -> str:
    """Normalize compact and ISO dates to one ISO date."""

    text = str(unwrap_principle(value) or "")[:10]
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def read_json_object(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one JSON object and retain the exact failure class."""

    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(value, Mapping):
        return None, "json_object_required"
    return dict(value), None


def artifact_freshness(payload: Mapping[str, Any], number: int, run_date: str) -> JsonDict:
    """Reject an old bootstrap that happens to use the expected result path."""

    observed_number = _experiment_number(
        unwrap_principle(payload.get("experiment_id", payload.get("experiment")))
    )
    observed_date = _normalized_date(payload.get("run_date"))
    status = str(unwrap_principle(payload.get("status")) or "").lower()
    verdict = str(unwrap_principle(payload.get("honest_verdict")) or "")
    terminal = bool(verdict) and status not in {"", "in_progress", "running", "bootstrap"}
    expected = {"experiment_id": number, "run_date": _normalized_date(run_date)}
    observed = {
        "experiment_id": observed_number,
        "run_date": observed_date,
        "status": status,
        "terminal_verdict_present": bool(verdict),
    }
    return _check(
        "artifact_freshness",
        expected,
        observed,
        observed_number == number and observed_date == expected["run_date"] and terminal,
    )


def _derive_verdict_class(payload: Mapping[str, Any]) -> str:
    """Use the declared class, with a narrow fallback for conductor gate artifacts."""

    declared = str(unwrap_principle(payload.get("verdict_class")) or "")
    if declared in CLOSED_VERDICT_CLASSES:
        return declared
    verdict = str(unwrap_principle(payload.get("honest_verdict")) or "").lower()
    status = str(unwrap_principle(payload.get("status")) or "").lower()
    if "blocked" in verdict or status == "blocked":
        return "blocked"
    if "null" in verdict or "no_improvement" in verdict:
        return "null"
    return "partial"


def _gate_result(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize producer and conductor-created gate summaries."""

    rows = unwrap_principle(payload.get("gates_evaluated"))
    if isinstance(rows, list) and rows:
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            normalized.append(
                {
                    "upstream": row.get("upstream"),
                    "artifact_field": row.get("artifact_field"),
                    "op": row.get("op"),
                    "expected": row.get("expected", row.get("value")),
                    "observed": row.get("actual", row.get("observed")),
                    "passed": row.get("passed") is True,
                }
            )
        return {"passed": all(row["passed"] for row in normalized), "rows": normalized}
    summary = unwrap_principle(payload.get("gate_check_summary"))
    if isinstance(summary, Mapping):
        return {
            "passed": summary.get("passed"),
            "failed_check": summary.get("failed_check"),
            "expected": summary.get("expected"),
            "observed": summary.get("observed"),
        }
    return {"passed": None, "summary": summary}


def _extract_run_command(prompt: Any) -> str | None:
    """Read the declared command from a roadmap prompt."""

    match = re.search(r"^\s*Run command:\s*(.+)$", str(prompt or ""), re.MULTILINE)
    if not match:
        return None
    command = match.group(1).strip()
    if "&&" in command:
        command = command.rsplit("&&", 1)[1].strip()
    return command


def _normalize_gate(row: Mapping[str, Any]) -> JsonDict:
    """Normalize one executable gate without changing field spellings."""

    return {
        "upstream": str(unwrap_principle(row.get("upstream")) or ""),
        "artifact_field": str(unwrap_principle(row.get("artifact_field")) or ""),
        "op": str(unwrap_principle(row.get("op")) or ""),
        "value": unwrap_principle(row.get("value", row.get("expected"))),
    }


def parse_roadmap_tasks(document: Mapping[str, Any]) -> list[JsonDict]:
    """Parse executable task order, deliverables, gates, and prior failures."""

    tasks_value = unwrap_principle(document.get("tasks"))
    if not isinstance(tasks_value, list):
        raise ValueError("roadmap tasks list is required")
    rows: list[JsonDict] = []
    for order, task in enumerate(tasks_value, 1):
        if not isinstance(task, Mapping):
            raise ValueError(f"malformed roadmap task at order {order}")
        task_id = str(unwrap_principle(task.get("id")) or "")
        number = _experiment_number(task_id)
        deliverable = str(unwrap_principle(task.get("deliverable")) or "")
        if number is None or not task_id or not deliverable:
            raise ValueError(f"malformed roadmap task at order {order}")
        gates_value = unwrap_principle(task.get("gated_on"))
        gates = gates_value if isinstance(gates_value, list) else []
        priors_value = unwrap_principle(task.get("prior_failures"))
        priors = priors_value if isinstance(priors_value, list) else []
        rows.append(
            {
                "order": order,
                "number": number,
                "task_id": task_id,
                "title": str(unwrap_principle(task.get("title")) or ""),
                "deliverable": deliverable,
                "gates": [_normalize_gate(row) for row in gates if isinstance(row, Mapping)],
                "prior_failures": deepcopy(priors),
                "run_command": _extract_run_command(unwrap_principle(task.get("prompt"))),
            }
        )
    return rows


def _task_id_from_deliverable(number: int, deliverable: str) -> str:
    """Derive the task ID encoded by a standard result path."""

    stem = Path(deliverable).stem
    suffix = stem.removeprefix(f"experiment_{number}_").replace("_", "-")
    return f"exp{number}-{suffix}"


def _parse_condition(text: str) -> tuple[str, Any]:
    """Parse the two comparison operators used by the roadmap gate table."""

    match = re.fullmatch(r"\s*(==|>=)\s*(.+?)\s*", text)
    if not match:
        return text.strip(), None
    return match.group(1), yaml.safe_load(match.group(2))


def parse_design_tasks(
    text: str, low: int, high: int, *, require_deliverables: bool = True
) -> list[JsonDict]:
    """Parse task sections independently from the executable YAML."""

    sections = list(
        re.finditer(
            r"^### Exp(?P<number>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^### Exp\d+:|^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    )
    rows: list[JsonDict] = []
    for order, match in enumerate(
        [item for item in sections if low <= int(item.group("number")) <= high], 1
    ):
        number = int(match.group("number"))
        body = match.group("body")
        deliverable_match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", body)
        if not deliverable_match and require_deliverables:
            raise ValueError(f"Exp{number} design section is missing deliverable")
        deliverable = deliverable_match.group(1) if deliverable_match else None
        prior_match = re.search(r"```ya?ml\s*\n(prior_failures:.*?\n)```", body, re.DOTALL)
        priors: list[Any] = []
        if prior_match:
            prior_doc = yaml.safe_load(prior_match.group(1))
            if isinstance(prior_doc, Mapping) and isinstance(prior_doc.get("prior_failures"), list):
                priors = deepcopy(prior_doc["prior_failures"])
        rows.append(
            {
                "order": order,
                "number": number,
                "task_id": (
                    _task_id_from_deliverable(number, deliverable)
                    if deliverable
                    else f"exp{number}-" + re.sub(r"[^a-z0-9]+", "-", match.group("title").lower()).strip("-")
                ),
                "title": match.group("title").strip(),
                "deliverable": deliverable,
                "gates": [],
                "prior_failures": priors,
                "run_command": None,
            }
        )
    by_number = {row["number"]: row for row in rows}
    gate_pattern = re.compile(
        r"^\|\s*Exp(?P<down>\d+)\s*\|\s*`exp(?P<up>\d+)\.(?P<field>[^`]+)`\s*"
        r"\|\s*`(?P<condition>[^`]+)`\s*\|$",
        re.MULTILINE,
    )
    for match in gate_pattern.finditer(text):
        down = int(match.group("down"))
        up = int(match.group("up"))
        if down not in by_number or up not in by_number:
            continue
        op, value = _parse_condition(match.group("condition"))
        by_number[down]["gates"].append(
            {
                "upstream": by_number[up]["task_id"],
                "artifact_field": match.group("field"),
                "op": op,
                "value": value,
            }
        )
    return rows


def compare_task_contracts(
    design_tasks: Sequence[Mapping[str, Any]], roadmap_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare task presence and exact per-task contract fields."""

    design_by_number = {int(row["number"]): row for row in design_tasks}
    roadmap_by_number = {int(row["number"]): row for row in roadmap_tasks}
    ordered_numbers = [int(row["number"]) for row in design_tasks]
    ordered_numbers.extend(
        int(row["number"])
        for row in roadmap_tasks
        if int(row["number"]) not in design_by_number
    )
    rows: list[JsonDict] = []
    for number in ordered_numbers:
        design = design_by_number.get(number)
        roadmap = roadmap_by_number.get(number)
        presence = "both" if design and roadmap else "document_only" if design else "yaml_only"
        comparisons = {
            "order": bool(design and roadmap and design.get("order") == roadmap.get("order")),
            "task_id": bool(design and roadmap and design.get("task_id") == roadmap.get("task_id")),
            "deliverable": bool(
                design and roadmap and design.get("deliverable") == roadmap.get("deliverable")
            ),
            "gates": bool(design and roadmap and design.get("gates") == roadmap.get("gates")),
            "prior_failures": bool(
                design
                and roadmap
                and design.get("prior_failures") == roadmap.get("prior_failures")
            ),
        }
        rows.append(
            {
                "number": number,
                "presence": presence,
                "document": deepcopy(dict(design)) if design else None,
                "yaml": deepcopy(dict(roadmap)) if roadmap else None,
                "comparisons": comparisons,
                "passed": presence == "both" and all(comparisons.values()),
            }
        )
    return rows


def build_gate_contract_rows(
    design_tasks: Sequence[Mapping[str, Any]], roadmap_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare every gate by downstream task and gate position."""

    design_by_number = {int(row["number"]): row for row in design_tasks}
    roadmap_by_number = {int(row["number"]): row for row in roadmap_tasks}
    rows: list[JsonDict] = []
    for number in sorted(set(design_by_number) | set(roadmap_by_number)):
        expected_gates = list(design_by_number.get(number, {}).get("gates", []))
        observed_gates = list(roadmap_by_number.get(number, {}).get("gates", []))
        for index in range(max(len(expected_gates), len(observed_gates))):
            expected = expected_gates[index] if index < len(expected_gates) else None
            observed = observed_gates[index] if index < len(observed_gates) else None
            gate = observed or expected or {}
            rows.append(
                {
                    "downstream_number": number,
                    "downstream_task_id": (
                        roadmap_by_number.get(number, design_by_number.get(number, {})).get(
                            "task_id"
                        )
                    ),
                    "gate_index": index,
                    "artifact_field": gate.get("artifact_field"),
                    "expected": deepcopy(expected),
                    "observed": deepcopy(observed),
                    "passed": expected == observed,
                }
            )
    return rows


def _conductor_records(text: str) -> list[JsonDict]:
    """Parse the four-column conductor table while ignoring prose."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 4 or not re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
            continue
        rows.append({"timestamp": parts[0], "title": parts[1], "status": parts[2], "detail": parts[3]})
    return rows


def _matching_conductor_record(records: Sequence[Mapping[str, Any]], title: str) -> JsonDict | None:
    """Return the last row whose truncated title matches the roadmap title."""

    matches = [
        dict(row)
        for row in records
        if title.startswith(str(row.get("title") or ""))
        or str(row.get("title") or "").startswith(title)
    ]
    return matches[-1] if matches else None


def build_v601_terminal_rows(
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
    conductor_text: str,
    run_date: str,
    artifact_hashes: Mapping[int, str | None] | None = None,
) -> list[JsonDict]:
    """Keep one V601 row with separate artifact and conductor states."""

    records = _conductor_records(conductor_text)
    hashes = artifact_hashes or {}
    rows: list[JsonDict] = []
    for task in tasks:
        number = int(task["number"])
        payload = payloads.get(number)
        conductor = _matching_conductor_record(records, str(task.get("title") or ""))
        fresh = artifact_freshness(payload, number, run_date) if payload else None
        rows.append(
            {
                "task_id": task.get("task_id"),
                "number": number,
                "title": task.get("title"),
                "deliverable": task.get("deliverable"),
                "artifact_present": payload is not None,
                "artifact_sha256": hashes.get(number),
                "artifact_fresh": bool(fresh and fresh["passed"]),
                "artifact_freshness_check": fresh,
                "honest_verdict": (
                    unwrap_principle(payload.get("honest_verdict")) if payload else None
                ),
                "verdict_class": _derive_verdict_class(payload) if payload else None,
                "gate_result": _gate_result(payload) if payload else None,
                "conductor_terminal_row_present": conductor is not None,
                "conductor_terminal_row": conductor,
                "source_complete": bool(payload and fresh and fresh["passed"] and conductor),
            }
        )
    return rows


def _stored_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Read stored flag rows without treating the Boolean stamp as a new flag."""

    value = unwrap_principle(payload.get("corrigendum_pending"))
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def build_stored_vs_fresh_adversarial_rows(
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
    reports: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Preserve stored and current verifier outputs as separate evidence."""

    rows: list[JsonDict] = []
    for task in tasks:
        number = int(task["number"])
        stored = _stored_flags(payloads.get(number, {}))
        fresh_value = reports.get(number, {}).get("flags", [])
        fresh = [dict(row) for row in fresh_value if isinstance(row, Mapping)]
        stored_pairs = sorted((str(row.get("kind")), str(row.get("severity"))) for row in stored)
        fresh_pairs = sorted((str(row.get("kind")), str(row.get("severity"))) for row in fresh)
        rows.append(
            {
                "task_id": task.get("task_id"),
                "number": number,
                "stored_flagged_adversarial": payloads.get(number, {}).get(
                    "flagged_adversarial", False
                )
                is True,
                "stored_flags": stored,
                "stored_flag_kinds": sorted({str(row.get("kind")) for row in stored}),
                "fresh_flags": fresh,
                "fresh_flag_kinds": sorted({str(row.get("kind")) for row in fresh}),
                "disagreement": stored_pairs != fresh_pairs,
            }
        )
    return rows


def inspect_no_llm_evidence(source_text: str, payload: Mapping[str, Any], command: str) -> JsonDict:
    """Prove no new LLM call from code and receipts, never from a substrate name."""

    risk_markers: set[str] = set()
    tokenizer_only = False
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        tree = None
        risk_markers.add("invalid_python")
    llama_calls: list[ast.Call] = []
    unsafe_call_names = {"create_completion", "generate", "eval", "OwnedLlamaCppProcess"}
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            for module in modules:
                if module.startswith(("llama_cpp", "transformers", "torch", "openai", "anthropic")):
                    risk_markers.add(module.split(".")[0])
                if "llama_cpp_process" in module or "llama_server" in module:
                    risk_markers.add(module)
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith("Llama"):
                    llama_calls.append(node)
                if any(marker in name for marker in unsafe_call_names):
                    risk_markers.add(name)
        if llama_calls:
            tokenizer_only = all(
                any(
                    keyword.arg == "vocab_only"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in call.keywords
                )
                for call in llama_calls
            ) and not any(marker != "llama_cpp" for marker in risk_markers)
            if tokenizer_only:
                risk_markers.discard("llama_cpp")
            else:
                risk_markers.add("Llama")
    count_fields = {
        key: unwrap_principle(payload.get(key))
        for key in ("generated_answer_count", "token_likelihood_call_count", "model_call_count")
        if key in payload
    }
    positive_count = any(isinstance(value, (int, float)) and value > 0 for value in count_fields.values())
    gpu_values = [
        unwrap_principle(payload.get(key))
        for key in ("gpu_receipts", "lease_receipts", "process_receipts", "accelerator_samples")
    ]
    gpu_count = sum(len(value) if isinstance(value, list) else int(bool(value)) for value in gpu_values)
    command_risky = bool(re.search(r"\b(llama-server|--live|CARNOT_FORCE_LIVE)\b", command))
    mechanically_safe = not risk_markers and not positive_count and gpu_count == 0 and not command_risky
    rows_value = unwrap_principle(payload.get("rows"))
    source_row_count = len(rows_value) if isinstance(rows_value, list) else 0
    return {
        "declared_inference_substrate": unwrap_principle(payload.get("inference_substrate")),
        "command_receipt": command,
        "model_call_count_fields": count_fields,
        "model_call_count": 0 if mechanically_safe else None,
        "gpu_receipt_count": gpu_count,
        "source_row_count": source_row_count,
        "tokenizer_only_vocab_load": tokenizer_only,
        "risk_markers": sorted(risk_markers),
        "command_risky": command_risky,
        "mechanically_no_new_llm_inference": mechanically_safe,
        "eligible": mechanically_safe,
    }


def build_requalification_rows(
    root: Path,
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
    adversarial_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Inspect every V601 artifact carrying a duration concern."""

    by_number = {int(row["number"]): row for row in adversarial_rows}
    rows: list[JsonDict] = []
    for task in tasks:
        number = int(task["number"])
        flags = by_number.get(number, {})
        flag_kinds = set(flags.get("stored_flag_kinds", [])) | set(
            flags.get("fresh_flag_kinds", [])
        )
        if "DURATION_TOO_SHORT" not in flag_kinds:
            continue
        artifact_path = Path(str(task["deliverable"]))
        suffix = artifact_path.stem.removeprefix(f"experiment_{number}_")
        module_path = Path(f"python/carnot/experiment_{number}_{suffix}.py")
        source_text = (root / module_path).read_text(encoding="utf-8") if (root / module_path).is_file() else ""
        proof = inspect_no_llm_evidence(
            source_text,
            payloads.get(number, {}),
            str(task.get("run_command") or ""),
        )
        rows.append(
            {
                "task_id": task.get("task_id"),
                "number": number,
                "artifact_path": artifact_path.as_posix(),
                "module_path": module_path.as_posix(),
                "module_sha256": sha256_path(root / module_path),
                "duration_s": unwrap_principle(payloads.get(number, {}).get("duration_s")),
                **proof,
            }
        )
    return rows


def build_prior_failure_contract_rows(
    tasks: Sequence[Mapping[str, Any]], artifacts: Mapping[int, Mapping[str, Any]]
) -> list[JsonDict]:
    """Check each prior-failure block against its primary honest verdict."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    for task in tasks:
        for index, value in enumerate(task.get("prior_failures", [])):
            prior = dict(value) if isinstance(value, Mapping) else {}
            missing = [field for field in required if field not in prior]
            prior_number = _experiment_number(prior.get("experiment_id"))
            primary = artifacts.get(prior_number) if prior_number is not None else None
            primary_verdict = (
                unwrap_principle(primary.get("honest_verdict")) if primary else None
            )
            verdict_exact = bool(primary and prior.get("verdict") == primary_verdict)
            addressed = isinstance(prior.get("addressed_by"), str) and bool(
                prior.get("addressed_by", "").strip()
            )
            retirement = prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "prior_index": index,
                    "experiment_id": prior.get("experiment_id"),
                    "prior_experiment_id": prior_number,
                    "declared_verdict": prior.get("verdict"),
                    "primary_verdict": primary_verdict,
                    "primary_artifact_present": primary is not None,
                    "missing_subfields": missing,
                    "verdict_exact": verdict_exact,
                    "addressed_by_nonempty": addressed,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "passed": not missing and verdict_exact and addressed and retirement,
                }
            )
    return rows


def retired_experiment_ids(manifest: Mapping[str, Any]) -> set[int]:
    """Return active retirement IDs while honoring append-only reversal rows."""

    retired: set[int] = set()
    unretired: set[int] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        values = unwrap_principle(manifest.get(section))
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            candidates = [value.get("experiment_id"), *value.get("experiment_ids", [])]
            reversals = value.get("un_retired_experiment_ids", [])
            for candidate in candidates:
                number = _experiment_number(candidate)
                if number is not None:
                    retired.add(number)
            for candidate in reversals:
                number = _experiment_number(candidate)
                if number is not None:
                    unretired.add(number)
    return retired - unretired


def build_retired_dependency_rows(
    tasks: Sequence[Mapping[str, Any]], retired_ids: set[int]
) -> list[JsonDict]:
    """Emit only gate edges that name an actively retired upstream."""

    rows: list[JsonDict] = []
    for task in tasks:
        for gate in task.get("gates", []):
            upstream = str(gate.get("upstream") or "")
            number = _experiment_number(upstream)
            if number in retired_ids:
                rows.append(
                    {
                        "downstream_task_id": task.get("task_id"),
                        "upstream_task_id": upstream,
                        "upstream_experiment_id": number,
                        "retired": True,
                        "passed": False,
                    }
                )
    return rows


def _run_git(root: Path, args: tuple[str, ...], allow_failure: bool = False) -> str | None:
    """Read one Git fact and fail closed when a required fact is unavailable."""

    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        if allow_failure:
            return None
        raise RuntimeError((completed.stderr or completed.stdout or "git command failed").strip())
    return completed.stdout.strip()


def build_activation_copy_receipt(
    root: Path,
    *,
    stage_commit: str = V602_STAGE_COMMIT,
    activation_commit: str = V602_ACTIVATION_COMMIT,
    deletion_commit: str = V602_DELETION_COMMIT,
    run_git: GitRunner = _run_git,
) -> JsonDict:
    """Prove the committed stage blob became active before staging deletion."""

    try:
        stage_full = run_git(root, ("rev-parse", stage_commit), False)
        activation_full = run_git(root, ("rev-parse", activation_commit), False)
        deletion_full = run_git(root, ("rev-parse", deletion_commit), False)
        stage_blob = run_git(
            root, ("rev-parse", f"{stage_commit}:research-roadmap-next.yaml"), False
        )
        parent_stage_blob = run_git(
            root, ("rev-parse", f"{activation_commit}^:research-roadmap-next.yaml"), False
        )
        activation_blob = run_git(
            root, ("rev-parse", f"{activation_commit}:research-roadmap.yaml"), False
        )
        deletion_parent_blob = run_git(
            root, ("rev-parse", f"{deletion_commit}^:research-roadmap-next.yaml"), False
        )
        head_blob = run_git(root, ("rev-parse", "HEAD:research-roadmap.yaml"), False)
        deleted_value = run_git(
            root, ("cat-file", "-e", f"{deletion_commit}:research-roadmap-next.yaml"), True
        )
        stage_before_activation = (
            run_git(
                root,
                ("merge-base", "--is-ancestor", stage_commit, activation_commit),
                True,
            )
            is not None
        )
        activation_before_deletion = (
            run_git(
                root,
                ("merge-base", "--is-ancestor", activation_commit, deletion_commit),
                True,
            )
            is not None
        )
    except RuntimeError as exc:
        return {"passed": False, "error": str(exc)}
    copy_equal = len({stage_blob, parent_stage_blob, activation_blob, deletion_parent_blob}) == 1
    deleted_after_copy = deleted_value is None and activation_before_deletion
    passed = copy_equal and deleted_after_copy and stage_before_activation and head_blob == activation_blob
    return {
        "stage_commit": stage_full,
        "activation_commit": activation_full,
        "deletion_commit": deletion_full,
        "stage_blob": stage_blob,
        "activation_parent_stage_blob": parent_stage_blob,
        "activated_roadmap_blob": activation_blob,
        "deletion_parent_stage_blob": deletion_parent_blob,
        "head_roadmap_blob": head_blob,
        "stage_before_activation": stage_before_activation,
        "activation_before_deletion": activation_before_deletion,
        "copy_equal": copy_equal,
        "deleted_after_copy": deleted_after_copy,
        "passed": passed,
    }


def milestone_check(document: Mapping[str, Any]) -> JsonDict:
    """Require the active V602 milestone instead of a stale roadmap."""

    observed = str(unwrap_principle(document.get("milestone")) or "")
    return _check("active_v602_milestone", V602_MILESTONE, observed, observed == V602_MILESTONE)


def _load_verifier(root: Path) -> Callable[[Path], Mapping[str, Any]]:
    """Load the current checked-in verifier rather than a cached package copy."""

    path = root / VERIFIER_PATH
    spec = importlib.util.spec_from_file_location("carnot_exp6874_adversarial_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("adversarial verifier import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact


def _fresh_reports(
    root: Path,
    tasks: Sequence[Mapping[str, Any]],
    verifier: Callable[[Path], Mapping[str, Any]] | None,
) -> dict[int, JsonDict]:
    """Run the current verifier once per exact existing V601 deliverable."""

    verify = verifier or _load_verifier(root)
    reports: dict[int, JsonDict] = {}
    verifier_hash = sha256_path(root / VERIFIER_PATH) or "missing"
    for task in tasks:
        number = int(task["number"])
        path = root / str(task["deliverable"])
        if not path.is_file():
            continue
        key = (verifier_hash, sha256_path(path) or "missing")
        if key not in _VERIFY_CACHE:
            _VERIFY_CACHE[key] = dict(verify(path))
        reports[number] = deepcopy(_VERIFY_CACHE[key])
    return reports


def _load_yaml_mapping(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one YAML mapping with an explicit malformed-source result."""

    if not path.is_file():
        return None, "missing"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, f"invalid_yaml:{type(exc).__name__}"
    if not isinstance(value, Mapping):
        return None, "yaml_mapping_required"
    return dict(value), None


def _load_prior_artifacts(root: Path, tasks: Sequence[Mapping[str, Any]]) -> tuple[dict[int, JsonDict], list[JsonDict]]:
    """Resolve each exact prior experiment to its primary result artifact."""

    numbers = {
        number
        for task in tasks
        for prior in task.get("prior_failures", [])
        if isinstance(prior, Mapping)
        if (number := _experiment_number(prior.get("experiment_id"))) is not None
    }
    artifacts: dict[int, JsonDict] = {}
    source_rows: list[JsonDict] = []
    for number in sorted(numbers):
        candidates = [
            path
            for path in sorted((root / "results").glob(f"experiment_{number}_*.json"))
            if "." not in path.stem
        ]
        path = candidates[0] if len(candidates) == 1 else None
        payload, error = read_json_object(path) if path else (None, "primary_artifact_ambiguous")
        relative = path.relative_to(root).as_posix() if path else None
        source_rows.append(
            {
                "experiment_id": number,
                "path": relative,
                "sha256": sha256_path(path) if path else None,
                "error": error,
            }
        )
        if payload is not None:
            artifacts[number] = payload
    return artifacts, source_rows


def _unexecuted_design_rows(
    root: Path,
    design_tasks: Sequence[Mapping[str, Any]],
    yaml_tasks: Sequence[Mapping[str, Any]],
    conductor_text: str,
    historical_result_paths: set[str],
) -> list[JsonDict]:
    """Prove exact V601 design-only paths never executed."""

    yaml_numbers = {int(row["number"]) for row in yaml_tasks}
    records = _conductor_records(conductor_text)
    rows: list[JsonDict] = []
    for task in design_tasks:
        if int(task["number"]) in yaml_numbers:
            continue
        conductor = _matching_conductor_record(records, str(task.get("title") or ""))
        matching_paths = sorted(
            path
            for path in historical_result_paths
            if re.search(rf"(?:^|/)experiment_{int(task['number'])}_[^/]+\.json$", path)
        )
        never = not matching_paths and conductor is None
        rows.append(
            {
                "number": task.get("number"),
                "task_id": task.get("task_id"),
                "deliverable": task.get("deliverable"),
                "yaml_task_present": False,
                "conductor_run_present": conductor is not None,
                "result_artifact_present": bool(matching_paths),
                "historical_result_paths": matching_paths,
                "execution_state": "never_executed" if never else "evidence_conflict",
                "passed": never,
            }
        )
    return rows


def _source_hash_rows(root: Path, extra: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Hash required files and keep missing paths visible."""

    rows: dict[str, JsonDict] = {}
    for name, relative in {**BASE_SOURCE_PATHS, **OWN_SOURCE_PATHS, **extra}.items():
        rows[name] = {"path": relative.as_posix(), "sha256": sha256_path(root / relative)}
    return rows


def _branch_closures(payloads: Mapping[int, Mapping[str, Any]]) -> tuple[bool, bool]:
    """Recompute the two V601 method closures from their controlling fields."""

    exp6869 = payloads.get(6869, {})
    exp6870 = payloads.get(6870, {})
    exp6873 = payloads.get(6873, {})
    semantic = (
        _derive_verdict_class(exp6869) == "null"
        and unwrap_principle(exp6869.get("semantic_contrast_rule_ready_score")) == 0
        and _derive_verdict_class(exp6870) == "blocked"
        and _experiment_number(exp6870.get("failed_upstream")) == 6869
        and unwrap_principle(exp6870.get("failed_observed")) == 0
        and unwrap_principle(exp6870.get("failed_expected")) == 1
    )
    reliability = (
        _derive_verdict_class(exp6873) == "null"
        and unwrap_principle(exp6873.get("scientific_claim_eligible")) is False
        and _gate_result(exp6873).get("failed_check") == "quarantine_beats_read_only_every_order"
    )
    return semantic, reliability


def _ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the single downstream readiness bit from row evidence."""

    terminal = artifact.get("v601_terminal_task_rows", [])
    parity = artifact.get("v602_document_yaml_parity_rows", [])
    gates = artifact.get("v602_gate_contract_rows", [])
    priors = artifact.get("prior_failure_contract_rows", [])
    unexecuted = artifact.get("v601_unexecuted_design_task_rows", [])
    preconditions = artifact.get("preconditions_checked", {})
    receipt = artifact.get("v602_activation_copy_receipt", {})
    return bool(
        isinstance(terminal, list)
        and len(terminal) == 9
        and all(row.get("source_complete") is True for row in terminal)
        and artifact.get("fixed_sequence_semantic_branch_closed") is True
        and artifact.get("reliability_update_branch_closed") is True
        and isinstance(unexecuted, list)
        and len(unexecuted) == 4
        and all(row.get("passed") is True for row in unexecuted)
        and isinstance(preconditions, Mapping)
        and preconditions.get("passed") is True
        and isinstance(receipt, Mapping)
        and receipt.get("passed") is True
        and isinstance(parity, list)
        and len(parity) == 11
        and all(row.get("passed") is True for row in parity)
        and isinstance(gates, list)
        and all(row.get("passed") is True for row in gates)
        and isinstance(priors, list)
        and bool(priors)
        and all(row.get("passed") is True for row in priors)
        and artifact.get("retired_dependency_rows") == []
    )


def _checksum_value(value: Any) -> Any:
    """Remove elapsed time and the checksum itself from reproducibility content."""

    if isinstance(value, Mapping):
        return {
            key: _checksum_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_checksum_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic content in the terminal artifact."""

    return sha256_bytes(canonical_json(_checksum_value(dict(artifact))))


def build_artifact(
    root: Path,
    run_date: str,
    *,
    verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    run_git: GitRunner = _run_git,
) -> JsonDict:
    """Build a complete positive or blocked V602 evidence contract."""

    started = time.perf_counter()
    source_errors: list[JsonDict] = []
    roadmap, roadmap_error = _load_yaml_mapping(root / ROADMAP_PATH)
    exclusion, exclusion_error = _load_yaml_mapping(root / EXCLUSION_PATH)
    design_path = root / DESIGN_PATH
    conductor_path = root / CONDUCTOR_LOG_PATH
    design_text = design_path.read_text(encoding="utf-8") if design_path.is_file() else ""
    conductor_text = conductor_path.read_text(encoding="utf-8") if conductor_path.is_file() else ""
    for name, error in (
        ("active_v602_roadmap", roadmap_error),
        ("exclusion_manifest", exclusion_error),
        ("v602_design", None if design_text else "missing_or_empty"),
        ("conductor_log", None if conductor_text else "missing_or_empty"),
        ("adversarial_verifier", None if (root / VERIFIER_PATH).is_file() else "missing"),
        ("substrate_alias_lint", None if (root / ALIAS_LINT_PATH).is_file() else "missing"),
    ):
        if error:
            source_errors.append({"source": name, "error": error})

    try:
        v601_yaml_text = run_git(
            root, ("show", f"{V601_ACTIVATION_COMMIT}:research-roadmap.yaml"), False
        )
        v601_design_text = run_git(
            root,
            (
                "show",
                f"{V601_ACTIVATION_COMMIT}:openspec/change-proposals/research-roadmap-vNEXT.md",
            ),
            False,
        )
        v601_document = yaml.safe_load(v601_yaml_text or "")
        if not isinstance(v601_document, Mapping):
            raise ValueError("V601 committed roadmap is not a mapping")
        v601_tasks = parse_roadmap_tasks(v601_document)
        v601_design_tasks = parse_design_tasks(
            v601_design_text or "", 6865, 6877, require_deliverables=False
        )
        historical_results_text = run_git(
            root,
            ("ls-tree", "-r", "--name-only", V602_ACTIVATION_COMMIT, "results"),
            False,
        )
        historical_result_paths = set((historical_results_text or "").splitlines())
    except (RuntimeError, ValueError, yaml.YAMLError) as exc:
        source_errors.append({"source": "v601_git_sources", "error": str(exc)})
        v601_yaml_text = ""
        v601_design_text = ""
        v601_tasks = []
        v601_design_tasks = []
        historical_result_paths = set()

    v602_tasks = parse_roadmap_tasks(roadmap) if roadmap is not None else []
    v602_design_tasks = parse_design_tasks(design_text, 6874, 6884) if design_text else []
    payloads: dict[int, JsonDict] = {}
    artifact_hashes: dict[int, str | None] = {}
    artifact_sources: dict[str, Path] = {}
    for task in v601_tasks:
        number = int(task["number"])
        relative = Path(str(task["deliverable"]))
        payload, error = read_json_object(root / relative)
        artifact_hashes[number] = sha256_path(root / relative)
        artifact_sources[f"v601_exp{number}"] = relative
        if error:
            source_errors.append({"source": relative.as_posix(), "error": error})
        elif payload is not None:
            payloads[number] = payload

    try:
        fresh_reports = _fresh_reports(root, v601_tasks, verifier)
    except (RuntimeError, OSError, ImportError) as exc:
        source_errors.append({"source": "fresh_adversarial_verifier", "error": str(exc)})
        fresh_reports = {}
    terminal_rows = build_v601_terminal_rows(
        v601_tasks, payloads, conductor_text, run_date, artifact_hashes
    )
    adversarial_rows = build_stored_vs_fresh_adversarial_rows(
        v601_tasks, payloads, fresh_reports
    )
    requalification_rows = build_requalification_rows(
        root, v601_tasks, payloads, adversarial_rows
    )
    semantic_closed, reliability_closed = _branch_closures(payloads)
    v601_parity = compare_task_contracts(v601_design_tasks, v601_tasks)
    v601_mismatches = [row for row in v601_parity if row["presence"] != "both"]
    unexecuted = _unexecuted_design_rows(
        root, v601_design_tasks, v601_tasks, conductor_text, historical_result_paths
    )

    v602_parity = compare_task_contracts(v602_design_tasks, v602_tasks)
    gate_rows = build_gate_contract_rows(v602_design_tasks, v602_tasks)
    prior_artifacts, prior_sources = _load_prior_artifacts(root, v602_tasks)
    prior_rows = build_prior_failure_contract_rows(v602_tasks, prior_artifacts)
    retired_rows = build_retired_dependency_rows(
        v602_tasks, retired_experiment_ids(exclusion or {})
    )
    activation_receipt = build_activation_copy_receipt(root, run_git=run_git)
    active_check = milestone_check(roadmap or {})
    expected_v601_ids = list(range(6865, 6874))
    observed_v601_ids = [int(row["number"]) for row in v601_tasks]
    source_complete = (
        not source_errors
        and observed_v601_ids == expected_v601_ids
        and len(terminal_rows) == 9
        and all(row["source_complete"] for row in terminal_rows)
    )
    precondition_checks = [
        _check("required_source_readability", [], source_errors, not source_errors),
        active_check,
        _check("v601_task_range", expected_v601_ids, observed_v601_ids, observed_v601_ids == expected_v601_ids),
        _check("v601_terminal_source_completeness", 9, sum(row["source_complete"] for row in terminal_rows), source_complete),
        _check("v602_activation_copy_receipt", True, activation_receipt.get("passed"), activation_receipt.get("passed") is True),
    ]
    preconditions = gate_summary(precondition_checks)
    source_hashes = _source_hash_rows(root, artifact_sources)
    source_hashes["v601_activated_roadmap_git"] = {
        "git_object": f"{V601_ACTIVATION_COMMIT}:research-roadmap.yaml",
        "sha256": sha256_bytes((v601_yaml_text or "").encode("utf-8")) if v601_yaml_text else None,
    }
    source_hashes["v601_design_git"] = {
        "git_object": f"{V601_ACTIVATION_COMMIT}:openspec/change-proposals/research-roadmap-vNEXT.md",
        "sha256": sha256_bytes((v601_design_text or "").encode("utf-8")) if v601_design_text else None,
    }
    for row in prior_sources:
        source_hashes[f"prior_exp{row['experiment_id']}"] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "error": row["error"],
        }

    combined_rows = [
        {"milestone": V601_MILESTONE, "task_id": row["task_id"], "number": row["number"], "task_state": row}
        for row in terminal_rows
    ]
    combined_rows.extend(
        {
            "milestone": V602_MILESTONE,
            "task_id": (row.get("yaml") or row.get("document") or {}).get("task_id"),
            "number": row["number"],
            "task_state": row,
        }
        for row in v602_parity
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_6874.v602_evidence_substrate_manifest_contract.v1",
        "experiment_id": 6874,
        "run_date": _normalized_date(run_date),
        "status": "complete_pending_gate_reduction",
        "field_principles": {},
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": source_hashes,
        "rows": combined_rows,
        "v601_terminal_task_rows": terminal_rows,
        "v601_stored_vs_fresh_adversarial_rows": adversarial_rows,
        "deterministic_substrate_requalification_rows": requalification_rows,
        "fixed_sequence_semantic_branch_closed": semantic_closed,
        "reliability_update_branch_closed": reliability_closed,
        "v601_design_yaml_mismatch_rows": v601_mismatches,
        "v601_unexecuted_design_task_rows": unexecuted,
        "v602_document_yaml_parity_rows": v602_parity,
        "v602_activation_copy_receipt": activation_receipt,
        "v602_gate_contract_rows": gate_rows,
        "prior_failure_contract_rows": prior_rows,
        "retired_dependency_rows": retired_rows,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v602_evidence_contract_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v602_evidence_substrate_manifest_contract",
    }
    ready = _ready_from_artifact(artifact)
    checks = [
        _check("v601_evidence_source_complete", True, source_complete, source_complete),
        _check("fixed_sequence_semantic_branch_closed", True, semantic_closed, semantic_closed),
        _check("reliability_update_branch_closed", True, reliability_closed, reliability_closed),
        _check(
            "v601_design_only_tasks_never_executed",
            [6874, 6875, 6876, 6877],
            [row["number"] for row in unexecuted if row["passed"]],
            len(unexecuted) == 4 and all(row["passed"] for row in unexecuted),
        ),
        _check(
            "v602_document_yaml_parity",
            {"task_count": 11, "experiment_range": [6874, 6884], "all_fields_equal": True},
            {
                "document_task_count": len(v602_design_tasks),
                "yaml_task_count": len(v602_tasks),
                "experiment_range": [
                    min((row["number"] for row in v602_parity), default=None),
                    max((row["number"] for row in v602_parity), default=None),
                ],
                "all_fields_equal": all(row["passed"] for row in v602_parity),
            },
            len(v602_parity) == 11 and all(row["passed"] for row in v602_parity),
        ),
        _check("v602_gate_contract", True, all(row["passed"] for row in gate_rows), all(row["passed"] for row in gate_rows)),
        _check("prior_failure_contract", True, bool(prior_rows) and all(row["passed"] for row in prior_rows), bool(prior_rows) and all(row["passed"] for row in prior_rows)),
        _check("retired_dependencies", [], retired_rows, not retired_rows),
        _check("v602_evidence_contract_ready_score", 1, int(ready), ready),
    ]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["v602_evidence_contract_ready_score"] = int(ready)
    if ready:
        artifact["status"] = "complete"
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "complete_positive_v602_evidence_substrate_manifest_contract_ready"
    else:
        artifact["status"] = "complete_blocked"
    principles = dict(FIELD_PRINCIPLES)
    for row in gate_rows:
        field = row.get("artifact_field")
        if field:
            principles[str(field)] = "This exact upstream field spelling controls one V602 gate."
    artifact["field_principles"] = principles
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, readiness reduction, principles, and checksum."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    gate_fields = {
        str(row.get("artifact_field"))
        for row in artifact.get("v602_gate_contract_rows", [])
        if isinstance(row, Mapping) and row.get("artifact_field")
    }
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not (REQUIRED_ARTIFACT_FIELDS | gate_fields).issubset(
        principles
    ):
        errors.append("field_principles_missing")
    ready = int(_ready_from_artifact(artifact))
    if artifact.get("v602_evidence_contract_ready_score") != ready:
        errors.append("readiness_recomputation_mismatch")
    if ready:
        if artifact.get("verdict_class") != "positive":
            errors.append("ready_verdict_class_mismatch")
    else:
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if artifact.get("honest_verdict") != "complete_blocked_v602_evidence_substrate_manifest_contract":
            errors.append("blocked_honest_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace only the requested output after a complete JSON serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _date_argument(value: str) -> str:
    """Reject ambiguous dates before reading or writing evidence."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Write the contract to the requested path and validate the stored shape."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    write_json_atomic(args.output, artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ready": artifact["v602_evidence_contract_ready_score"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path
    raise SystemExit(main())
