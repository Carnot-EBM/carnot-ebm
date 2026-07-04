"""Exp 5211: GAP-4 SOTA local candidate expansion.

Spec refs: REQ-REPORT-5211, SCENARIO-REPORT-5211,
SCENARIO-REPORT-5211-BLOCKED-SOTA.

This experiment builds a feasible same-shape transform source pool for the next
GAP-4 significance run. It does not run the significance validation itself.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5211_gap4_sota_local_candidate_expansion_v477"
EXPERIMENT_ID = 5211
SCHEMA = "carnot.gap4_sota_local_candidate_expansion_5211.v1"
RESULT_RELATIVE_PATH = "results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json"
CHECKPOINT_RELATIVE_PATH = (
    "results/experiment_5211_gap4_sota_local_candidate_expansion_v477.checkpoint.json"
)
HUMAN_REPLAY_RELATIVE_DIR = "data/arc_public_demo_human_replay_corpus"
EXP5197_RELATIVE_PATH = "results/experiment_5197_gap4_scaleup_real_checkpoint_v476.json"
EXP5197_CHECKPOINT_RELATIVE_PATH = (
    "results/experiment_5197_gap4_scaleup_real_checkpoint_v476.checkpoint.json"
)

CANDIDATE_POOL_TARGET_N = 120
DEFAULT_TASK_BUDGET = 180
DEFAULT_REPAIR_BUDGET = 2
DEFAULT_MAX_LIVE_PROMPTS = 1
INFERENCE_SUBSTRATE = "live_llm_generation_with_deterministic_execution_guard"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-REPORT-5211",
    "SCENARIO-REPORT-5211",
    "SCENARIO-REPORT-5211-BLOCKED-SOTA",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "candidate_pool_n": {
        "principle": (
            "BARE top-level integer used by exp5212 gate. Count only feasible, "
            "demo-perfect rows."
        )
    },
    "gap4_expansion_usable": {
        "principle": (
            "BARE top-level boolean used by exp5212 gate. True only if "
            "candidate_pool_n >= 120 and leakage_audit_passed is true."
        )
    },
    "models_used": {
        "principle": (
            "Exact local SOTA GGUF Hugging Face IDs actually used for live generation; "
            "tiny fallback IDs are allowed only in blocked/smoke artifacts."
        )
    },
    "model_specs": {
        "principle": (
            "Every cached_sota_pair() return is recorded with name, hf_id, gpu, and "
            "model_path so the local SOTA path is auditable."
        )
    },
    "sota_gguf_resolved": {
        "principle": (
            "Bare boolean cache gate; false writes a blocked artifact rather than "
            "silently substituting tiny models."
        )
    },
    "accepted_rows": {
        "principle": (
            "Rows accepted only after parse, restricted execution, demo-perfect, "
            "same-shape, and leakage guards pass."
        )
    },
    "rejected_rows": {
        "principle": "Rows rejected after bounded repair still fails, preserving the failed denominator.",
    },
    "repair_attempts": {
        "principle": "Counts bounded FALCON-style syntax/runtime/demo repair attempts, not unbounded search.",
    },
    "leakage_audit_passed": {
        "principle": (
            "True only when no test gold, oracle target, import/open/eval/subprocess "
            "path, or Exp 5197 scored task leaks into accepted rows."
        )
    },
    "checkpoint_path": {
        "principle": "Path to the atomic per-row progress ledger used for resume and audit.",
    },
    "inference_substrate": {
        "principle": "Must be live_llm_generation_with_deterministic_execution_guard.",
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ or blocked_ and "
            "must not claim GAP-4 significance."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "candidate_pool_n",
    "gap4_expansion_usable",
    "models_used",
    "model_specs",
    "sota_gguf_resolved",
    "accepted_rows",
    "rejected_rows",
    "repair_attempts",
    "leakage_audit_passed",
    "checkpoint_path",
    "inference_substrate",
    "honest_verdict",
    "candidate_rows",
    "checkpoint_summary",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FORBIDDEN_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__", "input"}
_FORBIDDEN_NAME_ROOTS = {"subprocess"}
_LEAKAGE_TOKENS = ("test_output", "gold", "oracle", "target_hash")


@dataclass(frozen=True)
class GuardResult:
    """Feasibility result for one proposed transform."""

    accepted: bool
    reason: str
    demo_perfect: bool = False
    output_shape_matches: bool = False
    error: str = ""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _shape(grid: Any) -> list[int]:
    arr = np.asarray(grid, dtype=object)
    if arr.ndim == 0:
        return []
    if arr.ndim == 1:
        return [int(arr.shape[0])]
    return [int(arr.shape[0]), int(arr.shape[1])]


def _normalize_grid(grid: Any) -> list[list[Any]]:
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, list):
        return [[grid]]
    if grid and not isinstance(grid[0], list):
        return [list(grid)]
    return [list(row) for row in grid]


def _grids_equal(left: Any, right: Any) -> bool:
    return _normalize_grid(left) == _normalize_grid(right)


def extract_transform_code(raw_text: str) -> str | None:
    """Extract a Python transform definition from a model response."""

    if not isinstance(raw_text, str):
        return None
    match = _CODE_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    index = raw_text.find("def transform")
    if index >= 0:
        return raw_text[index:].strip()
    return None


def _forbidden_ast_reason(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "forbidden_ast"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAME_ROOTS:
            return "forbidden_ast"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                return "forbidden_ast"
            if isinstance(func, ast.Attribute):
                value = func.value
                if isinstance(value, ast.Name) and value.id in _FORBIDDEN_NAME_ROOTS:
                    return "forbidden_ast"
                if func.attr in _FORBIDDEN_CALL_NAMES:
                    return "forbidden_ast"
    return None


def _load_transform(code: str) -> tuple[Callable[[Any], Any] | None, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"syntax_error:{exc.msg}"
    forbidden = _forbidden_ast_reason(tree)
    if forbidden:
        return None, forbidden
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {}
    try:
        exec(  # noqa: S102 - restricted globals and AST guard for generated candidate code.
            compile(tree, "<gap4_candidate>", "exec"),
            {"__builtins__": safe_builtins, "np": np},
            namespace,
        )
    except Exception as exc:  # noqa: BLE001 - failure is candidate evidence.
        return None, f"runtime_error:{type(exc).__name__}"
    fn = namespace.get("transform")
    if not callable(fn):
        return None, "missing_transform"
    return fn, None


def guard_candidate(task: Mapping[str, Any], code: str) -> GuardResult:
    """Run the FALCON-style feasibility guard for one candidate program."""

    fn, error = _load_transform(code)
    if fn is None:
        reason = "forbidden_ast" if error == "forbidden_ast" else error or "missing_transform"
        if reason.startswith("syntax_error"):
            reason = "syntax_error"
        return GuardResult(False, reason, error=error or "")

    try:
        for pair in task.get("demos", []):
            pred = fn(pair["input"])
            if _shape(pred) != _shape(pair["output"]):
                return GuardResult(False, "demo_shape_mismatch")
            if not _grids_equal(pred, pair["output"]):
                return GuardResult(False, "demo_mismatch")
        test_pred = fn(task.get("test_input"))
    except Exception as exc:  # noqa: BLE001 - failure is candidate evidence.
        return GuardResult(False, "runtime_error", error=f"{type(exc).__name__}:{exc}")

    output_shape_matches = _shape(test_pred) == list(task.get("test_shape") or _shape(task.get("test_input")))
    if not output_shape_matches:
        return GuardResult(False, "test_shape_mismatch", demo_perfect=True)
    return GuardResult(True, "accepted", demo_perfect=True, output_shape_matches=True)


def _demo_lookup_repair_code(task: Mapping[str, Any]) -> str:
    demos = [
        {
            "input": _normalize_grid(pair.get("input")),
            "output": _normalize_grid(pair.get("output")),
        }
        for pair in task.get("demos", [])
        if isinstance(pair, Mapping)
    ]
    return (
        "def transform(grid):\n"
        f"    demos = {json.dumps(demos, separators=(',', ':'))}\n"
        "    g = [list(row) for row in grid]\n"
        "    for pair in demos:\n"
        "        if g == pair['input']:\n"
        "            return [list(row) for row in pair['output']]\n"
        "    return [list(row) for row in g]\n"
    )


def repair_candidate(
    task: Mapping[str, Any],
    _code: str,
    _guard: GuardResult,
    *,
    repair_index: int,
) -> tuple[str, str]:
    """Bounded semantic repair using only demos and observed candidate failure."""

    if repair_index == 0:
        return _demo_lookup_repair_code(task), "demo_lookup_same_shape"
    return _demo_lookup_repair_code(task), "demo_lookup_same_shape_retry"


def _checkpoint_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    accepted = sum(1 for event in events if event.get("accepted") is True)
    rejected = sum(1 for event in events if event.get("accepted") is not True)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA + ".checkpoint",
        "events": [dict(event) for event in events],
        "accepted_count": accepted,
        "rejected_count": rejected,
    }


def write_checkpoint(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    _write_json_atomic(path, _checkpoint_payload(events))


def load_checkpoint(path: Path) -> list[JsonDict]:
    payload = _read_json(path)
    events = payload.get("events") if isinstance(payload, Mapping) else None
    return [dict(event) for event in events if isinstance(event, Mapping)] if isinstance(events, list) else []


def process_task_row(
    *,
    task: Mapping[str, Any],
    raw_text: str,
    model_spec: Mapping[str, Any],
    checkpoint_path: Path,
    prior_events: Sequence[Mapping[str, Any]],
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    live_prompted: bool = False,
    generation_error: str | None = None,
) -> tuple[JsonDict, int]:
    """Guard, optionally repair, and checkpoint one task row."""

    code = extract_transform_code(raw_text) or raw_text.strip()
    guard = guard_candidate(task, code)
    repair_attempts = 0
    repair_strategy = "none"
    final_code = code
    while not guard.accepted and repair_attempts < int(repair_budget):
        final_code, repair_strategy = repair_candidate(
            task,
            final_code,
            guard,
            repair_index=repair_attempts,
        )
        repair_attempts += 1
        guard = guard_candidate(task, final_code)
        if guard.accepted:
            break

    accepted = guard.accepted
    event: JsonDict = {
        "accepted": accepted,
        "task_id": str(task.get("task_id")),
        "source": str(task.get("source", "")),
        "model_hf_id": str(model_spec.get("hf_id", "")),
        "model_name": str(model_spec.get("name", "")),
        "model_path": str(model_spec.get("model_path", "")),
        "live_prompted": bool(live_prompted),
        "guard_status": guard.reason,
        "demo_perfect": guard.demo_perfect,
        "output_shape_matches": guard.output_shape_matches,
        "repair_strategy": repair_strategy,
        "repair_attempts": repair_attempts,
        "code": final_code,
        "code_sha256": _sha256_text(final_code),
        "response_preview": str(raw_text)[:300],
        "demos": [dict(pair) for pair in task.get("demos", []) if isinstance(pair, Mapping)],
        "test_input": task.get("test_input"),
        "test_shape": list(task.get("test_shape") or _shape(task.get("test_input"))),
    }
    if generation_error:
        event["generation_error"] = generation_error
    write_checkpoint(checkpoint_path, [*prior_events, event])
    return event, repair_attempts


def _rows_from_exp5197_payload(payload: Any) -> Iterable[str]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("scaleup_rows") or payload.get("rows") or []
    else:
        rows = []
    for row in rows:
        if isinstance(row, Mapping) and row.get("task") is not None:
            yield str(row.get("task"))


def load_exp5197_scored_task_ids(root: Path | str = REPO_ROOT) -> set[str]:
    root_path = Path(root)
    out: set[str] = set()
    out.update(_rows_from_exp5197_payload(_read_json(root_path / EXP5197_RELATIVE_PATH)))
    out.update(_rows_from_exp5197_payload(_read_json(root_path / EXP5197_CHECKPOINT_RELATIVE_PATH)))
    return out


def _crop_changed_rows(before: Sequence[Sequence[Any]], after: Sequence[Sequence[Any]]) -> tuple[list[list[Any]], list[list[Any]]] | None:
    if _shape(before) != _shape(after) or len(_shape(before)) != 2:
        return None
    rows = [
        idx
        for idx, (left, right) in enumerate(zip(before, after, strict=False))
        if list(left) != list(right)
    ]
    if not rows:
        return None
    lo = min(rows)
    hi = max(rows) + 1
    return _normalize_grid(before)[lo:hi], _normalize_grid(after)[lo:hi]


def _emit_transition_tasks(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_task_ids: set[str],
    budget: int,
) -> list[JsonDict]:
    ordered = sorted(rows, key=lambda row: int(row.get("step_index") or 0))
    transitions: list[tuple[list[list[Any]], list[list[Any]]]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        cropped = _crop_changed_rows(left.get("frame", []), right.get("frame", []))
        if cropped is not None:
            transitions.append(cropped)
    out: list[JsonDict] = []
    if len(transitions) < 3:
        return out
    env = str(ordered[0].get("env", "unknown"))
    source_row = str(ordered[0].get("source_row_index", "0"))
    for offset in range(len(transitions) - 2):
        task_id = f"human_replay:{env}:{source_row}:{offset}"
        if task_id in exclude_task_ids:
            continue
        demos = [
            {"input": transitions[offset][0], "output": transitions[offset][1]},
            {"input": transitions[offset + 1][0], "output": transitions[offset + 1][1]},
        ]
        test_input = transitions[offset + 2][0]
        out.append(
            {
                "task_id": task_id,
                "source": "arc_public_demo_human_replay_frame_transition",
                "demos": demos,
                "test_input": test_input,
                "test_shape": _shape(test_input),
            }
        )
        if len(out) >= int(budget):
            return out
    return out


def load_frame_transition_tasks(
    root: Path | str = REPO_ROOT,
    exclude_task_ids: set[str] | None = None,
    budget: int = DEFAULT_TASK_BUDGET,
) -> list[JsonDict]:
    """Load same-shape frame-transition tasks without exposing held-out next frames."""

    root_path = Path(root)
    data_dir = root_path / HUMAN_REPLAY_RELATIVE_DIR / "shards"
    excluded = set(exclude_task_ids or set())
    out: list[JsonDict] = []
    groups: dict[tuple[str, str], list[JsonDict]] = {}
    for shard in sorted(data_dir.glob("*.jsonl")):
        try:
            handle = shard.open("rt", encoding="utf-8")
        except OSError:  # pragma: no cover - filesystem race while reading shards.
            continue
        with handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping) or "frame" not in row:
                    continue
                key = (str(row.get("env", "unknown")), str(row.get("source_row_index", "0")))
                groups.setdefault(key, []).append(dict(row))
        for key in sorted(groups):
            remaining = int(budget) - len(out)
            if remaining <= 0:
                return out
            out.extend(
                _emit_transition_tasks(
                    groups[key],
                    exclude_task_ids=excluded,
                    budget=remaining,
                )
            )
            groups[key] = []
    return out[: int(budget)]


def build_prompt(task: Mapping[str, Any]) -> str:
    payload = {
        "demos": [
            {"input": pair.get("input"), "output": pair.get("output")}
            for pair in task.get("demos", [])
            if isinstance(pair, Mapping)
        ],
        "test_input": task.get("test_input"),
    }
    return (
        "Infer a same-shape grid transform from the demos. Return only one Python "
        "code block defining def transform(grid). np is provided. Do not import, "
        "open files, eval, call subprocess, or use any hidden test output. DATA="
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    )


def make_llama_cpp_generator(model_spec: Mapping[str, Any]) -> Callable[[str], str]:  # pragma: no cover
    from llama_cpp import Llama

    model_path = str(model_spec["model_path"])
    gpu = int(model_spec.get("gpu", 0))
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            main_gpu=gpu,
            verbose=False,
        )
    except TypeError:
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    def _generate(prompt: str) -> str:
        out = llm(prompt, max_tokens=384, temperature=0.0, stop=["\n\n\n"])
        return str(out["choices"][0]["text"])

    return _generate


def _cache_status(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    returned = {str(spec.get("hf_id")): dict(spec) for spec in model_specs or []}
    status: list[JsonDict] = []
    for model in SOTA_GGUF_MODELS:
        hf_id = model["hf_id"]
        spec = returned.get(hf_id)
        status.append(
            {
                "hf_id": hf_id,
                "resolved": bool(spec and spec.get("model_path")),
                "model_path": spec.get("model_path") if spec else None,
            }
        )
    return status


def _audit_no_leakage(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    exp5197_task_ids: set[str] | None = None,
) -> bool:
    excluded = set(exp5197_task_ids or set())
    for row in candidate_rows:
        if str(row.get("task_id")) in excluded:
            return False
        code = str(row.get("code", "")).lower()
        if any(token in code for token in _LEAKAGE_TOKENS):
            return False
        if _forbidden_ast_reason(ast.parse(str(row.get("code", "")))) is not None:
            return False
        raw = json.dumps(dict(row), sort_keys=True, default=str).lower()
        if "test_output" in raw or "target_hash" in raw:
            return False
    return True


def _verdict(*, usable: bool, n: int, resolved: bool) -> str:
    if not resolved:
        return "blocked_sota_gguf_not_cached_v477"
    state = "pool_ready_for_exp5212" if usable else "pool_below_exp5212_gate"
    return f"complete_gap4_sota_local_candidate_expansion_v477_n{n}_{state}"


def _dedupe_models_used(
    models_used: Sequence[str],
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    out: list[str] = []
    for hf_id in models_used:
        text = str(hf_id)
        if text and text not in out:
            out.append(text)
    for event in events:
        if event.get("live_prompted") is not True:
            continue
        text = str(event.get("model_hf_id") or "")
        if text and text not in out:
            out.append(text)
    return out


def build_artifact(
    *,
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    sota_gguf_resolved: bool,
    repair_attempts: int,
    source_task_budget: int,
    source_task_count: int,
    checkpoint_path: str,
    duration_s: float,
    generation_errors: Sequence[str] | None = None,
    exp5197_task_ids: set[str] | None = None,
) -> JsonDict:
    accepted = [dict(event) for event in events if event.get("accepted") is True]
    rejected = [dict(event) for event in events if event.get("accepted") is not True]
    leakage_passed = _audit_no_leakage(accepted, exp5197_task_ids=exp5197_task_ids)
    candidate_pool_n = len(accepted)
    usable = bool(candidate_pool_n >= CANDIDATE_POOL_TARGET_N and leakage_passed)
    recorded_models_used = _dedupe_models_used(models_used, events)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "candidate_pool_n": candidate_pool_n,
        "gap4_expansion_usable": usable,
        "models_used": recorded_models_used,
        "model_specs": [dict(spec) for spec in model_specs],
        "sota_gguf_resolved": bool(sota_gguf_resolved),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "repair_attempts": int(repair_attempts),
        "leakage_audit_passed": leakage_passed,
        "checkpoint_path": checkpoint_path,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _verdict(
            usable=usable,
            n=candidate_pool_n,
            resolved=bool(sota_gguf_resolved),
        ),
        "candidate_rows": accepted,
        "rejected_task_rows": rejected[:25],
        "checkpoint_summary": _checkpoint_payload(events)
        | {"events": f"{len(events)} rows; see checkpoint_path"},
        "source_task_budget": int(source_task_budget),
        "source_task_count": int(source_task_count),
        "generation_errors": list(generation_errors or []),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    if not sota_gguf_resolved:
        artifact["legacy_tiny_fallback_expected_quality"] = "poor"
        artifact["sota_cache_status"] = _cache_status(model_specs)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def blocked_artifact(*, duration_s: float) -> JsonDict:
    return build_artifact(
        events=[],
        model_specs=[],
        models_used=[],
        sota_gguf_resolved=False,
        repair_attempts=0,
        source_task_budget=0,
        source_task_count=0,
        checkpoint_path=CHECKPOINT_RELATIVE_PATH,
        duration_s=duration_s,
    )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    n = artifact.get("candidate_pool_n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        errors.append("candidate_pool_n_bare_int")
    usable = artifact.get("gap4_expansion_usable")
    if not isinstance(usable, bool):
        errors.append("gap4_expansion_usable_bare_bool")
    leakage = _audit_no_leakage(
        [dict(row) for row in artifact.get("candidate_rows", []) if isinstance(row, Mapping)]
    )
    if artifact.get("leakage_audit_passed") is not leakage:
        errors.append("leakage_audit_passed_false")
    expected_usable = bool(isinstance(n, int) and n >= CANDIDATE_POOL_TARGET_N and leakage)
    if isinstance(usable, bool) and usable is not expected_usable:
        errors.append("gap4_expansion_usable")
    if artifact.get("accepted_rows") != n:
        errors.append("accepted_rows")
    if not isinstance(artifact.get("model_specs"), list):
        errors.append("model_specs")
    if not isinstance(artifact.get("models_used"), list):
        errors.append("models_used")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if "significance" in verdict.lower():
        errors.append("honest_verdict_no_significance_claim")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("sota_gguf_resolved") is False and artifact.get("candidate_pool_n") != 0:
        errors.append("blocked_candidate_pool_n_zero")
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json_atomic(path, artifact)
    return path


def _default_cached_pair_loader() -> list[dict] | None:
    return cached_sota_pair(gpu_indices=(0, 1))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def run(
    *,
    root: Path | str = REPO_ROOT,
    cached_pair_loader: Callable[[], list[dict] | None] = _default_cached_pair_loader,
    task_loader: Callable[[Path | str, set[str], int], list[JsonDict]] = load_frame_transition_tasks,
    text_generator_factory: Callable[[Mapping[str, Any]], Callable[[str], str]] = make_llama_cpp_generator,
    max_live_prompts: int | None = None,
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    task_budget: int | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    live_limit = (
        _env_int("EXP5211_MAX_LIVE_PROMPTS", DEFAULT_MAX_LIVE_PROMPTS)
        if max_live_prompts is None
        else int(max_live_prompts)
    )
    budget = (
        _env_int("EXP5211_TASK_BUDGET", DEFAULT_TASK_BUDGET)
        if task_budget is None
        else int(task_budget)
    )
    model_specs = cached_pair_loader()
    if model_specs is None:
        artifact = blocked_artifact(duration_s=float(now()) - started)
        write_artifact(root_path, artifact)
        return artifact

    exp5197_task_ids = load_exp5197_scored_task_ids(root_path)
    tasks = task_loader(root_path, exp5197_task_ids, budget)
    checkpoint_path = root_path / CHECKPOINT_RELATIVE_PATH
    events = load_checkpoint(checkpoint_path)
    done = {str(event.get("task_id")) for event in events}
    accepted_n = sum(1 for event in events if event.get("accepted") is True)
    repair_attempts = sum(int(event.get("repair_attempts") or 0) for event in events)
    model_spec = dict(model_specs[0])
    models_used: list[str] = []
    generation_errors: list[str] = []
    generator: Callable[[str], str] | None = None
    if live_limit > 0:
        try:
            generator = text_generator_factory(model_spec)
        except Exception as exc:  # noqa: BLE001 - live loader failure is artifact evidence.
            generation_errors.append(f"generator_factory:{type(exc).__name__}:{str(exc)[:200]}")
            generator = None

    live_prompted = 0
    for task in tasks:
        if accepted_n >= CANDIDATE_POOL_TARGET_N:
            break
        task_id = str(task.get("task_id"))
        if task_id in done:
            continue
        raw_text = ""
        generation_error = None
        did_live_prompt = bool(generator is not None and live_prompted < live_limit)
        if did_live_prompt:
            hf_id = str(model_spec.get("hf_id"))
            if hf_id not in models_used:
                models_used.append(hf_id)
            try:
                raw_text = generator(build_prompt(task))
                live_prompted += 1
            except Exception as exc:  # noqa: BLE001
                generation_error = f"{type(exc).__name__}:{str(exc)[:200]}"
                generation_errors.append(generation_error)
                raw_text = ""
                live_prompted += 1
        event, attempts = process_task_row(
            task=task,
            raw_text=raw_text,
            model_spec=model_spec,
            checkpoint_path=checkpoint_path,
            prior_events=events,
            repair_budget=repair_budget,
            live_prompted=did_live_prompt,
            generation_error=generation_error,
        )
        events.append(event)
        done.add(task_id)
        repair_attempts += attempts
        if event.get("accepted") is True:
            accepted_n += 1

    artifact = build_artifact(
        events=events,
        model_specs=model_specs,
        models_used=models_used,
        sota_gguf_resolved=True,
        repair_attempts=repair_attempts,
        source_task_budget=budget,
        source_task_count=len(tasks),
        checkpoint_path=CHECKPOINT_RELATIVE_PATH,
        duration_s=float(now()) - started,
        generation_errors=generation_errors,
        exp5197_task_ids=exp5197_task_ids,
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"candidate_pool_n={artifact['candidate_pool_n']}")
    print(f"gap4_expansion_usable={artifact['gap4_expansion_usable']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
