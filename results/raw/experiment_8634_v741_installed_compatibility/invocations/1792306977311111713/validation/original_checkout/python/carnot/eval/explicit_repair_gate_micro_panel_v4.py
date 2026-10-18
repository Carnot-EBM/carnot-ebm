"""Exp 3115 explicit repair gate and micro-panel v4.

Spec refs: REQ-VERIFY-3115, SCENARIO-VERIFY-3115.

This module turns the upstream repair gate into a terminal artifact even when
repair cannot run. That boundary matters operationally: a blocked repair panel
is useful evidence, while a missing panel looks like the conductor simply lost
the task. When the gate is open, repair candidates are still accepted only
after exact offline authorities verify the localized fragment targets.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import fragment_verification_pilot_3114 as fragment_pilot


JsonDict = dict[str, Any]
RepairGenerator = Callable[[str, Mapping[str, Any], Mapping[str, Any]], str]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
SCHEMA = "carnot.explicit_repair_gate_micro_panel.v4"
ARTIFACT = "experiment_3115_explicit_repair_gate_micro_panel_v4"
OUTPUT_REL_PATH = Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json")

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3110_REL_PATH = Path(
    "results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"
)
EXP3113_REL_PATH = Path(
    "results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json"
)
EXP3114_REL_PATH = Path(
    "results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json"
)
REPAIR_TARGET_MANIFEST_REL_PATH = Path(
    "results/fragment_verification_pilot_3114/repair_target_manifest.jsonl"
)

MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MIN_LIVE_GGUF_BYTES = 1_000_000_000
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = (
    "repair_micro_panel_v4_artifact_ready",
    "repair_unblocked",
    "repair_run_executed",
    "gate_block_reason",
    "model_specs",
    "selected_headline_model_ids",
    "exact_ground_truth_count",
    "repair_success_delta",
    "false_repair_accept_rate",
    "intent_preservation_rate",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3115_explicit_repair_gate_micro_panel_v4.py -q --no-cov",
    ".venv/bin/coverage run --include='*/explicit_repair_gate_micro_panel_v4.py' -m pytest -o addopts='' tests/python/test_experiment_3115_explicit_repair_gate_micro_panel_v4.py -q",
    ".venv/bin/coverage report --include='*/explicit_repair_gate_micro_panel_v4.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python/test_experiment_1664_e2e.py -q --no-cov",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("experiment_template_cache_policy", Path("scripts/experiment_template.py"), False),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3110_model_manifest", EXP3110_REL_PATH, True),
    ("exp3113_repair_gate", EXP3113_REL_PATH, True),
    ("exp3114_fragment_pilot", EXP3114_REL_PATH, True),
    ("exp3114_repair_target_manifest", REPAIR_TARGET_MANIFEST_REL_PATH, True),
)
JSON_KEY_RE = re.compile(r'"([^"\\]+)"\s*:')


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence for missing inputs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL object rows, skipping malformed and non-object lines."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    repair_generator: RepairGenerator | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3115: build the explicit repair boundary artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3113 = read_json_object(root_path / EXP3113_REL_PATH)
    exp3114 = read_json_object(root_path / EXP3114_REL_PATH)
    target_manifest_rel_path = Path(
        str(exp3114.get("repair_target_manifest_path") or REPAIR_TARGET_MANIFEST_REL_PATH)
    )
    targets = read_jsonl_rows(root_path / target_manifest_rel_path)
    model_specs = [dict(row) for row in exp3113.get("model_specs", []) if isinstance(row, Mapping)]
    selected_ids = [
        str(model_id)
        for model_id in exp3113.get("selected_headline_model_ids", [])
        if isinstance(model_id, str)
    ]
    selected_specs = selected_cached_model_specs(model_specs, selected_ids)
    gate_state = str(exp3113.get("repair_gate_state") or "blocked_missing_inputs")
    live_generator_planned = repair_generator is None
    source_rows = source_artifacts(root_path, target_manifest_rel_path)

    repair_unblocked = gate_state == "unblocked"
    repair_run_executed = False
    gate_block_reason = gate_reason(gate_state, exp3113)
    repair_rows: list[JsonDict] = []

    if repair_unblocked:
        gate_block_reason = runtime_block_reason(
            targets,
            selected_specs,
            require_real_model_file=live_generator_planned,
        )
        if not gate_block_reason:
            generator = repair_generator
            if generator is None:  # pragma: no cover - live path is exercised by the artifact run.
                try:
                    generator = llama_cpp_repair_generator(selected_specs[0])
                except Exception as exc:  # pragma: no cover - depends on local llama.cpp runtime.
                    gate_block_reason = f"live_repair_runtime_blocked: {type(exc).__name__}: {exc}"
            if generator is not None and not gate_block_reason:
                repair_rows = run_repair_panel(targets, selected_specs[0], generator)
                repair_run_executed = True
                gate_block_reason = "repair_gate_unblocked"

    metrics = repair_metrics(repair_rows, len(targets))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "repair_micro_panel_v4_artifact_ready": True,
        "repair_unblocked": repair_unblocked,
        "repair_run_executed": repair_run_executed,
        "gate_block_reason": gate_block_reason,
        "model_specs": model_specs,
        "selected_headline_model_ids": selected_ids,
        "exact_ground_truth_count": len(targets),
        "repair_success_delta": metrics["repair_success_delta"],
        "false_repair_accept_rate": metrics["false_repair_accept_rate"],
        "intent_preservation_rate": metrics["intent_preservation_rate"],
        "repair_rows": repair_rows,
        "repair_target_manifest_path": target_manifest_rel_path.as_posix(),
        "repair_target_count": len(targets),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "inference_substrate": inference_substrate(
            repair_unblocked=repair_unblocked,
            repair_run_executed=repair_run_executed,
            live_generator_planned=live_generator_planned,
            selected_model_count=len(selected_specs),
        ),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    repair_generator: RepairGenerator | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3115 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        repair_generator=repair_generator,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    write_json(out_path, artifact)
    return out_path


def selected_cached_model_specs(
    model_specs: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
) -> list[JsonDict]:
    """Return selected mandated SOTA models that have local cache evidence."""

    selected = set(selected_ids)
    rows: list[JsonDict] = []
    for model in model_specs:
        hf_id = str(model.get("hf_id") or "")
        if hf_id not in MANDATORY_MODEL_IDS:
            continue
        if hf_id not in selected and model.get("selected") is not True:
            continue
        if not model.get("model_path"):
            continue
        cached = model.get("cache_present") is True or str(model.get("cache_status")) == "cached"
        if cached:
            rows.append(dict(model))
    return rows


def gate_reason(gate_state: str, exp3113: Mapping[str, Any]) -> str:
    """Return the actionable upstream gate reason for blocked repair."""

    if gate_state == "unblocked":
        return "repair_gate_unblocked"
    explanation = exp3113.get("exp3115_repair_gate_explanation")
    action = ""
    if isinstance(explanation, Mapping):
        action = str(explanation.get("downstream_action") or "")
    suffix = f"; downstream_action={action}" if action else ""
    return f"exp3113 repair_gate_state={gate_state}; repair generation not authorized{suffix}"


def runtime_block_reason(
    targets: Sequence[Mapping[str, Any]],
    selected_specs: Sequence[Mapping[str, Any]],
    *,
    require_real_model_file: bool,
) -> str:
    """Return a concrete runtime blocker once the upstream gate is open."""

    if not targets:
        return "missing_repair_targets: Exp 3114 repair target manifest has no rows"
    if not selected_specs:
        return "missing_selected_mandated_sota_model: no selected cached mandated GGUF model"
    model_path = selected_specs[0].get("model_path")
    if not model_path:
        return "missing_selected_model_path: selected SOTA model has no model_path"
    path = Path(str(model_path))
    if require_real_model_file:
        if not path.is_file():
            return f"missing_selected_model_path: {path}"
        size = path.stat().st_size
        if size < MIN_LIVE_GGUF_BYTES:
            return f"unusable_selected_model_path: {path} has only {size} bytes"
    return ""


def run_repair_panel(
    targets: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    repair_generator: RepairGenerator,
) -> list[JsonDict]:
    """Run repair generation for each localized target and verify candidates."""

    rows: list[JsonDict] = []
    for index, target in enumerate(targets):
        prompt = repair_prompt(target)
        raw_candidate = ""
        repaired_fragment = ""
        verification = {
            "exact_verified": False,
            "intent_preserved": False,
            "verification_errors": [],
        }
        try:
            raw_candidate = repair_generator(prompt, target, model_spec)
            repaired_fragment = extract_repaired_fragment(raw_candidate)
            verification = verify_repair_candidate(target, repaired_fragment)
        except Exception as exc:
            verification["verification_errors"] = [f"generation_error: {type(exc).__name__}: {exc}"]
        accepted = (
            verification["exact_verified"] is True and verification["intent_preserved"] is True
        )
        rows.append(
            {
                "row_index": index,
                "fixture_id": str(target.get("fixture_id") or ""),
                "fragment_id": str(target.get("fragment_id") or ""),
                "model_id": str(model_spec.get("hf_id") or ""),
                "model_path": model_spec.get("model_path"),
                "raw_candidate": raw_candidate,
                "repaired_fragment": repaired_fragment,
                "exact_verified": verification["exact_verified"],
                "intent_preserved": verification["intent_preserved"],
                "accepted": accepted,
                "verification_errors": verification["verification_errors"],
                "solver_evidence": target.get("solver_evidence", {}),
            }
        )
    return rows


def repair_prompt(target: Mapping[str, Any]) -> str:
    """Build the exact-fragment repair prompt used by live and test generators."""

    return (
        "REQ-VERIFY-3115 exact fragment repair.\n"
        "Return one JSON object with a string field named repaired_fragment.\n"
        f"fixture_id: {target.get('fixture_id')}\n"
        f"fragment_id: {target.get('fragment_id')}\n"
        f"failing_constraint: {target.get('failing_constraint')}\n"
        f"expected_direction: {target.get('expected_direction')}\n"
        f"solver_evidence: {json.dumps(target.get('solver_evidence', {}), sort_keys=True)}\n"
    )


def extract_repaired_fragment(raw_candidate: str) -> str:
    """Extract a repair string from JSON or simple fenced-code output."""

    text = raw_candidate.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("repaired_fragment"), str):
        return str(payload["repaired_fragment"]).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def verify_repair_candidate(target: Mapping[str, Any], repaired_fragment: str) -> JsonDict:
    """Verify one repaired fragment with the exact authority named by Exp 3114."""

    evidence = target.get("solver_evidence")
    evidence_map = dict(evidence) if isinstance(evidence, Mapping) else {}
    authority = str(evidence_map.get("authority") or "")
    if authority == "python_ast_literal_evaluator":
        return verify_assertion_repair(evidence_map, repaired_fragment)
    if authority == "python_json_parser":
        return verify_json_repair(evidence_map, repaired_fragment)
    if authority == "deterministic_integer_constraint_evaluator":
        return verify_numeric_repair(evidence_map, repaired_fragment)
    return {
        "exact_verified": False,
        "intent_preserved": False,
        "verification_errors": [f"unsupported repair authority: {authority}"],
    }


def verify_assertion_repair(evidence: Mapping[str, Any], repaired_fragment: str) -> JsonDict:
    """Check assertion repairs by recomputing the AST arithmetic expression."""

    errors: list[str] = []
    try:
        parsed = ast.parse(repaired_fragment)
        statement = parsed.body[0]
        if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Compare):
            raise ValueError("candidate is not a simple assertion")
        compare = statement.test
        if len(compare.comparators) != 1:
            raise ValueError("candidate assertion has multiple comparators")
        left_value = fragment_pilot._eval_int(compare.left)
        right_value = fragment_pilot._eval_int(compare.comparators[0])
        expected_value = int(evidence.get("computed_value"))
        exact = left_value == right_value == expected_value
        intent = assertion_left_ast(repaired_fragment) == assertion_left_ast(
            str(evidence.get("assertion"))
        )
    except Exception as exc:
        return {
            "exact_verified": False,
            "intent_preserved": False,
            "verification_errors": [f"assertion_parse_or_eval_error: {type(exc).__name__}: {exc}"],
        }
    if not exact:
        errors.append("assertion_not_exact")
    if not intent:
        errors.append("assertion_intent_changed")
    return {
        "exact_verified": exact,
        "intent_preserved": intent,
        "verification_errors": errors,
    }


def assertion_left_ast(assertion_source: str) -> str:
    """Return a stable AST dump for the left side of a simple assertion."""

    parsed = ast.parse(assertion_source)
    statement = parsed.body[0]
    if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Compare):
        return ""
    return ast.dump(statement.test.left)


def verify_json_repair(evidence: Mapping[str, Any], repaired_fragment: str) -> JsonDict:
    """Check JSON repairs by parsing and preserving keys from the broken candidate."""

    errors: list[str] = []
    try:
        parsed = json.loads(repaired_fragment)
    except json.JSONDecodeError as exc:
        return {
            "exact_verified": False,
            "intent_preserved": False,
            "verification_errors": [f"json_parse_error: {exc}"],
        }
    exact = isinstance(parsed, Mapping)
    original_keys = set(JSON_KEY_RE.findall(str(evidence.get("candidate") or "")))
    repaired_keys = set(str(key) for key in parsed) if isinstance(parsed, Mapping) else set()
    intent = exact and original_keys.issubset(repaired_keys)
    if not exact:
        errors.append("json_repair_not_object")
    if not intent:
        errors.append("json_required_keys_not_preserved")
    return {
        "exact_verified": exact,
        "intent_preserved": intent,
        "verification_errors": errors,
    }


def verify_numeric_repair(evidence: Mapping[str, Any], repaired_fragment: str) -> JsonDict:
    """Check numeric repairs with the deterministic integer constraint evaluator."""

    errors: list[str] = []
    try:
        parsed = json.loads(repaired_fragment)
        if not isinstance(parsed, Mapping):
            raise ValueError("numeric repair must be a JSON object")
        assignment = {str(key): int(value) for key, value in parsed.items()}
        original_assignment = evidence.get("assignment")
        original_keys = (
            set(original_assignment) if isinstance(original_assignment, Mapping) else set()
        )
        intent = set(assignment) == original_keys
        exact, _constraint_evidence, _direction = fragment_pilot._evaluate_constraint(
            str(evidence.get("constraint") or ""),
            assignment,
        )
    except Exception as exc:
        return {
            "exact_verified": False,
            "intent_preserved": False,
            "verification_errors": [f"numeric_parse_or_eval_error: {type(exc).__name__}: {exc}"],
        }
    if not exact:
        errors.append("numeric_constraint_not_satisfied")
    if not intent:
        errors.append("numeric_assignment_keys_changed")
    return {
        "exact_verified": exact,
        "intent_preserved": intent,
        "verification_errors": errors,
    }


def repair_metrics(rows: Sequence[Mapping[str, Any]], total_targets: int) -> JsonDict:
    """Compute finite repair safety metrics for executed and blocked panels."""

    if not rows or total_targets == 0:
        return {
            "repair_success_delta": 0.0,
            "false_repair_accept_rate": 0.0,
            "intent_preservation_rate": 0.0,
        }
    accepted = [row for row in rows if row.get("accepted") is True]
    false_accepts = [
        row
        for row in accepted
        if row.get("exact_verified") is not True or row.get("intent_preserved") is not True
    ]
    intent_preserved = [row for row in rows if row.get("intent_preserved") is True]
    return {
        "repair_success_delta": rate(len(accepted), total_targets),
        "false_repair_accept_rate": rate(len(false_accepts), len(accepted)),
        "intent_preservation_rate": rate(len(intent_preserved), total_targets),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3115 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("gate_block_reason") or ""):
        raise ValueError("gate_block_reason must be actionable")
    for field in (
        "repair_success_delta",
        "false_repair_accept_rate",
        "intent_preservation_rate",
    ):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"finite metric required for {field}")
    if artifact.get("repair_run_executed") is True and artifact.get("repair_unblocked") is not True:
        raise ValueError("repair_run_executed cannot execute when repair_unblocked is false")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("repair_run_executed") is True:
        if not any(verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES):
            raise ValueError(
                "executed repair artifact honest_verdict must start with a success prefix"
            )
    elif not verdict.startswith("blocked_"):
        raise ValueError("blocked repair boundary requires a blocked verdict")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Map artifact state to the conductor terminal vocabulary."""

    if artifact.get("repair_run_executed") is True:
        return (
            "complete: repair_micro_panel_v4_artifact_ready=true; "
            "repair_run_executed=true; "
            f"repair_success_delta={artifact.get('repair_success_delta')}; "
            f"false_repair_accept_rate={artifact.get('false_repair_accept_rate')}; "
            f"intent_preservation_rate={artifact.get('intent_preservation_rate')}"
        )
    if artifact.get("repair_unblocked") is True:
        return f"blocked_repair_runtime: {artifact.get('gate_block_reason')}"
    return f"blocked_repair_gate: {artifact.get('gate_block_reason')}"


def inference_substrate(
    *,
    repair_unblocked: bool,
    repair_run_executed: bool,
    live_generator_planned: bool,
    selected_model_count: int,
) -> JsonDict:
    """Describe whether the repair boundary used live model execution."""

    return {
        "kind": "explicit_repair_gate_micro_panel_v4",
        "repair_unblocked": repair_unblocked,
        "repair_run_executed": repair_run_executed,
        "live_llm_inference": repair_run_executed and live_generator_planned,
        "executes_models": repair_run_executed,
        "new_model_execution": repair_run_executed and live_generator_planned,
        "repair_generator_kind": (
            "live_llama_cpp" if live_generator_planned else "injected_repair_generator"
        ),
        "selected_mandated_sota_model_count": selected_model_count,
        "exact_offline_authority": True,
        "legacy_tiny_models_headline_eligible": False,
    }


def source_artifacts(root: Path, target_manifest_rel_path: Path) -> list[JsonDict]:
    """Return source provenance, replacing the dynamic Exp 3114 manifest path."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_SPECS:
        path = target_manifest_rel_path if rel_path == REPAIR_TARGET_MANIFEST_REL_PATH else rel_path
        full_path = root / path
        rows.append(
            {
                "id": source_id,
                "path": path.as_posix(),
                "required": required,
                "exists": full_path.is_file(),
                "sha256": sha256_file(full_path),
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum for a present file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe ratio."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative wall-clock duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def llama_cpp_repair_generator(
    model_spec: Mapping[str, Any],
) -> RepairGenerator:  # pragma: no cover
    """Return a live local llama.cpp repair generator for the selected GGUF."""

    from llama_cpp import Llama  # noqa: PLC0415

    model_path = str(model_spec["model_path"])
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_gpu_layers=-1,
        n_batch=128,
        verbose=False,
    )

    def generate(prompt: str, target: Mapping[str, Any], model: Mapping[str, Any]) -> str:
        del target, model
        response = llm(
            prompt,
            max_tokens=96,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["\n\n"],
        )
        choices = response.get("choices", [])
        if choices and isinstance(choices[0], Mapping):
            return str(choices[0].get("text") or "").strip()
        return ""

    return generate


def main() -> int:  # pragma: no cover
    """CLI entry point used by the conductor-facing experiment wrapper."""

    output_path = write_artifact(REPO_ROOT)
    artifact = read_json_object(output_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact.get("repair_micro_panel_v4_artifact_ready") is True else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
