"""Exp 4022 decentralization gate driven by Exp 4012 local best-of-N.

Spec refs: REQ-PHASE4-031, SCENARIO-PHASE4-031.

This module keeps the branch decision mechanical.  Exp 4012 is the diagnostic:
if high-k local sampling improves the local inducer, branch A should scale that
same-pool evidence.  If it does not, branch B only measures whether a
verifier-certified Codex corpus and a tiny sanity training signal exist.  Branch
B deliberately does not claim that a full local GGUF fine-tune would close the
gap when Exp 4012 shows no latent local-support lift.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "branch_taken",
    "exp4012_result_cited",
    "decentralization_next_step",
    "inference_substrate",
)

TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
BRANCH_A = "A_scale"
BRANCH_B = "B_distill_feasibility"
BLOCKED_BRANCH = "blocked"
MATERIAL_COVERAGE_GAIN = 0.02
INFERENCE_SUBSTRATE = (
    "cached_codex_gap4_traces_model_free_verifier_tiny_unigram_adapter_no_live_llm"
)
INVISIBLE_LEASH_PRINCIPLE = (
    "Invisible Leash: RLVR/distillation can sharpen behavior already in the base "
    "model support, but Exp 4012 showed no material local best-of-N lift here."
)

_BANNED_CODE_MARKERS = (
    "import os",
    "import sys",
    "subprocess",
    "open(",
    "np.load",
    "np.save",
    "fromfile",
    "__",
    "eval(",
    "exec(",
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and fail clearly if the artifact is malformed."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def choose_branch(exp4012_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the Exp 4022 branch and the exact Exp 4012 fields that drove it."""

    cited = {
        "local_beats_vote": bool(exp4012_result.get("local_beats_vote", False)),
        "coverage_gain_vs_3attempt": round(_as_float(exp4012_result.get("coverage_gain_vs_3attempt")), 4),
        "local_demo_perfect_coverage_bestofn": round(
            _as_float(exp4012_result.get("local_demo_perfect_coverage_bestofn")), 4
        ),
        "local_gated_pass2": round(_as_float(exp4012_result.get("local_gated_pass2")), 4),
        "k_samples_per_task": _as_int(exp4012_result.get("k_samples_per_task")),
    }
    materially_lifted = cited["coverage_gain_vs_3attempt"] >= MATERIAL_COVERAGE_GAIN
    if cited["local_beats_vote"] or materially_lifted:
        return BRANCH_A, cited
    return BRANCH_B, cited


def _load_pool_entries(pool_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in pool_paths:
        if not path.exists():
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in payload.get("entries", []) if isinstance(payload, dict) else []:
            if isinstance(entry, dict) and isinstance(entry.get("task"), str):
                entries.setdefault(str(entry["task"]), entry)
    return entries


def _demo_block(demos: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, pair in enumerate(demos, start=1):
        lines.append(f"Demo {index} INPUT:")
        lines.append(json.dumps(pair.get("input"), separators=(",", ":")))
        lines.append(f"Demo {index} OUTPUT:")
        lines.append(json.dumps(pair.get("output"), separators=(",", ":")))
    return "\n".join(lines)


def format_trace_instruction(task: str, pool_entry: dict[str, Any]) -> str:
    """Build the SFT prompt from public demos plus test input, never test gold."""

    demos = pool_entry.get("demos", [])
    if not isinstance(demos, list):
        demos = []
    return "\n".join(
        [
            "Infer the ARC transformation and write exactly one generic Python function.",
            "The function signature must be def transform(grid): and grid is a 2D numpy array.",
            f"Task: {task}",
            _demo_block(demos),
            "TEST INPUT:",
            json.dumps(pool_entry.get("test_input"), separators=(",", ":")),
            "Return only code for def transform(grid).",
        ]
    )


def _execution_validated(program: dict[str, Any]) -> bool:
    return (
        bool(program.get("demo_perfect"))
        and _as_float(program.get("demo_fit")) >= 1.0
        and isinstance(program.get("pred_grid"), list)
        and bool(program.get("pred_hash"))
        and isinstance(program.get("code"), str)
        and "def transform" in str(program.get("code"))
    )


def _code_is_safe_for_corpus(code: str) -> bool:
    lowered = code.lower()
    return not any(marker in lowered for marker in _BANNED_CODE_MARKERS)


def _hardcoded_grid_suspect(code: str) -> bool:
    return code.count("[") > 120 or "return arr([" in code or "np.array([[" in code


def _trace_from_program(
    program: dict[str, Any],
    *,
    source_artifact: Path,
    pool_entry: dict[str, Any],
) -> dict[str, Any]:
    code = str(program["code"]).strip()
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    suspect = _hardcoded_grid_suspect(code)
    return {
        "task": str(program["task"]),
        "source_artifact": str(source_artifact),
        "source_model": "codex",
        "instruction": format_trace_instruction(str(program["task"]), pool_entry),
        "response": f"```python\n{code}\n```",
        "verifier_certification": {
            "demo_perfect": True,
            "demo_fit": 1.0,
            "execution_validated": True,
            "pred_hash": str(program.get("pred_hash")),
        },
        "quality": {
            "code_chars": len(code),
            "prompt_chars": len(format_trace_instruction(str(program["task"]), pool_entry)),
            "hardcoded_grid_suspect": suspect,
            "quality_score": 0.75 if suspect else 1.0,
        },
        "code_sha256_16": code_hash,
    }


def harvest_distillation_traces(
    program_artifact_paths: Iterable[Path],
    pool_paths: Iterable[Path],
    *,
    max_traces: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Harvest demo-perfect, execution-validated Codex traces into SFT rows."""

    pool_by_task = _load_pool_entries(pool_paths)
    traces: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rejection_reasons = {
        "not_demo_perfect_or_not_executed": 0,
        "unsafe_code": 0,
        "missing_pool_entry": 0,
        "duplicate_trace": 0,
    }
    scanned = 0
    execution_validated = 0

    for artifact_path in program_artifact_paths:
        if not artifact_path.exists():
            continue
        try:
            payload = load_json(artifact_path)
        except Exception:
            continue
        for program in payload.get("programs", []):
            if not isinstance(program, dict):
                continue
            scanned += 1
            if not _execution_validated(program):
                rejection_reasons["not_demo_perfect_or_not_executed"] += 1
                continue
            execution_validated += 1
            code = str(program["code"])
            if not _code_is_safe_for_corpus(code):
                rejection_reasons["unsafe_code"] += 1
                continue
            task = str(program.get("task"))
            pool_entry = pool_by_task.get(task)
            if pool_entry is None:
                rejection_reasons["missing_pool_entry"] += 1
                continue
            code_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:16]
            dedupe_key = (task, code_hash)
            if dedupe_key in seen:
                rejection_reasons["duplicate_trace"] += 1
                continue
            seen.add(dedupe_key)
            traces.append(_trace_from_program(program, source_artifact=artifact_path, pool_entry=pool_entry))
            if max_traces is not None and len(traces) >= max_traces:
                break
        if max_traces is not None and len(traces) >= max_traces:
            break

    code_lengths = [int(trace["quality"]["code_chars"]) for trace in traces]
    suspect_count = sum(1 for trace in traces if trace["quality"]["hardcoded_grid_suspect"])
    report = {
        "n_programs_scanned": scanned,
        "n_execution_validated": execution_validated,
        "n_clean_traces": len(traces),
        "rejection_reasons": rejection_reasons,
        "median_code_chars": int(statistics.median(code_lengths)) if code_lengths else 0,
        "hardcoded_grid_suspect_count": suspect_count,
        "generic_trace_ratio": round((len(traces) - suspect_count) / len(traces), 4) if traces else 0.0,
    }
    return traces, report


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def tiny_sanity_finetune(traces: list[dict[str, Any]], *, subset_size: int = 8) -> dict[str, Any]:
    """Run a tiny add-one unigram adapter as a bounded feasibility proxy."""

    subset = traces[: max(0, int(subset_size))]
    token_stream: list[str] = []
    for trace in subset:
        token_stream.extend(_tokens(str(trace.get("response", ""))))
    vocab = sorted(set(token_stream))
    if not token_stream or not vocab:
        return {
            "ran": False,
            "method": "add_one_unigram_code_token_adapter",
            "subset_size": 0,
            "tokens_trained": 0,
            "loss_before": 0.0,
            "loss_after": 0.0,
            "loss_delta": 0.0,
            "full_llm_finetune": False,
            "claim_scope": "no full GGUF fine-tune was run",
        }

    counts = {token: 0 for token in vocab}
    for token in token_stream:
        counts[token] += 1
    vocab_size = len(vocab)
    total = len(token_stream)
    loss_before = math.log(vocab_size)
    loss_after = -sum(math.log((counts[token] + 1) / (total + vocab_size)) for token in token_stream) / total
    return {
        "ran": True,
        "method": "add_one_unigram_code_token_adapter",
        "subset_size": len(subset),
        "tokens_trained": total,
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "loss_delta": round(max(0.0, loss_before - loss_after), 6),
        "full_llm_finetune": False,
        "claim_scope": "tiny subset token adapter only; no full GGUF fine-tune was run",
    }


def build_blocked_artifact(reason: str, *, duration_s: float) -> dict[str, Any]:
    """Build the no-branch artifact required when Exp 4012 is unavailable."""

    return {
        "experiment": "experiment_4022_decentralization_gated",
        "honest_verdict": reason,
        "branch_taken": BLOCKED_BRANCH,
        "exp4012_result_cited": "unavailable",
        "decentralization_next_step": (
            "Regenerate results/experiment_4012_gap4_local_best_of_n.json before choosing "
            "scaling or distillation."
        ),
        "inference_substrate": "none_exp4012_precondition_blocked",
        "duration_s": round(float(duration_s), 3),
    }


def _next_step(branch_taken: str, corpus_report: dict[str, Any]) -> str:
    if branch_taken == BRANCH_A:
        return (
            "Run the bounded same-pool scaling confirmation on the lifted local model before "
            "starting any distillation."
        )
    if int(corpus_report.get("n_clean_traces", 0)) <= 0:
        return (
            "Do not train yet: regenerate verifier-certified traces with archived demos, then "
            "evaluate a stronger local base."
        )
    return (
        "Treat the current local model as representational-gap likely: do not run a full "
        "distillation pass on it; evaluate a stronger local base or second SOTA GGUF, and reuse "
        "this corpus only when a later high-k run shows latent support."
    )


def build_decentralization_artifact(
    exp4012_result: dict[str, Any],
    *,
    branch_taken: str,
    corpus_report: dict[str, Any],
    sanity_finetune: dict[str, Any],
    corpus_path: Path,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4022 terminal artifact from branch and corpus evidence."""

    _branch, cited = choose_branch(exp4012_result)
    clean = int(corpus_report.get("n_clean_traces", 0))
    delta = _as_float(sanity_finetune.get("loss_delta"))
    if branch_taken == BRANCH_A:
        verdict = "complete: A_scale_exp4012_positive_scaling_confirmation_required"
        support = "latent_support_possible"
    else:
        verdict = f"complete: B_distill_feasibility_exp4012_no_lift_clean_traces{clean}_sanity_delta{delta:.3f}"
        support = "representational_gap_likely"

    return {
        "experiment": "experiment_4022_decentralization_gated",
        "honest_verdict": verdict,
        "branch_taken": branch_taken,
        "exp4012_result_cited": cited,
        "decentralization_next_step": _next_step(branch_taken, corpus_report),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "distillation_corpus": {
            **corpus_report,
            "corpus_path": str(corpus_path),
            "verifier_certification_rule": "demo_perfect==true, demo_fit>=1.0, executed pred_grid/pred_hash present",
        },
        "sanity_finetune": sanity_finetune,
        "local_support_diagnostic": support,
        "prior_art_decider": INVISIBLE_LEASH_PRINCIPLE,
        "tulu3_rlvr_recipe_scope": (
            "Applicable only as a sharpening recipe after local best-of-N shows latent support."
        ),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return schema errors for the required Exp 4022 fields."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be a terminal-prefix string")

    branch = artifact.get("branch_taken")
    if branch not in {BRANCH_A, BRANCH_B, BLOCKED_BRANCH}:
        errors.append("branch_taken must be A_scale, B_distill_feasibility, or blocked")

    if branch != BLOCKED_BRANCH and not isinstance(artifact.get("exp4012_result_cited"), dict):
        errors.append("exp4012_result_cited must cite the Exp4012 numeric fields")
    if branch == BLOCKED_BRANCH and artifact.get("exp4012_result_cited") != "unavailable":
        errors.append("blocked artifacts must cite exp4012_result_cited as unavailable")

    for field in ("decentralization_next_step", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    return errors


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write deterministic JSONL for the harvested fine-tune corpus."""

    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")
    return path


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Write stable JSON for the terminal Exp 4022 result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
