"""Exp 1311 verifier-backed SOTA answer-stability micro-slice.

Spec: REQ-VERIFY-1311,
      SCENARIO-VERIFY-1311
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1311_sota_constraintbench_satquest_answer_stability.json"
)
DEFAULT_RUN_DATE = "20260505"
MANDATED_HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "answer_stability_score",
    "cross_model_disagreement_rate",
    "constraintbench_items",
    "satquest_items",
    "pysat_verified_rate",
    "feasibility_rate",
    "unknown_or_abstain_rate",
    "headline_result_allowed",
    "models_used",
    "honest_verdict",
)
_BOUNDED_LABELS = {"SAT", "UNSAT", "UNKNOWN", "ABSTAIN"}
_LABEL_RE = re.compile(
    r"\b(UNSATISFIABLE|UNSAT|SATISFIABLE|SAT|UNKNOWN|UNDETERMINED|ABSTAIN)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MicroItem:
    """One tiny deterministic verifier fixture."""

    item_id: str
    family: str
    prompt: str
    expected_label: str
    num_variables: int
    clauses: tuple[tuple[int, ...], ...] | None
    compact_encoding: bool = False


@dataclass(frozen=True)
class RawGeneration:
    """Raw bounded-label generation returned by either llama.cpp or a test double."""

    text: str
    token_count: int
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Deterministic SAT/UNSAT/UNKNOWN check for one parsed model label."""

    verifier_label: str
    verified: bool
    feasible: bool | None
    verifier_backend: str


CachedPairFn = Callable[..., list[dict[str, Any]] | None]
GenerationFn = Callable[[dict[str, Any], MicroItem, int, str, int], RawGeneration]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_micro_slice() -> list[MicroItem]:
    """Build the fixed 10-item ConstraintBench/SATQuest-inspired fixture set."""
    return [
        MicroItem(
            "cb_sat_schedule",
            "constraintbench",
            "Choose Boolean tasks x1,x2,x3. Constraints: x1, x2, and x3 must all hold.",
            "SAT",
            3,
            ((1,), (2,), (3,)),
        ),
        MicroItem(
            "cb_unsat_capacity",
            "constraintbench",
            "Choose x1. Constraints: x1 must hold and x1 must not hold.",
            "UNSAT",
            1,
            ((1,), (-1,)),
        ),
        MicroItem(
            "cb_unknown_missing_bound",
            "constraintbench",
            "A routing plan is feasible only if the capacity bound B is provided. B is omitted.",
            "UNKNOWN",
            0,
            None,
        ),
        MicroItem(
            "cb_compact_sat",
            "constraintbench",
            "Compact encoding: n=3; clauses=[1|-2],[2|3],[-1|3]. Is it feasible?",
            "SAT",
            3,
            ((1, -2), (2, 3), (-1, 3)),
            compact_encoding=True,
        ),
        MicroItem(
            "cb_compact_unsat",
            "constraintbench",
            "Compact encoding: n=2; clauses=[1],[-1|2],[-2]. Is it feasible?",
            "UNSAT",
            2,
            ((1,), (-1, 2), (-2,)),
            compact_encoding=True,
        ),
        MicroItem(
            "sq_sat_xor",
            "satquest",
            "CNF with two variables: (x1 or x2) and (not x1 or not x2).",
            "SAT",
            2,
            ((1, 2), (-1, -2)),
        ),
        MicroItem(
            "sq_unsat_unit_conflict",
            "satquest",
            "CNF with one variable: (x1) and (not x1).",
            "UNSAT",
            1,
            ((1,), (-1,)),
        ),
        MicroItem(
            "sq_unknown_hidden_clause",
            "satquest",
            "CNF query says one clause is hidden by the generator cache. Classify only if known.",
            "UNKNOWN",
            0,
            None,
        ),
        MicroItem(
            "sq_compact_sat_dimacs",
            "satquest",
            "DIMACS-lite: p cnf 3 3; 1 -2 0; 2 -3 0; 3 0.",
            "SAT",
            3,
            ((1, -2), (2, -3), (3,)),
            compact_encoding=True,
        ),
        MicroItem(
            "sq_compact_unsat_dimacs",
            "satquest",
            "DIMACS-lite: p cnf 2 3; 1 2 0; -1 0; -2 0.",
            "UNSAT",
            2,
            ((1, 2), (-1,), (-2,)),
            compact_encoding=True,
        ),
    ]


def parse_final_label(text: str) -> str:
    """Parse only the bounded final labels accepted by REQ-VERIFY-1311."""
    mapped: list[str] = []
    for match in _LABEL_RE.finditer(text):
        raw = match.group(1).upper()
        if raw in {"SATISFIABLE", "SAT"}:
            mapped.append("SAT")
        elif raw in {"UNSATISFIABLE", "UNSAT"}:
            mapped.append("UNSAT")
        elif raw in {"UNKNOWN", "UNDETERMINED"}:
            mapped.append("UNKNOWN")
        else:
            mapped.append("ABSTAIN")

    labels = set(mapped)
    return mapped[0] if len(labels) == 1 else "ABSTAIN"


def _pure_python_cnf_sat(item: MicroItem) -> bool:
    clauses = item.clauses or ()
    for bits in itertools.product((False, True), repeat=item.num_variables):
        assignment = {index + 1: value for index, value in enumerate(bits)}
        if all(
            any(assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)] for lit in clause)
            for clause in clauses
        ):
            return True
    return False


def _z3_cnf_sat(item: MicroItem) -> bool:
    import z3  # noqa: PLC0415

    variables = {index: z3.Bool(f"x{index}") for index in range(1, item.num_variables + 1)}
    solver = z3.Solver()
    for clause in item.clauses or ():
        literals = [
            variables[abs(lit)] if lit > 0 else z3.Not(variables[abs(lit)]) for lit in clause
        ]
        solver.add(z3.Or(*literals) if len(literals) > 1 else literals[0])
    return solver.check() == z3.sat


def _cnf_sat_with_backend(item: MicroItem, backend: str) -> tuple[bool, str]:
    if backend == "pure_python":
        return _pure_python_cnf_sat(item), "pure_python_cnf"
    try:
        return _z3_cnf_sat(item), "z3"
    except Exception:
        return _pure_python_cnf_sat(item), "pure_python_cnf"


def verify_item_label(
    item: MicroItem, parsed_label: str, *, backend: str = "auto"
) -> VerificationResult:
    """Verify a parsed model label against the deterministic fixture oracle."""
    normalized = parsed_label if parsed_label in _BOUNDED_LABELS else "ABSTAIN"
    if item.clauses is None or item.expected_label == "UNKNOWN":
        return VerificationResult(
            verifier_label="UNKNOWN",
            verified=normalized in {"UNKNOWN", "ABSTAIN"},
            feasible=None,
            verifier_backend="ambiguous_fixture",
        )

    sat, backend_used = _cnf_sat_with_backend(item, backend)
    verifier_label = "SAT" if sat else "UNSAT"
    return VerificationResult(
        verifier_label=verifier_label,
        verified=normalized == verifier_label,
        feasible=sat,
        verifier_backend=backend_used,
    )


def build_prompt(item: MicroItem, perturbation_index: int) -> str:
    """Return one of two deterministic bounded-label prompt perturbations."""
    if perturbation_index == 0:
        prefix = "Return exactly one label: SAT, UNSAT, or UNKNOWN."
    else:
        prefix = "Classify feasibility using only SAT, UNSAT, or UNKNOWN."
    return f"{prefix}\nItem {item.item_id}: {item.prompt}\nFinal label:"


def _completion_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    choices = result.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return str(choices[0].get("text", ""))
    return ""


def _completion_token_count(result: Any, text: str, llm: Any) -> int:
    if isinstance(result, dict):
        completion_tokens = (result.get("usage") or {}).get("completion_tokens")
        if isinstance(completion_tokens, int):
            return max(0, completion_tokens)
    tokenize = getattr(llm, "tokenize", None)
    if callable(tokenize):
        try:
            return len(tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            pass
    return len(text.split()) if text.strip() else 0


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    gc.collect()


def _import_llama_class() -> tuple[bool, type[Any] | None, str | None]:
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _response_row(
    spec: dict[str, Any],
    item: MicroItem,
    perturbation_index: int,
    prompt: str,
    raw: RawGeneration,
    generation_source: str,
) -> dict[str, Any]:
    parsed = parse_final_label(raw.text) if raw.error is None else "ABSTAIN"
    return {
        "model_name": spec.get("name"),
        "hf_id": spec.get("hf_id"),
        "gpu": spec.get("gpu"),
        "item_id": item.item_id,
        "family": item.family,
        "expected_label": item.expected_label,
        "compact_encoding": item.compact_encoding,
        "perturbation_index": perturbation_index,
        "prompt_sha16": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "raw_output": raw.text,
        "parsed_label": parsed,
        "token_count": raw.token_count,
        "elapsed_seconds": round(raw.elapsed_seconds, 6),
        "error": raw.error,
        "generation_source": generation_source,
    }


def _collect_with_generation_fn(
    specs: list[dict[str, Any]],
    items: list[MicroItem],
    generation_fn: GenerationFn,
    *,
    generation_source: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for item in items:
            for perturbation_index in (0, 1):
                prompt = build_prompt(item, perturbation_index)
                try:
                    raw = generation_fn(spec, item, perturbation_index, prompt, max_tokens)
                except Exception as exc:
                    raw = RawGeneration("", 0, error=f"{type(exc).__name__}: {exc}")
                rows.append(
                    _response_row(spec, item, perturbation_index, prompt, raw, generation_source)
                )
    return rows


def _collect_with_llama(
    specs: list[dict[str, Any]],
    items: list[MicroItem],
    *,
    llama_class: type[Any],
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        llm: Any | None = None
        try:
            llm = llama_class(
                model_path=spec["model_path"],
                n_gpu_layers=-1,
                n_ctx=384,
                seed=1311,
                main_gpu=int(spec["gpu"]),
                verbose=False,
            )
            for item in items:
                for perturbation_index in (0, 1):
                    prompt = build_prompt(item, perturbation_index)
                    started = time.monotonic()
                    result = llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        echo=False,
                        stop=["\n", "</s>", "<eos>"],
                    )
                    elapsed = max(time.monotonic() - started, 0.0)
                    text = _completion_text(result)
                    token_count = _completion_token_count(result, text, llm)
                    raw = RawGeneration(text=text, token_count=token_count, elapsed_seconds=elapsed)
                    rows.append(
                        _response_row(
                            spec,
                            item,
                            perturbation_index,
                            prompt,
                            raw,
                            "live_sota_llamacpp",
                        )
                    )
        except Exception as exc:
            for item in items:
                for perturbation_index in (0, 1):
                    prompt = build_prompt(item, perturbation_index)
                    raw = RawGeneration("", 0, error=f"{type(exc).__name__}: {exc}")
                    rows.append(
                        _response_row(
                            spec,
                            item,
                            perturbation_index,
                            prompt,
                            raw,
                            "live_sota_llamacpp",
                        )
                    )
        finally:
            if llm is not None:
                _close_llama(llm)
    return rows


def _resolved_specs(raw_specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_specs, list) or len(raw_specs) != 2:
        return []
    resolved: list[dict[str, Any]] = []
    for spec in raw_specs:
        hf_id = spec.get("hf_id")
        model_path = spec.get("model_path")
        if hf_id not in MANDATED_HEADLINE_MODEL_IDS or not model_path:
            return []
        resolved.append(
            {
                "name": spec.get("name"),
                "hf_id": hf_id,
                "gpu": spec.get("gpu"),
                "model_path": str(model_path),
            }
        )
    return resolved


def _base_artifact(
    *, project_root: Path, run_date: str, status: str = "in_progress"
) -> dict[str, Any]:
    return {
        "artifact": "experiment_1311_sota_constraintbench_satquest_answer_stability",
        "schema_version": 1,
        "run_date": run_date,
        "status": status,
        "answer_stability_score": None,
        "cross_model_disagreement_rate": None,
        "constraintbench_items": 0,
        "satquest_items": 0,
        "pysat_verified_rate": None,
        "feasibility_rate": None,
        "unknown_or_abstain_rate": None,
        "headline_result_allowed": False,
        "models_used": [],
        "honest_verdict": "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "gpu_indices": [0, 1],
            "preferred_quant": "Q4_K_M",
            "items_requested": "deterministic_local_micro_slice",
        },
        "mandated_headline_model_ids": list(MANDATED_HEADLINE_MODEL_IDS),
        "resolved_model_specs": [],
        "responses": [],
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    if not counts:
        return "ABSTAIN"
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else "ABSTAIN"


def _attach_verification(
    rows: list[dict[str, Any]],
    items: list[MicroItem],
    *,
    verifier_backend: str,
) -> None:
    item_by_id = {item.item_id: item for item in items}
    for row in rows:
        result = verify_item_label(
            item_by_id[str(row["item_id"])],
            str(row["parsed_label"]),
            backend=verifier_backend,
        )
        row["verifier_label"] = result.verifier_label
        row["verified"] = result.verified
        row["feasible"] = result.feasible
        row["verifier_backend"] = result.verifier_backend


def _metric_payload(
    rows: list[dict[str, Any]],
    items: list[MicroItem],
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (str(row["hf_id"]), str(row["item_id"]))
        groups.setdefault(key, []).append(str(row["parsed_label"]))
    stable_count = sum(1 for labels in groups.values() if len(set(labels)) == 1)

    disagreements = 0
    meaningful_disagreements = 0
    for item in items:
        labels_by_model = []
        for spec in specs:
            labels = groups.get((str(spec["hf_id"]), item.item_id), [])
            labels_by_model.append(_majority_label(labels))
        if len(set(labels_by_model)) > 1:
            disagreements += 1
            if all(label in {"SAT", "UNSAT"} for label in labels_by_model):
                meaningful_disagreements += 1

    total = len(rows)
    verified_count = sum(1 for row in rows if row.get("verified") is True)
    unknown_or_abstain = sum(1 for row in rows if row.get("parsed_label") in {"UNKNOWN", "ABSTAIN"})
    sat_opportunities = [row for row in rows if row.get("verifier_label") == "SAT"]
    feasible_hits = sum(1 for row in sat_opportunities if row.get("verified") is True)

    return {
        "answer_stability_score": _rate(stable_count, len(groups)),
        "cross_model_disagreement_rate": _rate(disagreements, len(items)),
        "meaningful_disagreement_rate": _rate(meaningful_disagreements, len(items)),
        "pysat_verified_rate": _rate(verified_count, total),
        "feasibility_rate": _rate(feasible_hits, len(sat_opportunities)),
        "unknown_or_abstain_rate": _rate(unknown_or_abstain, total),
        "stable_model_item_pairs": stable_count,
        "model_item_pairs": len(groups),
        "cross_model_disagreement_items": disagreements,
        "meaningful_disagreement_items": meaningful_disagreements,
        "verification_evaluable_count": total,
    }


def _blocked_artifact(
    artifact: dict[str, Any],
    *,
    reason: str,
    verdict: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact.update(
        {
            "status": "blocked",
            "blocked_reason": reason,
            "headline_result_allowed": False,
            "honest_verdict": verdict,
        }
    )
    if extra:
        artifact.update(extra)
    return artifact


def build_answer_stability_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "injected",
    items: list[MicroItem] | None = None,
    max_tokens: int = 6,
    verifier_backend: str = "auto",
) -> dict[str, Any]:
    """Build the Exp 1311 artifact, running live SOTA generation only when available."""
    root = Path(project_root)
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")

    try:
        raw_specs = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:
        return _blocked_artifact(
            artifact,
            reason="cached_sota_pair_exception",
            verdict="blocked_cached_sota_pair_exception",
            extra={"cached_sota_pair_error": f"{type(exc).__name__}: {exc}"},
        )

    specs = _resolved_specs(raw_specs)
    if not specs:
        return _blocked_artifact(
            artifact,
            reason="cached_sota_pair_not_loadable",
            verdict="blocked_cached_sota_pair_not_loadable",
        )

    fixture_items = list(items) if items is not None else build_micro_slice()
    artifact["resolved_model_specs"] = specs
    artifact["models_used"] = [str(spec["hf_id"]) for spec in specs]
    artifact["constraintbench_items"] = sum(
        item.family == "constraintbench" for item in fixture_items
    )
    artifact["satquest_items"] = sum(item.family == "satquest" for item in fixture_items)
    artifact["compact_encoding_items"] = sum(item.compact_encoding for item in fixture_items)
    artifact["item_count"] = len(fixture_items)
    artifact["perturbations_per_model_item"] = 2

    if generation_fn is None:
        import_ok, llama_class, import_error = llama_importer()
        artifact["llama_cpp_import_ok"] = import_ok
        artifact["llama_cpp_import_error"] = import_error
        if not import_ok or llama_class is None:
            return _blocked_artifact(
                artifact,
                reason="llama_cpp_import_failed",
                verdict="blocked_llama_cpp_import_failed",
                extra={"llama_cpp_import_error": import_error},
            )
        rows = _collect_with_llama(
            specs, fixture_items, llama_class=llama_class, max_tokens=max_tokens
        )
    else:
        artifact["llama_cpp_import_ok"] = None
        artifact["llama_cpp_import_error"] = None
        rows = _collect_with_generation_fn(
            specs,
            fixture_items,
            generation_fn,
            generation_source=generation_source,
            max_tokens=max_tokens,
        )

    _attach_verification(rows, fixture_items, verifier_backend=verifier_backend)
    artifact["responses"] = rows
    artifact.update(_metric_payload(rows, fixture_items, specs))

    expected_response_count = len(specs) * len(fixture_items) * 2
    generation_errors = sum(1 for row in rows if row.get("error"))
    live_generation = rows and all(
        row.get("generation_source") == "live_sota_llamacpp" for row in rows
    )
    artifact["generation_errors"] = generation_errors
    artifact["expected_response_count"] = expected_response_count
    artifact["observed_response_count"] = len(rows)
    if rows and generation_errors == len(rows):
        return _blocked_artifact(
            artifact,
            reason="sota_generation_failed",
            verdict="blocked_sota_generation_failed",
        )
    artifact["headline_result_allowed"] = (
        live_generation
        and len(rows) == expected_response_count
        and generation_errors == 0
        and all(spec["hf_id"] in MANDATED_HEADLINE_MODEL_IDS for spec in specs)
    )
    artifact["honest_verdict"] = (
        "sota_constraint_satquest_stability_audit_complete"
        if artifact["headline_result_allowed"]
        else "non_headline_constraint_satquest_stability_audit_complete"
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "injected",
    items: list[MicroItem] | None = None,
) -> dict[str, Any]:
    """Write the in-progress marker and final Exp 1311 artifact."""
    root = Path(project_root)
    out = Path(output_path)
    _write_json(out, _base_artifact(project_root=root, run_date=run_date))
    artifact = build_answer_stability_artifact(
        project_root=root,
        run_date=run_date,
        cached_pair_fn=cached_pair_fn,
        llama_importer=llama_importer,
        generation_fn=generation_fn,
        generation_source=generation_source,
        items=items,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover - covered through run_experiment injection tests.
    run_experiment(project_root=Path.cwd(), run_date=DEFAULT_RUN_DATE)


if __name__ == "__main__":  # pragma: no cover
    main()
