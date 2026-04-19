#!/usr/bin/env python3
"""Experiment 244: formal claim corpus from live semantic and prompt-side traces.

Writes:
- ``data/research/formal_claim_corpus_244.jsonl``
- ``results/experiment_244_results.json``

Spec: REQ-VERIFY-056, REQ-VERIFY-057,
SCENARIO-VERIFY-063, SCENARIO-VERIFY-064
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

RUN_DATE = "20260413"
EXPERIMENT_LABEL = "Exp 244"
SCHEMA_VERSION = "carnot.formal_claim_corpus.v1"
SOURCE_221 = Path("results/experiment_221_results.json")
SOURCE_235 = Path("results/experiment_235_results.json")
SOURCE_211 = Path("data/research/constraint_ir_benchmark_211.jsonl")
SOURCE_214 = Path("data/research/semantic_failure_corpus_214.jsonl")
STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "as",
    "at",
    "by",
    "for",
    "has",
    "have",
    "how",
    "if",
    "in",
    "is",
    "of",
    "or",
    "per",
    "the",
    "to",
    "total",
    "with",
}


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
CORPUS_PATH = REPO_ROOT / "data" / "research" / "formal_claim_corpus_244.jsonl"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_244_results.json"


def resolve_path(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if isinstance(row, dict)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _parse_number(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    if (
        "/" in token
        and token.count("/") == 1
        and all(part.strip("-").isdigit() for part in token.split("/"))
    ):
        numerator, denominator = token.split("/", maxsplit=1)
        if denominator == "0":
            return None
        return round(int(numerator) / int(denominator), 6)
    try:
        return round(float(token), 6)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text):
        parsed = _parse_number(match)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _extract_identifiers(text: str) -> list[str]:
    identifiers = [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if token.lower() not in STOPWORDS
    ]
    return _ordered_unique(identifiers)


def _safe_eval_expression(expression: str) -> float | None:
    sanitized = expression.strip()
    if not sanitized:
        return None
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", sanitized):
        return None
    try:
        parsed = ast.parse(sanitized, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div),
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("unsupported expression")

    try:
        return round(_eval(parsed), 6)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _equation_details(text: str) -> dict[str, Any] | None:
    if "=" not in text:
        return None
    match = re.search(
        r"(?P<lhs>[\d\.\s\+\-\*\/\(\)]+)\s*=\s*(?P<rhs>-?\d+(?:\.\d+)?(?:/\d+)?)",
        text,
    )
    if match is None:
        return None
    lhs = match.group("lhs").strip()
    rhs = _parse_number(match.group("rhs"))
    if rhs is None:
        return None
    computed = _safe_eval_expression(lhs)
    if computed is None:
        return None
    return {
        "relation_type": "equation",
        "candidate_solver_route": "arithmetic",
        "operands": _extract_numbers(lhs) + [rhs],
        "bound_variables": _extract_identifiers(text),
        "target": "final_answer" if "answer" in text.lower() else "derived_quantity",
        "formalization_status": "formalized",
        "equation_holds": abs(computed - rhs) < 1e-6,
    }


def _comparison_details(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    for phrase, relation in (
        ("greater than", "greater_than"),
        ("less than", "less_than"),
        ("more than", "greater_than"),
        ("fewer than", "less_than"),
        ("at least", "greater_or_equal"),
        ("at most", "less_or_equal"),
    ):
        if phrase not in lowered:
            continue
        return {
            "relation_type": relation,
            "candidate_solver_route": "comparison",
            "operands": _extract_numbers(text),
            "bound_variables": _extract_identifiers(text),
            "target": _extract_identifiers(text)[0]
            if _extract_identifiers(text)
            else "comparison_target",
            "formalization_status": "formalized",
        }
    return None


def _attribute_details(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"(?P<subject>[A-Za-z][A-Za-z' _-]+?)\s+"
        r"(?:has|have|had|is|are|eats|eat|costs|cost|contains|contained|"
        r"traveled|travelled|saw|see|waited|wait|plays|play)\b.*?"
        r"(?P<number>-?\d+(?:\.\d+)?(?:/\d+)?)",
        text,
    )
    if match is None:
        return None
    number = _parse_number(match.group("number"))
    if number is None:
        return None
    subject = slugify(match.group("subject"))
    return {
        "relation_type": "attribute_equals",
        "candidate_solver_route": "arithmetic",
        "operands": [number],
        "bound_variables": [subject],
        "target": subject,
        "formalization_status": "formalized",
    }


def normalize_semantic_claim(text: str, *, is_final: bool) -> dict[str, Any]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return {
            "relation_type": "empty_claim",
            "candidate_solver_route": "not_formalizable",
            "operands": [],
            "bound_variables": [],
            "target": "unknown",
            "formalization_status": "not_formalizable",
        }
    for candidate in (
        _equation_details(cleaned),
        _comparison_details(cleaned),
        _attribute_details(cleaned),
    ):
        if candidate is not None:
            return candidate
    target = (
        "final_answer"
        if is_final or cleaned.lower().startswith("answer:")
        else "unknown_claim_target"
    )
    return {
        "relation_type": "answer_binding" if target == "final_answer" else "unparsed_claim",
        "candidate_solver_route": "not_formalizable",
        "operands": _extract_numbers(cleaned),
        "bound_variables": _extract_identifiers(cleaned),
        "target": target,
        "formalization_status": "not_formalizable",
    }


def _value_is_arithmetic(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return all(isinstance(item, (int, float, str)) for item in value)
    if not isinstance(value, str):
        return False
    return bool(
        re.search(r"[\d][\s\+\-\*\/]", value) or re.search(r"[A-Za-z_]+\s*[\+\-\*\/]", value)
    )


def _constraint_route(constraint_type: str, relation: str, value: object) -> str:
    if constraint_type == "semantic_property":
        return "execution_oracle"
    if constraint_type in {"count_exact", "word_count_range"} or relation == "between":
        return "cardinality"
    if constraint_type in {
        "json_exact_keys",
        "yaml_exact_keys",
        "no_extra_keys",
        "enum_membership",
        "must_include_token",
        "forbidden_token",
    } or relation in {"contains", "not_contains", "in", "subset_equals"}:
        return "set_membership"
    if _value_is_arithmetic(value) or constraint_type in {
        "base_rate_binding",
        "count_expansion",
        "derived_quantity",
        "derived_value",
        "discounted_total",
        "equal_partition",
        "equal_split",
        "final_answer_binding",
        "net_difference",
        "semantic_modifier",
    }:
        return "arithmetic"
    return "boolean_entailment"


def normalize_prompt_constraint(constraint: dict[str, Any]) -> dict[str, Any]:
    relation = str(constraint.get("relation") or "equals")
    value = constraint.get("value")
    return {
        "relation_type": relation,
        "candidate_solver_route": _constraint_route(
            str(constraint.get("type") or ""),
            relation,
            value,
        ),
        "operands": _extract_numbers(json.dumps(value))
        if isinstance(value, (list, dict))
        else _extract_numbers(str(value)),
        "bound_variables": _extract_identifiers(str(value)),
        "target": str(constraint.get("target") or "unknown_target"),
        "formalization_status": "formalized",
    }


def _claim_text_from_constraint(constraint: dict[str, Any]) -> str:
    return (
        f"{constraint.get('target', 'target')} "
        f"{constraint.get('relation', 'equals')} "
        f"{constraint.get('value')}"
    )


def _gold_verdict_from_status(status: str) -> str:
    if status in {"satisfied", "supported"}:
        return "supported"
    if status == "violated":
        return "violated"
    return "abstain"


def build_row(
    *,
    row_id: str,
    source_family: str,
    prompt: str,
    response: str,
    claim_id: str,
    claim_role: str,
    claim_text: str,
    normalized_claim: dict[str, Any],
    gold_verdict: str,
    localization: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "schema_version": SCHEMA_VERSION,
        "run_date": RUN_DATE,
        "source_family": source_family,
        "prompt": prompt,
        "response": response,
        "claim": {
            "claim_id": claim_id,
            "claim_role": claim_role,
            "claim_text": claim_text,
            "relation_type": normalized_claim["relation_type"],
            "operands": normalized_claim["operands"],
            "bound_variables": normalized_claim["bound_variables"],
            "target": normalized_claim["target"],
            "candidate_solver_route": normalized_claim["candidate_solver_route"],
            "formalization_status": normalized_claim["formalization_status"],
        },
        "gold_verdict": gold_verdict,
        "localization": localization,
        "provenance": provenance,
    }


def build_exp221_rows(
    payload: dict[str, Any],
    benchmark_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_slug = slugify(str(run.get("model_name") or "unknown-model"))
        for case in run.get("cases", []):
            evaluation = case.get("evaluation", {})
            case_id = str(case.get("case_id") or evaluation.get("example_id") or "")
            example = benchmark_by_id.get(case_id, {})
            prompt = str(example.get("prompt") or "")
            response = str(evaluation.get("raw_response") or case.get("response") or "")
            constraints = {
                str(item.get("constraint_id") or ""): item
                for item in example.get("gold_atomic_constraints", [])
                if isinstance(item, dict)
            }
            for result in evaluation.get("constraint_results", []):
                if not isinstance(result, dict):
                    continue
                claim_id = str(result.get("constraint_id") or "")
                constraint = constraints.get(
                    claim_id, {"constraint_id": claim_id, "type": result.get("type")}
                )
                normalized = normalize_prompt_constraint(constraint)
                row = build_row(
                    row_id=f"exp244-exp221-{model_slug}-{case_id}-{claim_id}",
                    source_family="prompt_side_live_trace",
                    prompt=prompt,
                    response=response,
                    claim_id=claim_id,
                    claim_role="prompt_constraint",
                    claim_text=_claim_text_from_constraint(constraint),
                    normalized_claim=normalized,
                    gold_verdict=_gold_verdict_from_status(str(result.get("status") or "")),
                    localization={
                        "seed_constraint_ids": [claim_id],
                        "depends_on": list(constraint.get("depends_on", []))
                        if isinstance(constraint.get("depends_on"), list)
                        else [],
                        "taxonomy_hint": str(result.get("family") or "none"),
                    },
                    provenance={
                        "source_artifact": "results/experiment_221_results.json",
                        "source_run_date": str(payload.get("run_date") or ""),
                        "source_experiment": 221,
                        "source_case_id": case_id,
                        "source_claim_ref": claim_id,
                        "model_name": str(run.get("model_name") or ""),
                        "model_hf_id": str(run.get("model_hf_id") or ""),
                        "mode": "verify_only",
                        "response_mode": str(case.get("response_mode") or ""),
                        "benchmark_artifact": "data/research/constraint_ir_benchmark_211.jsonl",
                        "benchmark_example_id": str(example.get("example_id") or case_id),
                        "source_refs": list(example.get("source_refs", []))
                        if isinstance(example.get("source_refs"), list)
                        else [],
                    },
                )
                rows.append(row)
    return rows


def build_exp235_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_slug = slugify(str(run.get("model_name") or "unknown-model"))
        for case in run.get("cases", []):
            verification = case.get("verification", {})
            semantic = verification.get("semantic_verifier_v2", {})
            claims = {
                str(item.get("claim_id") or ""): item
                for item in semantic.get("claims", [])
                if isinstance(item, dict)
            }
            question = str(
                semantic.get("question_profile", {}).get("question")
                or verification.get("typed_reasoning", {}).get("question")
                or ""
            )
            for result in semantic.get("claim_results", []):
                if not isinstance(result, dict):
                    continue
                claim_id = str(result.get("claim_id") or "")
                claim = claims.get(claim_id, {})
                claim_text = str(result.get("text") or claim.get("text") or "")
                normalized = normalize_semantic_claim(
                    claim_text,
                    is_final=bool(result.get("is_final") or claim.get("is_final")),
                )
                legacy_types = [
                    str(item)
                    for item in result.get("legacy_violation_types", [])
                    if isinstance(item, str)
                ]
                row = build_row(
                    row_id=f"exp244-exp235-{model_slug}-{case.get('case_id')}-{claim_id}",
                    source_family="semantic_live_trace",
                    prompt=question,
                    response=str(case.get("response") or ""),
                    claim_id=claim_id,
                    claim_role="response_claim",
                    claim_text=claim_text,
                    normalized_claim=normalized,
                    gold_verdict=_gold_verdict_from_status(str(result.get("status") or "abstain")),
                    localization={
                        "focus_claim_id": str(semantic.get("focus_claim_id") or ""),
                        "is_focus_claim": claim_id == str(semantic.get("focus_claim_id") or ""),
                        "matched_clause_ids": list(result.get("matched_clause_ids", []))
                        if isinstance(result.get("matched_clause_ids"), list)
                        else [],
                        "missing_clause_ids": list(result.get("missing_clause_ids", []))
                        if isinstance(result.get("missing_clause_ids"), list)
                        else [],
                        "missing_target_keywords": list(result.get("missing_target_keywords", []))
                        if isinstance(result.get("missing_target_keywords"), list)
                        else [],
                        "supporting_claim_ids": list(result.get("supporting_claim_ids", []))
                        if isinstance(result.get("supporting_claim_ids"), list)
                        else [],
                        "taxonomy_hint": legacy_types[0] if legacy_types else "none",
                    },
                    provenance={
                        "source_artifact": "results/experiment_235_results.json",
                        "source_run_date": str(payload.get("run_date") or ""),
                        "source_experiment": 235,
                        "source_case_id": str(case.get("case_id") or ""),
                        "source_claim_ref": claim_id,
                        "model_name": str(run.get("model_name") or ""),
                        "model_hf_id": str(run.get("model_hf_id") or ""),
                        "mode": "verify_only",
                        "response_mode": str(case.get("response_mode") or ""),
                        "semantic_verdict": str(semantic.get("verdict") or "abstain"),
                    },
                )
                rows.append(row)
    return rows


def build_exp214_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("source_type") or "") != "live_trace":
            continue
        prompt = str(record.get("prompt") or "")
        response = str(record.get("response") or "")
        example_id = str(record.get("example_id") or "")
        for index, line in enumerate(response.splitlines(), start=1):
            claim_text = line.strip()
            if not claim_text:
                continue
            claim_id = f"cl{index}"
            normalized = normalize_semantic_claim(
                claim_text,
                is_final=claim_text.lower().startswith("answer:"),
            )
            gold_verdict = "abstain"
            if normalized["formalization_status"] == "formalized":
                gold_verdict = "supported" if normalized.get("equation_holds", True) else "violated"
            rows.append(
                build_row(
                    row_id=f"exp244-exp214-{example_id}-{claim_id}",
                    source_family="semantic_failure_live_trace",
                    prompt=prompt,
                    response=response,
                    claim_id=claim_id,
                    claim_role="response_claim",
                    claim_text=claim_text,
                    normalized_claim=normalized,
                    gold_verdict=gold_verdict,
                    localization={
                        "taxonomy_label": str(
                            record.get("gold_diagnosis", {}).get("taxonomy_label") or "unknown"
                        ),
                        "failure_mechanism": str(
                            record.get("gold_diagnosis", {}).get("failure_mechanism") or ""
                        ),
                        "expected_verifier_path": str(
                            record.get("expected_verifier_signal", {}).get("verifier_path") or ""
                        ),
                    },
                    provenance={
                        "source_artifact": "data/research/semantic_failure_corpus_214.jsonl",
                        "source_run_date": "20260412",
                        "source_experiment": 214,
                        "source_case_id": example_id,
                        "source_claim_ref": claim_id,
                        "diagnosis_source_artifact": str(record.get("source_artifact") or ""),
                        "source_refs": list(record.get("source_refs", []))
                        if isinstance(record.get("source_refs"), list)
                        else [],
                    },
                )
            )
    return rows


def build_corpus(repo_root: Path | None = None) -> list[dict[str, Any]]:
    root = repo_root or REPO_ROOT
    benchmark_by_id = {
        str(row.get("example_id") or ""): row for row in load_jsonl(resolve_path(root, SOURCE_211))
    }
    rows = (
        build_exp221_rows(load_json(resolve_path(root, SOURCE_221)), benchmark_by_id)
        + build_exp235_rows(load_json(resolve_path(root, SOURCE_235)))
        + build_exp214_rows(load_jsonl(resolve_path(root, SOURCE_214)))
    )
    return sorted(rows, key=lambda row: str(row["row_id"]))


def build_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = Counter(str(row["claim"]["candidate_solver_route"]) for row in rows)
    formalization_counts = Counter(str(row["claim"]["formalization_status"]) for row in rows)
    gold_verdict_counts = Counter(str(row["gold_verdict"]) for row in rows)
    source_breakdown = Counter(str(row["source_family"]) for row in rows)
    source_artifact_breakdown = Counter(str(row["provenance"]["source_artifact"]) for row in rows)
    formalized = formalization_counts.get("formalized", 0)
    abstain_like = formalization_counts.get("abstain", 0) + formalization_counts.get(
        "not_formalizable", 0
    )
    total = len(rows)
    return {
        "experiment": EXPERIMENT_LABEL,
        "run_date": RUN_DATE,
        "title": "Formal claim corpus from live traces",
        "summary": {
            "n_rows": total,
            "route_counts": dict(sorted(route_counts.items())),
            "formalization_status_counts": dict(sorted(formalization_counts.items())),
            "gold_verdict_counts": dict(sorted(gold_verdict_counts.items())),
            "source_breakdown": dict(sorted(source_breakdown.items())),
            "source_artifact_breakdown": dict(sorted(source_artifact_breakdown.items())),
            "formalizable_rate": round(formalized / total, 6) if total else 0.0,
            "abstain_or_not_formalizable_rate": round(abstain_like / total, 6) if total else 0.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--corpus-path", type=Path, default=CORPUS_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args([] if argv is None else argv)
    rows = build_corpus(args.repo_root)
    results = build_results(rows)
    write_jsonl(args.corpus_path, rows)
    write_json(args.results_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
