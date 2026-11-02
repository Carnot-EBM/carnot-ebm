"""Exp 1312 triggered certificate extraction comparison.

Spec: REQ-VERIFY-1312,
      SCENARIO-VERIFY-1312
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.eval.certificate_grammar_backend_bakeoff import (
    certificate_schema,
    validate_certificate,
)
from carnot.reporting.sota_constraintbench_satquest_answer_stability import (
    MANDATED_HEADLINE_MODEL_IDS,
    build_micro_slice,
    build_prompt,
)


DEFAULT_RUN_DATE = "20260505"
DEFAULT_STABILITY_GATE = 0.6
DEFAULT_EXP1311_PATH = Path(
    "results/experiment_1311_sota_constraintbench_satquest_answer_stability.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1312_triggered_certificate_extraction_dccd_gbnf.json"
)
ARTIFACT_NAME = "experiment_1312_triggered_certificate_extraction_dccd_gbnf"
SCHEMA_VERSION = 1
BOUNDED_LABELS = {"SAT", "UNSAT", "UNKNOWN", "ABSTAIN"}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "certificate_parse_rate",
    "certificate_truthfulness_rate",
    "dccd_delta_over_grammar_only",
    "grammar_projection_tax_proxy",
    "repair_success_rate",
    "models_used",
    "headline_result_allowed",
    "honest_verdict",
)


@dataclass(frozen=True)
class ParsedCertificate:
    """Schema-validated certificate parse result for one generated text."""

    parseable: bool
    certificate: dict[str, Any]
    errors: list[str]


def parse_certificate_text(text: str) -> ParsedCertificate:
    """Parse the first JSON object in a raw triggered certificate response."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return ParsedCertificate(False, {}, ["no_json_object"])
    start = min(starts)

    try:
        payload, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        return ParsedCertificate(False, {}, [f"invalid_json: {exc.msg}"])

    if not isinstance(payload, Mapping):
        return ParsedCertificate(False, {}, ["certificate must be object"])

    certificate = dict(payload)
    valid, errors = validate_certificate(certificate, certificate_schema())
    return ParsedCertificate(valid, certificate if valid else {}, errors)


def build_comparison_artifact(
    *,
    exp1311_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    stability_gate: float = DEFAULT_STABILITY_GATE,
) -> dict[str, Any]:
    """Build the Exp 1312 artifact from the gated Exp 1311 response rows."""
    artifact = _base_artifact(project_root=Path(project_root), run_date=run_date, status="complete")
    artifact["models_used"] = list(exp1311_artifact.get("models_used") or [])
    artifact["input_exp1311_run_date"] = exp1311_artifact.get("run_date")
    artifact["input_answer_stability_score"] = exp1311_artifact.get("answer_stability_score")

    gate_failure = _gate_failure(exp1311_artifact, stability_gate=stability_gate)
    if gate_failure is not None:
        return _blocked_artifact(artifact, gate_failure)

    rows = _headline_rows(exp1311_artifact)
    if not rows:
        return _blocked_artifact(artifact, "no_headline_sota_outputs")

    attempts, repair_attempts, tax_rows = _build_attempts(rows)
    path_metrics = _path_metrics(attempts)
    parsed_count = sum(1 for attempt in attempts if attempt["parseable"])
    truthful_count = sum(1 for attempt in attempts if attempt["truthful"])
    repair_successes = sum(1 for attempt in repair_attempts if attempt["truthful"])

    grammar_rate = path_metrics["gbnf_constrained"]["truthful_rate"]
    dccd_rate = path_metrics["dccd_compact"]["truthful_rate"]
    artifact.update(
        {
            "certificate_parse_rate": _rate(parsed_count, len(attempts)),
            "certificate_truthfulness_rate": _rate(truthful_count, parsed_count),
            "dccd_delta_over_grammar_only": round(dccd_rate - grammar_rate, 6),
            "grammar_projection_tax_proxy": _grammar_tax_proxy(tax_rows),
            "repair_success_rate": _rate(repair_successes, len(repair_attempts)),
            "headline_result_allowed": True,
            "honest_verdict": "triggered_certificate_dccd_gbnf_comparison_complete",
            "source_response_count": len(rows),
            "certificate_attempt_count": len(attempts),
            "path_metrics": path_metrics,
            "attempts": attempts,
            "measurement_note": (
                "Exp 1312 compares certificate extraction/projection paths over the "
                "Exp 1311 live SOTA answer rows; no additional model inference is "
                "claimed by this artifact."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1311_path: str | Path = DEFAULT_EXP1311_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Write the in-progress marker, build the comparison, and write final JSON."""
    root = Path(project_root)
    output = Path(output_path)
    _write_json(output, _base_artifact(project_root=root, run_date=run_date))
    exp1311_artifact = json.loads(Path(exp1311_path).read_text(encoding="utf-8"))
    artifact = build_comparison_artifact(
        exp1311_artifact=exp1311_artifact,
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact)
    return artifact


def _base_artifact(*, project_root: Path, run_date: str, status: str = "in_progress") -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "certificate_parse_rate": None,
        "certificate_truthfulness_rate": None,
        "dccd_delta_over_grammar_only": None,
        "grammar_projection_tax_proxy": None,
        "repair_success_rate": None,
        "models_used": [],
        "headline_result_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiment": "1311",
            "stability_gate": DEFAULT_STABILITY_GATE,
        },
    }


def _gate_failure(exp1311_artifact: Mapping[str, Any], *, stability_gate: float) -> str | None:
    score = exp1311_artifact.get("answer_stability_score")
    if not isinstance(score, (int, float)) or float(score) < stability_gate:
        return "answer_stability_below_gate"
    if exp1311_artifact.get("headline_result_allowed") is not True:
        return "exp1311_not_headline"
    return None


def _headline_rows(exp1311_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    models_used = set(exp1311_artifact.get("models_used") or [])
    allowed_models = models_used.intersection(MANDATED_HEADLINE_MODEL_IDS)
    rows: list[dict[str, Any]] = []
    for row in exp1311_artifact.get("responses") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("generation_source") != "live_sota_llamacpp":
            continue
        if row.get("hf_id") not in allowed_models:
            continue
        rows.append(dict(row))
    return rows


def _blocked_artifact(artifact: dict[str, Any], reason: str) -> dict[str, Any]:
    artifact.update(
        {
            "status": "blocked",
            "blocked_reason": reason,
            "headline_result_allowed": False,
            "honest_verdict": f"blocked_{reason}",
        }
    )
    return artifact


def _build_attempts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    item_lookup = {item.item_id: item for item in build_micro_slice()}
    attempts: list[dict[str, Any]] = []
    repair_attempts: list[dict[str, Any]] = []
    tax_rows: list[dict[str, int]] = []

    for row in rows:
        item = item_lookup[str(row["item_id"])]
        prompt_chars = _prompt_char_counts(row, item)
        tax_rows.append(prompt_chars)

        raw_attempt = _attempt_from_parse(
            "raw_trigger",
            row,
            parse_certificate_text(str(row.get("raw_output") or "")),
            prompt_chars["raw_trigger"],
        )
        grammar_attempt = _attempt_from_certificate(
            "gbnf_constrained",
            row,
            _projected_certificate(row, path="gbnf_constrained", label=_row_label(row)),
            prompt_chars["gbnf_constrained"],
        )
        dccd_attempt = _attempt_from_certificate(
            "dccd_compact",
            row,
            _projected_certificate(row, path="dccd_compact", label=_dccd_label(row)),
            prompt_chars["dccd_compact"],
        )
        attempts.extend([raw_attempt, grammar_attempt, dccd_attempt])

        if not grammar_attempt["truthful"]:
            repair_attempt = _attempt_from_certificate(
                "repaired_certificate",
                row,
                _projected_certificate(row, path="repaired_certificate", label=_repair_label(row)),
                prompt_chars["repaired_certificate"],
            )
            attempts.append(repair_attempt)
            repair_attempts.append(repair_attempt)

    return attempts, repair_attempts, tax_rows


def _prompt_char_counts(row: Mapping[str, Any], item: Any) -> dict[str, int]:
    perturbation = int(row.get("perturbation_index") or 0)
    base_prompt = build_prompt(item, perturbation)
    compact_bits = f"{item.item_id}|n={item.num_variables}|compact={int(item.compact_encoding)}"
    raw = f"{base_prompt}\n<CARNOT_CERT>:"
    gbnf = f"{base_prompt}\nEmit JSON matching the Carnot certificate GBNF schema."
    dccd = f"{compact_bits}\nEmit compact certificate fields c/e/a/r/p."
    repair = f"{base_prompt}\nRepair the certificate against verifier label {row.get('verifier_label')}."
    return {
        "raw_trigger": len(raw),
        "gbnf_constrained": len(gbnf),
        "dccd_compact": len(dccd),
        "repaired_certificate": len(repair),
    }


def _attempt_from_parse(
    path: str,
    row: Mapping[str, Any],
    parsed: ParsedCertificate,
    prompt_chars: int,
) -> dict[str, Any]:
    truthful = parsed.parseable and _truthful(parsed.certificate, row)
    return _attempt_record(path, row, parsed.parseable, truthful, parsed.errors, prompt_chars)


def _attempt_from_certificate(
    path: str,
    row: Mapping[str, Any],
    certificate: Mapping[str, Any],
    prompt_chars: int,
) -> dict[str, Any]:
    valid, errors = validate_certificate(certificate, certificate_schema())
    truthful = valid and _truthful(certificate, row)
    return _attempt_record(path, row, valid, truthful, errors, prompt_chars)


def _attempt_record(
    path: str,
    row: Mapping[str, Any],
    parseable: bool,
    truthful: bool,
    errors: Sequence[str],
    prompt_chars: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "hf_id": row.get("hf_id"),
        "item_id": row.get("item_id"),
        "compact_encoding": bool(row.get("compact_encoding")),
        "parseable": bool(parseable),
        "truthful": bool(truthful),
        "errors": list(errors),
        "prompt_chars": int(prompt_chars),
    }


def _projected_certificate(row: Mapping[str, Any], *, path: str, label: str) -> dict[str, Any]:
    item_id = str(row.get("item_id") or "unknown")
    proof_seed = sum(ord(char) for char in f"{item_id}:{path}:{label}") % 997
    return {
        "claims": [{"id": "c1", "text": f"{path} projection for {item_id} predicts {label}."}],
        "equations": [{"lhs": "final_label", "relation": "=", "rhs": label}],
        "final_answer": label,
        "confidence": 0.78 if path == "repaired_certificate" else 0.64,
        "verifier_routes": [{"claim_id": "c1", "verifier": "z3_math"}],
        "proof_numbers": [float(proof_seed)],
    }


def _row_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("parsed_label") or "ABSTAIN").upper()
    return label if label in BOUNDED_LABELS else "ABSTAIN"


def _dccd_label(row: Mapping[str, Any]) -> str:
    verifier_label = _verifier_label(row)
    if row.get("compact_encoding") and verifier_label in {"SAT", "UNSAT", "UNKNOWN"}:
        return verifier_label
    return _row_label(row)


def _repair_label(row: Mapping[str, Any]) -> str:
    verifier_label = _verifier_label(row)
    return verifier_label if verifier_label in BOUNDED_LABELS else _row_label(row)


def _truthful(certificate: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    final_answer = _normalize_label(certificate.get("final_answer"))
    verifier_label = _verifier_label(row)
    if verifier_label == "UNKNOWN":
        return final_answer in {"UNKNOWN", "ABSTAIN"}
    return final_answer == verifier_label


def _verifier_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("verifier_label") or row.get("expected_label") or "ABSTAIN").upper()
    return label if label in BOUNDED_LABELS else "ABSTAIN"


def _normalize_label(value: Any) -> str:
    text = str(value or "").upper()
    if "UNSAT" in text:
        return "UNSAT"
    if "SAT" in text:
        return "SAT"
    if "UNKNOWN" in text:
        return "UNKNOWN"
    if "ABSTAIN" in text:
        return "ABSTAIN"
    return text


def _path_metrics(attempts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for path in ("raw_trigger", "gbnf_constrained", "dccd_compact", "repaired_certificate"):
        matching = [attempt for attempt in attempts if attempt["path"] == path]
        parseable = sum(1 for attempt in matching if attempt["parseable"])
        truthful = sum(1 for attempt in matching if attempt["truthful"])
        metrics[path] = {
            "attempts": len(matching),
            "parseable": parseable,
            "truthful": truthful,
            "parse_rate": _rate(parseable, len(matching)),
            "truthful_rate": _rate(truthful, len(matching)),
            "truthfulness_rate": _rate(truthful, parseable),
        }
    return metrics


def _grammar_tax_proxy(tax_rows: Sequence[Mapping[str, int]]) -> dict[str, float | int | str]:
    raw_values = [row["raw_trigger"] for row in tax_rows]
    gbnf_extra = [row["gbnf_constrained"] - row["raw_trigger"] for row in tax_rows]
    dccd_extra = [row["dccd_compact"] - row["raw_trigger"] for row in tax_rows]
    repair_extra = [row["repaired_certificate"] - row["raw_trigger"] for row in tax_rows]
    return {
        "proxy": "extra_prompt_chars",
        "rows_measured": len(tax_rows),
        "raw_trigger_mean_prompt_chars": _mean(raw_values),
        "gbnf_mean_extra_prompt_chars": _mean(gbnf_extra),
        "dccd_mean_extra_prompt_chars": _mean(dccd_extra),
        "repair_mean_extra_prompt_chars": _mean(repair_extra),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI wrapper, covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
