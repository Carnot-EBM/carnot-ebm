"""Exp 2892 bounded VeriCoT-style exact-frontier expansion.

This runner keeps the VeriCoT idea that reasoning steps should be formalized
before they are trusted, but it deliberately avoids live autoformalization. A
row is promoted only when local bytes already expose the premises needed for a
deterministic formal check: either an Exp 2877 exact certificate can be replayed,
or a HaluEval answer states one unambiguous year that can be checked against the
reference year grounded in the prompt.

Rows that need broad natural-language theorem proving, missing entity grounding,
or generated-answer reasoning traces remain outside the frontier.

Spec: REQ-VERIFY-2892, SCENARIO-VERIFY-2892.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2892_vericot_exact_frontier_expansion_v1.json"
EXP2877_REL_PATH = Path("results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json")
EXP2878_REL_PATH = Path("results/experiment_2878_halueval_fever_error_verifiability_v1.json")
EXP2888_REL_PATH = Path("results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json")
HALUEVAL_MANIFEST_REL_PATH = Path("data/eval_manifests/halueval_20260522.jsonl")
FEVER_MANIFEST_REL_PATH = Path("data/eval_manifests/fever_20260522.jsonl")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "vericot_frontier_ready",
    "source_artifacts",
    "selection_rule",
    "n_candidate_rows",
    "n_vericot_supported_rows",
    "n_unsupported_rows",
    "unsupported_reasons",
    "solver_backend",
    "formal_checks",
    "autoformalization_llm_called",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

SELECTION_RULE = (
    "Load Exp 2877/2878 and the optional Exp 2888 TruthfulQA taxonomy manifest. "
    "Candidate rows are the local HaluEval and FEVER manifests plus TruthfulQA "
    "taxonomy rows when present. Promote only rows whose premises can be encoded "
    "deterministically: replayable Exp 2877 exact certificates, or HaluEval "
    "year-answer rows where the prompt contains the reference year and the "
    "candidate contains exactly one year. Label=0 year rows must answer with the "
    "reference year alone; label=1 year rows are supported only when the claimed "
    "year differs from the grounded reference. All other rows remain unsupported."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix verdict grounded in source readiness and promoted checks.",
    "vericot_frontier_ready": (
        "True only when prior local evidence is ready and at least one premise-grounded "
        "formal check is replayed or built."
    ),
    "source_artifacts": "Records Exp 2877/2878, optional Exp 2888, and the local manifests read.",
    "selection_rule": "Documents the deterministic row-selection boundary.",
    "n_candidate_rows": "All HaluEval/FEVER manifest rows plus optional TruthfulQA taxonomy rows.",
    "n_vericot_supported_rows": "Rows admitted to the bounded VeriCoT frontier.",
    "n_unsupported_rows": "Rows deliberately kept outside the frontier.",
    "unsupported_reasons": "Counts the exact reason unsupported rows were not promoted.",
    "solver_backend": "Identifies Z3 plus local anchor checks used for formal checks.",
    "formal_checks": "Per-supported-row premises, logical steps, solver status, and checksum.",
    "autoformalization_llm_called": "Always false; no live or remote LLM is used.",
    "tests_run": "Commands used to validate the module and artifact.",
    "duration_s": "Measured wall-clock runtime with no artificial delay.",
}

YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 2892 bounded local runner."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2877_path: Path | None = None
    exp2878_path: Path | None = None
    exp2888_path: Path | None = None
    halueval_manifest_path: Path | None = None
    fever_manifest_path: Path | None = None
    tests_run: tuple[str, ...] | list[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    run_date: str = RUN_DATE

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_exp2877_path(self) -> Path:
        return _resolve(self.repo_root, self.exp2877_path or EXP2877_REL_PATH)

    def resolved_exp2878_path(self) -> Path:
        return _resolve(self.repo_root, self.exp2878_path or EXP2878_REL_PATH)

    def resolved_exp2888_path(self) -> Path:
        return _resolve(self.repo_root, self.exp2888_path or EXP2888_REL_PATH)

    def resolved_halueval_manifest_path(self) -> Path:
        return _resolve(self.repo_root, self.halueval_manifest_path or HALUEVAL_MANIFEST_REL_PATH)

    def resolved_fever_manifest_path(self) -> Path:
        return _resolve(self.repo_root, self.fever_manifest_path or FEVER_MANIFEST_REL_PATH)


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Build and optionally write the Exp 2892 VeriCoT-frontier artifact."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    exp2877 = _load_json(cfg.resolved_exp2877_path())
    exp2878 = _load_json(cfg.resolved_exp2878_path())
    exp2888 = _load_json(cfg.resolved_exp2888_path())
    halueval_rows = _load_manifest_rows(cfg.resolved_halueval_manifest_path(), "halueval")
    fever_rows = _load_manifest_rows(cfg.resolved_fever_manifest_path(), "fever")
    truthfulqa_rows = _truthfulqa_rows(exp2888)
    manifest_rows = [*halueval_rows, *fever_rows]
    rows_by_id = {_stable_id(row): row for row in manifest_rows if _stable_id(row)}
    formal_checks: list[dict[str, Any]] = []
    supported_ids: set[str] = set()

    for certificate in exp2877.get("certificates", []):
        if not isinstance(certificate, dict):
            continue
        check = _build_certificate_check(certificate, rows_by_id)
        if check is None:
            continue
        formal_checks.append(check)
        supported_ids.add(check["stable_id"])

    for row in halueval_rows:
        stable_id = _stable_id(row)
        if stable_id in supported_ids:
            continue
        check = _build_halueval_year_check(row)
        if check is None:
            continue
        formal_checks.append(check)
        supported_ids.add(stable_id)

    unsupported_reasons: Counter[str] = Counter()
    for row in manifest_rows:
        if _stable_id(row) not in supported_ids:
            unsupported_reasons[_unsupported_manifest_reason(row)] += 1
    for _row in truthfulqa_rows:
        unsupported_reasons["unsupported_truthfulqa_taxonomy_has_no_logical_steps"] += 1

    source_artifacts = _source_artifacts(cfg, include_truthfulqa=bool(exp2888))
    source_ready = bool(
        exp2877.get("frontier_expansion_ready") and exp2878.get("error_verifiability_ready")
    )
    ready = bool(source_ready and formal_checks)
    artifact = {
        "honest_verdict": (
            "complete: deterministic VeriCoT frontier rows available"
            if ready
            else "complete: no deterministic VeriCoT frontier rows"
        ),
        "vericot_frontier_ready": ready,
        "source_artifacts": source_artifacts,
        "selection_rule": SELECTION_RULE,
        "n_candidate_rows": len(manifest_rows) + len(truthfulqa_rows),
        "n_vericot_supported_rows": len(formal_checks),
        "n_unsupported_rows": len(manifest_rows) + len(truthfulqa_rows) - len(formal_checks),
        "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
        "solver_backend": solver_backend(),
        "formal_checks": formal_checks,
        "autoformalization_llm_called": False,
        "tests_run": list(cfg.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": cfg.run_date,
        "duration_s": max(0.0, cfg.clock() - started),
    }
    validate_artifact(artifact)
    if write:
        write_artifact(cfg.resolved_output_path(), artifact)
    return artifact


def solver_backend() -> str:
    """Return the exact deterministic solver identity for this runner."""

    return f"z3-solver {z3.get_version_string()} + deterministic premise anchors"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required schema fields and supported/unsupported accounting."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260523")
    if artifact["autoformalization_llm_called"] is not False:
        raise ValueError("autoformalization_llm_called must be false")
    if not isinstance(artifact["source_artifacts"], list):
        raise ValueError("source_artifacts must be a list")
    if int(artifact["n_candidate_rows"]) != int(artifact["n_vericot_supported_rows"]) + int(
        artifact["n_unsupported_rows"]
    ):
        raise ValueError("candidate count must equal supported plus unsupported")
    if len(artifact["formal_checks"]) != int(artifact["n_vericot_supported_rows"]):
        raise ValueError("formal_checks count must equal n_vericot_supported_rows")
    if sum(int(value) for value in artifact["unsupported_reasons"].values()) != int(
        artifact["n_unsupported_rows"]
    ):
        raise ValueError("unsupported_reasons must sum to n_unsupported_rows")


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Persist a stable JSON artifact for the conductor."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_certificate_check(
    certificate: Mapping[str, Any],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    stable_id = str(certificate.get("stable_id") or "")
    row = rows_by_id.get(stable_id)
    if not row:
        return None
    constraint_type = str(certificate.get("constraint_type") or "")
    constraints = dict(certificate.get("constraints") or {})
    expected_status = str(
        certificate.get("expected_solver_status") or certificate.get("solver_status") or ""
    )
    if constraint_type in {"safe_prefix_year_answer", "year_contradiction"}:
        candidate_year = _coerce_int(constraints.get("candidate_year"))
        expected_year = _coerce_int(constraints.get("expected_year"))
        if candidate_year is None or expected_year is None:
            return None
        status = _z3_status(z3.IntVal(candidate_year) == z3.IntVal(expected_year))
        check = _base_check(
            row=row,
            check_type=constraint_type,
            source="exp2877_certificate_replay",
            solver_status=status,
            expected_status=expected_status,
            premises=[
                {"name": "prompt_grounded_expected_year", "value": expected_year},
                {"name": "candidate_claimed_year", "value": candidate_year},
            ],
            logical_steps=["candidate_year == expected_year"],
            z3_expression=f"{candidate_year} == {expected_year}",
            vericot_verdict=str(certificate.get("exact_verdict") or ""),
        )
    elif constraint_type == "arithmetic_like_date_order":
        claimed_year = _coerce_int(constraints.get("claimed_year"))
        comparison_year = _coerce_int(constraints.get("comparison_year"))
        if claimed_year is None or comparison_year is None:
            return None
        status = _z3_status(z3.IntVal(claimed_year) < z3.IntVal(comparison_year))
        check = _base_check(
            row=row,
            check_type=constraint_type,
            source="exp2877_certificate_replay",
            solver_status=status,
            expected_status=expected_status,
            premises=[
                {
                    "name": "claimed_first_year",
                    "entity": constraints.get("claimed_first"),
                    "value": claimed_year,
                },
                {
                    "name": "comparison_entity_year",
                    "entity": constraints.get("compare_against"),
                    "value": comparison_year,
                },
            ],
            logical_steps=["claimed_year < comparison_year"],
            z3_expression=f"{claimed_year} < {comparison_year}",
            vericot_verdict=str(certificate.get("exact_verdict") or ""),
        )
    elif constraint_type in {"anchored_entailment", "existential_entailment"}:
        evidence = dict(certificate.get("evidence") or {})
        prompt_anchors = [str(anchor) for anchor in evidence.get("prompt_anchors") or []]
        candidate_anchors = [str(anchor) for anchor in evidence.get("candidate_anchors") or []]
        candidate_text = _candidate_text(row)
        prompt_text = str(row.get("prompt") or "")
        if not prompt_anchors or not candidate_anchors:
            return None
        anchors_present = all(anchor in prompt_text for anchor in prompt_anchors) and all(
            anchor in candidate_text for anchor in candidate_anchors
        )
        status = _z3_status(z3.BoolVal(anchors_present))
        check = _base_check(
            row=row,
            check_type=constraint_type,
            source="exp2877_certificate_replay",
            solver_status=status,
            expected_status=expected_status,
            premises=[
                {"name": "prompt_anchors", "anchors": prompt_anchors},
                {"name": "candidate_anchors", "anchors": candidate_anchors},
            ],
            logical_steps=["all_prompt_and_candidate_anchors_present"],
            z3_expression=f"anchor_presence == {str(anchors_present).lower()}",
            vericot_verdict=str(certificate.get("exact_verdict") or ""),
        )
    else:
        return None
    if check["solver_status"] != check["expected_solver_status"]:
        return None
    return _with_check_hash(check)


def _build_halueval_year_check(row: Mapping[str, Any]) -> dict[str, Any] | None:
    reference = str(row.get("reference") or "").strip()
    if not YEAR_RE.fullmatch(reference):
        return None
    prompt = str(row.get("prompt") or "")
    if reference not in prompt:
        return None
    candidate = _candidate_text(row)
    candidate_years = _years(candidate)
    if len(candidate_years) != 1:
        return None
    expected_year = int(reference)
    candidate_year = candidate_years[0]
    label = _coerce_label(row.get("label"))
    if label == 0 and candidate.strip() != reference:
        return None
    if label == 1 and candidate_year == expected_year:
        return None
    if label not in {0, 1}:
        return None
    status = _z3_status(z3.IntVal(candidate_year) == z3.IntVal(expected_year))
    expected_status = "sat" if label == 0 else "unsat"
    check = _base_check(
        row=row,
        check_type="halueval_year_answer",
        source="deterministic_manifest_year_parser",
        solver_status=status,
        expected_status=expected_status,
        premises=[
            {"name": "prompt_grounded_expected_year", "anchor": reference, "value": expected_year},
            {
                "name": "candidate_claimed_year",
                "anchor": str(candidate_year),
                "value": candidate_year,
            },
        ],
        logical_steps=["candidate_year == expected_year"],
        z3_expression=f"{candidate_year} == {expected_year}",
        vericot_verdict="safe_prefix_supported" if label == 0 else "contradiction_verified",
    )
    return _with_check_hash(check)


def _unsupported_manifest_reason(row: Mapping[str, Any]) -> str:
    dataset_key = str(row.get("_dataset_key") or row.get("dataset") or "").lower()
    if dataset_key == "halueval":
        reference = str(row.get("reference") or "").strip()
        candidate_years = _years(_candidate_text(row))
        label = _coerce_label(row.get("label"))
        if (
            YEAR_RE.fullmatch(reference)
            and reference in str(row.get("prompt") or "")
            and len(candidate_years) == 1
            and label == 1
            and candidate_years[0] == int(reference)
        ):
            return "unsupported_year_only_does_not_establish_entity_grounding"
    return "unsupported_no_deterministic_vericot_template"


def _base_check(
    *,
    row: Mapping[str, Any],
    check_type: str,
    source: str,
    solver_status: str,
    expected_status: str,
    premises: list[dict[str, Any]],
    logical_steps: list[str],
    z3_expression: str,
    vericot_verdict: str,
) -> dict[str, Any]:
    return {
        "stable_id": _stable_id(row),
        "dataset": str(row.get("dataset") or row.get("_dataset_key") or ""),
        "label": _coerce_label(row.get("label")),
        "check_type": check_type,
        "source": source,
        "premise_grounded": True,
        "premises": premises,
        "logical_steps": logical_steps,
        "z3_expression": z3_expression,
        "solver_status": solver_status,
        "expected_solver_status": expected_status,
        "solver_backend": solver_backend(),
        "vericot_verdict": vericot_verdict,
        "candidate_or_claim": _candidate_text(row),
    }


def _with_check_hash(check: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(check)
    payload["formal_check_sha256"] = _checksum(payload)
    return payload


def _load_manifest_rows(path: Path, dataset_key: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path) if path.is_file() else []
    for row in rows:
        row["_dataset_key"] = dataset_key
    return rows


def _truthfulqa_rows(exp2888: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not exp2888.get("truthfulqa_taxonomy_ready"):
        return []
    return [dict(row) for row in exp2888.get("materialized_rows", []) if isinstance(row, dict)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    loaded.append(payload)
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_artifacts(config: ExperimentConfig, *, include_truthfulqa: bool) -> list[str]:
    artifacts = [str(EXP2877_REL_PATH), str(EXP2878_REL_PATH)]
    if include_truthfulqa:
        artifacts.append(str(EXP2888_REL_PATH))
    artifacts.extend([str(HALUEVAL_MANIFEST_REL_PATH), str(FEVER_MANIFEST_REL_PATH)])
    return artifacts


def _candidate_text(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate") or row.get("claim") or row.get("best_answer") or "").strip()


def _stable_id(row: Mapping[str, Any]) -> str:
    return str(row.get("stable_id") or "")


def _years(text: str) -> list[int]:
    return [int(match.group(0)) for match in YEAR_RE.finditer(text)]


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _coerce_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip()
    return int(text) if text in {"0", "1"} else None


def _z3_status(expression: Any) -> str:
    solver = z3.Solver()
    solver.add(expression)
    return str(solver.check())


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - command wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


if __name__ == "__main__":  # pragma: no cover - command wrapper.
    raise SystemExit(main())


__all__ = [
    "EXP2877_REL_PATH",
    "EXP2878_REL_PATH",
    "EXP2888_REL_PATH",
    "ExperimentConfig",
    "FIELD_PRINCIPLES",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "SELECTION_RULE",
    "main",
    "run_experiment",
    "solver_backend",
    "validate_artifact",
    "write_artifact",
]
