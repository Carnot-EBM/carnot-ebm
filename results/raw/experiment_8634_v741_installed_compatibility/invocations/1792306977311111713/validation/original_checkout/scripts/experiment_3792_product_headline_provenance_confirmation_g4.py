#!/usr/bin/env python3
"""Exp 3792: confirm product-headline G4 provenance from primary artifacts.

This runner exists because the old product headline was demoted after its
numbers traced to prose rather than a primary result JSON. The task here is
narrow: read the checked-in artifacts, report whether the two surviving modest
HumanEval positives have seed/checksum provenance, and leave operator-curated
documents untouched so the operator can decide whether to restore any headline.

Spec: REQ-PUBLISH-3792, SCENARIO-PUBLISH-3792
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "results"
    / "experiment_3792_product_headline_provenance_confirmation_g4.json"
)
AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
RESTORABLE = "restorable"
RESTORABLE_WITH_CAVEAT = "restorable_with_caveat"
NOT_YET_HEADLINE_ELIGIBLE = "not_yet_headline_eligible"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _matches_expected(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12


def _extract_n(data: dict[str, Any], n_key: str | None) -> int | None:
    if n_key and isinstance(data.get(n_key), int):
        return int(data[n_key])
    for key in ("n", "sample_size", "dataset_size", "problems_count", "num_problems"):
        if isinstance(data.get(key), int):
            return int(data[key])
    text = " ".join(str(data.get(key, "")) for key in ("title", "honest_verdict"))
    match = re.search(r"\b(\d+)\s+HumanEval\b", text)
    return int(match.group(1)) if match else None


def _source_substrate(data: dict[str, Any]) -> str:
    for key in ("inference_substrate", "inference_mode"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("inference_mode")
        if isinstance(value, str) and value:
            return value
    if data.get("model_used"):
        return "model_used_field_present_redacted_for_aggregation_hygiene"
    return "not_recorded"


def _row_caveat(seed_present: bool, checksum_present: bool, numbers_match: bool) -> str:
    missing = []
    if not seed_present:
        missing.append("missing random_seed")
    if not checksum_present:
        missing.append("missing reproducibility_checksum")
    if not numbers_match:
        missing.append("number discrepancy versus north-star")
    if missing:
        return "; ".join(missing)
    return (
        "G4 seed/checksum present; still needs clean full-HumanEval live evidence "
        "before any operator-facing product headline."
    )


def build_provenance_row(
    *,
    root: Path,
    number: str,
    relative_path: Path,
    before_key: str,
    after_key: str,
    expected_before: float,
    expected_after: float,
    n_key: str | None = None,
) -> dict[str, Any]:
    """Build one table row without inferring through prose.

    Missing artifacts and mismatched numbers are failures, not opportunities to
    fill gaps from the north-star text. That is the anti-fabrication boundary
    this runner is meant to enforce.
    """
    source = (root / relative_path).resolve()
    if not source.exists():
        return {
            "number": number,
            "source_artifact": str(source),
            "n": None,
            "observed_before": None,
            "observed_after": None,
            "headline_numbers_match_north_star": False,
            "seed_present": False,
            "checksum_present": False,
            "substrate": "artifact_not_found",
            "g4_pass": False,
            "caveat": "artifact_not_found_cannot_confirm_g4",
        }

    data = _load_json(source)
    observed_before = data.get(before_key)
    observed_after = data.get(after_key)

    numbers_match = _matches_expected(observed_before, expected_before) and _matches_expected(
        observed_after,
        expected_after,
    )
    seed_present = data.get("random_seed") is not None
    checksum_present = bool(data.get("reproducibility_checksum"))
    g4_pass = bool(numbers_match and seed_present and checksum_present)

    return {
        "number": number,
        "source_artifact": str(source),
        "n": _extract_n(data, n_key),
        "observed_before": observed_before,
        "observed_after": observed_after,
        "headline_numbers_match_north_star": numbers_match,
        "seed_present": seed_present,
        "checksum_present": checksum_present,
        "substrate": _source_substrate(data),
        "g4_pass": g4_pass,
        "caveat": _row_caveat(seed_present, checksum_present, numbers_match),
    }


def _exp227_contrast(root: Path) -> dict[str, Any]:
    path = (root / "results" / "experiment_227_results.json").resolve()
    if not path.exists():
        return {"source_artifact": str(path), "available": False}
    data = _load_json(path)
    stats = data.get("statistics") if isinstance(data.get("statistics"), dict) else {}
    baseline = stats.get("baseline") if isinstance(stats.get("baseline"), dict) else {}
    repair = stats.get("verify_repair") if isinstance(stats.get("verify_repair"), dict) else {}
    improvement = stats.get("improvement") if isinstance(stats.get("improvement"), dict) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return {
        "source_artifact": str(path),
        "available": True,
        "n": metadata.get("sample_size"),
        "baseline_pass_at_1": baseline.get("pass_at_1"),
        "verify_repair_pass_at_1": repair.get("pass_at_1"),
        "improvement_delta": improvement.get("delta"),
        "n_repaired": repair.get("n_repaired"),
    }


def _headline_status(rows: list[dict[str, Any]]) -> str:
    if not all(row["g4_pass"] for row in rows):
        return NOT_YET_HEADLINE_ELIGIBLE
    if any(row["caveat"] != "none" for row in rows):
        return RESTORABLE_WITH_CAVEAT
    return RESTORABLE


def _terminal_status(product_headline_restorable: str) -> str:
    if product_headline_restorable == NOT_YET_HEADLINE_ELIGIBLE:
        return "not_yet_eligible"
    return product_headline_restorable


def build_artifact(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    rows = [
        build_provenance_row(
            root=root,
            number="exp1999_humaneval_repair_0.66_to_0.84",
            relative_path=Path("results/experiment_1999_code_verification_humaneval.json"),
            before_key="baseline_pass_rate",
            after_key="repair_pass_rate",
            expected_before=0.66,
            expected_after=0.84,
            n_key="dataset_size",
        ),
        build_provenance_row(
            root=root,
            number="exp2090_crane_rigid_0.70_to_crane_0.85",
            relative_path=Path("results/experiment_2090_crane_humaneval.json"),
            before_key="rigid_pass_rate",
            after_key="crane_pass_rate",
            expected_before=0.70,
            expected_after=0.85,
        ),
    ]
    exp1999_row, exp2090_row = rows
    product_headline_restorable = _headline_status(rows)
    terminal_status = _terminal_status(product_headline_restorable)
    honest_verdict = (
        "complete: product_headline_provenance_confirmed_"
        f"exp1999_g4_{str(exp1999_row['g4_pass']).lower()}_"
        f"exp2090_g4_{str(exp2090_row['g4_pass']).lower()}_"
        f"headline_{terminal_status}_operator_curated_doc_unedited"
    )

    exp1999_data = {}
    exp2090_data = {}
    exp1999_path = root / "results" / "experiment_1999_code_verification_humaneval.json"
    exp2090_path = root / "results" / "experiment_2090_crane_humaneval.json"
    if exp1999_path.exists():
        exp1999_data = _load_json(exp1999_path)
    if exp2090_path.exists():
        exp2090_data = _load_json(exp2090_path)

    duration_s = round(time.perf_counter() - started, 6)
    return {
        "experiment": 3792,
        "schema": "carnot.product_headline_provenance_confirmation_g4.v1",
        "honest_verdict": honest_verdict,
        "inference_substrate": AGGREGATION_SUBSTRATE,
        "provenance_table": rows,
        "exp1999_g4_pass": bool(exp1999_row["g4_pass"]),
        "exp2090_g4_pass": bool(exp2090_row["g4_pass"]),
        "product_headline_restorable": product_headline_restorable,
        "operator_curated_doc_unedited": True,
        "cited_upstream_artifacts": [
            str((root / "results" / "experiment_227_results.json").resolve()),
            str(exp1999_path.resolve()),
            str(exp2090_path.resolve()),
        ],
        "refuted_exp227_contrast": _exp227_contrast(root),
        "random_seed": {
            "exp1999": exp1999_data.get("random_seed"),
            "exp2090": exp2090_data.get("random_seed"),
        },
        "reproducibility_checksum": {
            "exp1999": exp1999_data.get("reproducibility_checksum"),
            "exp2090": exp2090_data.get("reproducibility_checksum"),
        },
        "duration_s": duration_s,
    }


def write_artifact(artifact: dict[str, Any], path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> int:
    try:
        import yaml  # noqa: F401
    except ImportError:
        artifact = {
            "experiment": 3792,
            "honest_verdict": "blocked_interpreter_yaml_unavailable",
            "inference_substrate": AGGREGATION_SUBSTRATE,
            "provenance_table": [],
            "exp1999_g4_pass": False,
            "exp2090_g4_pass": False,
            "product_headline_restorable": NOT_YET_HEADLINE_ELIGIBLE,
            "operator_curated_doc_unedited": True,
            "cited_upstream_artifacts": [],
            "random_seed": {"exp1999": None, "exp2090": None},
            "reproducibility_checksum": {"exp1999": None, "exp2090": None},
            "duration_s": 0.0,
        }
        write_artifact(artifact, ARTIFACT_PATH)
        print(artifact["honest_verdict"])
        return 1

    artifact = build_artifact(PROJECT_ROOT)
    write_artifact(artifact, ARTIFACT_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
