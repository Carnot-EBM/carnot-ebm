#!/usr/bin/env python3
"""Exp 3799: reconfirm product-headline G4 provenance after the rerun.

The runner reads checked-in result artifacts only. It answers one operator
question: after Exp 3798 reran the formerly unproven Exp 1999 code-repair
number, which product numbers now satisfy the narrow G4 provenance rule, and
does that make the product headline eligible again?

Spec: REQ-PUBLISH-3799, SCENARIO-PUBLISH-3799
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
    / "experiment_3799_product_headline_provenance_reconfirmation.json"
)

AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
RESTORABLE = "restorable"
RESTORABLE_WITH_CAVEAT = "restorable_with_caveat"
NOT_YET_HEADLINE_ELIGIBLE = "not_yet_headline_eligible"

EXP3798_RELATIVE = Path("results/experiment_3798_g4_product_headline_restoration.json")
EXP2090_RELATIVE = Path("results/experiment_2090_crane_humaneval.json")
RERUN_NUMBER = "exp3798_rerun_code_repair_baseline_0.13_repair_0.13_delta_0.0pp"
CRANE_NUMBER = "exp2090_crane_rigid_0.70_to_crane_0.85"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _absolute(root: Path, relative_path: Path) -> Path:
    return (root / relative_path).resolve()


def _seed_present(data: dict[str, Any]) -> bool:
    return data.get("random_seed") is not None


def _checksum_present(data: dict[str, Any]) -> bool:
    return bool(data.get("reproducibility_checksum"))


def _non_trivial_n(data: dict[str, Any]) -> int | None:
    n = data.get("n")
    if isinstance(n, int) and n > 0:
        return n
    for key in ("sample_size", "dataset_size", "problems_count", "num_problems"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    text = " ".join(str(data.get(key, "")) for key in ("title", "honest_verdict"))
    match = re.search(r"\b(\d+)\s+HumanEval\b", text)
    return int(match.group(1)) if match else None


def _sanitized_substrate(data: dict[str, Any]) -> str:
    value = data.get("inference_substrate")
    if isinstance(value, str) and value:
        return value
    value = data.get("inference_mode")
    if isinstance(value, str) and value:
        return value
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("inference_mode")
        if isinstance(value, str) and value:
            return value
    if data.get("model_used") or data.get("model_specs"):
        return "model_provenance_present_redacted_for_aggregation_hygiene"
    return "not_recorded"


def _numbers_match(data: dict[str, Any], before_key: str, after_key: str) -> bool:
    before = data.get(before_key)
    after = data.get(after_key)
    return (
        isinstance(before, (int, float))
        and isinstance(after, (int, float))
        and abs(float(before) - 0.70) < 1e-12
        and abs(float(after) - 0.85) < 1e-12
    )


def _missing_rerun_row(source: Path) -> dict[str, Any]:
    return {
        "number": RERUN_NUMBER,
        "source_artifact": str(source),
        "n": None,
        "seed_present": False,
        "checksum_present": False,
        "substrate": "artifact_not_found",
        "positive_control_passed": False,
        "g4_provenance_complete": False,
        "baseline_pass1": None,
        "repair_pass1": None,
        "repair_delta_pp": None,
        "g4_pass": False,
        "caveat": "artifact_not_found_cannot_confirm_g4",
    }


def _rerun_caveat(
    *,
    seed_present: bool,
    checksum_present: bool,
    n: int | None,
    positive_control_passed: bool,
    g4_provenance_complete: bool,
    data: dict[str, Any],
) -> str:
    caveats: list[str] = []
    if not seed_present:
        caveats.append("missing_random_seed")
    if not checksum_present:
        caveats.append("missing_reproducibility_checksum")
    if n is None:
        caveats.append("missing_non_trivial_n")
    if not positive_control_passed:
        caveats.append("positive_control_failed")
    if not g4_provenance_complete:
        caveats.append("g4_provenance_incomplete")
    if data.get("flagged_adversarial") is True:
        caveats.append("upstream_flagged_adversarial")
    if data.get("repair_delta_pp") == 0.0:
        caveats.append("zero_delta_headline_stays_demoted")
    if data.get("product_headline_restorable") == "stays_demoted":
        caveats.append("upstream_headline_stays_demoted")
    return "; ".join(caveats) if caveats else "none"


def build_rerun_code_repair_row(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source = _absolute(root, EXP3798_RELATIVE)
    if not source.exists():
        return _missing_rerun_row(source)

    data = _load_json(source)
    seed = _seed_present(data)
    checksum = _checksum_present(data)
    n = _non_trivial_n(data)
    positive_control = data.get("positive_control_passed") is True
    provenance_complete = data.get("g4_provenance_complete") is True
    g4_pass = bool(seed and checksum and n is not None and positive_control and provenance_complete)
    caveat = _rerun_caveat(
        seed_present=seed,
        checksum_present=checksum,
        n=n,
        positive_control_passed=positive_control,
        g4_provenance_complete=provenance_complete,
        data=data,
    )

    return {
        "number": RERUN_NUMBER,
        "source_artifact": str(source),
        "n": n,
        "seed_present": seed,
        "checksum_present": checksum,
        "substrate": _sanitized_substrate(data),
        "positive_control_passed": positive_control,
        "g4_provenance_complete": provenance_complete,
        "baseline_pass1": data.get("baseline_pass1"),
        "repair_pass1": data.get("repair_pass1"),
        "repair_delta_pp": data.get("repair_delta_pp"),
        "g4_pass": g4_pass,
        "caveat": caveat,
    }


def _missing_crane_row(source: Path) -> dict[str, Any]:
    return {
        "number": CRANE_NUMBER,
        "source_artifact": str(source),
        "n": None,
        "seed_present": False,
        "checksum_present": False,
        "substrate": "artifact_not_found",
        "positive_control_passed": None,
        "observed_before": None,
        "observed_after": None,
        "g4_pass": False,
        "caveat": "artifact_not_found_cannot_confirm_g4",
    }


def build_exp2090_crane_row(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source = _absolute(root, EXP2090_RELATIVE)
    if not source.exists():
        return _missing_crane_row(source)

    data = _load_json(source)
    seed = _seed_present(data)
    checksum = _checksum_present(data)
    n = _non_trivial_n(data)
    numbers_match = _numbers_match(data, "rigid_pass_rate", "crane_pass_rate")
    g4_pass = bool(seed and checksum and n is not None and numbers_match)
    caveats = []
    if not seed:
        caveats.append("missing_random_seed")
    if not checksum:
        caveats.append("missing_reproducibility_checksum")
    if n is None:
        caveats.append("missing_non_trivial_n")
    if not numbers_match:
        caveats.append("number_discrepancy_vs_exp3792")

    return {
        "number": CRANE_NUMBER,
        "source_artifact": str(source),
        "n": n,
        "seed_present": seed,
        "checksum_present": checksum,
        "substrate": _sanitized_substrate(data),
        "positive_control_passed": None,
        "observed_before": data.get("rigid_pass_rate"),
        "observed_after": data.get("crane_pass_rate"),
        "g4_pass": g4_pass,
        "caveat": "; ".join(caveats) if caveats else "none",
    }


def headline_status(rerun_row: dict[str, Any], crane_row: dict[str, Any]) -> str:
    if not rerun_row["g4_pass"] or not crane_row["g4_pass"]:
        return NOT_YET_HEADLINE_ELIGIBLE
    rerun_caveat = str(rerun_row.get("caveat", ""))
    if "zero_delta_headline_stays_demoted" in rerun_caveat:
        return NOT_YET_HEADLINE_ELIGIBLE
    if "upstream_flagged_adversarial" in rerun_caveat:
        return NOT_YET_HEADLINE_ELIGIBLE
    if rerun_caveat != "none" or crane_row.get("caveat") != "none":
        return RESTORABLE_WITH_CAVEAT
    return RESTORABLE


def terminal_status(product_headline_restorable: str) -> str:
    if product_headline_restorable == NOT_YET_HEADLINE_ELIGIBLE:
        return "not_yet_eligible"
    return product_headline_restorable


def _load_if_exists(root: Path, relative_path: Path) -> dict[str, Any]:
    path = _absolute(root, relative_path)
    return _load_json(path) if path.exists() else {}


def _blocked_missing_exp3798(rerun_row: dict[str, Any]) -> bool:
    return rerun_row.get("caveat") == "artifact_not_found_cannot_confirm_g4"


def build_artifact(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    rerun_row = build_rerun_code_repair_row(root)
    crane_row = build_exp2090_crane_row(root)
    product_status = headline_status(rerun_row, crane_row)

    if _blocked_missing_exp3798(rerun_row):
        verdict = "blocked: exp3798_did_not_produce_clean_artifact_headline_stays_demoted"
    else:
        verdict = (
            "complete: product_headline_provenance_reconfirmed_"
            f"rerun_g4_{str(rerun_row['g4_pass']).lower()}_"
            f"exp2090_g4_{str(crane_row['g4_pass']).lower()}_"
            f"headline_{terminal_status(product_status)}_operator_curated_doc_unedited"
        )

    exp3798_data = _load_if_exists(root, EXP3798_RELATIVE)
    exp2090_data = _load_if_exists(root, EXP2090_RELATIVE)
    duration_s = max(round(time.perf_counter() - started, 6), 0.0001)

    return {
        "experiment": 3799,
        "schema": "carnot.product_headline_provenance_reconfirmation.v1",
        "honest_verdict": verdict,
        "inference_substrate": AGGREGATION_SUBSTRATE,
        "provenance_table": [rerun_row, crane_row],
        "rerun_code_repair_g4_pass": bool(rerun_row["g4_pass"]),
        "exp2090_g4_pass": bool(crane_row["g4_pass"]),
        "product_headline_restorable": product_status,
        "operator_curated_doc_unedited": True,
        "cited_upstream_artifacts": [
            str(_absolute(root, EXP3798_RELATIVE)),
            str(_absolute(root, EXP2090_RELATIVE)),
        ],
        "random_seed": {
            "exp3798": exp3798_data.get("random_seed"),
            "exp2090": exp2090_data.get("random_seed"),
        },
        "reproducibility_checksum": {
            "exp3798": exp3798_data.get("reproducibility_checksum"),
            "exp2090": exp2090_data.get("reproducibility_checksum"),
        },
        "duration_s": duration_s,
    }


def write_artifact(artifact: dict[str, Any], path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _interpreter_preconditions_ok() -> bool:
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


def _interpreter_blocked_artifact() -> dict[str, Any]:
    return {
        "experiment": 3799,
        "schema": "blocked_interpreter_precondition",
        "honest_verdict": "blocked_interpreter_yaml_unavailable",
        "inference_substrate": AGGREGATION_SUBSTRATE,
        "provenance_table": [],
        "rerun_code_repair_g4_pass": False,
        "exp2090_g4_pass": False,
        "product_headline_restorable": NOT_YET_HEADLINE_ELIGIBLE,
        "operator_curated_doc_unedited": True,
        "cited_upstream_artifacts": [],
        "random_seed": {"exp3798": None, "exp2090": None},
        "reproducibility_checksum": {"exp3798": None, "exp2090": None},
        "duration_s": 0.0001,
    }


def main() -> int:
    if not _interpreter_preconditions_ok():
        artifact = _interpreter_blocked_artifact()
        write_artifact(artifact, ARTIFACT_PATH)
        print(artifact["honest_verdict"])
        return 1

    artifact = build_artifact(PROJECT_ROOT)
    write_artifact(artifact, ARTIFACT_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
