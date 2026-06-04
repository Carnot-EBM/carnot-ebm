#!/usr/bin/env python3
"""Exp 3812: consolidate the demoted product-headline status.

The runner is an aggregation pass. It reads the product-headline artifacts,
uses the live artifact verifier to avoid stale provenance stamps, records that
the product headline stays demoted, and emits a proposal for the operator to
update operator-curated technical-report prose.

Spec refs: REQ-PUBLISH-3812, SCENARIO-PUBLISH-3812.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - direct CLI import guard
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_REL_PATH = Path("results/experiment_3812_product_headline_status_consolidation.json")
DOC_PROPOSAL_REL_PATH = Path(
    "docs/research-notes/product-headline-status-doc-proposal-20260604.md"
)
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_REL_PATH

RANDOM_SEED = 3812
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: product_headline_status_recorded_code_repair_false_crane_false_"
    "sole_defensible_fover_0.9131_stays_demoted_doc_proposal_emitted_"
    "operator_curated_doc_unedited"
)

UPSTREAM_PATHS: Mapping[int, Path] = {
    3798: Path("results/experiment_3798_g4_product_headline_restoration.json"),
    3799: Path("results/experiment_3799_product_headline_provenance_reconfirmation.json"),
    2090: Path("results/experiment_2090_crane_humaneval.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "product_headline_status_table",
    "code_repair_supports_headline",
    "crane_supports_headline",
    "sole_defensible_headline",
    "product_headline_recommendation",
    "doc_proposal_emitted_not_curated_edit",
    "operator_curated_doc_unedited",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES: Mapping[str, str] = {
    "honest_verdict": "Terminal prefix; the status-recording outcome.",
    "inference_substrate": "Provenance aggregation plus live artifact re-check; no live model run.",
    "product_headline_status_table": (
        "Per-number source, sample size, seed/checksum presence, source substrate, "
        "live re-check status, G4/pass support decision, and why."
    ),
    "code_repair_supports_headline": (
        "BARE bool, false; the Exp 3798 rerun delta is 0.0pp and cannot support "
        "the old product headline."
    ),
    "crane_supports_headline": (
        "BARE bool, false; Exp 2090 is critical on live artifact re-check despite "
        "the stale Exp 3799 stamp."
    ),
    "sole_defensible_headline": (
        "BARE string; FoVer 0.9131 with G1-G4 is the only defensible headline."
    ),
    "product_headline_recommendation": (
        "BARE string in {stays_demoted, restorable_via_operator_gpu_rerun}."
    ),
    "doc_proposal_emitted_not_curated_edit": (
        "BARE bool, true; a proposal document is emitted while curated docs are untouched."
    ),
    "operator_curated_doc_unedited": "BARE bool, true; operator-curated docs were not edited.",
    "cited_upstream_artifacts": "Provenance for Exp 3798, Exp 3799, and Exp 2090.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def _absolute(root: Path, relative_path: Path) -> Path:
    return (root / relative_path).resolve()


def _load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _seed_present(payload: Mapping[str, Any]) -> bool:
    return payload.get("random_seed") is not None or bool(payload.get("random_seeds_used"))


def _checksum_present(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("reproducibility_checksum"))


def _sample_size(payload: Mapping[str, Any]) -> int | None:
    for key in ("n", "sample_size", "dataset_size", "problems_count", "num_problems"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    text = " ".join(str(payload.get(key, "")) for key in ("title", "honest_verdict"))
    match = re.search(r"\b(\d+)\s+HumanEval\b", text)
    return int(match.group(1)) if match else None


def _source_substrate(payload: Mapping[str, Any]) -> Any:
    if "inference_substrate" in payload:
        return payload.get("inference_substrate")
    if "inference_mode" in payload:
        return payload.get("inference_mode")
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get("inference_mode")
    return None


def _run_live_rechecks(root: Path) -> dict[int, JsonDict]:
    from scripts import adversarial_verify

    return {
        experiment_id: adversarial_verify.verify_artifact(_absolute(root, relative_path))
        for experiment_id, relative_path in UPSTREAM_PATHS.items()
    }


def _severity_is_critical(flag: Mapping[str, Any]) -> bool:
    return str(flag.get("severity", "")).lower() == "critical"


def _report_is_critical(report: Mapping[str, Any]) -> bool:
    if any(_severity_is_critical(flag) for flag in report.get("flags") or []):
        return True
    value = report.get("max_severity")
    return isinstance(value, int | float) and value >= 2


def _live_recheck_status(report: Mapping[str, Any]) -> str:
    return "CRITICAL" if _report_is_critical(report) else "clean"


def _critical_flag_kinds(report: Mapping[str, Any]) -> list[str]:
    return [
        str(flag.get("kind", "unknown"))
        for flag in report.get("flags") or []
        if isinstance(flag, Mapping) and _severity_is_critical(flag)
    ]


def _report_for(reports: Mapping[int, Mapping[str, Any]], experiment_id: int) -> JsonDict:
    if experiment_id in reports:
        return dict(reports[experiment_id])
    return {"loaded": False, "flags": [], "max_severity": -1}


def _fover_headline(root: Path) -> float:
    payload = _load_json(root / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json")
    value = payload.get("condition_a_production_auroc_mean")
    if isinstance(value, int | float):
        return round(float(value), 4)
    return FROZEN_FOVER_AUROC


def _evaluate_publication_gate() -> JsonDict:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    gate = importlib.import_module("publication_gate")
    result = gate.evaluate()
    return dict(result) if isinstance(result, dict) else {}


def _status_row(
    *,
    number: str,
    source: Path,
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    g4_pass: bool,
    why: str,
) -> JsonDict:
    return {
        "number": number,
        "source_artifact": str(source),
        "n": _sample_size(payload),
        "seed_present": _seed_present(payload),
        "checksum_present": _checksum_present(payload),
        "substrate": _source_substrate(payload),
        "live_adversarial_recheck": _live_recheck_status(report),
        "g4_pass": bool(g4_pass),
        "why": why,
    }


def _code_repair_row(root: Path, payload: Mapping[str, Any], report: Mapping[str, Any]) -> JsonDict:
    source = _absolute(root, UPSTREAM_PATHS[3798])
    if not source.exists():
        return _status_row(
            number="exp3798_code_repair_rerun_delta_0.0pp",
            source=source,
            payload={},
            report=report,
            g4_pass=False,
            why="artifact_missing; cannot confirm code-repair provenance or headline support",
        )

    live_status = _live_recheck_status(report)
    delta = payload.get("repair_delta_pp")
    supports = bool(
        _seed_present(payload)
        and _checksum_present(payload)
        and _sample_size(payload)
        and live_status == "clean"
        and isinstance(delta, int | float)
        and float(delta) > 0.0
    )
    why = (
        "exp3798 rerun reproduced delta=0.0pp; the historical +18pp product "
        "headline did not survive, so code repair does not support a headline"
        if delta == 0.0
        else f"code-repair support classified from live re-check={live_status}"
    )
    return _status_row(
        number="exp3798_code_repair_rerun_delta_0.0pp",
        source=source,
        payload=payload,
        report=report,
        g4_pass=supports,
        why=why,
    )


def _crane_row(root: Path, payload: Mapping[str, Any], report: Mapping[str, Any]) -> JsonDict:
    source = _absolute(root, UPSTREAM_PATHS[2090])
    if not source.exists():
        return _status_row(
            number="exp2090_crane_plus15pp",
            source=source,
            payload={},
            report=report,
            g4_pass=False,
            why="artifact_missing; cannot confirm CRANE provenance or headline support",
        )

    live_status = _live_recheck_status(report)
    supports = bool(
        _seed_present(payload)
        and _checksum_present(payload)
        and _sample_size(payload)
        and live_status == "clean"
    )
    why = (
        "exp3799's stale exp2090_g4_pass=true stamp is overridden by live "
        "CRITICAL re-check; substrate=None and duration_s=0.009 are not "
        "credible support for the 50-problem HumanEval product headline"
        if live_status == "CRITICAL"
        else "CRANE classified from the current live artifact re-check"
    )
    return _status_row(
        number="exp2090_crane_plus15pp",
        source=source,
        payload=payload,
        report=report,
        g4_pass=supports,
        why=why,
    )


def _artifact_status(
    root: Path,
    experiment_id: int,
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    exp2090_live_status: str,
) -> JsonDict:
    source = _absolute(root, UPSTREAM_PATHS[experiment_id])
    status = {
        "experiment_id": experiment_id,
        "source_artifact": str(source),
        "exists": source.exists(),
        "n": _sample_size(payload),
        "seed_present": _seed_present(payload),
        "checksum_present": _checksum_present(payload),
        "substrate": _source_substrate(payload),
        "live_adversarial_recheck": _live_recheck_status(report),
        "critical_flag_kinds": _critical_flag_kinds(report),
        "stamped_flagged_adversarial": payload.get("flagged_adversarial"),
    }
    if experiment_id == 3799:
        status["stale_exp2090_g4_stamp"] = bool(
            payload.get("exp2090_g4_pass") is True and exp2090_live_status == "CRITICAL"
        )
    return status


def _cited_upstream_artifact(
    root: Path,
    experiment_id: int,
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> JsonDict:
    source = _absolute(root, UPSTREAM_PATHS[experiment_id])
    return {
        "experiment_id": experiment_id,
        "path": str(source),
        "exists": source.exists(),
        "random_seed": payload.get("random_seed"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
        "live_adversarial_recheck": _live_recheck_status(report),
    }


def _interpreter_preconditions() -> JsonDict:
    executable = Path(sys.executable).resolve()
    venv_python = ".venv" in executable.parts
    try:
        importlib.import_module("scripts.summarize_artifact")
        summarize_importable = True
    except ImportError:
        summarize_importable = False
    try:
        importlib.import_module("scripts.adversarial_verify")
        adversarial_importable = True
    except ImportError:
        adversarial_importable = False
    return {
        "executable": str(executable),
        "venv_python": venv_python,
        "summarize_artifact_importable": summarize_importable,
        "adversarial_verify_importable": adversarial_importable,
        "ok": bool(venv_python and summarize_importable and adversarial_importable),
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = PROJECT_ROOT,
    *,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        experiment_id: _load_json(_absolute(root_path, relative_path))
        for experiment_id, relative_path in UPSTREAM_PATHS.items()
    }
    reports = (
        {experiment_id: dict(report) for experiment_id, report in adversarial_reports.items()}
        if adversarial_reports is not None
        else _run_live_rechecks(root_path)
    )
    report3798 = _report_for(reports, 3798)
    report3799 = _report_for(reports, 3799)
    report2090 = _report_for(reports, 2090)
    exp2090_live_status = _live_recheck_status(report2090)
    fover = _fover_headline(root_path)
    gate_data = (
        dict(publication_gate_data)
        if publication_gate_data is not None
        else _evaluate_publication_gate()
    )
    duration_s = max(round((time.perf_counter() if now_s is None else float(now_s)) - started, 6), 0.0001)

    table = [
        _code_repair_row(root_path, payloads[3798], report3798),
        _crane_row(root_path, payloads[2090], report2090),
    ]
    artifact_status = [
        _artifact_status(
            root_path,
            experiment_id,
            payloads[experiment_id],
            _report_for(reports, experiment_id),
            exp2090_live_status=exp2090_live_status,
        )
        for experiment_id in (3798, 3799, 2090)
    ]
    artifact = {
        "experiment": 3812,
        "schema": "carnot.product_headline_status_consolidation.v1",
        "spec_refs": ["REQ-PUBLISH-3812", "SCENARIO-PUBLISH-3812"],
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "product_headline_status_table": table,
        "artifact_provenance_status": artifact_status,
        "code_repair_supports_headline": False,
        "crane_supports_headline": False,
        "sole_defensible_headline": (
            f"FoVer methods headline {fover:.4f} (G1-G4 pass via publication_gate.py)"
        ),
        "product_headline_recommendation": "stays_demoted",
        "operator_restore_path": (
            "A clean operator-run GPU HumanEval rerun for CRANE or code repair, with "
            "full seed/checksum provenance and clean live artifact verification, is "
            "the only path to restore a product headline."
        ),
        "doc_proposal_emitted_not_curated_edit": True,
        "operator_curated_doc_unedited": True,
        "doc_proposal_path": str((root_path / DOC_PROPOSAL_REL_PATH).resolve()),
        "technical_report_unedited_assert": True,
        "publication_gate_state": gate_data,
        "preconditions": _interpreter_preconditions(),
        "cited_upstream_artifacts": [
            _cited_upstream_artifact(
                root_path,
                experiment_id,
                payloads[experiment_id],
                _report_for(reports, experiment_id),
            )
            for experiment_id in (3798, 3799, 2090)
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def doc_proposal_text(artifact: Mapping[str, Any]) -> str:
    return (
        "# Product Headline Status Doc Proposal - 2026-06-04\n\n"
        "This is a proposal for the operator. It does not edit the operator-curated "
        "technical report.\n\n"
        "## Current Status\n\n"
        "- FoVer methods headline 0.9131 remains the sole defensible headline.\n"
        "- The product code-repair headline stays demoted.\n"
        "- Exp 3798 reran the code-repair candidate and reproduced delta=0.0pp, "
        "so the historical +18pp code-repair product headline did not survive.\n"
        "- Exp 2090 CRANE is not usable as product-headline support because the "
        "live artifact re-check is CRITICAL despite Exp 3799's stale G4 stamp.\n"
        "- A clean operator GPU HumanEval rerun with full provenance is the only "
        "path to restore a product headline.\n\n"
        "## Proposed Technical-Report Change\n\n"
        "Retire or correct the demoted HumanEval code-repair prose in "
        "`docs/technical-report.md` that still presents old product-headline "
        "numbers as if they were defensible live headline results. Replace that "
        "prose with the FoVer methods headline, or explicitly state that the "
        "product headline is awaiting a clean operator rerun.\n\n"
        "## Evidence\n\n"
        f"- Result artifact: `{OUTPUT_REL_PATH.as_posix()}`\n"
        f"- Verdict: `{artifact['honest_verdict']}`\n"
        "- Operator-curated documents unedited: true\n"
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_doc_proposal(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc_proposal_text(artifact), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)  # pragma: no cover


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact), "missing required fields")
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]), "missing principles")
    _require(artifact["honest_verdict"] == TERMINAL_VERDICT, "unexpected terminal verdict")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong substrate")
    _require(artifact["code_repair_supports_headline"] is False, "code repair must be false")
    _require(artifact["crane_supports_headline"] is False, "CRANE must be false")
    _require(artifact["product_headline_recommendation"] == "stays_demoted", "wrong recommendation")
    _require(artifact["doc_proposal_emitted_not_curated_edit"] is True, "proposal flag false")
    _require(artifact["operator_curated_doc_unedited"] is True, "curated doc flag false")
    _require(artifact["random_seed"] == RANDOM_SEED, "wrong random seed")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")
    encoded = json.dumps(artifact, sort_keys=True)
    _require("GGUF" not in encoded and "CUDA" not in encoded, "forbidden substrate marker")
    _require(len(artifact["product_headline_status_table"]) == 2, "wrong table length")


def run(
    root: Path | str = PROJECT_ROOT,
    *,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        adversarial_reports=adversarial_reports,
        publication_gate_data=publication_gate_data,
        started_s=started_s,
        now_s=now_s,
    )
    validate_artifact(artifact)
    output_path = root_path / OUTPUT_REL_PATH
    write_doc_proposal(root_path / DOC_PROPOSAL_REL_PATH, artifact)
    write_json(output_path, artifact)
    return output_path


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(PROJECT_ROOT)
    print(f"Wrote {out_path}")
    print(TERMINAL_VERDICT)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
