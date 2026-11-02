"""Prepare the operator-facing G4 correction proposal for the technical report.

Spec refs: REQ-REPORT-3782, SCENARIO-REPORT-3782.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 3782
OUTPUT_REL_PATH = Path("results/experiment_3782_technical_report_g4_correction_prep.json")
PROPOSAL_REL_PATH = Path(
    "docs/research-notes/technical-report-g4-correction-proposal-20260604.md"
)
TECHNICAL_REPORT_REL_PATH = Path("docs/technical-report.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
EXP227_REL_PATH = Path("results/experiment_227_results.json")
EXP1999_REL_PATH = Path("results/experiment_1999_code_verification_humaneval.json")
EXP2090_REL_PATH = Path("results/experiment_2090_crane_humaneval.json")
OPERATOR_CURATED_DOCS = (
    ("technical_report", TECHNICAL_REPORT_REL_PATH),
    ("north_star", NORTH_STAR_REL_PATH),
)

INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: provenance aggregation + "
    "drafting, no live model)."
)
TERMINAL_VERDICT = (
    "complete: "
    "g4_correction_prepped_unsupported_numbers_identified_real_numbers_confirmed_"
    "proposal_written_operator_curated_doc_unedited"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "unsupported_numbers_identified",
    "real_numbers_confirmed",
    "proposed_correction_written",
    "operator_curated_doc_unedited",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the correction-prep outcome.",
    "inference_substrate": (
        "Provenance aggregation and drafting over upstream artifacts; no fresh "
        "inference."
    ),
    "unsupported_numbers_identified": (
        "Lists the refuted prose numbers (8%->80%, 0%->36%, +3.0pp) plus that "
        "Exp 227 shows delta=0.0 -- the G4 catch being corrected."
    ),
    "real_numbers_confirmed": (
        "The provenance-confirmed Exp 1999 (+18pp) and Exp 2090 (+15pp) numbers "
        "plus whether each passes G4 using seed, checksum, and n."
    ),
    "proposed_correction_written": (
        "BARE bool, true -- the old-to-new diff was written to a research note "
        "for the operator."
    ),
    "operator_curated_doc_unedited": (
        "BARE bool, true -- docs/technical-report.md and operator-curated inputs "
        "were not edited."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the corrected numbers; anti-fabrication audit trail."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def run(
    repo_root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the proposal note and terminal artifact, returning the artifact path."""

    started = time.time() if started_s is None else started_s
    before_docs = snapshot_operator_curated_docs(repo_root)
    artifact = build_artifact(repo_root, before_docs, started_s=started, now_s=now_s)
    out_path = repo_root / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = compact_verify_report(verify_report)
    artifact["adversarial_verify_clean"] = report_is_clean(verify_report)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def build_artifact(
    repo_root: Path,
    before_docs: Mapping[str, str],
    *,
    started_s: float,
    now_s: float | None = None,
) -> JsonDict:
    """Build the aggregation artifact from primary source files."""

    exp227 = load_json(repo_root / EXP227_REL_PATH)
    exp1999 = load_json(repo_root / EXP1999_REL_PATH)
    exp2090 = load_json(repo_root / EXP2090_REL_PATH)
    technical_report = (repo_root / TECHNICAL_REPORT_REL_PATH).read_text(encoding="utf-8")
    old_paragraph = find_trajectory_paragraph(technical_report)
    exp227_summary = summarize_exp227(exp227)
    confirmed = [
        summarize_positive(
            1999,
            EXP1999_REL_PATH,
            exp1999,
            baseline_key="baseline_pass_rate",
            improved_key="repair_pass_rate",
            method="repair",
        ),
        summarize_positive(
            2090,
            EXP2090_REL_PATH,
            exp2090,
            baseline_key="rigid_pass_rate",
            improved_key="crane_pass_rate",
            method="CRANE constrained decoding",
        ),
    ]
    proposal_text = render_proposal(old_paragraph, exp227_summary, confirmed)
    write_proposal(repo_root, proposal_text)

    artifact: JsonDict = {
        "honest_verdict": TERMINAL_VERDICT,
        "schema": "carnot.technical_report_g4_correction_prep.v1",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "unsupported_numbers_identified": {
            "refuted_prose_numbers": ["8%->80%", "0%->36%", "+3.0pp"],
            "exp227": exp227_summary,
            "g4_catch": (
                "The cited Exp 227 source is flat, so the unsupported prose "
                "numbers must not be used as a product headline."
            ),
        },
        "real_numbers_confirmed": confirmed,
        "proposal_path": PROPOSAL_REL_PATH.as_posix(),
        "proposed_correction_written": (repo_root / PROPOSAL_REL_PATH).exists(),
        "operator_curated_doc_unedited": operator_curated_docs_unchanged(
            repo_root, before_docs
        ),
        "operator_curated_docs_checked": [
            rel.as_posix() for _, rel in OPERATOR_CURATED_DOCS
        ],
        "cited_upstream_artifacts": cited_upstream_artifacts(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed_seconds(started_s, now_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "headline_scope_note": (
            "These code-positive artifacts are proposal evidence only; they are "
            "not promoted as the product headline until the operator accepts a "
            "clean full-HumanEval repair-run artifact."
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def summarize_exp227(payload: Mapping[str, Any]) -> JsonDict:
    """Return the flat Exp 227 result used to refute the prose headline."""

    stats = payload["statistics"]
    improvement = stats["improvement"]
    repair_stats = stats.get("repair_stats", {})
    verify_repair = stats["verify_repair"]
    metadata = payload.get("metadata", {})
    return {
        "experiment_id": 227,
        "n": metadata.get("sample_size") or payload.get("cohort", {}).get("case_count"),
        "baseline_pass_at_1": stats["baseline"]["pass_at_1"],
        "verify_repair_pass_at_1": verify_repair["pass_at_1"],
        "improvement_delta": improvement["delta"],
        "delta_pp": round(float(improvement["delta"]) * 100.0, 1),
        "n_repaired": repair_stats.get("n_repaired", verify_repair.get("n_repaired")),
    }


def summarize_positive(
    experiment_id: int,
    rel_path: Path,
    payload: Mapping[str, Any],
    *,
    baseline_key: str,
    improved_key: str,
    method: str,
) -> JsonDict:
    """Return one surviving code-positive result with explicit G4 status."""

    baseline = float(payload[baseline_key])
    improved = float(payload[improved_key])
    n, n_source, n_is_structured = extract_n(payload)
    missing = []
    if not has_random_seed(payload):
        missing.append("random_seed")
    if not payload.get("reproducibility_checksum"):
        missing.append("reproducibility_checksum")
    if not n_is_structured:
        missing.append("structured_n")
    return {
        "experiment_id": experiment_id,
        "path": rel_path.as_posix(),
        "method": method,
        "baseline": baseline,
        "improved": improved,
        "delta_pp": round((improved - baseline) * 100.0, 1),
        "n": n,
        "n_source": n_source,
        "random_seed_present": has_random_seed(payload),
        "reproducibility_checksum_present": bool(payload.get("reproducibility_checksum")),
        "g4_passes": not missing,
        "g4_missing_fields": missing,
        "headline_ready": False,
    }


def extract_n(payload: Mapping[str, Any]) -> tuple[int | None, str, bool]:
    """Extract n/sample-size and whether it is structured rather than prose."""

    for key in ("n", "sample_size", "dataset_size"):
        value = payload.get(key)
        if isinstance(value, int):
            return value, key, True
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("sample_size"), int):
        return metadata["sample_size"], "metadata.sample_size", True
    cohort = payload.get("cohort")
    if isinstance(cohort, Mapping) and isinstance(cohort.get("case_count"), int):
        return cohort["case_count"], "cohort.case_count", True
    verdict = str(payload.get("honest_verdict", ""))
    match = re.search(r"\b(\d+)\s+HumanEval\b", verdict)
    if match:
        return int(match.group(1)), "honest_verdict_text", False
    return None, "missing", False


def has_random_seed(payload: Mapping[str, Any]) -> bool:
    """Return true if the artifact carries a seed field."""

    if payload.get("random_seed") is not None:
        return True
    metadata = payload.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("run_seed") is not None


def render_proposal(
    old_paragraph: str,
    exp227_summary: Mapping[str, Any],
    confirmed: list[Mapping[str, Any]],
) -> str:
    """Render the operator-action correction proposal note."""

    old_clause = find_old_clause(old_paragraph)
    new_clause = corrected_clause(exp227_summary, confirmed)
    corrected_paragraph = old_paragraph.replace(old_clause, new_clause)
    return (
        "# Technical Report G4 Correction Proposal - 2026-06-04\n\n"
        "## OPERATOR ACTION Proposal\n\n"
        "This note prepares the Public Documentation Discipline correction. It does "
        "not edit `docs/technical-report.md`.\n\n"
        "## Unsupported Numbers Identified\n\n"
        "- Remove the unsupported `8%->80%`, `0%->36%`, and `+3.0pp` code-repair "
        "claims from the trajectory framing.\n"
        "- Exp 227 reports 0.0pp delta and 0 repaired cases; it cannot support the "
        "demoted prose headline.\n\n"
        "## Real Numbers Confirmed\n\n"
        "- Exp 1999 reports 0.66 -> 0.84 over n=50 (+18pp), but the primary "
        "artifact lacks random_seed and reproducibility_checksum, so it does not "
        "fully pass G4 for a headline.\n"
        "- Exp 2090 reports 0.70 -> 0.85 over n=50 (+15pp), with random_seed=42 "
        "and reproducibility_checksum=bfb0acdb53773a49, but n is in the artifact "
        "verdict text rather than a structured n field; keep it scoped until a "
        "clean live-GPU full-HumanEval repair run lands.\n\n"
        "## Exact Old Text\n\n"
        f"{old_paragraph}\n\n"
        "## Proposed Corrected Paragraph\n\n"
        f"{corrected_paragraph}\n\n"
        "## Old -> New Diff\n\n"
        "```diff\n"
        f"-{old_clause}\n"
        f"+{new_clause}\n"
        "```\n"
    )


def corrected_clause(
    exp227_summary: Mapping[str, Any], confirmed: list[Mapping[str, Any]]
) -> str:
    """Build the replacement clause for the trajectory paragraph."""

    by_id = {int(row["experiment_id"]): row for row in confirmed}
    exp1999 = by_id[1999]
    exp2090 = by_id[2090]
    return (
        "found that the prior code-repair trajectory claim fails G4: Exp 227 "
        f"reports {percent(exp227_summary['baseline_pass_at_1'])} -> "
        f"{percent(exp227_summary['verify_repair_pass_at_1'])} with "
        f"{exp227_summary['delta_pp']:.1f}pp delta and "
        f"{exp227_summary['n_repaired']} repaired cases (n={exp227_summary['n']}), "
        "while the surviving narrower positives are Exp 1999 "
        f"{exp1999['baseline']:.2f} -> {exp1999['improved']:.2f} "
        f"(+{exp1999['delta_pp']:.0f}pp, n={exp1999['n']}, missing seed/checksum) "
        "and Exp 2090 "
        f"{exp2090['baseline']:.2f} -> {exp2090['improved']:.2f} "
        f"(+{exp2090['delta_pp']:.0f}pp, n={exp2090['n']}, seed/checksum present "
        "but n not structured), so code repair should remain scoped until a "
        "clean live-GPU full-HumanEval repair run lands; typed constraint "
        "verification (+4.9pp) remains separately cited"
    )


def find_trajectory_paragraph(text: str) -> str:
    """Find the technical-report trajectory paragraph."""

    for paragraph in re.split(r"\n\s*\n", text):
        if paragraph.startswith("The trajectory of this project is:"):
            return paragraph.strip()
    raise ValueError("technical-report trajectory paragraph not found")


def find_old_clause(paragraph: str) -> str:
    """Find the exact old trajectory clause to replace."""

    pattern = re.compile(
        r"proved that code verification \(\+3\.0pp HumanEval\) and typed "
        r"constraint verification \(\+4\.9pp\) work on "
        r"(?:live GPU inference|inference artifacts)"
    )
    match = pattern.search(paragraph)
    if not match:
        raise ValueError("old code-repair trajectory clause not found")
    return match.group(0)


def cited_upstream_artifacts(repo_root: Path) -> list[JsonDict]:
    """Return source-file provenance with stable hashes."""

    return [
        {
            "experiment_id": 227,
            "path": EXP227_REL_PATH.as_posix(),
            "sha256": sha256_path(repo_root / EXP227_REL_PATH),
            "fields_imported": [
                "metadata.sample_size",
                "statistics.baseline.pass_at_1",
                "statistics.verify_repair.pass_at_1",
                "statistics.improvement.delta",
                "statistics.repair_stats.n_repaired",
            ],
        },
        {
            "experiment_id": 1999,
            "path": EXP1999_REL_PATH.as_posix(),
            "sha256": sha256_path(repo_root / EXP1999_REL_PATH),
            "fields_imported": [
                "baseline_pass_rate",
                "repair_pass_rate",
                "dataset_size",
            ],
        },
        {
            "experiment_id": 2090,
            "path": EXP2090_REL_PATH.as_posix(),
            "sha256": sha256_path(repo_root / EXP2090_REL_PATH),
            "fields_imported": [
                "rigid_pass_rate",
                "crane_pass_rate",
                "pass_rate_delta",
                "random_seed",
                "reproducibility_checksum",
                "honest_verdict",
            ],
        },
    ]


def snapshot_operator_curated_docs(repo_root: Path) -> dict[str, str]:
    """Read operator-curated inputs before writing generated outputs."""

    return {
        label: (repo_root / rel_path).read_text(encoding="utf-8")
        for label, rel_path in OPERATOR_CURATED_DOCS
    }


def operator_curated_docs_unchanged(
    repo_root: Path, before_docs: Mapping[str, str]
) -> bool:
    """Return true when curated documents match their pre-run content."""

    for label, rel_path in OPERATOR_CURATED_DOCS:
        before = before_docs.get(label, before_docs.get(rel_path.as_posix()))
        if before is None:
            return False
        if (repo_root / rel_path).read_text(encoding="utf-8") != before:
            return False
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact shape and anti-fabrication fields."""

    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "honest_verdict")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _ensure(
        artifact.get("operator_curated_doc_unedited") is True,
        "operator_curated_doc_unedited",
    )
    _ensure(artifact.get("proposed_correction_written") is True, "proposal")
    _ensure(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact), "required fields")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles")
    _ensure(set(REQUIRED_ARTIFACT_FIELDS) <= set(principles), "field principles")
    unsupported = artifact.get("unsupported_numbers_identified")
    _ensure(isinstance(unsupported, Mapping), "unsupported numbers")
    _ensure(unsupported.get("exp227", {}).get("improvement_delta") == 0.0, "exp227 delta")
    _ensure(unsupported.get("exp227", {}).get("n_repaired") == 0, "exp227 repair count")
    confirmed = artifact.get("real_numbers_confirmed")
    _ensure(isinstance(confirmed, list) and len(confirmed) == 2, "real numbers")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _ensure(float(artifact.get("duration_s", 0.0)) > 0.0, "duration_s")
    _ensure(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Exp 3782 artifact: {message}")


def write_proposal(repo_root: Path, text: str) -> None:
    """Write the generated research-note proposal."""

    path = repo_root / PROPOSAL_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> JsonDict:
    """Load one JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep the adversarial verifier report compact but auditable."""

    flags = list(report.get("flags", []))
    return {
        "max_severity": max((severity_rank(flag.get("severity")) for flag in flags), default=0),
        "flags": flags,
    }


def report_is_clean(report: Mapping[str, Any] | None) -> bool:
    """Return true when no critical adversarial flag is present."""

    if not isinstance(report, Mapping):
        return True
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in report.get("flags", [])
    )


def severity_rank(severity: Any) -> int:
    """Map verifier severity labels to sortable integers."""

    return {"info": 0, "warn": 1, "critical": 2}.get(str(severity).lower(), -1)


def percent(value: Any) -> str:
    """Format a rate as a one-decimal percentage."""

    return f"{float(value) * 100.0:.1f}%"


def elapsed_seconds(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration with a nonzero floor."""

    now = time.time() if now_s is None else now_s
    return round(max(now - started_s, 0.0001), 6)


def sha256_path(path: Path) -> str:
    """Return a file SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum over payload content."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
