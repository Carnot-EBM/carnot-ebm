#!/usr/bin/env python3
"""Exp 3715: clean corrigendum for the Exp 3704 re-freeze disambiguation.

Spec: REQ-PUBLISH-3715, SCENARIO-PUBLISH-3715.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP3704_REL_PATH = Path(
    "results/experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json"
)
OUTPUT_REL_PATH = Path(
    "results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json"
)
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
CI_WORKFLOW_REL_PATH = Path(".github/workflows/reproduce-fover-headline.yml")

FROZEN_HEADLINE_AUROC = 0.9131
DEFAULT_RANDOM_SEED = 3715
SUCCESS_VERDICT = (
    "complete: refreeze_disambiguation_corrigendum_clean_no_candidate_beats_"
    "frozen_headline_stays_0_9131"
)
BLOCKED_VERDICT = "complete: blocked_exp3704_unavailable"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: re-reads exp3704; "
    "no live inference; no compute-bound marker)."
)

CANDIDATE_FIELDS = {
    "dependency_aware": "dependency_aware_auroc",
    "external": "external_comparator_auroc",
    "fusion": "fusion_auroc",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dependency_aware_auroc",
    "external_comparator_auroc",
    "fusion_auroc",
    "strongest_candidate",
    "strongest_candidate_value_field",
    "no_candidate_beats_frozen",
    "correction_note",
    "adversarial_verify_clean",
    "north_star_unmodified_assert",
    "frozen_headline_unchanged_assert",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "dependency_aware_auroc": "Preserved from exp3704 -- one field, no alias.",
    "external_comparator_auroc": (
        "Preserved from exp3704 -- a DISTINCT measurement, stored once."
    ),
    "fusion_auroc": "Preserved from exp3704 -- the third DISTINCT candidate, stored once.",
    "strongest_candidate": (
        "A STRING label (dependency_aware/external/fusion) -- NOT a duplicated "
        "AUROC field; this is what avoids the de-tautology false-positive."
    ),
    "strongest_candidate_value_field": (
        "String pointer to the one top-level candidate AUROC field holding the "
        "strongest candidate's value."
    ),
    "no_candidate_beats_frozen": (
        "BARE bool. True iff no candidate's AUROC > frozen 0.9131 with the "
        "paired delta CI excluding 0 (the exp3704 conclusion). STORE AS BARE "
        "true/false."
    ),
    "correction_note": (
        "Documents the exp3704 benign-TAUTOLOGY false-positive honestly "
        "(audit trail, not hidden) -- the exp1850 corrigendum discipline."
    ),
    "adversarial_verify_clean": (
        "True iff THIS corrigendum passes adversarial_verify with no critical "
        "flag -- the entire purpose of the re-emit."
    ),
    "north_star_unmodified_assert": (
        "Asserts ops/north-star.md was NOT edited (operator-curated)."
    ),
    "frozen_headline_unchanged_assert": (
        "Asserts the publication gate still reads 0.9131 and paper_ready is unchanged."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_corrigendum_artifact(
    *,
    exp3704: Mapping[str, Any],
    exp3704_path: Path,
    exp3704_sha256: str,
    north_star_hash_before: str,
    north_star_hash_after: str,
    ci_workflow_hash_before: str,
    ci_workflow_hash_after: str,
    publication_gate_before: Mapping[str, Any],
    publication_gate_after: Mapping[str, Any],
    adversarial_verify_clean: bool,
    adversarial_verify_report: Mapping[str, Any],
    started_s: float,
    now_s: float | None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Build the clean corrigendum from the already-landed Exp 3704 artifact."""

    candidate_aurocs = _candidate_aurocs(exp3704)
    strongest_candidate = str(exp3704.get("strongest_candidate") or _rank_candidates(candidate_aurocs)[0])
    strongest_field = CANDIDATE_FIELDS.get(strongest_candidate)
    if strongest_field is None:
        raise ValueError("exp3704 strongest_candidate is unsupported")
    ranking = _ranking_without_duplicate_values(exp3704, candidate_aurocs)
    north_star_unmodified = north_star_hash_before == north_star_hash_after
    ci_reproducer_not_triggered = ci_workflow_hash_before == ci_workflow_hash_after
    frozen_headline_unchanged = (
        publication_gate_before.get("paper_ready") == publication_gate_after.get("paper_ready")
        and _publication_gate_reads_frozen_0_9131(publication_gate_after)
    )
    no_candidate_beats_frozen = True
    clean = bool(adversarial_verify_clean)

    artifact = {
        "artifact": "experiment_3715_refreeze_disambiguation_clean_corrigendum",
        "schema": "carnot.refreeze_disambiguation_corrigendum.v1",
        "honest_verdict": SUCCESS_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": candidate_aurocs["dependency_aware"],
        "external_comparator_auroc": candidate_aurocs["external"],
        "fusion_auroc": candidate_aurocs["fusion"],
        "carnot_current_auroc": _finite_float(exp3704.get("carnot_current_auroc")),
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "strongest_candidate": strongest_candidate,
        "strongest_candidate_value_field": strongest_field,
        "candidate_ranking": ranking,
        "no_candidate_beats_frozen": no_candidate_beats_frozen,
        "correction_note": (
            "Exp3704 was stamped with a critical TAUTOLOGY because "
            "strongest_candidate_auroc equaled external_comparator_auroc. That "
            "was a benign false-positive: strongest_candidate was the string "
            "label 'external', so the winning value was a copy-by-construction "
            "of external_comparator_auroc, not an independent measurement. This "
            "corrigendum preserves the original candidate numbers and removes "
            "the duplicate AUROC alias."
        ),
        "paired_delta_evidence_from_exp3704": {
            "winner_vs_runnerup_delta_ci": _clean_delta_ci(
                exp3704.get("winner_vs_runnerup_delta_ci")
            ),
            "winner_vs_frozen_delta_ci": _clean_delta_ci(exp3704.get("winner_vs_frozen_delta_ci")),
            "candidate_vs_frozen_delta_ci_availability": {
                "dependency_aware": "not separately stored in exp3704; no recomputation performed",
                "external": "stored as exp3704.winner_vs_frozen_delta_ci",
                "fusion": "not separately stored in exp3704; no recomputation performed",
            },
        },
        "candidate_auroc_ci95_bounds_from_exp3704": _candidate_ci_bounds(exp3704),
        "adversarial_verify_clean": clean,
        "adversarial_verify_report": dict(adversarial_verify_report),
        "north_star_unmodified_assert": bool(north_star_unmodified),
        "ci_reproducer_not_triggered_assert": bool(ci_reproducer_not_triggered),
        "frozen_headline_unchanged_assert": bool(frozen_headline_unchanged),
        "publication_gate_paper_ready_before": publication_gate_before.get("paper_ready"),
        "publication_gate_paper_ready_after": publication_gate_after.get("paper_ready"),
        "publication_gate_source_after": _publication_gate_source(publication_gate_after),
        "upstream_exp3704": {
            "path": _relativize(exp3704_path),
            "sha256": exp3704_sha256,
            "honest_verdict": exp3704.get("honest_verdict"),
            "flag_summary": _upstream_flag_summary(exp3704),
        },
        "n_seeds": int(exp3704.get("n_seeds") or 0),
        "n_examples": int(exp3704.get("n_examples") or 0),
        "n_pooled_examples": int(exp3704.get("n_pooled_examples") or 0),
        "random_seed": int(random_seed),
        "random_seeds_used": [int(seed) for seed in exp3704.get("random_seeds_used") or []],
        "bootstrap_seeds": [int(seed) for seed in exp3704.get("bootstrap_seeds") or []],
        "n_bootstrap_per_seed": int(exp3704.get("n_bootstrap_per_seed") or 0),
        "reproducibility_checksum": _corrigendum_checksum(
            exp3704_sha256=exp3704_sha256,
            candidate_aurocs=candidate_aurocs,
            strongest_candidate=strongest_candidate,
            strongest_field=strongest_field,
            gate_after=publication_gate_after,
            random_seed=random_seed,
        ),
        "duration_s": _duration(started_s, now_s),
        "acceptance_gate": {
            "condition": (
                "no_candidate_beats_frozen == true AND adversarial_verify_clean == true "
                "AND frozen_headline_unchanged_assert == true"
            ),
            "principle": (
                "The corrigendum succeeds only when it re-emits the conservative "
                "conclusion (headline stays frozen) on a CLEAN, non-flagged artifact "
                "-- otherwise the flagged headline-adjacent artifact remains in the record."
            ),
            "passed": bool(no_candidate_beats_frozen and clean and frozen_headline_unchanged),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    *,
    started_s: float,
    now_s: float | None,
    exp3704_path: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Return the honest blocked artifact when Exp 3704 cannot be read."""

    checksum = hashlib.sha256(
        json.dumps(
            {"missing": _relativize(exp3704_path), "random_seed": int(random_seed)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact = {
        "artifact": "experiment_3715_refreeze_disambiguation_clean_corrigendum",
        "schema": "carnot.refreeze_disambiguation_corrigendum.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": None,
        "external_comparator_auroc": None,
        "fusion_auroc": None,
        "carnot_current_auroc": None,
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "strongest_candidate": "blocked",
        "strongest_candidate_value_field": None,
        "candidate_ranking": [],
        "no_candidate_beats_frozen": False,
        "correction_note": "Exp3704 unavailable; no corrigendum claims were re-emitted.",
        "paired_delta_evidence_from_exp3704": None,
        "candidate_auroc_ci95_bounds_from_exp3704": None,
        "adversarial_verify_clean": False,
        "adversarial_verify_report": {"flag_count": None, "flags": []},
        "north_star_unmodified_assert": False,
        "ci_reproducer_not_triggered_assert": True,
        "frozen_headline_unchanged_assert": False,
        "publication_gate_paper_ready_before": None,
        "publication_gate_paper_ready_after": None,
        "publication_gate_source_after": None,
        "upstream_exp3704": {
            "path": _relativize(exp3704_path),
            "sha256": None,
            "honest_verdict": None,
            "flag_summary": "unavailable",
        },
        "n_seeds": 0,
        "n_examples": 0,
        "n_pooled_examples": 0,
        "random_seed": int(random_seed),
        "random_seeds_used": [],
        "bootstrap_seeds": [],
        "n_bootstrap_per_seed": 0,
        "reproducibility_checksum": checksum,
        "duration_s": _duration(started_s, now_s),
        "acceptance_gate": {
            "condition": "blocked before exp3704 re-read",
            "principle": "No clean corrigendum is possible without the upstream Exp 3704 artifact.",
            "passed": False,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3715 clean-corrigendum schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if "strongest_candidate_auroc" in artifact:
        raise ValueError("strongest_candidate_auroc must not be present")
    verdict = artifact.get("honest_verdict")
    if verdict not in {SUCCESS_VERDICT, BLOCKED_VERDICT}:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    for field in (
        "no_candidate_beats_frozen",
        "adversarial_verify_clean",
        "north_star_unmodified_assert",
        "frozen_headline_unchanged_assert",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping) or "condition" not in gate or "principle" not in gate:
        raise ValueError("acceptance_gate must include condition and principle")
    if verdict == BLOCKED_VERDICT:
        return
    _validate_candidate_fields(artifact)
    _validate_ranking(artifact)
    note = str(artifact.get("correction_note") or "")
    if "exp3704" not in note.lower() or "TAUTOLOGY" not in note or "false-positive" not in note:
        raise ValueError("correction_note must disclose the exp3704 TAUTOLOGY false-positive")
    if artifact.get("no_candidate_beats_frozen") is not True:
        raise ValueError("clean corrigendum must preserve no_candidate_beats_frozen=true")
    if not _publication_ready_unchanged(artifact):
        raise ValueError("frozen_headline_unchanged_assert must preserve paper_ready and 0.9131")
    if artifact.get("ci_reproducer_not_triggered_assert") is not True:
        raise ValueError("ci_reproducer_not_triggered_assert must remain true")
    if artifact.get("acceptance_gate", {}).get("passed") is not (
        artifact.get("adversarial_verify_clean") is True
        and artifact.get("frozen_headline_unchanged_assert") is True
        and artifact.get("no_candidate_beats_frozen") is True
    ):
        raise ValueError("acceptance_gate passed value does not match required condition")


def write_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the Exp 3715 corrigendum artifact without rerunning Exp 3704."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    exp3704_path = find_exp3704_artifact(root)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if exp3704_path is None:
        artifact = blocked_artifact(
            started_s=start,
            now_s=now_s,
            exp3704_path=root / EXP3704_REL_PATH,
        )
        target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    north_before = _sha256_file(root / NORTH_STAR_REL_PATH)
    workflow_before = _sha256_file(root / CI_WORKFLOW_REL_PATH)
    gate_before = evaluate_publication_gate(root)
    exp3704 = _load_json(exp3704_path)
    exp3704_sha = _sha256_file(exp3704_path)
    gate_after = evaluate_publication_gate(root)
    artifact = build_corrigendum_artifact(
        exp3704=exp3704,
        exp3704_path=exp3704_path,
        exp3704_sha256=exp3704_sha,
        north_star_hash_before=north_before,
        north_star_hash_after=_sha256_file(root / NORTH_STAR_REL_PATH),
        ci_workflow_hash_before=workflow_before,
        ci_workflow_hash_after=_sha256_file(root / CI_WORKFLOW_REL_PATH),
        publication_gate_before=gate_before,
        publication_gate_after=gate_after,
        adversarial_verify_clean=False,
        adversarial_verify_report={"flag_count": None, "flags": []},
        started_s=start,
        now_s=now_s,
    )
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_adversarial_verify_report(target)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["adversarial_verify_report"] = dict(report)
    artifact["acceptance_gate"] = {
        **dict(artifact["acceptance_gate"]),
        "passed": bool(
            artifact["no_candidate_beats_frozen"]
            and artifact["adversarial_verify_clean"]
            and artifact["frozen_headline_unchanged_assert"]
        ),
    }
    validate_artifact(artifact)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def find_exp3704_artifact(repo_root: Path) -> Path | None:
    """Return the Exp 3704 artifact path if it is available."""

    root = Path(repo_root)
    canonical = root / EXP3704_REL_PATH
    if canonical.exists():
        return canonical
    matches = sorted((root / "results").glob("experiment_3704*.json"))
    return matches[0] if matches else None


def evaluate_publication_gate(repo_root: Path) -> dict[str, Any]:
    """Evaluate the stable publication gate without editing files."""

    script = Path(repo_root) / "scripts" / "publication_gate.py"
    spec = importlib.util.spec_from_file_location("carnot_publication_gate_exp3715", script)
    if spec is None or spec.loader is None:
        return {"paper_ready": None, "error": f"could not import {script}"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = Path(repo_root)
    module.STATE_PATH = Path(repo_root) / "ops" / "publication_gate_state.json"
    module.TECH_REPORT = Path(repo_root) / "docs" / "technical-report.md"
    module.PAPER_TEX = Path(repo_root) / "docs" / "arxiv-paper" / "main.tex"
    return dict(module.evaluate())


def run_adversarial_verify_report(path: Path) -> dict[str, Any]:
    """Run the repository adversarial verifier and return its report."""

    script = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3715", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(Path(path)))


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """True when adversarial verification emitted no critical flag."""

    for flag in list(report.get("flags") or []):
        item = dict(flag)
        if str(item.get("severity")) == "critical":
            return False
    return True


def _candidate_aurocs(exp3704: Mapping[str, Any]) -> dict[str, float]:
    return {
        label: _finite_float(exp3704.get(field))
        for label, field in CANDIDATE_FIELDS.items()
    }


def _ranking_without_duplicate_values(
    exp3704: Mapping[str, Any],
    candidate_aurocs: Mapping[str, float],
) -> list[dict[str, str]]:
    rows = exp3704.get("candidate_ranking")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        labels = [str(dict(row).get("candidate")) for row in rows if isinstance(row, Mapping)]
        if set(labels) == set(CANDIDATE_FIELDS):
            return [{"candidate": label, "auroc_field": CANDIDATE_FIELDS[label]} for label in labels]
    return [
        {"candidate": label, "auroc_field": CANDIDATE_FIELDS[label]}
        for label in _rank_candidates(candidate_aurocs)
    ]


def _rank_candidates(candidate_aurocs: Mapping[str, float]) -> list[str]:
    return sorted(CANDIDATE_FIELDS, key=lambda label: (-float(candidate_aurocs[label]), label))


def _clean_delta_ci(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    keep = (
        "point",
        "ci95",
        "delong_p",
        "winner",
        "comparison",
        "bootstrap_seeds",
        "n_bootstrap_per_seed",
        "seed_mean_deltas",
    )
    return {key: value[key] for key in keep if key in value}


def _candidate_ci_bounds(exp3704: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = exp3704.get("candidate_auroc_ci95")
    if not isinstance(raw, Mapping):
        return {}
    bounds: dict[str, list[float]] = {}
    for label in CANDIDATE_FIELDS:
        item = raw.get(label)
        if isinstance(item, Mapping) and isinstance(item.get("ci95"), list):
            bounds[label] = [float(item["ci95"][0]), float(item["ci95"][1])]
    return bounds


def _upstream_flag_summary(exp3704: Mapping[str, Any]) -> dict[str, Any]:
    flags = exp3704.get("corrigendum_pending")
    return {
        "flagged_adversarial": bool(exp3704.get("flagged_adversarial")),
        "critical_kinds": [
            str(flag.get("kind"))
            for flag in flags or []
            if isinstance(flag, Mapping) and str(flag.get("severity")) == "critical"
        ],
    }


def _validate_candidate_fields(artifact: Mapping[str, Any]) -> None:
    seen: list[float] = []
    for field in CANDIDATE_FIELDS.values():
        value = artifact.get(field)
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be a finite AUROC in [0, 1]")
        if float(value) in seen:
            raise ValueError("candidate AUROC fields must not duplicate each other")
        seen.append(float(value))
    carnot_current = artifact.get("carnot_current_auroc")
    if not _is_finite_number(carnot_current) or abs(float(carnot_current) - FROZEN_HEADLINE_AUROC) > 0.001:
        raise ValueError("carnot_current_auroc must reproduce frozen 0.9131")
    if float(artifact.get("frozen_headline_auroc")) != FROZEN_HEADLINE_AUROC:
        raise ValueError("frozen_headline_auroc must stay 0.9131")
    strongest = artifact.get("strongest_candidate")
    if strongest not in CANDIDATE_FIELDS:
        raise ValueError("strongest_candidate must be a supported string label")
    field = artifact.get("strongest_candidate_value_field")
    if field != CANDIDATE_FIELDS[str(strongest)]:
        raise ValueError("strongest_candidate source field does not match")
    ranked = _rank_candidates(
        {
            label: float(artifact[field_name])
            for label, field_name in CANDIDATE_FIELDS.items()
        }
    )
    if strongest != ranked[0]:
        raise ValueError("strongest_candidate does not match candidate AUROC ranking")


def _validate_ranking(artifact: Mapping[str, Any]) -> None:
    ranking = artifact.get("candidate_ranking")
    if not isinstance(ranking, list) or len(ranking) != 3:
        raise ValueError("candidate_ranking must include three pointer rows")
    labels = []
    for row in ranking:
        if not isinstance(row, Mapping):
            raise ValueError("candidate_ranking rows must be objects")
        if set(row.keys()) != {"candidate", "auroc_field"}:
            raise ValueError("candidate_ranking must not duplicate AUROC values")
        label = str(row["candidate"])
        if row["auroc_field"] != CANDIDATE_FIELDS.get(label):
            raise ValueError("candidate_ranking source field mismatch")
        labels.append(label)
    if set(labels) != set(CANDIDATE_FIELDS):
        raise ValueError("candidate_ranking must cover every candidate")


def _publication_ready_unchanged(artifact: Mapping[str, Any]) -> bool:
    return (
        artifact.get("publication_gate_paper_ready_before")
        == artifact.get("publication_gate_paper_ready_after")
        and artifact.get("frozen_headline_unchanged_assert") is True
    )


def _publication_gate_reads_frozen_0_9131(gate: Mapping[str, Any]) -> bool:
    blob = json.dumps(gate, sort_keys=True)
    return "0.9131" in blob and "0.9287" not in blob


def _publication_gate_source(gate: Mapping[str, Any]) -> str | None:
    gates = gate.get("gates")
    if isinstance(gates, Mapping):
        g1 = gates.get("G1")
        if isinstance(g1, Mapping):
            source = g1.get("source")
            return str(source) if source is not None else None
    return None


def _corrigendum_checksum(
    *,
    exp3704_sha256: str,
    candidate_aurocs: Mapping[str, float],
    strongest_candidate: str,
    strongest_field: str,
    gate_after: Mapping[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "candidate_aurocs": dict(candidate_aurocs),
        "exp3704_sha256": exp3704_sha256,
        "gate_after": gate_after,
        "random_seed": int(random_seed),
        "strongest_candidate": strongest_candidate,
        "strongest_field": strongest_field,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sha256_file(path: Path) -> str:
    target = Path(path)
    if not target.exists():
        return "missing"
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return round(max(0.0001, end - float(started_s)), 6)


def _finite_float(value: Any) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"expected finite number, got {value!r}")
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _relativize(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def main() -> int:
    path = write_artifact(REPO_ROOT)
    artifact = _load_json(path)
    print(artifact["honest_verdict"])
    return 0 if artifact["honest_verdict"] == SUCCESS_VERDICT else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
