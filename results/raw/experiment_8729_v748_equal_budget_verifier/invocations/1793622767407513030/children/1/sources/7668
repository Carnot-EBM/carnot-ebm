#!/usr/bin/env python3
"""Exp 3681: prepare the operator-only FoVer headline re-freeze package.

Spec: REQ-PUBLISH-040, SCENARIO-PUBLISH-040, SCENARIO-PUBLISH-040B.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_REL_PATH = Path(
    "results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json"
)
EXP3680_REL_PATH = Path("results/experiment_3680_dependency_aware_dual_condition_integrity.json")
FROZEN_SOURCE_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
CI_WORKFLOW_REL_PATH = Path(".github/workflows/reproduce-fover-headline.yml")

READY_VERDICT = "complete: refreeze_package_ready_for_operator_frozen_headline_unchanged"
BLOCKED_VERDICT = "complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable"
TERMINAL_VERDICTS = (READY_VERDICT, BLOCKED_VERDICT)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: re-runs the "
    "cached-corpus reproducer; no LLM load)."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "reproducer_extended",
    "existing_0_9131_reproduction_still_green",
    "candidate_reproduction_asserts_in_ci",
    "draft_ci_workflow_assertion_bounds",
    "operator_checklist",
    "north_star_unmodified_assert",
    "ci_workflow_unmodified_assert",
    "frozen_headline_unchanged_assert",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "reproducer_extended": (
        "True iff scripts/reproduce_fover_headline.py was extended ADDITIVELY "
        "to compute+assert the candidate number while leaving the 0.9131 path green."
    ),
    "existing_0_9131_reproduction_still_green": (
        "True iff the unchanged 0.9131 reproduction path still passes -- proves "
        "the additive change did not regress G2."
    ),
    "candidate_reproduction_asserts_in_ci": (
        "True iff the new dependency-aware reproduction lands inside exp3680's "
        "CI -- the new number is independently recomputable."
    ),
    "draft_ci_workflow_assertion_bounds": (
        "The exact CI-workflow assertion bounds the operator would set (drafted "
        "in-artifact, NOT applied) -- the operator's re-freeze input."
    ),
    "operator_checklist": (
        "The ordered OPERATOR-ACTION steps (north-star sec-1 edit, CI-workflow "
        "update, trigger run) -- the re-freeze is operator-only."
    ),
    "north_star_unmodified_assert": (
        "Asserts ops/north-star.md was NOT edited (operator-curated)."
    ),
    "ci_workflow_unmodified_assert": (
        "Asserts the .github reproducer workflow was NOT edited and the run was "
        "NOT triggered (operator-only external action)."
    ),
    "frozen_headline_unchanged_assert": (
        "Asserts the publication gate still reads 0.9131 and paper_ready is "
        "unchanged -- no silent substitution."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def _round_metric(value: float | int, digits: int = 6) -> float:
    return round(float(value), digits)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.001, end - float(started_s))


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_exp3680_artifact(repo_root: Path) -> dict[str, Any]:
    """Load the candidate G1 source artifact."""

    path = Path(repo_root) / EXP3680_REL_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exp3680_candidate_confirmed(exp3680_artifact: Mapping[str, Any]) -> bool:
    """Return true only for the explicit bare G1-candidate confirmation."""

    return bool(exp3680_artifact.get("dependency_aware_g1_rigor_confirmed") is True)


def reproducer_import_status() -> tuple[bool, ModuleType | None]:
    """Import the G2 reproducer without executing it."""

    try:
        module = importlib.import_module("scripts.reproduce_fover_headline")
    except ImportError:
        return False, None
    return True, module


def reproducer_has_candidate_extension(module: ModuleType | None) -> bool:
    """Check the additive candidate function and CLI flag are present."""

    if module is None:
        return False
    functions_present = all(
        hasattr(module, name)
        for name in (
            "run_dependency_aware_candidate_reproduction",
            "check_dependency_aware_candidate_ci",
            "dependency_aware_candidate_bounds_from_artifact",
        )
    )
    source_path = Path(getattr(module, "__file__", ""))
    flag_present = source_path.exists() and "--dependency-aware-candidate" in source_path.read_text(
        encoding="utf-8"
    )
    return bool(functions_present and flag_present)


def run_frozen_reproduction(module: ModuleType, repo_root: Path) -> dict[str, Any]:
    """Run the unchanged frozen 0.9131 reproducer path."""

    return dict(module.run_reproduction(Path(repo_root)))


def run_candidate_reproduction(module: ModuleType, repo_root: Path) -> dict[str, Any]:
    """Run the additive dependency-aware candidate reproducer path."""

    return dict(module.run_dependency_aware_candidate_reproduction(Path(repo_root)))


def evaluate_publication_gate(repo_root: Path) -> dict[str, Any]:
    """Evaluate publication_gate.py against repo_root without editing files."""

    script = Path(repo_root) / "scripts" / "publication_gate.py"
    spec = importlib.util.spec_from_file_location("carnot_publication_gate_exp3681", script)
    if spec is None or spec.loader is None:
        return {"paper_ready": None, "error": f"could not import {script}"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = Path(repo_root)
    module.STATE_PATH = Path(repo_root) / "ops" / "publication_gate_state.json"
    module.TECH_REPORT = Path(repo_root) / "docs" / "technical-report.md"
    module.PAPER_TEX = Path(repo_root) / "docs" / "arxiv-paper" / "main.tex"
    return dict(module.evaluate())


def draft_ci_workflow_assertion_bounds(exp3680_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact assertion bounds to put in the CI workflow later."""

    from scripts import reproduce_fover_headline as reproducer

    bounds = reproducer.dependency_aware_candidate_bounds_from_artifact(exp3680_artifact)
    production = dict(bounds["production_auroc_dependency_aware"])
    learning = dict(bounds["learning_contribution_dependency_aware"])
    production_ci = production["ci95"]
    learning_ci = learning["ci95"]
    production["assertion"] = (
        f"{production_ci[0]} <= production_auroc_dependency_aware <= {production_ci[1]}"
    )
    learning["assertion"] = (
        f"{learning_ci[0]} <= learning_contribution_dependency_aware <= {learning_ci[1]}"
    )
    return {
        "source_artifact": EXP3680_REL_PATH.as_posix(),
        "production_auroc_dependency_aware": production,
        "learning_contribution_dependency_aware": learning,
        "not_applied_to_workflow": True,
    }


def build_operator_checklist(bounds: Mapping[str, Any]) -> list[str]:
    """Build the operator-only re-freeze checklist."""

    production = dict(bounds.get("production_auroc_dependency_aware") or {})
    learning = dict(bounds.get("learning_contribution_dependency_aware") or {})
    production_ci = production.get("ci95")
    learning_ci = learning.get("ci95")
    candidate_point = production.get("headline_candidate_point", production.get("point"))
    learning_point = learning.get("point")
    return [
        (
            "OPERATOR-ACTION: Edit ops/north-star.md Section 1 methods headline "
            f"from FoVer AUROC 0.9131 to dependency-aware FoVer AUROC "
            f"{candidate_point} with CI {production_ci}, learning contribution "
            f"{learning_point} with CI {learning_ci}, source artifact "
            f"{EXP3680_REL_PATH.as_posix()}."
        ),
        (
            "OPERATOR-ACTION: Update .github/workflows/reproduce-fover-headline.yml "
            f"to run the dependency-aware candidate assertion and use production "
            f"bounds {production_ci} plus learning-contribution bounds {learning_ci}."
        ),
        (
            "OPERATOR-ACTION: Trigger the GitHub Actions FoVer headline reproducer "
            "run and record the green run before changing the publication gate. "
            "The frozen 0.9131 stays the headline until this checklist is complete "
            "and the CI reproducer re-runs green on the new number."
        ),
    ]


def frozen_reproduction_green(module: ModuleType | None, result: Mapping[str, Any]) -> bool:
    """Return true when the default reproducer still passes the frozen CI."""

    if module is None or not hasattr(module, "check_acceptance_ci"):
        return False
    if str(result.get("honest_verdict", "")).startswith("blocked"):
        return False
    cond_a_in_ci, lc_in_ci = module.check_acceptance_ci(dict(result))
    return bool(cond_a_in_ci and lc_in_ci)


def publication_gate_reads_frozen_0_9131(repo_root: Path, gate: Mapping[str, Any]) -> bool:
    """Check the stable publication gate still resolves to the frozen source."""

    gates = dict(gate.get("gates") or {})
    sources = {
        str(dict(gates.get(name) or {}).get("source") or "")
        for name in ("G1", "G4")
    }
    if any("experiment_2850_fover_dual_condition_integrity_v4.json" in source for source in sources):
        source_path = Path(repo_root) / FROZEN_SOURCE_REL_PATH
        if source_path.exists():
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            return round(float(payload.get("condition_a_production_auroc_mean")), 4) == 0.9131
        return True
    detail_blob = json.dumps(gates, sort_keys=True)
    return "0.9131" in detail_blob and "0.925" not in detail_blob


def classify_honest_verdict(
    *,
    g1_candidate_confirmed: bool,
    reproducer_importable: bool,
    reproducer_extended: bool,
    existing_0_9131_reproduction_still_green: bool,
    candidate_reproduction_asserts_in_ci: bool,
    north_star_unmodified_assert: bool,
    ci_workflow_unmodified_assert: bool,
    frozen_headline_unchanged_assert: bool,
) -> str:
    """Map all acceptance booleans to the closed Exp 3681 verdict set."""

    ready = all(
        (
            g1_candidate_confirmed,
            reproducer_importable,
            reproducer_extended,
            existing_0_9131_reproduction_still_green,
            candidate_reproduction_asserts_in_ci,
            north_star_unmodified_assert,
            ci_workflow_unmodified_assert,
            frozen_headline_unchanged_assert,
        )
    )
    return READY_VERDICT if ready else BLOCKED_VERDICT


def build_artifact(
    *,
    repo_root: Path,
    started_s: float,
    now_s: float | None,
    exp3680_artifact: Mapping[str, Any],
    reproducer_importable: bool,
    reproducer_extended: bool,
    frozen_reproduction_result: Mapping[str, Any],
    candidate_reproduction_result: Mapping[str, Any],
    publication_gate_before: Mapping[str, Any],
    publication_gate_after: Mapping[str, Any],
    north_star_hash_before: str,
    north_star_hash_after: str,
    ci_workflow_hash_before: str,
    ci_workflow_hash_after: str,
    github_run_triggered: bool,
    reproducer_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Assemble the Exp 3681 operator package artifact."""

    root = Path(repo_root)
    exp3680_confirmed = exp3680_candidate_confirmed(exp3680_artifact)
    bounds = (
        draft_ci_workflow_assertion_bounds(exp3680_artifact)
        if exp3680_artifact
        else {
            "source_artifact": EXP3680_REL_PATH.as_posix(),
            "production_auroc_dependency_aware": None,
            "learning_contribution_dependency_aware": None,
            "not_applied_to_workflow": True,
        }
    )
    checklist = build_operator_checklist(bounds)
    existing_green = frozen_reproduction_green(reproducer_module, frozen_reproduction_result)
    if reproducer_module is None and frozen_reproduction_result.get("condition_a_production_auroc_mean"):
        from scripts import reproduce_fover_headline as reproducer

        existing_green = frozen_reproduction_green(reproducer, frozen_reproduction_result)
    candidate_in_ci = bool(candidate_reproduction_result.get("candidate_reproduction_asserts_in_ci"))
    north_star_unmodified = north_star_hash_before == north_star_hash_after
    ci_workflow_unmodified = (
        ci_workflow_hash_before == ci_workflow_hash_after and not bool(github_run_triggered)
    )
    frozen_headline_unchanged = (
        publication_gate_before.get("paper_ready") == publication_gate_after.get("paper_ready")
        and publication_gate_reads_frozen_0_9131(root, publication_gate_after)
    )

    verdict = classify_honest_verdict(
        g1_candidate_confirmed=exp3680_confirmed,
        reproducer_importable=reproducer_importable,
        reproducer_extended=reproducer_extended,
        existing_0_9131_reproduction_still_green=existing_green,
        candidate_reproduction_asserts_in_ci=candidate_in_ci,
        north_star_unmodified_assert=north_star_unmodified,
        ci_workflow_unmodified_assert=ci_workflow_unmodified,
        frozen_headline_unchanged_assert=frozen_headline_unchanged,
    )
    checksum_payload = {
        "verdict": verdict,
        "exp3680_checksum": exp3680_artifact.get("reproducibility_checksum"),
        "frozen_checksum": frozen_reproduction_result.get("reproducibility_checksum"),
        "candidate_checksum": candidate_reproduction_result.get("reproducibility_checksum"),
        "north_star_hash_before": north_star_hash_before,
        "north_star_hash_after": north_star_hash_after,
        "ci_workflow_hash_before": ci_workflow_hash_before,
        "ci_workflow_hash_after": ci_workflow_hash_after,
        "paper_ready_before": publication_gate_before.get("paper_ready"),
        "paper_ready_after": publication_gate_after.get("paper_ready"),
    }
    artifact = {
        "artifact": "experiment_3681_g2_reproducer_prep_operator_refreeze_package",
        "schema": "carnot.g2_reproducer_prep_operator_refreeze_package.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducer_extended": bool(reproducer_extended),
        "existing_0_9131_reproduction_still_green": bool(existing_green),
        "candidate_reproduction_asserts_in_ci": bool(candidate_in_ci),
        "draft_ci_workflow_assertion_bounds": bounds,
        "operator_checklist": checklist,
        "north_star_unmodified_assert": bool(north_star_unmodified),
        "ci_workflow_unmodified_assert": bool(ci_workflow_unmodified),
        "frozen_headline_unchanged_assert": bool(frozen_headline_unchanged),
        "random_seed": int(exp3680_artifact.get("random_seed") or 42),
        "reproducibility_checksum": _json_checksum(checksum_payload),
        "duration_s": _round_metric(_duration(started_s, now_s)),
        "exp3680_dependency_aware_g1_rigor_confirmed": bool(exp3680_confirmed),
        "reproducer_importable": bool(reproducer_importable),
        "github_actions_run_triggered": bool(github_run_triggered),
        "publication_gate_paper_ready_before": publication_gate_before.get("paper_ready"),
        "publication_gate_paper_ready_after": publication_gate_after.get("paper_ready"),
        "frozen_reproduction_result": dict(frozen_reproduction_result),
        "candidate_reproduction_result": dict(candidate_reproduction_result),
        "source_artifacts": [
            EXP3680_REL_PATH.as_posix(),
            FROZEN_SOURCE_REL_PATH.as_posix(),
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3681 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError(f"unsupported honest_verdict: {artifact.get('honest_verdict')!r}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    for field in (
        "reproducer_extended",
        "existing_0_9131_reproduction_still_green",
        "candidate_reproduction_asserts_in_ci",
        "north_star_unmodified_assert",
        "ci_workflow_unmodified_assert",
        "frozen_headline_unchanged_assert",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    if not isinstance(artifact.get("operator_checklist"), list):
        raise ValueError("operator_checklist must be a list")
    if not all(str(step).startswith("OPERATOR-ACTION:") for step in artifact["operator_checklist"]):
        raise ValueError("all operator_checklist steps must be marked OPERATOR-ACTION")


def _blocked_result(reason: str, exc: Exception | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"honest_verdict": f"blocked_{reason}"}
    if exc is not None:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def write_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the prep checks and write the terminal Exp 3681 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    north_before = _sha256_file(root / NORTH_STAR_REL_PATH)
    workflow_before = _sha256_file(root / CI_WORKFLOW_REL_PATH)
    gate_before = evaluate_publication_gate(root)
    exp3680_artifact = load_exp3680_artifact(root)
    importable, module = reproducer_import_status()
    extended = reproducer_has_candidate_extension(module)

    frozen_result: dict[str, Any] = {}
    candidate_result: dict[str, Any] = {}
    if exp3680_candidate_confirmed(exp3680_artifact) and importable and extended and module is not None:
        try:
            frozen_result = run_frozen_reproduction(module, root)
        except Exception as exc:  # noqa: BLE001 - terminal artifact must fail closed.
            frozen_result = _blocked_result("frozen_reproduction_unavailable", exc)
        try:
            candidate_result = run_candidate_reproduction(module, root)
        except Exception as exc:  # noqa: BLE001
            candidate_result = _blocked_result("candidate_reproduction_unavailable", exc)
    else:
        frozen_result = _blocked_result("preconditions_not_met")
        candidate_result = _blocked_result("preconditions_not_met")

    gate_after = evaluate_publication_gate(root)
    north_after = _sha256_file(root / NORTH_STAR_REL_PATH)
    workflow_after = _sha256_file(root / CI_WORKFLOW_REL_PATH)
    artifact = build_artifact(
        repo_root=root,
        started_s=start,
        now_s=now_s,
        exp3680_artifact=exp3680_artifact,
        reproducer_importable=importable,
        reproducer_extended=extended,
        frozen_reproduction_result=frozen_result,
        candidate_reproduction_result=candidate_result,
        publication_gate_before=gate_before,
        publication_gate_after=gate_after,
        north_star_hash_before=north_before,
        north_star_hash_after=north_after,
        ci_workflow_hash_before=workflow_before,
        ci_workflow_hash_after=workflow_after,
        github_run_triggered=False,
        reproducer_module=module,
    )
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    root = REPO_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output = write_artifact(root)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
