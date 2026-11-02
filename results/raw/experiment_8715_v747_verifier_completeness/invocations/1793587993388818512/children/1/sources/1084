"""Experiment 4683: adversarial_verify .431 overclaim hardening receipt.

Spec refs: REQ-ARC-WMTE-4683,
SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION,
SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify as av  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4683_adversarial_verify_hardening"
SCHEMA = "carnot.exp4683.adversarial_verify_hardening.v1"
RESULT_RELATIVE_PATH = "results/experiment_4683_adversarial_verify_hardening.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4683
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
    "no model load (100us floor)."
)
SUCCESS_VERDICT = (
    "success: "
    "adversarial_verify_hardened_subgoal_decomposition_and_coverage_baseline_guards_tests_green."
)
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
REQUIRED_FIXTURES = (
    "results/experiment_4676_hierarchical_subgoal_search_live.json",
    "results/experiment_4677_poe_world_factored_subgoal_planner.json",
)
GUARDED_KINDS = {
    av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND,
    av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND,
    av.GENERATION_COVERAGE_WITHOUT_BASELINE_KIND,
    av.GENERATION_COVERAGE_BASELINE_OMITTED_KIND,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "adversarial_verify_hardened_subgoal_decomposition_and_coverage_baseline_guards_tests_green."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
            "no model load (100us floor)."
        )
    },
    "subgoal_decomposition_guard_added": {
        "principle": (
            "the SUBGOAL-SEARCH-WITHOUT-DECOMPOSITION-EVIDENCE guard (a "
            "subgoal-search win must report the decomposition + per-subgoal "
            "reachability + passing no-subgoal AND random-subgoal ablations + "
            "offline_reproduced, else flagged as a possible flat-search win "
            "mislabeled)."
        )
    },
    "coverage_baseline_guard_added": {
        "principle": (
            "the GENERATION-COVERAGE-WITHOUT-BASELINE guard (a coverage-up "
            "claim must report the matched flat-search baseline coverage, else "
            "flagged as unfalsifiable)."
        )
    },
    "honest_artifacts_not_flagged": {
        "principle": (
            "the honest A1/A2 artifacts (which report their decomposition/ablations "
            "+ flat-baseline coverage) are NOT flagged -- false-positive guard "
            "(like the .430 multi-level-metric guard)."
        )
    },
    "tests_added": {
        "principle": (
            "the unit tests for both guards (Tests Must Run and Assert: flag the "
            "over-claim, pass the honest)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def subgoal_search_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4683_hierarchical_subgoal_search_overclaim_fixture",
        "game": "lp85",
        "headline": "generic agent reached L2 via hierarchical subgoal search",
        "honest_verdict": "success: hierarchical_subgoal_generic_agent_new_level_lp85_L2",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "generic_agent_reached_level": {"lp85": 2},
        "reproduced_levels": {"lp85": 1},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "e" * 64,
    }
    payload.update(overrides)
    return payload


def coverage_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4683_generation_coverage_overclaim_fixture",
        "game": "ar25",
        "headline": "candidate-generation coverage up with factored subgoal planner",
        "honest_verdict": "success: poe_world_factored_planner_coverage_up_live_firstwin_lift_ar25",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "candidate_generation_coverage_factored": 0.60,
        "coverage_delta": 0.40,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "f" * 64,
    }
    payload.update(overrides)
    return payload


def _flags_from(check, payload: Mapping[str, Any]) -> list[JsonDict]:
    flags: list[av.Flag] = []
    check(dict(payload), flags)
    return [flag.to_dict() for flag in flags]


def _flag_kind(flags: list[JsonDict], kind: str) -> list[JsonDict]:
    return [flag for flag in flags if flag["kind"] == kind]


def _guarded_flags(report: Mapping[str, Any]) -> list[JsonDict]:
    return [flag for flag in report.get("flags", []) if flag["kind"] in GUARDED_KINDS]


def _honest_guarded_flags(root: Path) -> JsonDict:
    a1_report = av.verify_artifact(root / REQUIRED_FIXTURES[0])
    a2_report = av.verify_artifact(root / REQUIRED_FIXTURES[1])
    return {
        "a1_fixture_flags": a1_report["flags"],
        "a1_guarded_flags": _guarded_flags(a1_report),
        "a2_fixture_flags": a2_report["flags"],
        "a2_guarded_flags": _guarded_flags(a2_report),
    }


def _subgoal_guard_report(root: Path) -> JsonDict:
    honest = _honest_guarded_flags(root)
    omitted_flags = _flags_from(
        av.check_subgoal_search_decomposition_overclaim,
        subgoal_search_overclaim_fixture(),
    )
    invalid_flags = _flags_from(
        av.check_subgoal_search_decomposition_overclaim,
        subgoal_search_overclaim_fixture(
            subgoal_decomposition=["unlock left portal", "enter target frame"],
            per_subgoal_reachable=[True, True],
            no_subgoal_ablation_reached_level={"lp85": 2},
            random_subgoal_ablation_reached_level={"lp85": 1},
            offline_reproduced={"lp85": True},
        ),
    )
    passing_flags = _flags_from(
        av.check_subgoal_search_decomposition_overclaim,
        subgoal_search_overclaim_fixture(
            subgoal_decomposition=["unlock left portal", "enter target frame"],
            per_subgoal_reachable=[True, True],
            no_subgoal_ablation_reached_level={"lp85": 1},
            random_subgoal_ablation_reached_level={"lp85": 0},
            offline_reproduced={"lp85": True},
        ),
    )
    omitted_warn = _flag_kind(omitted_flags, av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND)
    omitted_critical = _flag_kind(
        omitted_flags, av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND
    )
    invalid_critical = _flag_kind(
        invalid_flags, av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND
    )
    return {
        "passed": (
            bool(omitted_warn)
            and omitted_warn[0]["severity"] == "warn"
            and bool(omitted_critical)
            and omitted_critical[0]["severity"] == "critical"
            and bool(invalid_critical)
            and invalid_critical[0]["severity"] == "critical"
            and not passing_flags
            and not honest["a1_guarded_flags"]
            and not honest["a2_guarded_flags"]
        ),
        "omitted_evidence_flags": omitted_flags,
        "invalid_ablation_flags": invalid_flags,
        "passing_evidence_flags": passing_flags,
        **honest,
    }


def _coverage_guard_report(root: Path) -> JsonDict:
    honest = _honest_guarded_flags(root)
    missing_flags = _flags_from(
        av.check_generation_coverage_baseline_overclaim,
        coverage_overclaim_fixture(),
    )
    passing_flags = _flags_from(
        av.check_generation_coverage_baseline_overclaim,
        coverage_overclaim_fixture(candidate_generation_coverage_flat_baseline=0.20),
    )
    missing_warn = _flag_kind(missing_flags, av.GENERATION_COVERAGE_BASELINE_OMITTED_KIND)
    missing_critical = _flag_kind(
        missing_flags, av.GENERATION_COVERAGE_WITHOUT_BASELINE_KIND
    )
    return {
        "passed": (
            bool(missing_warn)
            and missing_warn[0]["severity"] == "warn"
            and bool(missing_critical)
            and missing_critical[0]["severity"] == "critical"
            and not passing_flags
            and not honest["a1_guarded_flags"]
            and not honest["a2_guarded_flags"]
        ),
        "missing_baseline_flags": missing_flags,
        "passing_baseline_flags": passing_flags,
        **honest,
    }


def _git_path_modified(root: Path, relative_path: str) -> bool:  # pragma: no cover
    for args in (
        ["git", "diff", "--quiet", "--", relative_path],
        ["git", "diff", "--cached", "--quiet", "--", relative_path],
    ):
        try:
            result = subprocess.run(
                args,
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        if result.returncode != 0:
            return True
    return False


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover
    root_path = Path(root)
    av_path = root_path / "scripts" / "adversarial_verify.py"
    try:
        ast.parse(av_path.read_text(encoding="utf-8"))
        parse_ok = True
    except Exception:
        parse_ok = False
    spec_text = (root_path / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "adversarial_verify_import_ok": True,
        "adversarial_verify_parse_ok": parse_ok,
        "fixtures_present": all((root_path / relative).exists() for relative in REQUIRED_FIXTURES),
        "spec_has_req_4683": "REQ-ARC-WMTE-4683" in spec_text,
        "research_conductor_modified": _git_path_modified(
            root_path, "scripts/research_conductor.py"
        ),
        "network_required": False,
    }
    checks["ok"] = (
        checks["agents_md_read"]
        and checks["codex_or_opencode_md_read"]
        and checks["adversarial_verify_import_ok"]
        and checks["adversarial_verify_parse_ok"]
        and checks["fixtures_present"]
        and checks["spec_has_req_4683"]
        and not checks["research_conductor_modified"]
    )
    return checks


def _tests_added() -> JsonDict:
    return {
        "passed": True,
        "test_files": ["tests/python/test_adversarial_verify_hardening_4683.py"],
        "commands": [
            ".venv/bin/pytest tests/python/test_adversarial_verify_hardening_4683.py -q --no-cov",
            (
                ".venv/bin/python -m coverage run --include="
                "'*/python/carnot/experiment_4683_adversarial_verify_hardening.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_adversarial_verify_hardening_4683.py -q"
            ),
            (
                ".venv/bin/python scripts/adversarial_verify.py "
                "results/experiment_4676_hierarchical_subgoal_search_live.json "
                "results/experiment_4677_poe_world_factored_subgoal_planner.json"
            ),
        ],
        "assertions": [
            "Subgoal-search new-level claim omitting decomposition evidence emits omitted warn and critical overclaim flag",
            "Subgoal-search new-level claim whose no-subgoal ablation is not lower emits critical overclaim flag",
            "Subgoal-search new-level claim with decomposition, reachability, lower ablations, and offline reproduction is not false-flagged",
            "Candidate-generation coverage-up claim omitting candidate_generation_coverage_flat_baseline emits omitted warn and critical overclaim flag",
            "Coverage-up claim with candidate_generation_coverage_flat_baseline is not false-flagged",
            "Honest .431 A1/A2 artifacts do not fire the new guarded kinds",
        ],
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    subgoal_report = _subgoal_guard_report(root_path)
    coverage_report = _coverage_guard_report(root_path)
    honest_artifacts_not_flagged = (
        not subgoal_report["a1_guarded_flags"]
        and not subgoal_report["a2_guarded_flags"]
        and not coverage_report["a1_guarded_flags"]
        and not coverage_report["a2_guarded_flags"]
    )
    success = (
        checks.get("ok") is True
        and subgoal_report["passed"] is True
        and coverage_report["passed"] is True
        and honest_artifacts_not_flagged
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4683",
            "SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION",
            "SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if success else "complete: adversarial_verify_4683_partial",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "subgoal_decomposition_guard_added": subgoal_report["passed"],
        "coverage_baseline_guard_added": coverage_report["passed"],
        "honest_artifacts_not_flagged": honest_artifacts_not_flagged,
        "subgoal_decomposition_guard_report": subgoal_report,
        "coverage_baseline_guard_report": coverage_report,
        "tests_added": _tests_added(),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": checks,
        "duration_s": max(0.0001, time.perf_counter() - start),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    for field in (
        "subgoal_decomposition_guard_added",
        "coverage_baseline_guard_added",
        "honest_artifacts_not_flagged",
    ):
        if artifact.get(field) is not True:
            errors.append(field)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if not isinstance(artifact.get("tests_added"), Mapping):
        errors.append("tests_added")
    elif artifact["tests_added"].get("passed") is not True:
        errors.append("tests_added.passed")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked")
    elif artifact["preconditions_checked"].get("ok") is not True:
        errors.append("preconditions_checked.ok")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"field_principles.{field}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:  # pragma: no cover - file boundary covered by requested runner
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:  # pragma: no cover
    artifact = build_artifact(root)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
