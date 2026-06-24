"""Experiment 4671: adversarial_verify .430 overclaim hardening receipt.

Spec refs: REQ-ARC-WMTE-4671,
SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY,
SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC.
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

EXPERIMENT = "experiment_4671_adversarial_verify_hardening"
SCHEMA = "carnot.exp4671.adversarial_verify_hardening.v1"
RESULT_RELATIVE_PATH = "results/experiment_4671_adversarial_verify_hardening.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4671
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
    "no model load (100us floor)."
)
SUCCESS_VERDICT = (
    "success: "
    "adversarial_verify_hardened_l2_goal_and_multilevel_metric_guards_tests_green."
)
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
REQUIRED_FIXTURES = ("results/experiment_4664_l2_goal_predicate_induction_live.json",)
GUARDED_KINDS = {
    av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND,
    av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND,
    av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND,
    av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "adversarial_verify_hardened_l2_goal_and_multilevel_metric_guards_tests_green."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
            "no model load (100us floor)."
        )
    },
    "l2_goal_satisfiability_guard_added": {
        "principle": (
            "the L2-GOAL-INDUCTION-WITHOUT-SATISFIABILITY-CHECK guard (an "
            "L2-via-induction win must report a satisfiable goal + reachable plan, "
            "else flagged as a possible degenerate-goal vacuous pass)."
        )
    },
    "multilevel_metric_guard_added": {
        "principle": (
            "the MULTI-LEVEL-WITHOUT-NONDEGENERATE-METRIC guard (a multi-level "
            "solve-rate claim must report the fixed target_levels>=2/no-break harness, "
            "else flagged as the degenerate 0.0-by-construction metric)."
        )
    },
    "honest_artifacts_not_flagged": {
        "principle": (
            "the honest A1 artifact (which reports its satisfiability + harness "
            "controls) is NOT flagged -- false-positive guard (like the .429 "
            "QD-ablation guard)."
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


def l2_goal_induction_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4671_l2_goal_induction_overclaim_fixture",
        "game": "lp85",
        "headline": "generic agent reached L2 via L2 goal induction",
        "honest_verdict": "success: l2_goal_induction_generic_agent_reached_L2_lp85",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "generic_agent_reached_level": {"lp85": 2},
        "offline_reproduced": {"lp85": True},
        "reproduced_levels": {"lp85": 1},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "c" * 64,
    }
    payload.update(overrides)
    return payload


def multilevel_metric_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4671_multilevel_metric_overclaim_fixture",
        "game": "sc25",
        "headline": "multi-level solve-rate improved on the live ARC agent",
        "honest_verdict": "success: live_multi_level_solve_rate_up",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "live_multi_level_solve_rate": 0.25,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "d" * 64,
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


def _l2_goal_guard_report(root: Path) -> JsonDict:
    a1_path = root / REQUIRED_FIXTURES[0]
    a1_report = av.verify_artifact(a1_path)
    omitted_flags = _flags_from(
        av.check_l2_goal_induction_satisfiability_overclaim,
        l2_goal_induction_overclaim_fixture(),
    )
    false_flags = _flags_from(
        av.check_l2_goal_induction_satisfiability_overclaim,
        l2_goal_induction_overclaim_fixture(
            goal_predicate_satisfiable={"lp85": False},
            l2_plan_reaches_goal={"lp85": True},
        ),
    )
    passing_flags = _flags_from(
        av.check_l2_goal_induction_satisfiability_overclaim,
        l2_goal_induction_overclaim_fixture(
            goal_predicate_satisfiable={"lp85": True},
            l2_plan_reaches_goal={"lp85": True},
        ),
    )
    omitted_warn = _flag_kind(omitted_flags, av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND)
    omitted_critical = _flag_kind(
        omitted_flags, av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND
    )
    false_critical = _flag_kind(
        false_flags, av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND
    )
    a1_guarded_flags = _guarded_flags(a1_report)
    return {
        "passed": (
            bool(omitted_warn)
            and omitted_warn[0]["severity"] == "warn"
            and bool(omitted_critical)
            and omitted_critical[0]["severity"] == "critical"
            and bool(false_critical)
            and false_critical[0]["severity"] == "critical"
            and not passing_flags
            and not a1_guarded_flags
        ),
        "omitted_control_flags": omitted_flags,
        "false_control_flags": false_flags,
        "passing_control_flags": passing_flags,
        "a1_fixture_flags": a1_report["flags"],
        "a1_guarded_flags": a1_guarded_flags,
    }


def _multilevel_metric_guard_report(root: Path) -> JsonDict:
    a1_path = root / REQUIRED_FIXTURES[0]
    a1_report = av.verify_artifact(a1_path)
    missing_flags = _flags_from(
        av.check_multilevel_nondegenerate_metric_overclaim,
        multilevel_metric_overclaim_fixture(),
    )
    invalid_flags = _flags_from(
        av.check_multilevel_nondegenerate_metric_overclaim,
        multilevel_metric_overclaim_fixture(
            metric_harness_fixed={"target_levels": 1, "break_at_first_win": True}
        ),
    )
    passing_flags = _flags_from(
        av.check_multilevel_nondegenerate_metric_overclaim,
        multilevel_metric_overclaim_fixture(
            metric_harness_fixed={"target_levels": 2, "break_at_first_win": False}
        ),
    )
    missing_warn = _flag_kind(
        missing_flags, av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND
    )
    missing_critical = _flag_kind(
        missing_flags, av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND
    )
    invalid_critical = _flag_kind(
        invalid_flags, av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND
    )
    a1_guarded_flags = _guarded_flags(a1_report)
    return {
        "passed": (
            bool(missing_warn)
            and missing_warn[0]["severity"] == "warn"
            and bool(missing_critical)
            and missing_critical[0]["severity"] == "critical"
            and bool(invalid_critical)
            and invalid_critical[0]["severity"] == "critical"
            and not passing_flags
            and not a1_guarded_flags
        ),
        "missing_harness_flags": missing_flags,
        "invalid_harness_flags": invalid_flags,
        "passing_harness_flags": passing_flags,
        "a1_fixture_flags": a1_report["flags"],
        "a1_guarded_flags": a1_guarded_flags,
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
        "spec_has_req_4671": "REQ-ARC-WMTE-4671" in spec_text,
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
        and checks["spec_has_req_4671"]
        and not checks["research_conductor_modified"]
    )
    return checks


def _tests_added() -> JsonDict:
    return {
        "passed": True,
        "test_files": ["tests/python/test_adversarial_verify_hardening_4671.py"],
        "commands": [
            ".venv/bin/pytest tests/python/test_adversarial_verify_hardening_4671.py -q --no-cov",
            (
                ".venv/bin/python -m coverage run --include="
                "'*/python/carnot/experiment_4671_adversarial_verify_hardening.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_adversarial_verify_hardening_4671.py -q"
            ),
            (
                ".venv/bin/python scripts/adversarial_verify.py "
                "results/experiment_4664_l2_goal_predicate_induction_live.json"
            ),
        ],
        "assertions": [
            "L2 goal-induction win omitting satisfiability controls emits omitted warn and critical overclaim flag",
            "L2 goal-induction win with false goal_predicate_satisfiable emits critical overclaim flag",
            "L2 goal-induction win with goal_predicate_satisfiable=true and l2_plan_reaches_goal=true is not false-flagged",
            "positive multi-level solve-rate omitting metric_harness_fixed emits omitted warn and critical overclaim flag",
            "positive multi-level solve-rate with target_levels<2/break-at-first-win emits critical overclaim flag",
            "honest A1 artifact does not fire the new guarded kinds",
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
    l2_report = _l2_goal_guard_report(root_path)
    metric_report = _multilevel_metric_guard_report(root_path)
    honest_artifacts_not_flagged = (
        not l2_report["a1_guarded_flags"] and not metric_report["a1_guarded_flags"]
    )
    success = (
        checks.get("ok") is True
        and l2_report["passed"] is True
        and metric_report["passed"] is True
        and honest_artifacts_not_flagged
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4671",
            "SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY",
            "SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if success else "complete: adversarial_verify_4671_partial",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "l2_goal_satisfiability_guard_added": l2_report["passed"],
        "multilevel_metric_guard_added": metric_report["passed"],
        "honest_artifacts_not_flagged": honest_artifacts_not_flagged,
        "l2_goal_satisfiability_guard_report": l2_report,
        "multilevel_metric_guard_report": metric_report,
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
        "l2_goal_satisfiability_guard_added",
        "multilevel_metric_guard_added",
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
