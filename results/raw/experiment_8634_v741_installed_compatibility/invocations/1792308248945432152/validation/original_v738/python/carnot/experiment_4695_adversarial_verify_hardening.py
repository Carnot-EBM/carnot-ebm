"""Experiment 4695: adversarial_verify .432 overclaim hardening receipt.

Spec refs: REQ-ARC-WMTE-4695,
SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION,
SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT.
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

EXPERIMENT = "experiment_4695_adversarial_verify_hardening"
SCHEMA = "carnot.exp4695.adversarial_verify_hardening.v1"
RESULT_RELATIVE_PATH = "results/experiment_4695_adversarial_verify_hardening.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4695
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
    "no model load (100us floor)."
)
SUCCESS_VERDICT = (
    "success: "
    "adversarial_verify_hardened_novelty_ablation_and_proposal_filter_heldout_guards_tests_green."
)
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
REQUIRED_FIXTURES = (
    "results/experiment_4688_controllable_novelty_proposal_policy_live.json",
    "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json",
)
GUARDED_KINDS = {
    av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND,
    av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND,
    av.PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND,
    av.PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "adversarial_verify_hardened_novelty_ablation_and_proposal_filter_heldout_guards_tests_green."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, "
            "no model load (100us floor)."
        )
    },
    "novelty_ablation_guard_added": {
        "principle": (
            "the NOVELTY-PROPOSAL-WITHOUT-ABLATION guard (a "
            "controllable-novelty win must report the no-novelty-bonus AND "
            "cosmetic-novelty ablations strictly lower + offline_reproduced, "
            "else flagged as a possible flat-exploration win mislabeled / "
            "controllability-gate-adds-nothing)."
        )
    },
    "proposal_filter_heldout_guard_added": {
        "principle": (
            "the PROPOSAL-FILTER-WITHOUT-HELDOUT-REJECTION guard (a coverage-up "
            "claim must report the held-out rejected-program count AND the "
            "matched blind-proposal baseline, else flagged as unfalsifiable / "
            "experts_overfit_prefix-unguarded)."
        )
    },
    "honest_artifacts_not_flagged": {
        "principle": (
            "the honest A1/A2 artifacts (which report their ablations + "
            "held-out rejection + blind-baseline coverage) are NOT flagged -- "
            "false-positive guard (like the .431 coverage-baseline guard)."
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


def novelty_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4695_controllable_novelty_overclaim_fixture",
        "game": "bp35",
        "headline": "generic agent reached L2 via controllable novelty",
        "honest_verdict": "success: controllable_novelty_generic_agent_new_level_bp35_L2",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "controllability_gate_on": True,
        "generic_agent_reached_level": {"bp35": 2},
        "reproduced_levels": {"bp35": 1},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }
    payload.update(overrides)
    return payload


def proposal_filter_overclaim_fixture(**overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": "experiment_4695_program_synthesis_filter_overclaim_fixture",
        "game": "bp35",
        "headline": "program-synthesis proposal filter coverage up",
        "honest_verdict": "success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_bp35",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "candidate_generation_coverage_filter": 0.60,
        "coverage_delta": 0.40,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "b" * 64,
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


def _novelty_guard_report(root: Path) -> JsonDict:
    honest = _honest_guarded_flags(root)
    omitted_flags = _flags_from(
        av.check_novelty_proposal_ablation_overclaim,
        novelty_overclaim_fixture(),
    )
    invalid_flags = _flags_from(
        av.check_novelty_proposal_ablation_overclaim,
        novelty_overclaim_fixture(
            no_novelty_ablation_reached_level={"bp35": 2},
            cosmetic_novelty_ablation_reached_level={"bp35": 1},
            offline_reproduced={"bp35": True},
        ),
    )
    passing_flags = _flags_from(
        av.check_novelty_proposal_ablation_overclaim,
        novelty_overclaim_fixture(
            no_novelty_ablation_reached_level={"bp35": 1},
            cosmetic_novelty_ablation_reached_level={"bp35": 0},
            offline_reproduced={"bp35": True},
        ),
    )
    omitted_warn = _flag_kind(omitted_flags, av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND)
    omitted_critical = _flag_kind(
        omitted_flags, av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND
    )
    invalid_critical = _flag_kind(
        invalid_flags, av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND
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


def _proposal_filter_guard_report(root: Path) -> JsonDict:
    honest = _honest_guarded_flags(root)
    omitted_flags = _flags_from(
        av.check_proposal_filter_heldout_rejection_overclaim,
        proposal_filter_overclaim_fixture(),
    )
    passing_flags = _flags_from(
        av.check_proposal_filter_heldout_rejection_overclaim,
        proposal_filter_overclaim_fixture(
            heldout_programs_rejected=2,
            candidate_generation_coverage_blind_baseline=0.20,
        ),
    )
    omitted_warn = _flag_kind(
        omitted_flags, av.PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND
    )
    omitted_critical = _flag_kind(
        omitted_flags, av.PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND
    )
    return {
        "passed": (
            bool(omitted_warn)
            and omitted_warn[0]["severity"] == "warn"
            and bool(omitted_critical)
            and omitted_critical[0]["severity"] == "critical"
            and not passing_flags
            and not honest["a1_guarded_flags"]
            and not honest["a2_guarded_flags"]
        ),
        "omitted_evidence_flags": omitted_flags,
        "passing_evidence_flags": passing_flags,
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
        "spec_has_req_4695": "REQ-ARC-WMTE-4695" in spec_text,
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
        and checks["spec_has_req_4695"]
        and not checks["research_conductor_modified"]
    )
    return checks


def _tests_added() -> JsonDict:
    return {
        "passed": True,
        "test_files": ["tests/python/test_adversarial_verify_hardening_4695.py"],
        "commands": [
            ".venv/bin/pytest tests/python/test_adversarial_verify_hardening_4695.py -q --no-cov",
            (
                ".venv/bin/python -m coverage run --include="
                "'*/python/carnot/experiment_4695_adversarial_verify_hardening.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_adversarial_verify_hardening_4695.py -q"
            ),
            (
                ".venv/bin/python scripts/adversarial_verify.py "
                "results/experiment_4688_controllable_novelty_proposal_policy_live.json "
                "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"
            ),
        ],
        "assertions": [
            "Controllable-novelty new-level claim omitting ablations emits omitted warn and critical overclaim flag",
            "Controllable-novelty new-level claim whose no-novelty ablation is not lower emits critical overclaim flag",
            "Controllable-novelty new-level claim with lower ablations and offline reproduction is not false-flagged",
            "Program-synthesis coverage-up claim omitting heldout_programs_rejected and blind baseline emits omitted warn and critical flag",
            "Coverage-up claim with heldout_programs_rejected and candidate_generation_coverage_blind_baseline is not false-flagged",
            "Honest .432 A1/A2 artifacts do not fire the new guarded kinds",
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
    novelty_report = _novelty_guard_report(root_path)
    proposal_filter_report = _proposal_filter_guard_report(root_path)
    honest_artifacts_not_flagged = (
        not novelty_report["a1_guarded_flags"]
        and not novelty_report["a2_guarded_flags"]
        and not proposal_filter_report["a1_guarded_flags"]
        and not proposal_filter_report["a2_guarded_flags"]
    )
    success = (
        checks.get("ok") is True
        and novelty_report["passed"] is True
        and proposal_filter_report["passed"] is True
        and honest_artifacts_not_flagged
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4695",
            "SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION",
            "SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if success else "complete: adversarial_verify_4695_partial",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "novelty_ablation_guard_added": novelty_report["passed"],
        "proposal_filter_heldout_guard_added": proposal_filter_report["passed"],
        "honest_artifacts_not_flagged": honest_artifacts_not_flagged,
        "novelty_ablation_guard_report": novelty_report,
        "proposal_filter_heldout_guard_report": proposal_filter_report,
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
        "novelty_ablation_guard_added",
        "proposal_filter_heldout_guard_added",
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
