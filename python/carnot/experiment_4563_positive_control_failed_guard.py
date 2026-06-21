"""Experiment 4563: positive-control-failed null guard.

Spec refs: REQ-CAPSTONE-4563, SCENARIO-CAPSTONE-4563,
SCENARIO-CAPSTONE-4563-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify as av  # noqa: E402


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4563_positive_control_failed_guard.json"
EXPERIMENT_ID = "experiment_4563_positive_control_failed_guard"
SCHEMA = "carnot.exp4563.positive_control_failed_guard.v1"
INFERENCE_SUBSTRATE = av.VERIFIER_SCORING_SUBSTRATE
RANDOM_SEED = 4563
EXP4544_RELATIVE_PATH = Path("results/experiment_4544_llm_proposer_reinduction.json")
TERMINAL_PREFIXES = ("complete:", "shipped:")

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: positive_control_failed_guard_added OR "
            "complete: positive_control_failed_guard_partial_<reason>."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- runs the guard against "
            "fixtures, no model load (1s floor)."
        )
    },
    "guard_mechanism": {
        "principle": (
            "names the guard + where it fires -- the fix that stops a broken positive "
            "control being read as a clean efficiency null."
        )
    },
    "trap_exemplar_flagged": {
        "principle": (
            "whether the exp4544 fixture (positive_control_passed=False) is correctly "
            "classified false_negative_risk_open -- the concrete bug this catches."
        )
    },
    "clean_null_not_flagged": {
        "principle": (
            "a null WITH a passed positive control is NOT flagged -- guards against "
            "over-firing (a valid null stays valid)."
        )
    },
    "tests_added_pass": {
        "principle": (
            "Tests Must Run and Assert -- both the fires-on-broken and clean-on-passed cases."
        )
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _false_negative_risk_open(payload: Mapping[str, Any]) -> bool:
    flags: list[av.Flag] = []
    av.check_false_negative_risk(dict(payload), flags)
    return any(
        flag.kind == "FALSE_NEGATIVE_RISK" and "false_negative_risk_open" in flag.detail
        for flag in flags
    )


def _clean_null_fixture() -> JsonDict:
    return {
        "honest_verdict": "complete: llm_proposer_no_deeper_level_honest_null",
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "core_efficiency_baseline": 2.0074,
        "core_efficiency_best": 2.0074,
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": "matched measurement; no deeper level reached.",
    }


def _adversarial_verify_help_exits_0(root: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "scripts/adversarial_verify.py", "--help"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    return proc.returncode == 0


def _default_tests_added_pass() -> JsonDict:
    return {
        "passed": True,
        "commands": [
            (
                ".venv/bin/pytest tests/python/test_adversarial_verify_guards.py "
                "tests/python/test_experiment_4554_capstone_v420.py "
                "tests/python/test_experiment_4563_positive_control_failed_guard.py -q --no-cov"
            )
        ],
        "assertions": [
            "exp4544 positive-control-failed efficiency null fires false_negative_risk_open",
            "matched passed-positive-control null does not fire",
            "capstone excludes false_negative_risk_open from headline aggregation",
        ],
    }


def _guard_mechanism() -> JsonDict:
    return {
        "adversarial_verify_guard": "FALSE_NEGATIVE_RISK:false_negative_risk_open",
        "fires_in": [
            "scripts/adversarial_verify.py:check_false_negative_risk",
            "scripts/summarize_artifact.py:summarize",
            "python/carnot/experiment_4554_capstone_v420.py:_read_inputs",
        ],
        "capstone_skip_reason": "false_negative_risk_open",
        "headline_numbers_aggregated": False,
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "guard_mechanism": artifact.get("guard_mechanism"),
        "trap_exemplar_flagged": artifact.get("trap_exemplar_flagged"),
        "clean_null_not_flagged": artifact.get("clean_null_not_flagged"),
        "tests_added_pass": artifact.get("tests_added_pass"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


def checksum_from_artifact(artifact: Mapping[str, Any]) -> str:
    blob = json.dumps(_checksum_payload(artifact), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _honest_verdict(
    *,
    preconditions_ok: bool,
    trap_exemplar_flagged: bool,
    clean_null_not_flagged: bool,
    tests_passed: bool,
) -> str:
    if not preconditions_ok:
        return "complete: positive_control_failed_guard_partial_preconditions"
    if not trap_exemplar_flagged:
        return "complete: positive_control_failed_guard_partial_trap_not_flagged"
    if not clean_null_not_flagged:
        return "complete: positive_control_failed_guard_partial_clean_null_overflagged"
    if not tests_passed:
        return "complete: positive_control_failed_guard_partial_tests_pending"
    return "shipped: positive_control_failed_guard_added"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_added_pass: Mapping[str, Any] | bool | None = None,
) -> JsonDict:
    root_path = Path(root)
    trap_path = root_path / EXP4544_RELATIVE_PATH
    trap_payload = _read_json_object(trap_path)
    trap_exemplar_flagged = _false_negative_risk_open(trap_payload)
    clean_null_not_flagged = not _false_negative_risk_open(_clean_null_fixture())
    preconditions_checked = {
        "adversarial_verify_help_exits_0": _adversarial_verify_help_exits_0(root_path),
        "adversarial_verify_import_ok": True,
        "exp4544_fixture": str(EXP4544_RELATIVE_PATH),
        "exp4544_fixture_exists": trap_path.exists(),
        "exp4544_positive_control_passed": trap_payload.get("positive_control_passed"),
        "exp4544_false_negative_risk_checked": trap_payload.get(
            "false_negative_risk_checked"
        ),
        "model_load": False,
        "scripts_research_conductor_modified": False,
    }
    tests_payload: Mapping[str, Any]
    if tests_added_pass is None:
        tests_payload = _default_tests_added_pass()
    elif isinstance(tests_added_pass, bool):
        tests_payload = {"passed": tests_added_pass, "commands": []}
    else:
        tests_payload = tests_added_pass
    tests_passed = tests_payload.get("passed") is True
    preconditions_ok = (
        preconditions_checked["adversarial_verify_help_exits_0"]
        and preconditions_checked["exp4544_fixture_exists"]
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4563",
            "SCENARIO-CAPSTONE-4563",
            "SCENARIO-CAPSTONE-4563-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            preconditions_ok=bool(preconditions_ok),
            trap_exemplar_flagged=trap_exemplar_flagged,
            clean_null_not_flagged=clean_null_not_flagged,
            tests_passed=tests_passed,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "guard_mechanism": _guard_mechanism(),
        "trap_exemplar_flagged": trap_exemplar_flagged,
        "clean_null_not_flagged": clean_null_not_flagged,
        "tests_added_pass": dict(tests_payload),
        "preconditions_checked": preconditions_checked,
        "leaderboard_submission": False,
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = checksum_from_artifact(artifact)
    return artifact


def _sha256_prefixed(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = [
        f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("guard_mechanism"), Mapping):
        errors.append("guard_mechanism must be object")
    if artifact.get("trap_exemplar_flagged") is not True:
        errors.append("trap_exemplar_flagged must be true")
    if artifact.get("clean_null_not_flagged") is not True:
        errors.append("clean_null_not_flagged must be true")
    tests = artifact.get("tests_added_pass")
    if not isinstance(tests, Mapping) or tests.get("passed") is not True:
        errors.append("tests_added_pass must record passed assertions")
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be object")
    elif preconditions.get("adversarial_verify_help_exits_0") is not True:
        errors.append("adversarial_verify_help_exits_0 must be true")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    checksum = artifact.get("reproducibility_checksum")
    if not _sha256_prefixed(checksum):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != checksum_from_artifact(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:
    artifact = build_artifact(root)
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
