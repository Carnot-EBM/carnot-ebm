"""Experiment 4732: adversarial_verify lever exercise-evidence guard receipt.

Spec refs: REQ-ARC-WMTE-4732, SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify as av  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4732_adversarial_verify_exercise_evidence_guard"
SCHEMA = "carnot.exp4732.adversarial_verify_exercise_evidence_guard.v1"
RESULT_RELATIVE_PATH = "results/experiment_4732_adversarial_verify_exercise_evidence_guard.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
PINNING_TEST_PATH = "tests/python/test_adversarial_verify_lever_exercise_evidence.py"
RANDOM_SEED = 4732
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- guard dev + a pytest run, no model load "
    "(100us floor)."
)
SUCCESS_VERDICT = "success: lever_exercise_evidence_guard_shipped_pinned."
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
REQUIRED_FIXTURES = {
    "go_explore_dead_archive": (
        "results/experiment_4701_amortized_exploration_prior_go_explore_live.json"
    ),
    "exp4710_cnn_dict_candidate_bug": (
        "results/experiment_4710_online_action_learning_arms_online_scratch.json"
    ),
    "a4_byte_identical_all_arms": (
        "results/experiment_4715_online_action_learning_driver_corrected.json"
    ),
    "trustworthy_nondegenerate_null": (
        "results/experiment_4726_online_action_learning_driver_valid_test.json"
    ),
}
PROMPT_EXP4710_ALIAS = "results/experiment_4710_online_action_learning_arms_summary.json"
ADVERSARIAL_VERIFY_TESTS = (
    "tests/python/test_adversarial_verify_guards.py",
    "tests/python/test_adversarial_verify_degenerate_controls.py",
    "tests/python/test_adversarial_verify_degenerate_separation.py",
    "tests/python/test_adversarial_verify_lever_exercise_evidence.py",
    "tests/python/test_adversarial_verify_hardening_4695.py",
    "tests/python/test_adversarial_verify_hardening_4683.py",
    "tests/python/test_adversarial_verify_hardening_4671.py",
    "tests/python/test_adversarial_verify_hardening_4659.py",
    "tests/python/test_adversarial_verify_hardening_4647.py",
    "tests/python/test_adversarial_verify_hardening_4635.py",
    "tests/python/test_adversarial_verify_hardening_4623.py",
    "tests/python/test_adversarial_verify_hardening_4611.py",
    "tests/python/test_adversarial_verify_arc_self_solve.py",
    "tests/python/test_adversarial_verify_baseline_identity_tautology.py",
    "tests/python/test_adversarial_verify_hardening_4707.py",
    "tests/python/test_adversarial_verify_control_treatment_null.py",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success: lever_exercise_evidence_guard_shipped_pinned."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- guard dev + a pytest run, "
            "no model load (100us floor)."
        )
    },
    "check_added": {
        "principle": (
            "LEVER_EXERCISE_EVIDENCE_DEGENERATE -- the new guard name; mechanizes "
            "the operator's manual silent-dead-code catch."
        )
    },
    "dead_code_exemplars_flagged": {
        "principle": (
            "the three dead-code nulls (Go-Explore (1,64,64); exp4710 CNN "
            "dict-candidate; .434 A4 byte-identical arms) are flagged in retrospect -- "
            "the guard works."
        )
    },
    "trustworthy_null_not_flagged": {
        "principle": (
            "a genuine non-degenerate null is NOT flagged -- the false-positive "
            "carve-out works (a real null must survive)."
        )
    },
    "existing_suite_green": {
        "principle": (
            "the existing adversarial_verify tests stay green -- no existing check "
            "weakened (the never-weaken-the-linter rule)."
        )
    },
    "pinning_test_path": {
        "principle": (
            "tests/python/test_adversarial_verify_lever_exercise_evidence.py -- "
            "the standing pin."
        )
    },
    "verifier_is_oracle": {"principle": "false -- a linter check invokes no oracle."},
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (adversarial_verify importable, exemplars present); "
            "pre-empts missing-resource fabrication."
        )
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


def _run_command(root: Path, args: list[str], timeout_s: int = 120) -> JsonDict:
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    return {
        "command": " ".join(args),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_s": max(0.0001, time.perf_counter() - start),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _lever_flags(path: Path) -> list[JsonDict]:
    report = av.verify_artifact(path)
    return [
        flag
        for flag in report.get("flags", [])
        if flag.get("kind") == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_text = (root_path / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    fixtures_present = {
        name: (root_path / relative).exists() for name, relative in REQUIRED_FIXTURES.items()
    }
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "adversarial_verify_importable": hasattr(
            av, "check_lever_exercise_evidence"
        ),
        "guard_constant_present": getattr(
            av, "LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND", ""
        )
        == "LEVER_EXERCISE_EVIDENCE_DEGENERATE",
        "fixtures_present": fixtures_present,
        "prompt_exp4710_alias_present": (root_path / PROMPT_EXP4710_ALIAS).exists(),
        "prompt_exp4710_alias_resolved_to": REQUIRED_FIXTURES[
            "exp4710_cnn_dict_candidate_bug"
        ],
        "pinning_test_present": (root_path / PINNING_TEST_PATH).exists(),
        "spec_has_req_4732": "REQ-ARC-WMTE-4732" in spec_text,
        "network_required": False,
    }
    checks["ok"] = (
        checks["agents_md_read"]
        and checks["codex_or_opencode_md_read"]
        and checks["adversarial_verify_importable"]
        and checks["guard_constant_present"]
        and all(fixtures_present.values())
        and checks["pinning_test_present"]
        and checks["spec_has_req_4732"]
    )
    return checks


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    start = time.perf_counter()
    root_path = Path(root)
    checks = check_preconditions(root_path)
    dead_code_reports = {
        name: _lever_flags(root_path / relative)
        for name, relative in REQUIRED_FIXTURES.items()
        if name != "trustworthy_nondegenerate_null"
    }
    trustworthy_flags = _lever_flags(
        root_path / REQUIRED_FIXTURES["trustworthy_nondegenerate_null"]
    )
    pinning_test = _run_command(
        root_path,
        [sys.executable, "-m", "pytest", PINNING_TEST_PATH, "-q", "--no-cov"],
        timeout_s=120,
    )
    existing_suite = _run_command(
        root_path,
        [sys.executable, "-m", "pytest", *ADVERSARIAL_VERIFY_TESTS, "-q", "--no-cov"],
        timeout_s=180,
    )
    dead_code_exemplars_flagged = {
        "passed": all(bool(flags) for flags in dead_code_reports.values()),
        "reports": dead_code_reports,
    }
    trustworthy_null_not_flagged = {
        "passed": trustworthy_flags == [],
        "artifact": REQUIRED_FIXTURES["trustworthy_nondegenerate_null"],
        "flags": trustworthy_flags,
    }
    success = (
        checks.get("ok") is True
        and dead_code_exemplars_flagged["passed"] is True
        and trustworthy_null_not_flagged["passed"] is True
        and pinning_test["passed"] is True
        and existing_suite["passed"] is True
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4732", "SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE"],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if success else "complete: lever_guard_partial",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "check_added": av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND,
        "dead_code_exemplars_flagged": dead_code_exemplars_flagged,
        "trustworthy_null_not_flagged": trustworthy_null_not_flagged,
        "existing_suite_green": {
            "passed": existing_suite["passed"],
            "command": existing_suite["command"],
            "returncode": existing_suite["returncode"],
            "stdout_tail": existing_suite["stdout_tail"],
            "stderr_tail": existing_suite["stderr_tail"],
        },
        "pinning_test_path": PINNING_TEST_PATH,
        "pinning_test_green": pinning_test,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": checks,
        "duration_s": max(0.0001, time.perf_counter() - start),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("check_added") != "LEVER_EXERCISE_EVIDENCE_DEGENERATE":
        errors.append("check_added")
    if artifact.get("dead_code_exemplars_flagged", {}).get("passed") is not True:
        errors.append("dead_code_exemplars_flagged")
    if artifact.get("trustworthy_null_not_flagged", {}).get("passed") is not True:
        errors.append("trustworthy_null_not_flagged")
    if artifact.get("existing_suite_green", {}).get("passed") is not True:
        errors.append("existing_suite_green")
    if artifact.get("pinning_test_path") != PINNING_TEST_PATH:
        errors.append("pinning_test_path")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("preconditions_checked", {}).get("ok") is not True:
        errors.append("preconditions_checked")
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


def write_artifact(root: Path | str = REPO_ROOT, artifact: Mapping[str, Any] | None = None) -> Path:
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
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
