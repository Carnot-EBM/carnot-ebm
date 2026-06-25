"""Experiment 4743: adversarial_verify ARC null carve-out hardening receipt.

Spec refs: REQ-ARC-WMTE-4743, SCENARIO-ARC-WMTE-4743-CARVEOUT-HARDENING.
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

EXPERIMENT = "experiment_4743_adversarial_verify_carveout_hardening"
SCHEMA = "carnot.exp4743.adversarial_verify_carveout_hardening.v1"
RESULT_RELATIVE_PATH = "results/experiment_4743_adversarial_verify_carveout_hardening.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
PINNING_TEST_PATH = "tests/python/test_adversarial_verify_carveout_hardening.py"
EXP4726_PATH = "results/experiment_4726_online_action_learning_driver_valid_test.json"
EXP4727_PATH = "results/experiment_4727_active_probe_disambiguation.json"
RANDOM_SEED = 4743
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- guard dev + a pytest run, no model load "
    "(100us floor)."
)
SUCCESS_VERDICT = "success: adversarial_verify_carveout_hardening_shipped_pinned."
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "adversarial_verify_carveout_hardening_shipped_pinned."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- guard dev + a pytest run, "
            "no model load (100us floor)."
        )
    },
    "tautology_carveout_added": {
        "principle": (
            "the non-degenerate-zero-lift carve-out -- mechanizes the fix so an "
            "honest generation null (the .436 A1/A2 shape) is not quarantined "
            "like .435 A1."
        )
    },
    "exercise_evidence_extension_added": {
        "principle": (
            "the declared-but-unrun extension (probe_actions_taken==0 etc.) -- "
            "catches the .435 A2-class no-op the existing guard missed."
        )
    },
    "a1_exemplar_downgraded_to_warn": {
        "principle": (
            "exp4726 (non-degenerate + null markers) downgrades CRITICAL->WARN -- "
            "the honest-null-not-quarantined fix works."
        )
    },
    "a2_exemplar_flagged": {
        "principle": (
            "exp4727 (declared probe + 0 probe_actions) is flagged "
            "LEVER_EXERCISE_EVIDENCE_DEGENERATE -- the declared-but-unrun catch works."
        )
    },
    "positive_exercise_null_not_flagged": {
        "principle": (
            "a genuine null with positive exercise evidence is NOT flagged -- the "
            "false-positive carve-out works (a real null must survive)."
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
            "tests/python/test_adversarial_verify_carveout_hardening.py -- the "
            "standing pin."
        )
    },
    "verifier_is_oracle": {"principle": "false -- a linter check invokes no oracle."},
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (adversarial_verify importable, exemplars "
            "present); pre-empts missing-resource fabrication."
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


def _run_command(root: Path, args: list[str], timeout_s: int = 180) -> JsonDict:
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


def _artifact_flags(root: Path, relative_path: str) -> list[JsonDict]:
    report = av.verify_artifact(root / relative_path)
    return list(report.get("flags", []))


def _lever_flags_from_payload(payload: JsonDict) -> list[JsonDict]:
    flags: list[av.Flag] = []
    av.check_lever_exercise_evidence(payload, flags)
    return [
        flag.to_dict()
        for flag in flags
        if flag.kind == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]


def _positive_exercise_payload() -> JsonDict:
    return {
        "experiment": "arc_active_probe_positive_exercise_null",
        "schema": "carnot.arc.active_probe.valid_null.v1",
        "honest_verdict": "complete: active_probe_no_first_win_lift_honest_null",
        "inference_substrate": "active-probe replay over cached ARC transitions",
        "duration_s": 1.0,
        "target_game": "bp35",
        "probe_actions_taken": 3,
        "hypothesis_posterior_built": True,
        "posterior_entropy_reduction": 0.25,
        "active_probe_result": {
            "active_probe": True,
            "probe_actions_taken": 3,
            "hypothesis_posterior_built": True,
            "posterior_entropy_reduction": 0.25,
            "budget": 10,
        },
        "baseline_first_win": 0.04,
        "active_probe_first_win": 0.04,
        "active_probe_delta": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "7" * 64,
    }


def _adversarial_verify_tests(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.glob("tests/python/test_adversarial_verify_*.py"))


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_text = (root_path / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "adversarial_verify_importable": hasattr(av, "check_tautology")
        and hasattr(av, "check_lever_exercise_evidence"),
        "exp4726_present": (root_path / EXP4726_PATH).exists(),
        "exp4727_present": (root_path / EXP4727_PATH).exists(),
        "pinning_test_present": (root_path / PINNING_TEST_PATH).exists(),
        "spec_has_req_4743": "REQ-ARC-WMTE-4743" in spec_text,
        "network_required": False,
    }
    checks["ok"] = all(
        checks[key]
        for key in (
            "agents_md_read",
            "codex_or_opencode_md_read",
            "adversarial_verify_importable",
            "exp4726_present",
            "exp4727_present",
            "pinning_test_present",
            "spec_has_req_4743",
        )
    )
    return checks


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    start = time.perf_counter()
    root_path = Path(root)
    checks = check_preconditions(root_path)

    exp4726_flags = _artifact_flags(root_path, EXP4726_PATH)
    exp4726_tautology = [flag for flag in exp4726_flags if flag.get("kind") == "TAUTOLOGY"]
    exp4726_critical_tautology = [
        flag for flag in exp4726_tautology if flag.get("severity") == "critical"
    ]
    exp4726_critical = [flag for flag in exp4726_flags if flag.get("severity") == "critical"]
    a1_exemplar = {
        "passed": bool(exp4726_tautology)
        and not exp4726_critical_tautology
        and not exp4726_critical,
        "artifact": EXP4726_PATH,
        "tautology_flags": exp4726_tautology,
        "critical_flags": exp4726_critical,
        "not_quarantined": not exp4726_critical,
    }

    exp4727_flags = _artifact_flags(root_path, EXP4727_PATH)
    exp4727_lever_flags = [
        flag
        for flag in exp4727_flags
        if flag.get("kind") == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]
    a2_exemplar = {
        "passed": bool(exp4727_lever_flags)
        and any("probe_actions_taken=0" in flag.get("detail", "") for flag in exp4727_lever_flags),
        "artifact": EXP4727_PATH,
        "lever_flags": exp4727_lever_flags,
    }

    positive_exercise_flags = _lever_flags_from_payload(_positive_exercise_payload())
    positive_exercise = {
        "passed": positive_exercise_flags == [],
        "lever_flags": positive_exercise_flags,
        "probe_actions_taken": 3,
        "posterior_entropy_reduction": 0.25,
    }

    pinning_test = _run_command(
        root_path,
        [sys.executable, "-m", "pytest", PINNING_TEST_PATH, "-q", "--no-cov"],
        timeout_s=120,
    )
    suite_paths = _adversarial_verify_tests(root_path)
    existing_suite = _run_command(
        root_path,
        [sys.executable, "-m", "pytest", *suite_paths, "-q", "--no-cov"],
        timeout_s=240,
    )

    tautology_carveout_added = {
        "passed": a1_exemplar["passed"],
        "markers": [
            "arms_non_degenerate=true",
            "null_delta_methodology_note",
            "positive_control_passed=true",
        ],
        "critical_to_warn": True,
    }
    exercise_evidence_extension_added = {
        "passed": a2_exemplar["passed"] and positive_exercise["passed"],
        "kind": av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND,
        "declared_but_unrun": "probe_actions_taken==0 / hypothesis_posterior_built==False / posterior_entropy_reduction==0.0",
    }
    success = (
        checks.get("ok") is True
        and tautology_carveout_added["passed"] is True
        and exercise_evidence_extension_added["passed"] is True
        and pinning_test["passed"] is True
        and existing_suite["passed"] is True
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4743",
            "SCENARIO-ARC-WMTE-4743-CARVEOUT-HARDENING",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if success else "complete: carveout_hardening_partial",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "tautology_carveout_added": tautology_carveout_added,
        "exercise_evidence_extension_added": exercise_evidence_extension_added,
        "a1_exemplar_downgraded_to_warn": a1_exemplar,
        "a2_exemplar_flagged": a2_exemplar,
        "positive_exercise_null_not_flagged": positive_exercise,
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
    if artifact.get("tautology_carveout_added", {}).get("passed") is not True:
        errors.append("tautology_carveout_added")
    if artifact.get("exercise_evidence_extension_added", {}).get("passed") is not True:
        errors.append("exercise_evidence_extension_added")
    if artifact.get("a1_exemplar_downgraded_to_warn", {}).get("passed") is not True:
        errors.append("a1_exemplar_downgraded_to_warn")
    if artifact.get("a2_exemplar_flagged", {}).get("passed") is not True:
        errors.append("a2_exemplar_flagged")
    if artifact.get("positive_exercise_null_not_flagged", {}).get("passed") is not True:
        errors.append("positive_exercise_null_not_flagged")
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
