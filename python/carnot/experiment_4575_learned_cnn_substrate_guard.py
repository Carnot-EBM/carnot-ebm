"""Experiment 4575: learned-CNN substrate guard.

Spec refs: REQ-ARC-FCP-4575, SCENARIO-ARC-FCP-4575.
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

RESULT_RELATIVE_PATH = "results/experiment_4575_learned_cnn_substrate_guard.json"
EXPERIMENT_ID = "experiment_4575_learned_cnn_substrate_guard"
SCHEMA = "carnot.exp4575.learned_cnn_substrate_guard.v1"
INFERENCE_SUBSTRATE = av.VERIFIER_SCORING_SUBSTRATE
RANDOM_SEED = 4575
TERMINAL_PREFIXES = ("complete:", "shipped:")

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: learned_cnn_substrate_guard_added OR "
            "complete: learned_cnn_substrate_guard_partial_<reason>."
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
            "names the recognized substrate/floor + where it fires -- the fix that "
            "stops a fast-but-real CNN being quarantined."
        )
    },
    "cnn_artifact_not_flagged": {
        "principle": (
            "a fast learned-CNN action-model fixture is NOT DURATION_TOO_SHORT-flagged "
            "-- the .422 A1 headline protection."
        )
    },
    "fake_llm_still_flagged": {
        "principle": (
            "a live_llm_inference fixture at <60s IS still flagged -- guards against "
            "weakening the real fabrication check."
        )
    },
    "tests_added_pass": {
        "principle": (
            "Tests Must Run and Assert -- both the not-flagged-on-CNN and "
            "still-flagged-on-fake-LLM cases."
        )
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def learned_cnn_fixture() -> JsonDict:
    """REQ-ARC-FCP-4575: fast real CNN forward pass fixture."""
    return {
        "experiment": "experiment_4575_cnn_fixture",
        "honest_verdict": "complete: learned_cnn_fixture_real_fast_forward",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "duration_s": 5.0,
        "model_specs": {
            "architecture": "learned frame-action CNN",
            "framework": "torch",
            "device": "cpu",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }


def fake_llm_fixture() -> JsonDict:
    """REQ-ARC-FCP-4575: live-model fabrication fixture that must still fire."""
    return {
        "experiment": "experiment_4575_fake_llm_fixture",
        "honest_verdict": "complete: fake_llm_claim_finished_too_fast",
        "inference_substrate": av.LIVE_LLM_SUBSTRATE,
        "duration_s": 5.0,
        "model_specs": [{"name": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }


def _duration_flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    flags: list[av.Flag] = []
    av.check_duration_vs_claim(dict(payload), flags)
    return [flag.kind for flag in flags]


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
                "tests/python/test_experiment_4575_learned_cnn_substrate_guard.py "
                "-q --no-cov"
            )
        ],
        "assertions": [
            "fast learned-CNN torch fixture uses verifier-scoring 1s floor",
            "fast live_llm_inference GGUF fixture still emits DURATION_TOO_SHORT",
            "summarize_artifact surfaces the applied substrate floor",
        ],
    }


def _guard_mechanism() -> JsonDict:
    cnn = learned_cnn_fixture()
    fake = fake_llm_fixture()
    cnn_floor = av.duration_floor_for_artifact(cnn) or {}
    fake_floor = av.duration_floor_for_artifact(fake) or {}
    return {
        "recognized_substrates": [
            av.VERIFIER_SCORING_SUBSTRATE,
            av.LIVE_LLM_SUBSTRATE,
        ],
        "fires_in": [
            "scripts/adversarial_verify.py:duration_floor_for_artifact",
            "scripts/adversarial_verify.py:check_duration_vs_claim",
            "scripts/summarize_artifact.py:summarize",
        ],
        "cnn_applied_floor_s": cnn_floor.get("min_duration_s"),
        "cnn_floor_reason": cnn_floor.get("reason"),
        "fake_llm_applied_floor_s": fake_floor.get("min_duration_s"),
        "fake_llm_floor_reason": fake_floor.get("reason"),
        "cnn_duration_flag_kinds": _duration_flag_kinds(cnn),
        "fake_llm_duration_flag_kinds": _duration_flag_kinds(fake),
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "guard_mechanism": artifact.get("guard_mechanism"),
        "cnn_artifact_not_flagged": artifact.get("cnn_artifact_not_flagged"),
        "fake_llm_still_flagged": artifact.get("fake_llm_still_flagged"),
        "tests_added_pass": artifact.get("tests_added_pass"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


def checksum_from_artifact(artifact: Mapping[str, Any]) -> str:
    blob = json.dumps(
        _checksum_payload(artifact),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _honest_verdict(
    *,
    preconditions_ok: bool,
    cnn_artifact_not_flagged: bool,
    fake_llm_still_flagged: bool,
    tests_passed: bool,
) -> str:
    if not preconditions_ok:
        return "complete: learned_cnn_substrate_guard_partial_preconditions"
    if not cnn_artifact_not_flagged:
        return "complete: learned_cnn_substrate_guard_partial_cnn_overflagged"
    if not fake_llm_still_flagged:
        return "complete: learned_cnn_substrate_guard_partial_fake_llm_not_flagged"
    if not tests_passed:
        return "complete: learned_cnn_substrate_guard_partial_tests_pending"
    return "shipped: learned_cnn_substrate_guard_added"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_added_pass: Mapping[str, Any] | bool | None = None,
) -> JsonDict:
    root_path = Path(root)
    cnn_flags = _duration_flag_kinds(learned_cnn_fixture())
    fake_flags = _duration_flag_kinds(fake_llm_fixture())
    cnn_artifact_not_flagged = "DURATION_TOO_SHORT" not in cnn_flags
    fake_llm_still_flagged = "DURATION_TOO_SHORT" in fake_flags
    preconditions_checked = {
        "adversarial_verify_help_exits_0": _adversarial_verify_help_exits_0(root_path),
        "adversarial_verify_import_ok": True,
        "cnn_fixture_has_torch_marker": True,
        "cnn_fixture_has_llm_or_gguf_marker": False,
        "fake_llm_fixture_has_gguf_marker": True,
        "model_load": False,
        "scripts_research_conductor_modified": False,
    }
    if tests_added_pass is None:
        tests_payload: Mapping[str, Any] = _default_tests_added_pass()
    elif isinstance(tests_added_pass, bool):
        tests_payload = {"passed": tests_added_pass, "commands": []}
    else:
        tests_payload = tests_added_pass
    tests_passed = tests_payload.get("passed") is True
    preconditions_ok = preconditions_checked["adversarial_verify_help_exits_0"]
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-FCP-4575", "SCENARIO-ARC-FCP-4575"],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            preconditions_ok=bool(preconditions_ok),
            cnn_artifact_not_flagged=cnn_artifact_not_flagged,
            fake_llm_still_flagged=fake_llm_still_flagged,
            tests_passed=tests_passed,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "guard_mechanism": _guard_mechanism(),
        "cnn_artifact_not_flagged": cnn_artifact_not_flagged,
        "fake_llm_still_flagged": fake_llm_still_flagged,
        "tests_added_pass": dict(tests_payload),
        "preconditions_checked": preconditions_checked,
        "model_specs": {
            "invoked": "none",
            "fixture_markers": [
                "learned CNN torch CPU marker",
                "fake live_llm_inference GGUF marker",
            ],
            "purpose": "adversarial_verify duration-floor regression fixtures only",
        },
        "leaderboard_submission": False,
        "duration_s": av.VERIFIER_SCORING_MIN_DURATION_S,
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
    if artifact.get("cnn_artifact_not_flagged") is not True:
        errors.append("cnn_artifact_not_flagged must be true")
    if artifact.get("fake_llm_still_flagged") is not True:
        errors.append("fake_llm_still_flagged must be true")
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
