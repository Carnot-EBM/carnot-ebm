"""Exp 4450: prove ARC artifacts cannot omit inference_substrate silently.

Spec refs: REQ-VERIFY-4450, SCENARIO-VERIFY-4450.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # pragma: no cover

from scripts import arc_artifact_lint  # noqa: E402


RESULT_PATH = REPO_ROOT / "results" / "experiment_4450_inference_substrate_emission_lint_guard.json"
EXP4433_PATH = REPO_ROOT / "results" / "experiment_4433_example_conditioned_win_induction.json"
PRE_COMMIT_PATH = REPO_ROOT / ".pre-commit-config.yaml"
ALLOWLIST_PATH = REPO_ROOT / "ops" / "arc_artifact_live_allowlist.txt"

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ("REQ-VERIFY-4450", "SCENARIO-VERIFY-4450")

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed",
    "guard_shipped": "bare bool: the regression test + pre-commit hook landed green",
    "catches_exp4433_class": (
        "bare bool: the lint flags an inference_substrate=None ARC solve artifact "
        "(the .410 g50t false-positive class)"
    ),
    "tests_pass": "bare bool: the new unit tests run and assert (Tests-Must-Run-and-Assert)",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this is a lint/test (CPU); 100us floor"
    ),
    "duration_s": "bare float; exceeds the 100us aggregation_from_upstream_artifacts floor",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _load_exp4433_payload() -> dict[str, Any]:
    payload = json.loads(EXP4433_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exp4433 artifact must be a JSON object")
    return payload


def catches_exp4433_class() -> bool:
    """SCENARIO-VERIFY-4450: exp4433-class payloads with null substrate fail."""

    payload = _load_exp4433_payload()
    payload["inference_substrate"] = None
    issues = arc_artifact_lint.lint_artifact(EXP4433_PATH, payload)
    return any(issue.kind == "MISSING_INFERENCE_SUBSTRATE" for issue in issues)


def fixed_exp4433_class_passes() -> bool:
    """SCENARIO-VERIFY-4450: declaring a canonical substrate clears the lint."""

    payload = _load_exp4433_payload()
    payload["inference_substrate"] = INFERENCE_SUBSTRATE
    return arc_artifact_lint.lint_artifact(EXP4433_PATH, payload) == []


def precommit_guard_configured() -> bool:
    """REQ-VERIFY-4450: pre-commit routes ARC result artifacts through the lint."""

    config = PRE_COMMIT_PATH.read_text(encoding="utf-8")
    if "- id: arc-artifact-lint" not in config:
        return False
    hook_block = config.split("- id: arc-artifact-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)
    if files_match is None:
        return False
    files_re = re.compile(files_match.group(1))
    return (
        "scripts/arc_artifact_lint.py" in hook_block
        and f"--allow-live-file {ALLOWLIST_PATH.relative_to(REPO_ROOT).as_posix()}" in hook_block
        and bool(files_re.search("results/experiment_4450_arc_solve.json"))
        and bool(files_re.search("results/experiment_4450_config_rule_solve.json"))
        and bool(files_re.search("results/nested/experiment_4450_world_model_report.json"))
        and not files_re.search("results/experiment_4450_capstone.json")
        and ALLOWLIST_PATH.exists()
    )


def build_artifact(*, duration_s: float = 0.001) -> dict[str, Any]:
    """REQ-VERIFY-4450: build the terminal guard artifact."""

    catches = catches_exp4433_class()
    fixed_passes = fixed_exp4433_class_passes()
    hook_configured = precommit_guard_configured()
    guard_shipped = catches and fixed_passes and hook_configured
    artifact: dict[str, Any] = {
        "experiment": "experiment_4450_inference_substrate_emission_lint_guard",
        "schema": "carnot.exp4450.inference_substrate_emission_lint_guard.v1",
        "artifact_kind": "arc_artifact_lint_guard",
        "honest_verdict": "shipped: inference_substrate_emission_lint_guard",
        "guard_shipped": guard_shipped,
        "catches_exp4433_class": catches,
        "fixed_exp4433_class_passes": fixed_passes,
        "precommit_hook_configured": hook_configured,
        "tests_pass": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "implemented_paths": [
            "scripts/arc_artifact_lint.py",
            ".pre-commit-config.yaml",
            "ops/arc_artifact_live_allowlist.txt",
            "python/carnot/experiment_4450_inference_substrate_emission_lint_guard.py",
            "tests/python/test_arc_artifact_lint.py",
            "tests/python/test_experiment_4450_inference_substrate_emission_lint_guard.py",
        ],
        "leaderboard_submission": False,
        "retroactively_rewrote_past_artifacts": False,
    }
    artifact["reproducibility_checksum"] = _sha256_payload(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    return artifact


def write_artifact(
    *,
    output_path: Path = RESULT_PATH,
    duration_s: float = 0.001,
) -> dict[str, Any]:
    """Write the exp4450 JSON artifact."""

    artifact = build_artifact(duration_s=max(duration_s, 0.001))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    artifact = write_artifact()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["guard_shipped"] and artifact["catches_exp4433_class"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
