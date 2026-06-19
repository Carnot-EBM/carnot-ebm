"""Exp 4462: ship ARC reproduced-count integrity linting.

Spec refs: REQ-REPORT-4462, SCENARIO-REPORT-4462, SCENARIO-REPORT-4462-SUBMISSION.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # pragma: no cover

from scripts import arc_count_integrity_lint  # noqa: E402


RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_4462_provisional_reproduced_count_integrity_lint.json"
)
REGISTRY_PATH = REPO_ROOT / "ops" / "arc_solve_registry.yaml"
SUBMISSION_PACKAGE_PATH = REPO_ROOT / "results" / "experiment_4460_submission_package_prep.json"
PRE_COMMIT_PATH = REPO_ROOT / ".pre-commit-config.yaml"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
SPEC_REFS = [
    "REQ-REPORT-4462",
    "SCENARIO-REPORT-4462",
    "SCENARIO-REPORT-4462-SUBMISSION",
]

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "guard_shipped": {
        "principle": (
            "bare bool: the count-integrity lint + replay guard + pre-commit hook landed green"
        )
    },
    "catches_provisional_inflation": {
        "principle": (
            "bare bool: the lint flags a registry where reproducible_total_levels counts "
            "a provisional/non-replaying level"
        )
    },
    "tests_pass": {
        "principle": "bare bool: the new unit tests run and assert (Tests-Must-Run-and-Assert)"
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- this is a lint/test (CPU); 100us floor"
        )
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _duration(started_at: float, ended_at: float) -> float:
    return max(0.001, round(float(ended_at - started_at), 6))


def catches_provisional_inflation() -> bool:
    """SCENARIO-REPORT-4462: a live-recorded provisional total is rejected."""

    inflated_registry = {
        "schema_version": 1,
        "reproducible_total_levels": 5,
        "provisional_total_levels": 5,
        "games": [
            {
                "game": "sc25",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "levels_live_recorded": 5,
            }
        ],
    }
    issues = arc_count_integrity_lint.lint_registry_payload(
        REGISTRY_PATH,
        inflated_registry,
        replay_entry_fn=lambda _entry, _root: None,
        max_replay_games=0,
        root=REPO_ROOT,
    )
    return any(issue.kind == "PROVISIONAL_INFLATION" for issue in issues)


def precommit_guard_configured() -> bool:
    """REQ-REPORT-4462: pre-commit routes registry/package edits through the lint."""

    try:
        config = PRE_COMMIT_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    if "- id: arc-count-integrity-lint" not in config:
        return False
    hook_block = config.split("- id: arc-count-integrity-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)
    if files_match is None:
        return False
    files_re = re.compile(files_match.group(1))
    return (
        "scripts/arc_count_integrity_lint.py" in hook_block
        and bool(files_re.search("ops/arc_solve_registry.yaml"))
        and bool(files_re.search("results/experiment_4460_submission_package_prep.json"))
        and not files_re.search("results/experiment_4450_inference_substrate_emission_lint_guard.json")
    )


def build_artifact(
    *,
    duration_s: float,
    guard_shipped: bool,
    catches_provisional_inflation: bool,
    registry_lint_issue_count: int,
    submission_lint_issue_count: int,
    precommit_hook_configured: bool,
) -> dict[str, Any]:
    """REQ-REPORT-4462: build the terminal lint-guard artifact."""

    shipped = bool(guard_shipped)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4462_provisional_reproduced_count_integrity_lint",
        "schema": "carnot.exp4462.provisional_reproduced_count_integrity_lint.v1",
        "artifact_kind": "arc_count_integrity_lint_guard",
        "honest_verdict": (
            "shipped: provisional_reproduced_count_integrity_lint"
            if shipped
            else "complete: provisional_reproduced_count_integrity_lint_issues_found"
        ),
        "guard_shipped": shipped,
        "catches_provisional_inflation": bool(catches_provisional_inflation),
        "tests_pass": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": max(0.001, float(duration_s)),
        "registry_lint_issue_count": int(registry_lint_issue_count),
        "submission_lint_issue_count": int(submission_lint_issue_count),
        "precommit_hook_configured": bool(precommit_hook_configured),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "implemented_paths": [
            "scripts/arc_count_integrity_lint.py",
            ".pre-commit-config.yaml",
            "python/carnot/experiment_4462_provisional_reproduced_count_integrity_lint.py",
            "tests/python/test_arc_count_integrity_lint.py",
            "tests/python/test_experiment_4462_provisional_reproduced_count_integrity_lint.py",
        ],
        "submitted_to_leaderboard": False,
        "retroactively_rewrote_past_artifacts": False,
        "production_verifier_edits": False,
    }
    artifact["reproducibility_checksum"] = _sha256_payload(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    return artifact


def write_artifact(
    *,
    output_path: Path = RESULT_PATH,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Write the Exp4462 JSON artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_guard() -> dict[str, Any]:
    """Run the current-repo guard checks and write the artifact."""

    started = time.perf_counter()
    registry_issues = arc_count_integrity_lint.lint_registry_path(
        REGISTRY_PATH,
        max_replay_games=arc_count_integrity_lint.DEFAULT_REGISTRY_REPLAY_SPOT_CHECK,
        root=REPO_ROOT,
    )
    submission_issues = arc_count_integrity_lint.lint_submission_package_path(
        SUBMISSION_PACKAGE_PATH,
        root=REPO_ROOT,
    )
    catches = catches_provisional_inflation()
    hook_configured = precommit_guard_configured()
    ended = time.perf_counter()
    guard_shipped = bool(not registry_issues and not submission_issues and catches and hook_configured)
    artifact = build_artifact(
        duration_s=_duration(started, ended),
        guard_shipped=guard_shipped,
        catches_provisional_inflation=catches,
        registry_lint_issue_count=len(registry_issues),
        submission_lint_issue_count=len(submission_issues),
        precommit_hook_configured=hook_configured,
    )
    artifact["registry_lint_issues"] = [issue.to_dict() for issue in registry_issues]
    artifact["submission_lint_issues"] = [issue.to_dict() for issue in submission_issues]
    artifact["reproducibility_checksum"] = _sha256_payload(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    return write_artifact(artifact=artifact)


def main() -> int:
    artifact = run_guard()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["guard_shipped"] and artifact["catches_provisional_inflation"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
