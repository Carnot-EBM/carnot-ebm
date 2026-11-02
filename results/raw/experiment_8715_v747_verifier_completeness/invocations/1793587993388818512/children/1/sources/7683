#!/usr/bin/env python3
"""Exp 3716: ship the standalone Paper-v6 narrowing lint and record G3 status."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import paper_v6_narrowing_lint as narrowing_lint


RESULT_PATH = REPO_ROOT / "results" / "experiment_3716_ship_paper_v6_narrowing_lint.json"
LINT_REL_PATH = "scripts/paper_v6_narrowing_lint.py"
TEST_REL_PATH = "tests/python/test_paper_v6_narrowing_lint.py"
CONDUCTOR_REL_PATH = "scripts/research_conductor.py"
RANDOM_SEED = 3716
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a static text lint; "
    "no live inference; no compute-bound marker)."
)
COMPLETE_VERDICT = (
    "complete: paper_v6_narrowing_lint_shipped_g3_mechanically_enforced_current_paper_clean"
)
BLOCKED_VERDICT = "complete: blocked_no_paper_targets_on_disk"
PRECOMMIT_HOOK_STANZA = """- repo: local
  hooks:
    - id: paper-v6-narrowing-lint
      name: Paper-v6 narrowing lint
      entry: python scripts/paper_v6_narrowing_lint.py
      language: system
      files: ^(docs/arxiv-paper/main\\.tex|docs/technical-report\\.md|results/paper_v6_.*\\.json)$
"""

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: a static text lint; "
        "no live inference; no compute-bound marker)."
    ),
    "lint_script_path": "scripts/paper_v6_narrowing_lint.py -- the shipped mechanical G3 enforcer.",
    "forbidden_phrasings_count": (
        "How many forbidden phrasings + retracted numbers the lint checks -- "
        "coverage of the CLAUDE.md narrowing table."
    ),
    "current_paper_lint_clean": (
        "BARE bool. True iff the current paper targets pass the new lint "
        "(G3 currently passes). STORE AS BARE true/false."
    ),
    "pytest_passed": (
        "True iff the parametrized lint test (clean-passes / forbidden-fails / "
        "retracted-number-fails) is green."
    ),
    "g3_now_mechanically_enforced": (
        "BARE bool. True iff a standalone lint now defends G3 "
        "(was honor-discipline + inline scan). STORE AS BARE true/false."
    ),
    "precommit_hook_stanza": (
        "The recommended .pre-commit-config.yaml stanza for OPERATOR action -- "
        "not auto-wired (operator-curated config)."
    ),
    "conductor_unmodified_assert": "Asserts research_conductor.py was NOT modified.",
    "adversarial_verify_clean": "True iff no critical flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def venv_python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def conductor_unmodified() -> bool:
    unstaged = run_command(["git", "diff", "--quiet", "--", CONDUCTOR_REL_PATH])
    staged = run_command(["git", "diff", "--cached", "--quiet", "--", CONDUCTOR_REL_PATH])
    return unstaged["returncode"] == 0 and staged["returncode"] == 0


def current_paper_targets_present() -> bool:
    return any((REPO_ROOT / rel).exists() for rel in narrowing_lint.PAPER_TARGETS)


def run_current_lint() -> tuple[bool, dict[str, Any]]:
    cmd = [venv_python(), LINT_REL_PATH, "--verbose"]
    result = run_command(cmd)
    return result["returncode"] == 0, result


def run_focused_pytest() -> tuple[bool, dict[str, Any]]:
    cmd = [
        venv_python(),
        "-m",
        "pytest",
        "-o",
        "addopts=",
        TEST_REL_PATH,
        "-q",
    ]
    result = run_command(cmd)
    return result["returncode"] == 0, result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    stable = {
        "random_seed": RANDOM_SEED,
        "lint_sha256": file_sha256(REPO_ROOT / LINT_REL_PATH),
        "test_sha256": file_sha256(REPO_ROOT / TEST_REL_PATH),
        "forbidden_patterns": [
            {"name": spec.name, "regex": spec.regex, "why": spec.why}
            for spec in narrowing_lint.FORBIDDEN_PATTERNS
        ],
        "current_paper_lint_clean": payload.get("current_paper_lint_clean"),
        "pytest_passed": payload.get("pytest_passed"),
        "g3_now_mechanically_enforced": payload.get("g3_now_mechanically_enforced"),
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_artifact(artifact: Mapping[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_adversarial_verify() -> Any:
    script = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("adversarial_verify_exp3716", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    for flag in list(report.get("flags") or []):
        item = dict(flag)
        if str(item.get("severity")) == "critical":
            return False
    return True


def verify_artifact(path: Path) -> dict[str, Any]:
    module = load_adversarial_verify()
    return dict(module.verify_artifact(path))


def build_artifact() -> dict[str, Any]:
    start = time.perf_counter()
    paper_targets_present = current_paper_targets_present()
    current_clean, lint_result = run_current_lint() if paper_targets_present else (False, {})
    pytest_passed, pytest_result = run_focused_pytest()
    conductor_ok = conductor_unmodified()
    g3_enforced = bool(
        paper_targets_present
        and current_clean
        and pytest_passed
        and conductor_ok
        and (REPO_ROOT / LINT_REL_PATH).exists()
        and len(narrowing_lint.FORBIDDEN_PATTERNS) > 0
    )
    verdict = COMPLETE_VERDICT if g3_enforced else BLOCKED_VERDICT
    if paper_targets_present and not g3_enforced:
        verdict = "complete: paper_v6_narrowing_lint_shipped_but_acceptance_gate_unmet"

    artifact: dict[str, Any] = {
        "schema": "carnot.paper_v6_narrowing_lint.v1",
        "experiment": 3716,
        "experiment_id": 3716,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "lint_script_path": LINT_REL_PATH,
        "forbidden_phrasings_count": len(narrowing_lint.FORBIDDEN_PATTERNS),
        "current_paper_lint_clean": bool(current_clean),
        "pytest_passed": bool(pytest_passed),
        "g3_now_mechanically_enforced": bool(g3_enforced),
        "precommit_hook_stanza": PRECOMMIT_HOOK_STANZA,
        "operator_action_note": (
            "OPERATOR-ACTION: Review and add precommit_hook_stanza to "
            ".pre-commit-config.yaml when operator-curated commit policy approves it."
        ),
        "conductor_unmodified_assert": bool(conductor_ok),
        "adversarial_verify_clean": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": max(time.perf_counter() - start, 0.0001),
        "paper_targets_present": bool(paper_targets_present),
        "lint_command": lint_result,
        "pytest_command": pytest_result,
        "field_principles": FIELD_PRINCIPLES,
        "acceptance_gate": {
            "condition": (
                "g3_now_mechanically_enforced == true AND pytest_passed == true AND "
                "current_paper_lint_clean == true AND conductor_unmodified_assert == true"
            ),
            "principle": (
                "G3 is mechanically defended only when the standalone lint exists, "
                "its test is green, the current paper passes it, and the conductor is untouched."
            ),
            "passed": bool(
                g3_enforced and pytest_passed and current_clean and conductor_ok
            ),
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def main() -> int:
    artifact = build_artifact()
    write_artifact(artifact)
    report = verify_artifact(RESULT_PATH)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["adversarial_verify_report"] = {
        "flag_count": int(report.get("flag_count", 0)),
        "max_severity": report.get("max_severity"),
        "flags": list(report.get("flags") or []),
    }
    artifact["duration_s"] = max(float(artifact["duration_s"]), 0.0001)
    write_artifact(artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"OPERATOR-ACTION recommended hook stanza:\n{PRECOMMIT_HOOK_STANZA}")
    return 0 if artifact["acceptance_gate"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
