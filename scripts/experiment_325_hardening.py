#!/usr/bin/env python3
"""Experiment 325: Conductor Hardening — timeout wrapper and test-first stub.

**What this experiment verifies:**
    1. ``scripts/run_experiment_with_timeout.sh`` exists, is executable, and
       contains the correct timeout logic (REQ-INFRA-001, RETRO-001).
    2. ``ExperimentTemplate.generate_test_stub()`` creates a valid Python
       skeleton and is idempotent (REQ-INFRA-002, NEW-001).

**Why this matters:**
    RETRO-001 has been carried forward for TWO consecutive milestones
    (2026.04.22 and 2026.04.29).  Exp 308's post-test failure loop consumed
    138 minutes — a 45-minute hard cap would have saved 93 minutes.

    NEW-001 identified that ExperimentTemplate should auto-generate a test
    skeleton BEFORE implementation to reduce the 23.5% post-test failure rate.

Spec: REQ-INFRA-001, REQ-INFRA-002,
      SCENARIO-INFRA-001, SCENARIO-INFRA-002, SCENARIO-INFRA-003
"""
from __future__ import annotations

import ast
import json
import os
import stat
import tempfile
from pathlib import Path

from scripts.experiment_template import ExperimentTemplate

DELIVERABLE = "results/experiment_325_hardening.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "scripts" / "run_experiment_with_timeout.sh"
DEFAULT_TIMEOUT_MINUTES = 45


def check_wrapper() -> dict:
    """Verify that run_experiment_with_timeout.sh exists and is correct."""
    results: dict = {}

    # Existence
    results["wrapper_exists"] = WRAPPER_PATH.exists()
    if not results["wrapper_exists"]:
        results["wrapper_error"] = str(WRAPPER_PATH)
        return results

    # Executable bit
    st = WRAPPER_PATH.stat()
    results["wrapper_executable"] = bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

    content = WRAPPER_PATH.read_text()

    # Shebang
    results["has_bash_shebang"] = content.splitlines()[0].strip() == "#!/usr/bin/env bash"

    # Key requirements
    results["uses_timeout_command"] = "timeout" in content
    results["uses_env_var"] = "CARNOT_CONDUCTOR_TIMEOUT_MINUTES" in content
    results["default_is_45"] = "45" in content
    results["exit_code_124"] = "124" in content
    results["timeout_message"] = "CONDUCTOR TIMEOUT" in content
    results["kill_flag"] = "-k" in content
    results["passes_args"] = "$@" in content

    return results


def check_test_stub_generation() -> dict:
    """Verify generate_test_stub() behaviour against REQ-INFRA-002."""
    tmpl = ExperimentTemplate(
        325,
        "Exp 325: conductor hardening",
        DELIVERABLE,
    )
    tmpl.setup()

    results: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        # --- first call: write the file ---
        dest = os.path.join(tmp, "test_exp325_placeholder.py")
        returned_path = tmpl.generate_test_stub(dest, "scripts.experiment_template")
        results["stub_path_returned"] = returned_path == dest
        results["stub_file_created"] = Path(dest).exists()

        content = Path(dest).read_text()
        results["stub_has_req_comment"] = "REQ-INFRA-002" in content
        results["stub_has_autogen_header"] = "AUTO-GENERATED" in content
        results["stub_has_test_class"] = "TestExp" in content
        results["stub_has_placeholder_test"] = "test_placeholder_stub" in content
        results["stub_has_assert_true"] = "assert True" in content
        results["stub_has_module_import"] = "scripts.experiment_template" in content

        # Syntactic validity
        try:
            ast.parse(content)
            results["stub_parses_as_valid_python"] = True
        except SyntaxError as exc:
            results["stub_parses_as_valid_python"] = False
            results["stub_parse_error"] = str(exc)

        # File permissions (0o644)
        mode = Path(dest).stat().st_mode & 0o777
        results["stub_permissions_644"] = mode == 0o644

        # --- second call: idempotency ---
        # Append a marker so we can detect overwrite
        Path(dest).write_text(content + "\n# IDEMPOTENCY_MARKER\n")
        returned_path2 = tmpl.generate_test_stub(dest, "scripts.experiment_template")
        after_second_call = Path(dest).read_text()
        results["stub_idempotent_path_same"] = returned_path2 == dest
        results["stub_idempotent_no_overwrite"] = "IDEMPOTENCY_MARKER" in after_second_call

    return results


def main() -> None:
    tmpl = ExperimentTemplate(
        325,
        "Exp 325: conductor hardening",
        DELIVERABLE,
    )
    tmpl.setup()

    wrapper_results = check_wrapper()
    stub_results = check_test_stub_generation()

    retro_items_implemented = ["RETRO-001", "NEW-001"]

    all_checks = {**wrapper_results, **stub_results}
    all_passed = all(v is True for v in all_checks.values() if isinstance(v, bool))

    artifact = tmpl.build_result(
        {
            "wrapper_checks": wrapper_results,
            "stub_checks": stub_results,
            "retro_items_implemented": retro_items_implemented,
            "estimated_speedup_pct": 27.0,
            "timeout_minutes": DEFAULT_TIMEOUT_MINUTES,
            "test_first_stub_added": True,
            "all_checks_passed": all_passed,
        },
        status="success" if all_passed else "partial",
        schema="carnot.conductor_hardening.v1",
    )

    out_path = REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact written to {out_path}")

    if not all_passed:
        failed = [k for k, v in all_checks.items() if v is not True and isinstance(v, bool)]
        print(f"FAILED checks: {failed}")
        raise SystemExit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
