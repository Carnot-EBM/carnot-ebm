"""Exp 3488: clean-room regression-verify the FoVer G2 package + prepare the ask.

Why this experiment exists
--------------------------
Publication gate G2 (independent reproduction) is the SOLE remaining blocker to
``paper_ready`` (G1/G3/G4 met per ``ops/north-star.md`` §2). Exp 3476 (.320)
built a self-contained reproduction package (``dist/g2-fover-repro.tar.gz``).
Two pieces of autonomous work remain before a non-operator actually runs it:

  1. REGRESSION-VERIFY the package still reproduces the headline AUROC from an
     environment isolated from the working repo — catching any drift since .320.
  2. PREPARE THE LOWEST-FRICTION EXTERNAL ASK — a public ``workflow_dispatch``
     reproduction workflow, a one-paragraph reproducer invite, and an operator
     checklist whose terminal step is a single click.

This experiment does the above and emits the artifact. Per Operator-Only
External Publication it NEVER pushes, NEVER triggers external CI, and NEVER sets
``g2_met`` / ``g2_independent_reproducer`` true — only a confirmed non-operator
external run may flip those. It does NOT modify ``scripts/research_conductor.py``
or operator-curated docs (``ops/north-star.md`` / ``docs/index.html`` /
``README.md``).

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3488_fover_g2_clean_room_regression_verify_external_ask_v1.py

Spec: REQ-PUBLISH-039, SCENARIO-PUBLISH-039, SCENARIO-PUBLISH-039B
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.eval.fover_g2_package import TARBALL_REL  # noqa: E402
from carnot.eval.fover_g2_regression import (  # noqa: E402
    OPERATOR_CHECKLIST_REL,
    REPRO_WORKFLOW_REL,
    REPRODUCER_INVITE_REL,
    append_runbook,
    build_artifact,
    build_operator_checklist,
    build_repro_workflow_yaml,
    build_reproducer_invite,
    check_preconditions,
    maybe_ipfs_add,
    read_recorded_sha256,
    regression_verify,
    verify_sha256,
)

OUT_PATH = REPO_ROOT / "results" / (
    "experiment_3488_fover_g2_clean_room_regression_verify_external_ask_v1.json"
)

# Set to "1" to attempt the strongest isolation (fresh venv with pinned-wheel
# install) first; defaults off because that needs network for the wheels and the
# isolated-dir fallback is a faithful regression check with identical pins.
PREFER_FRESH_VENV = os.environ.get("G2_PREFER_FRESH_VENV") == "1"


def _write(artifact: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2, default=str) + "\n")


def main() -> int:
    """Regression-verify the package, prepare the external ask, emit the artifact."""
    start = time.time()

    # ------------------------------------------------------------------ #
    # STEP 0: PRECONDITIONS                                              #
    # ------------------------------------------------------------------ #
    preconditions = check_preconditions(REPO_ROOT)
    if not preconditions["ok"]:
        artifact = build_artifact(
            start_time=start,
            preconditions=preconditions,
            regression={},
            sha_check={"computed": None, "recorded": None, "verified": False},
            ipfs_result={"ipfs_available": False, "package_cid": None},
            workflow_path=None,
            invite_path=None,
            checklist_path=None,
            runbook_appended=False,
        )
        artifact["honest_verdict"] = "complete: " + preconditions["blocked_reason"]
        _write(artifact)
        print(f"honest_verdict: {artifact['honest_verdict']}")
        return 0

    tar_path = REPO_ROOT / TARBALL_REL

    # ------------------------------------------------------------------ #
    # STEP 1-2-3: REGRESSION RUN + INTEGRITY + CID                       #
    # ------------------------------------------------------------------ #
    print(f"Regression-verifying {tar_path.relative_to(REPO_ROOT)} ...")
    regression = regression_verify(tar_path, prefer_fresh_venv=PREFER_FRESH_VENV)
    print(
        f"  method={regression['clean_env_method']}  "
        f"cond_a={regression['condition_a_auroc']}  "
        f"lc={regression['learning_contribution']}  "
        f"within_ci={regression['condition_a_in_ci']}"
    )
    if regression.get("error"):
        print(f"  isolation error: {regression['error']}")
    if regression.get("stderr_tail"):
        print(f"  stderr tail: {regression['stderr_tail']}")

    recorded_sha = read_recorded_sha256(REPO_ROOT)
    sha_check = verify_sha256(tar_path, recorded_sha)
    print(
        f"  sha256 computed={str(sha_check['computed'])[:16]}...  "
        f"verified={sha_check['verified']}"
    )

    ipfs_result = maybe_ipfs_add(tar_path)
    print(
        f"  ipfs_available={ipfs_result['ipfs_available']}  "
        f"cid={ipfs_result['package_cid']}"
    )

    # ------------------------------------------------------------------ #
    # STEP 4-5: EXTERNAL ASK (workflow + invite + operator checklist)    #
    # ------------------------------------------------------------------ #
    workflow_path = REPO_ROOT / REPRO_WORKFLOW_REL
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(build_repro_workflow_yaml(), encoding="utf-8")

    invite_path = REPO_ROOT / REPRODUCER_INVITE_REL
    invite_path.parent.mkdir(parents=True, exist_ok=True)
    invite_path.write_text(
        build_reproducer_invite(sha_check["computed"], ipfs_result["package_cid"]),
        encoding="utf-8",
    )

    checklist_path = REPO_ROOT / OPERATOR_CHECKLIST_REL
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(
        build_operator_checklist(
            package_path=TARBALL_REL,
            package_sha256=sha_check["computed"],
            package_sha256_verified=bool(sha_check["verified"]),
            package_cid=ipfs_result["package_cid"],
            reproduced_auroc=regression["condition_a_auroc"],
            auroc_within_ci=bool(regression["condition_a_in_ci"]),
            clean_env_method=regression["clean_env_method"],
            workflow_path=REPRO_WORKFLOW_REL,
            invite_path=REPRODUCER_INVITE_REL,
        ),
        encoding="utf-8",
    )

    runbook_appended = append_runbook(
        REPO_ROOT,
        reproduced_auroc=regression["condition_a_auroc"],
        auroc_within_ci=bool(regression["condition_a_in_ci"]),
        clean_env_method=regression["clean_env_method"],
        package_sha256=sha_check["computed"],
        package_sha256_verified=bool(sha_check["verified"]),
        package_cid=ipfs_result["package_cid"],
    )

    # ------------------------------------------------------------------ #
    # STEP 6: EMIT ARTIFACT                                              #
    # ------------------------------------------------------------------ #
    artifact = build_artifact(
        start_time=start,
        preconditions=preconditions,
        regression=regression,
        sha_check=sha_check,
        ipfs_result=ipfs_result,
        workflow_path=REPRO_WORKFLOW_REL,
        invite_path=REPRODUCER_INVITE_REL,
        checklist_path=OPERATOR_CHECKLIST_REL,
        runbook_appended=runbook_appended,
    )
    _write(artifact)
    print(f"\nhonest_verdict: {artifact['honest_verdict']}")
    print(f"package_auroc_within_ci: {artifact['package_auroc_within_ci']}")
    print(f"package_sha256_verified: {artifact['package_sha256_verified']}")
    print(f"g2_met: {artifact['g2_met']}  external_run_pending: "
          f"{artifact['external_run_pending']}")
    print(f"artifact written to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
