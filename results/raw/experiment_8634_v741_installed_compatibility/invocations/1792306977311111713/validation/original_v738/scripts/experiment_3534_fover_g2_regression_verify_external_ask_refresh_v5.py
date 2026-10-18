"""Exp 3534: post-.325 drift check — re-confirm the FoVer G2 package + refresh the ask.

Why this experiment exists
--------------------------
After a full milestone of repository changes (.325) the self-contained reproduction
package (``dist/g2-fover-repro.tar.gz``) may have drifted: a source file the
package copies, a corpus file, or a pinned dep could have changed and the package
would silently reproduce a wrong AUROC for an external party. This experiment is the
milestone-boundary regression gate: same machinery as Exp 3510 (.323), but run AFTER
.325's changes to detect any drift before the operator sends the external ask.

Spec: REQ-PUBLISH-039C, SCENARIO-PUBLISH-039C
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

# v5-specific identifiers — only these differ from exp3510.
EXP_ID = 3534
ARTIFACT_NAME = "experiment_3534_fover_g2_regression_verify_external_ask_refresh_v5"
SCHEMA_NAME = "carnot.fover_g2_regression_verify_external_ask_refresh_v5"
OUT_PATH = REPO_ROOT / "results" / f"{ARTIFACT_NAME}.json"

# Per CLAUDE.md: random_seed is the run-date seed for determinism, NOT the exp number.
RANDOM_SEED = 20260531

PREFER_FRESH_VENV = os.environ.get("G2_PREFER_FRESH_VENV") == "1"

# The v5 clean verdict acknowledges the refreshed external-ask artifacts are "current"
# (updated at .325) rather than just "ready" (the .322 framing).
VERDICT_CLEAN = (
    "complete: fover_g2_package_regression_clean_external_ask_current_g2_operator_gated"
)


def _write(artifact: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2, default=str) + "\n")


def main() -> int:
    """Post-.325 drift check: re-verify the G2 package and refresh external-ask files."""
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
        artifact["artifact"] = ARTIFACT_NAME
        artifact["schema"] = SCHEMA_NAME
        artifact["experiment"] = EXP_ID
        artifact["random_seed"] = RANDOM_SEED
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
    # STEP 4-5: EXTERNAL ASK (refresh workflow + invite + operator checklist)
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
        exp_id=f"exp{EXP_ID}",
        run_date="2026-05-31",
        artifact_name=ARTIFACT_NAME,
    )

    # ------------------------------------------------------------------ #
    # STEP 6: EMIT ARTIFACT (patch v5 identifiers + determinism seed)    #
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
    # Patch the v5-specific identifiers (build_artifact uses v1 names by default).
    artifact["artifact"] = ARTIFACT_NAME
    artifact["schema"] = SCHEMA_NAME
    artifact["experiment"] = EXP_ID
    # Per task spec: random_seed is the run-date seed 20260531, not the exp number.
    artifact["random_seed"] = RANDOM_SEED
    # Use the v5 verdict string (acknowledges "current" refresh at .325).
    if artifact["honest_verdict"].startswith(
        "complete: fover_g2_package_regression_clean_external_ask"
    ) and "drift" not in artifact["honest_verdict"]:
        artifact["honest_verdict"] = VERDICT_CLEAN

    _write(artifact)
    print(f"\nhonest_verdict: {artifact['honest_verdict']}")
    print(f"package_auroc_within_ci: {artifact['package_auroc_within_ci']}")
    print(f"package_sha256_verified: {artifact['package_sha256_verified']}")
    print(
        f"g2_met: {artifact['g2_met']}  "
        f"external_run_pending: {artifact['external_run_pending']}"
    )
    print(f"artifact written to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
