"""Exp 3476: build + verify a self-contained FoVer G2 reproduction package.

Why this experiment exists
--------------------------
Publication gate G2 (independent reproduction) is the SOLE remaining blocker to
``paper_ready`` (G1/G3/G4 met per ``ops/north-star.md`` §2). G2 needs ">=1
reproducer who is NOT the operator". The lowest-friction path to that reproducer
is a SELF-CONTAINED package a true stranger runs in ONE command with zero Carnot
knowledge and no repo checkout — just the tarball.

This experiment:

  1. Assembles ``dist/g2-fover-repro/`` (harness + corpus + FR-11 state files +
     carnot source + pinned requirements.txt + pyproject + one-command run.sh +
     README), tars it to ``dist/g2-fover-repro.tar.gz``, and computes its sha256.
  2. Records the IPFS CID if a node is available (decentralization rule 3).
  3. VERIFIES the package by extracting the tarball into a fresh temp dir and
     running the one command in a clean Docker container (a different base image
     than the operator's box), confirming both headline numbers land in their
     published CIs.
  4. Appends the package path + checksum to
     ``ops/reproduction-runbook-fover-headline.md`` (never deletes).
  5. Emits the artifact. NEVER sets ``g2_independent_reproducer=true``.

What it NEVER does: push, trigger external CI, edit operator-curated docs
(north-star.md / index.html / README.md), or modify research_conductor.py.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3476_fover_g2_self_contained_external_package_v1.py

Spec: REQ-PUBLISH-038, SCENARIO-PUBLISH-038, SCENARIO-PUBLISH-038B
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.eval.fover_g2_package import (  # noqa: E402
    PACKAGE_NAME,
    TARBALL_REL,
    build_artifact,
    build_package_tree,
    check_preconditions,
    classify_ci,
    docker_is_available,
    make_tarball,
    maybe_ipfs_add,
    parse_harness_numbers,
    verify_package_in_docker,
)

OUT_PATH = REPO_ROOT / "results" / (
    "experiment_3476_fover_g2_self_contained_external_package_v1.json"
)
RUNBOOK_PATH = REPO_ROOT / "ops" / "reproduction-runbook-fover-headline.md"


def _parse_isolated_checksum(stdout: str) -> str | None:
    """Pull the harness-printed ``reproducibility_checksum:`` from stdout."""
    m = re.search(r"reproducibility_checksum:\s*(\S+)", stdout)
    return m.group(1) if m else None


def _append_runbook(package_path: str, package_sha256: str, cid: str | None,
                    reproduced: bool) -> None:
    """Append (never delete) the package path + checksum to the runbook."""
    if not RUNBOOK_PATH.exists():
        return
    cid_line = f"- IPFS CID: `{cid}`\n" if cid else ""
    section = (
        "\n## Self-contained reproduction package (exp3476, 2026-05-30)\n\n"
        "A single self-contained tarball now lets a true stranger reproduce the\n"
        "FoVer headline in one command, with no repo checkout and no Carnot\n"
        "knowledge. Unpack and run:\n\n"
        "```bash\n"
        "tar xzf g2-fover-repro.tar.gz && cd g2-fover-repro && bash run.sh\n"
        "```\n\n"
        "`run.sh` installs the pinned dependencies, installs the package, and runs\n"
        "the reproducer harness, which exits non-zero unless condition-A mean AUROC\n"
        "lands in [0.9027, 0.9235] AND learning_contribution mean in [0.0125, 0.0245].\n\n"
        f"- Package: `{package_path}`\n"
        f"- sha256: `{package_sha256}`\n"
        f"{cid_line}"
        f"- Clean-environment verification reproduced both numbers in CI: "
        f"{reproduced}\n\n"
        "G2 is still NOT closed by building + verifying this package — closure\n"
        "requires an actual external/CI run by a non-operator. Artifact:\n"
        "`results/experiment_3476_fover_g2_self_contained_external_package_v1.json`.\n"
    )
    with open(RUNBOOK_PATH, "a", encoding="utf-8") as f:
        f.write(section)


def main() -> int:
    """Build the package, verify it in a clean environment, emit the artifact."""
    start = time.time()

    # ------------------------------------------------------------------ #
    # STEP 0: PRECONDITIONS                                              #
    # ------------------------------------------------------------------ #
    preconditions = check_preconditions(REPO_ROOT)
    if not preconditions["ok"]:
        artifact = build_artifact(
            start_time=start,
            preconditions=preconditions,
            package_path=None,
            package_sha256=None,
            ipfs_result={"ipfs_available": False, "package_cid": None},
            clean_env_method=None,
            cond_a=None,
            lc=None,
            verification_attempted=False,
            isolated_checksum=None,
            manifest={},
        )
        # Preconditions failed: harness/corpus missing is the honest verdict.
        artifact["honest_verdict"] = "complete: " + preconditions["blocked_reason"]
        artifact["g2_status"] = "blocked_" + preconditions["blocked_reason"]
        _write(artifact)
        return 0

    # ------------------------------------------------------------------ #
    # STEP 1: BUILD PACKAGE TREE + TARBALL + SHA256                      #
    # ------------------------------------------------------------------ #
    dist_dir = REPO_ROOT / "dist"
    pkg_dir = dist_dir / PACKAGE_NAME
    import shutil

    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    print(f"Building self-contained package at {pkg_dir} ...")
    manifest = build_package_tree(REPO_ROOT, pkg_dir)
    print(
        f"  corpus_sha256={manifest['corpus_sha256'][:16]}...  "
        f"state_files_packaged={manifest['state_files_copied']}"
    )

    tar_path = REPO_ROOT / TARBALL_REL
    make_tarball(pkg_dir, tar_path)
    from carnot.eval.fover_g2_package import sha256_of_file

    package_sha256 = sha256_of_file(tar_path)
    package_path = str(tar_path.relative_to(REPO_ROOT))
    print(f"  tarball={package_path}  sha256={package_sha256[:16]}...")

    # ------------------------------------------------------------------ #
    # STEP 2: IPFS (non-blocking)                                        #
    # ------------------------------------------------------------------ #
    ipfs_result = maybe_ipfs_add(tar_path)
    print(
        f"  ipfs_available={ipfs_result['ipfs_available']}  "
        f"cid={ipfs_result['package_cid']}"
    )

    # ------------------------------------------------------------------ #
    # STEP 3: VERIFY IN CLEAN ENVIRONMENT (docker preferred)             #
    # ------------------------------------------------------------------ #
    clean_env_method: str | None = None
    cond_a: float | None = None
    lc: float | None = None
    isolated_checksum: str | None = None
    verification_attempted = False

    if docker_is_available():
        clean_env_method = "docker"
        verification_attempted = True
        print("  verifying extracted package in a clean Docker container ...")
        result = verify_package_in_docker(tar_path)
        if result.get("error"):
            print(f"  docker verification error: {result['error']}")
            verification_attempted = False
            clean_env_method = None
        else:
            stdout = result.get("stdout", "")
            cond_a, lc = parse_harness_numbers(stdout)
            isolated_checksum = _parse_isolated_checksum(stdout)
            exit_code = result.get("exit_code")
            print(
                f"  docker run exit_code={exit_code}  "
                f"cond_a={cond_a}  lc={lc}"
            )
            if result.get("stderr") and exit_code not in (0, None):
                print(f"  stderr tail: {result['stderr'][-500:]}")
    else:
        print("  Docker unavailable — package built but not clean-env verified.")

    # ------------------------------------------------------------------ #
    # STEP 4: APPEND RUNBOOK (never delete)                             #
    # ------------------------------------------------------------------ #
    _, _, reproduced = classify_ci(cond_a, lc)
    _append_runbook(package_path, package_sha256, ipfs_result["package_cid"], reproduced)

    # ------------------------------------------------------------------ #
    # STEP 5: EMIT ARTIFACT                                              #
    # ------------------------------------------------------------------ #
    artifact = build_artifact(
        start_time=start,
        preconditions=preconditions,
        package_path=package_path,
        package_sha256=package_sha256,
        ipfs_result=ipfs_result,
        clean_env_method=clean_env_method,
        cond_a=cond_a,
        lc=lc,
        verification_attempted=verification_attempted,
        isolated_checksum=isolated_checksum,
        manifest=manifest,
    )
    _write(artifact)
    print(f"\nhonest_verdict: {artifact['honest_verdict']}")
    print(f"g2_status: {artifact['g2_status']}")
    print(f"package_verified_reproduces: {artifact['package_verified_reproduces']}")
    print(f"artifact written to: {OUT_PATH}")
    return 0


def _write(artifact: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
