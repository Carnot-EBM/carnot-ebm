"""Self-contained FoVer G2 reproduction-package builder + clean-room verifier.

Why this module exists
----------------------
Publication gate **G2** (independent reproduction) is the *sole* remaining
blocker to ``paper_ready`` (G1/G3/G4 are met per ``ops/north-star.md`` §2). G2
requires ">=1 reproducer who is NOT the operator". The lowest-friction path to
that reproducer is a *self-contained package* a true stranger can run in **one
command** with **zero Carnot knowledge** and **no repo checkout** — just the
tarball.

This module is the testable core behind ``scripts/experiment_3476_*``. It:

1. Assembles a package directory (harness + corpus + FR-11 state files + the
   ``carnot`` source + a pinned ``requirements.txt`` + ``pyproject.toml`` + a
   one-command ``run.sh`` + a package ``README``).
2. Tars it to ``dist/g2-fover-repro.tar.gz`` and computes its sha256
   (content-addressed integrity + the G4 trace anchor).
3. Optionally records the IPFS CID (decentralization rule 3) when a node exists.
4. VERIFIES the package by extracting the tarball into a fresh temp dir and
   running the one command in a clean environment (Docker preferred, fresh venv
   fallback) — confirming both headline numbers land inside their published CIs.

It deliberately does NOT push, does NOT trigger external CI, and NEVER sets
``g2_independent_reproducer=true``. Only an actual external/CI run by a
non-operator may flip that. The package brings G2 to "one non-operator command
from met", and reports an honest ``g2_status``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Published acceptance CI (exp2837 5-seed dual-condition headline).
# ---------------------------------------------------------------------------

CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245

RANDOM_SEEDS = [42, 137, 271, 314, 1729]
N_EXAMPLES = 1000

HARNESS_REL = "scripts/reproduce_fover_headline.py"
CORPUS_REL = "data/fover_corpus.jsonl"

PACKAGE_NAME = "g2-fover-repro"
TARBALL_REL = "dist/g2-fover-repro.tar.gz"

# A stock public Python base image — deliberately NOT the operator's venv, so
# the recompute is from a fresh OS + interpreter + pip resolution.
DOCKER_BASE_IMAGE = "python:3.12-slim"

# The single command a stranger runs after unpacking the tarball. This is the
# "true-stranger test": one command, no repo, no Carnot knowledge.
ONE_COMMAND_REPRO = (
    "tar xzf g2-fover-repro.tar.gz && cd g2-fover-repro && bash run.sh"
)

# FR-11 session-memory state globs the FoVer scorer consults for condition A
# (production). Mirrors FR11_STATE_GLOBS in
# python/carnot/eval/fover_memory_leakage_v3.py. The package MUST include
# matching files or condition A collapses to architecture-only and
# learning_contribution ~ 0, which would fall outside the published CI.
FR11_STATE_GLOBS = (
    "data/constraint_memory.db",
    "data/fr11_*.json",
    "data/fr11_*.jsonl",
    "results/constraint_memory*.json",
    "results/constraint_patterns*.json",
    "results/fr11_*.json",
    "results/fr11_*.jsonl",
    "results/nexus_constraint_memory*.json",
    "results/session_memory_*/**/session_state.json",
    "results/exp_448_session_memory/**/session_state.json",
)


# ---------------------------------------------------------------------------
# Step 0 — preconditions (pure, easily testable)
# ---------------------------------------------------------------------------


def check_preconditions(repo_root: Path) -> dict[str, Any]:
    """Step 0 PRECONDITIONS: the harness + corpus must both be present.

    Principle: naming a missing resource up front pre-empts the fabrication mode
    where the agent silently lacks a resource and synthesizes a passing artifact
    instead of emitting a ``blocked_*`` verdict. A clean-env mechanism (Docker or
    a fresh venv) is probed separately and is non-blocking for the *build* step.
    """
    harness = repo_root / HARNESS_REL
    corpus = repo_root / CORPUS_REL
    if not harness.exists() or not corpus.exists():
        return {
            "ok": False,
            "blocked_reason": "blocked_fover_harness_or_corpus_missing",
            "harness_present": harness.exists(),
            "corpus_present": corpus.exists(),
        }
    return {"ok": True, "harness": str(harness), "corpus": str(corpus)}


# ---------------------------------------------------------------------------
# Pinned dependency resolution (pure-ish — reads pyproject + installed metadata)
# ---------------------------------------------------------------------------


def _dep_name(requirement: str) -> str:
    """Extract the bare distribution name from a PEP 508 requirement string.

    e.g. ``"scikit-learn>=1.4"`` -> ``"scikit-learn"``; ``"jax[cuda]>=0.4"`` ->
    ``"jax"``. We only need the name to look up the installed exact version.
    """
    # Strip environment markers (after ';') and extras (the '[...]' block), then
    # cut at the first version-specifier / whitespace character.
    base = requirement.split(";", 1)[0].strip()
    base = re.split(r"[\[<>=!~ ]", base, maxsplit=1)[0]
    return base.strip()


def read_pyproject_dependencies(repo_root: Path) -> list[str]:
    """Return the ``[project].dependencies`` list verbatim from pyproject.toml."""
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    return [str(d) for d in deps]


def build_requirements_txt(
    repo_root: Path,
    version_lookup: Callable[[str], str] | None = None,
) -> str:
    """Render a pinned ``requirements.txt`` from the pyproject dependency set.

    Each dependency is pinned to the EXACT version currently installed in the
    measuring environment (via ``importlib.metadata.version``). Exact pins are
    the difference between "reproduces on my machine" and "reproduces on a
    stranger's machine years from now": a range can resolve to a newer release
    whose numerics drift. If a dependency is not importable in this environment
    we fall back to the pyproject specifier so the line is never silently
    dropped (the stranger still gets a usable, if unpinned, requirement).
    """
    if version_lookup is None:
        version_lookup = importlib_metadata.version
    lines = [
        "# Pinned dependency set for the FoVer G2 self-contained reproduction.",
        "# Exact versions = the environment the headline AUROC 0.9131 was measured in.",
        "# Generated from pyproject.toml [project.dependencies]; do not hand-edit.",
    ]
    for req in read_pyproject_dependencies(repo_root):
        name = _dep_name(req)
        try:
            version = version_lookup(name)
            lines.append(f"{name}=={version}")
        except importlib_metadata.PackageNotFoundError:
            # Not installed here — keep the pyproject range rather than drop it.
            lines.append(req)
    return "\n".join(lines) + "\n"


def build_run_sh() -> str:
    """Return the one-command entry point a stranger runs after unpacking.

    The script installs the pinned dependencies first (exact reproducibility),
    then installs the ``carnot`` package WITHOUT re-resolving deps
    (``--no-deps``) so the pinned versions are the ones actually used, then runs
    the reproducer harness. The harness's ``main()`` returns non-zero unless BOTH
    headline numbers land in their published CIs, and ``set -e`` propagates that
    non-zero exit — so a zero exit from ``run.sh`` *is* the pass.
    """
    return (
        "#!/usr/bin/env bash\n"
        "# One-command FoVer G2 reproduction. Exits non-zero unless condition-A\n"
        "# mean AUROC lands in [0.9027, 0.9235] AND learning_contribution mean in\n"
        "# [0.0125, 0.0245]. A zero exit (echo $? -> 0) IS the independent-repro pass.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'cd "$HERE"\n'
        "export JAX_PLATFORMS=cpu\n"
        "python3 -m pip install --quiet --no-cache-dir -r requirements.txt\n"
        "python3 -m pip install --quiet --no-cache-dir --no-deps -e .\n"
        "python3 scripts/reproduce_fover_headline.py\n"
    )


def build_package_readme(corpus_sha256: str, package_sha256: str | None = None) -> str:
    """Return the package-level README a true stranger reads first.

    Plain-language: what this proves, the single command, and the exact output a
    green run produces. No Carnot internals, no experiment IDs — a stranger with
    zero project knowledge must be able to follow it.
    """
    pkg_line = (
        f"This package sha256: {package_sha256}\n" if package_sha256 else ""
    )
    return (
        "# FoVer headline — one-command independent reproduction\n"
        "\n"
        "This self-contained package lets anyone independently reproduce the\n"
        "headline result of the Carnot-EBM verifier ensemble on the FoVer\n"
        "step-error corpus. You need **no repository checkout** and **no prior\n"
        "knowledge** of the project. It is **CPU-only** (no GPU, no large model,\n"
        "no API keys) and takes a couple of minutes, most of which is pip install.\n"
        "\n"
        "## The one command\n"
        "\n"
        "```bash\n"
        f"{ONE_COMMAND_REPRO}\n"
        "```\n"
        "\n"
        "(If you have already unpacked the tarball, just run `bash run.sh` from\n"
        "inside the `g2-fover-repro/` directory.)\n"
        "\n"
        "## What a green run proves\n"
        "\n"
        "`run.sh` exits **non-zero unless** both numbers below land inside their\n"
        "published 95% confidence intervals, so a zero exit (`echo $?` -> `0`)\n"
        "*is* the pass:\n"
        "\n"
        "| Quantity | Must land in | Published value |\n"
        "|---|---|---|\n"
        "| condition-A (production) mean AUROC | `[0.9027, 0.9235]` | 0.9131 |\n"
        "| learning_contribution (FR-11 ablation) mean | `[0.0125, 0.0245]` | 0.0185 |\n"
        "\n"
        "Both over **n=1,000**, **5 seeds** `[42, 137, 271, 314, 1729]`.\n"
        "\n"
        "## Expected output (tail)\n"
        "\n"
        "```\n"
        "condition A (production)        mean AUROC: 0.9131\n"
        "condition B (architecture-only) mean AUROC: 0.8947\n"
        "learning contribution:                      0.0185\n"
        "\n"
        "condition A in CI [0.9027, 0.9235]: True\n"
        "learning_contribution in CI [0.0125, 0.0245]: True\n"
        "\n"
        "RESULT: PASS — FoVer headline reproduces within published CI\n"
        "```\n"
        "\n"
        "## Integrity\n"
        "\n"
        "```\n"
        f"sha256(data/fover_corpus.jsonl) = {corpus_sha256}\n"
        f"{pkg_line}"
        "```\n"
        "\n"
        "## What this package does NOT claim\n"
        "\n"
        "It does not by itself close gate G2 (independent reproduction). G2 closes\n"
        "when a person who is **not** the project operator runs this command and\n"
        "reports both numbers in range. If that is you: thank you — please report\n"
        "the two numbers, your platform, and your Python/library versions.\n"
    )


# ---------------------------------------------------------------------------
# Package-tree assembly + tarball
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Stream a file through SHA-256 — content-addressed integrity anchor."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_glob_set(repo_root: Path, pkg_dir: Path) -> int:
    """Copy the FR-11 state-file globs into the package tree. Returns the count.

    Condition A (production) needs these or it degrades to architecture-only and
    learning_contribution ~ 0, which falls outside the published CI.
    """
    copied = 0
    for pattern in FR11_STATE_GLOBS:
        for src in repo_root.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(repo_root)
            dst = pkg_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def build_package_tree(
    repo_root: Path,
    pkg_dir: Path,
    version_lookup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Assemble the self-contained package directory under ``pkg_dir``.

    Copies exactly what a stranger needs to run the one command: build metadata,
    the ``carnot`` source (sans prebuilt ``.so`` / ``__pycache__`` to stay
    small), the committed corpus, the FR-11 state files, the reproducer harness,
    a pinned ``requirements.txt``, a one-command ``run.sh``, and a package
    README. Returns a manifest with the corpus checksum + state-file count.
    """
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata + license/readme (pyproject reads these at install time).
    for name in ("pyproject.toml", "LICENSE", "NOTICE"):
        src = repo_root / name
        if src.exists():
            shutil.copy2(src, pkg_dir / name)

    # Package source. Exclude the prebuilt Rust .so (not needed by the FoVer
    # verifier-scoring path) and __pycache__ to keep the tarball small.
    shutil.copytree(
        repo_root / "python",
        pkg_dir / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )

    # The reproducer harness + the committed corpus.
    (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / HARNESS_REL, pkg_dir / HARNESS_REL)
    (pkg_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / CORPUS_REL, pkg_dir / CORPUS_REL)
    corpus_sha256 = sha256_of_file(pkg_dir / CORPUS_REL)

    state_files_copied = _copy_glob_set(repo_root, pkg_dir)

    # Pinned deps + one-command entry + package README.
    (pkg_dir / "requirements.txt").write_text(
        build_requirements_txt(repo_root, version_lookup), encoding="utf-8"
    )
    run_sh = pkg_dir / "run.sh"
    run_sh.write_text(build_run_sh(), encoding="utf-8")
    run_sh.chmod(0o755)
    # README named per-package; carnot's own README is intentionally NOT copied
    # (operator-curated) — the package README is purpose-built for a stranger.
    (pkg_dir / "README.md").write_text(
        build_package_readme(corpus_sha256), encoding="utf-8"
    )

    return {
        "corpus_sha256": corpus_sha256,
        "state_files_copied": state_files_copied,
        "package_dir": str(pkg_dir),
    }


def make_tarball(pkg_dir: Path, tar_path: Path) -> Path:
    """Tar ``pkg_dir`` into ``tar_path`` with a top-level ``g2-fover-repro/`` dir.

    The arcname is the package name so a stranger who runs ``tar xzf`` gets a
    single tidy directory (matching ``ONE_COMMAND_REPRO``).
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg_dir, arcname=PACKAGE_NAME)
    return tar_path


# ---------------------------------------------------------------------------
# IPFS (decentralization rule 3) — non-blocking
# ---------------------------------------------------------------------------


def maybe_ipfs_add(
    tar_path: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Record the tarball's IPFS CID if an IPFS node is available.

    Decentralization rule 3 wants published artifacts content-addressed so any
    party can verify integrity and become a mirror. We try a real ``ipfs add``
    (which pins to the local node when its daemon/repo is reachable); if that
    fails we fall back to ``ipfs add --only-hash`` which computes the identical
    CID without touching a datastore. Returns ``{ipfs_available, package_cid}``.
    Never raises — IPFS absence must not fail the task.
    """
    if shutil.which("ipfs") is None:
        return {"ipfs_available": False, "package_cid": None}
    for args in (
        ["ipfs", "add", "-Q", str(tar_path)],
        ["ipfs", "add", "-Q", "--only-hash", str(tar_path)],
    ):
        try:
            proc = runner(args, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            cid = proc.stdout.strip().splitlines()[-1].strip()
            return {"ipfs_available": True, "package_cid": cid}
    return {"ipfs_available": False, "package_cid": None}


# ---------------------------------------------------------------------------
# Clean-environment verification
# ---------------------------------------------------------------------------


def docker_is_available(runner: Callable[..., Any] = subprocess.run) -> bool:
    """Return True iff a usable Docker daemon is reachable (client + ``info``)."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = runner(["docker", "info"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def parse_harness_numbers(stdout: str) -> tuple[float | None, float | None]:
    """Extract (condition_a_mean, learning_contribution_mean) from harness stdout.

    The harness ``main()`` prints two human-readable lines we parse:
    ``"condition A (production)        mean AUROC: 0.9131"`` and
    ``"learning contribution:                      0.0185"``. Parsing the printed
    floats lets us record the isolated numbers in the artifact without modifying
    the harness. Returns ``(None, None)`` if a line is absent (e.g. on error).
    """
    cond_a: float | None = None
    lc: float | None = None
    for line in stdout.splitlines():
        m = re.search(r"condition A \(production\).*?:\s*([0-9]*\.?[0-9]+)", line)
        if m:
            cond_a = float(m.group(1))
            continue
        m = re.search(r"learning contribution:\s*([0-9]*\.?[0-9]+)", line)
        if m:
            lc = float(m.group(1))
    return cond_a, lc


def classify_ci(cond_a: Any, lc: Any) -> tuple[bool, bool, bool]:
    """Return (cond_a_in_ci, lc_in_ci, reproduced).

    The headline only "reproduces" when BOTH numbers land inside their published
    CIs; a single number in range is not sufficient. ``None`` is never in any CI.
    """
    cond_a_in_ci = (
        cond_a is not None
        and CONDITION_A_CI_LOW <= float(cond_a) <= CONDITION_A_CI_HIGH
    )
    lc_in_ci = (
        lc is not None
        and LEARNING_CONTRIB_CI_LOW <= float(lc) <= LEARNING_CONTRIB_CI_HIGH
    )
    return cond_a_in_ci, lc_in_ci, bool(cond_a_in_ci and lc_in_ci)


def verify_package_in_docker(
    tar_path: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Extract the tarball into a fresh temp dir and run the one command in Docker.

    This is the true-stranger test: a stock ``python:3.12-slim`` container (which
    shares nothing with the operator's venv) gets ONLY the extracted package
    mounted, then runs ``run.sh``. The container's exit code is the pass signal —
    ``run.sh`` propagates the harness's non-zero exit when either number is out of
    CI. Returns ``{exit_code, stdout, stderr, error?}``. Never raises on a Docker
    failure — captures it so the caller can fall back honestly.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_g2_pkg_verify_"))
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extractall(tar, tmpdir)
        extracted = tmpdir / PACKAGE_NAME
        if not (extracted / "run.sh").exists():
            return {"error": "package_missing_run_sh", "exit_code": None}
        proc = runner(
            [
                "docker", "run", "--rm",
                "-v", f"{extracted}:/pkg",
                "-w", "/pkg",
                "-e", "JAX_PLATFORMS=cpu",
                DOCKER_BASE_IMAGE,
                "bash", "run.sh",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr[-2000:],
        }
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"error": f"docker_exception_{type(exc).__name__}", "exit_code": None}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar, refusing any member that escapes ``dest`` (path traversal).

    Defense against a malicious tarball with ``../`` or absolute members. We
    built this tarball ourselves, but extraction code should never trust member
    paths — a stranger reusing this module on an untrusted tarball is protected.
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"unsafe tar member path: {member.name}")
    tar.extractall(dest)  # noqa: S202 — members validated above


# ---------------------------------------------------------------------------
# Verdict / status mapping (pure)
# ---------------------------------------------------------------------------


def determine_verdict_and_status(
    package_built: bool,
    clean_env_method: str | None,
    reproduced: bool,
    verification_attempted: bool,
) -> tuple[str, str]:
    """Map the build + verification outcome to a terminal verdict + g2_status.

    All branches are terminal ``complete:`` states — the experiment ran to a
    scientific conclusion in each. ``reproduced`` requires a real clean-env run
    that landed both numbers in their CIs; a built-but-unverified package is
    honestly distinguished from a verified one and from a still-failing one.
    """
    if not package_built:
        return (
            "complete: fover_g2_package_repro_failing_build_failed",
            "still_failing_build_failed",
        )
    if reproduced:
        return (
            "complete: fover_g2_self_contained_package_verified_external_run_pending",
            "self_contained_package_verified_external_run_pending",
        )
    if not verification_attempted:
        return (
            "complete: fover_g2_package_built_verification_unavailable",
            "package_built_verification_unavailable",
        )
    return (
        "complete: fover_g2_package_repro_failing_clean_env_out_of_ci",
        "still_failing_clean_env_out_of_ci",
    )


OPERATOR_ACTION_REQUIRED = (
    "Closing G2 requires a person who is NOT the operator to run the one-command "
    "package (or trigger the CI workflow) and report condition-A AUROC in "
    "[0.9027, 0.9235] and learning_contribution in [0.0125, 0.0245]. Per the "
    "Operator-Only External Publication discipline, autonomous work may build and "
    "verify the package but may not flip g2_independent_reproducer."
)


def build_artifact(
    *,
    start_time: float,
    preconditions: dict[str, Any],
    package_path: str | None,
    package_sha256: str | None,
    ipfs_result: dict[str, Any],
    clean_env_method: str | None,
    cond_a: float | None,
    lc: float | None,
    verification_attempted: bool,
    isolated_checksum: str | None,
    manifest: dict[str, Any],
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Assemble the final experiment artifact with principle annotations."""
    duration_s = clock() - start_time
    _, _, reproduced = classify_ci(cond_a, lc)
    package_built = bool(package_sha256)
    verdict, g2_status = determine_verdict_and_status(
        package_built, clean_env_method, reproduced, verification_attempted
    )
    return {
        "artifact": "experiment_3476_fover_g2_self_contained_external_package_v1",
        "schema": "carnot.fover_g2_self_contained_external_package_v1",
        "experiment": 3476,
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "package_path": package_path,
        "package_sha256": package_sha256,
        "package_cid": ipfs_result.get("package_cid"),
        "ipfs_available": ipfs_result.get("ipfs_available", False),
        "one_command_repro": ONE_COMMAND_REPRO,
        "clean_env_method": clean_env_method,
        "condition_a_auroc_isolated": cond_a,
        "learning_contribution_isolated": lc,
        "package_verified_reproduces": reproduced,
        "g2_status": g2_status,
        "g2_independent_reproducer": False,
        "operator_action_required": OPERATOR_ACTION_REQUIRED,
        "reproducibility_checksum": isolated_checksum or package_sha256,
        "random_seed": RANDOM_SEEDS,
        "duration_s": duration_s,
        "n_examples": N_EXAMPLES,
        "live_model_invoked": False,
        "preconditions_checked": preconditions,
        "corpus_sha256": manifest.get("corpus_sha256"),
        "state_files_packaged": manifest.get("state_files_copied"),
        "field_principles": {
            "honest_verdict": (
                "complete:/success:/passed:/shipped_ prefix required by CLAUDE.md "
                "Verdict Terminal-Prefix Discipline."
            ),
            "inference_substrate": (
                "verifier_ensemble_against_cached_candidates: scores the verifier "
                "ensemble against the labeled FoVer corpus; no live LLM is loaded, "
                "so adversarial_verify.py applies the 1s floor not the 60s floor."
            ),
            "package_path": (
                "dist/g2-fover-repro.tar.gz — the self-contained artifact a "
                "stranger runs with one command, no repo checkout."
            ),
            "package_sha256": (
                "Content-addressed checksum — integrity + the G4 numbers-trace "
                "anchor (the package is what produced the verified numbers)."
            ),
            "package_cid": (
                "IPFS CID if a node was available, else null (decentralization "
                "rule 3: content-addressed mirroring)."
            ),
            "ipfs_available": "Whether the CID could be computed this run.",
            "one_command_repro": (
                "The single command a stranger runs — the true-stranger test."
            ),
            "clean_env_method": (
                "'docker' | 'fresh_venv' — how the extracted package was verified, "
                "or null if no clean environment was available."
            ),
            "condition_a_auroc_isolated": (
                "Recomputed production AUROC from the extracted self-contained "
                "package in a clean environment."
            ),
            "learning_contribution_isolated": (
                "Recomputed FR-11 ablation; must land in [0.0125, 0.0245]."
            ),
            "package_verified_reproduces": (
                "Boolean: the extracted package reproduced BOTH numbers in their "
                "CIs from a clean environment (Docker/fresh venv)."
            ),
            "g2_status": (
                "Honest string: self_contained_package_verified_external_run_pending "
                "| package_built_verification_unavailable | still_failing_<cause>."
            ),
            "g2_independent_reproducer": (
                "MUST be false — only an actual external/CI run by a non-operator "
                "flips it true."
            ),
            "operator_action_required": (
                "The one-line note that closing G2 needs a non-operator to run the "
                "package (Operator-Only External Publication)."
            ),
            "reproducibility_checksum": (
                "Content hash anchoring the verified run (isolated harness "
                "checksum when available, else the package sha256)."
            ),
            "random_seed": "The published seeds [42,137,271,314,1729].",
            "duration_s": "Package build + clean-env install + CPU scoring wall time.",
        },
    }
