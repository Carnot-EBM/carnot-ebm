"""Clean-room regression verifier for the FoVer G2 package + external-ask author.

Why this module exists
----------------------
Publication gate **G2** (independent reproduction) is the *sole* remaining
blocker to ``paper_ready`` (G1/G3/G4 are met per ``ops/north-star.md`` §2).
Exp 3476 built a *self-contained* reproduction package
(``dist/g2-fover-repro.tar.gz``) that a true stranger runs in one command. The
two pieces of autonomous work that remain before a non-operator actually runs it
are:

1. **REGRESSION-VERIFY** that the package *still* reproduces the headline AUROC
   from a fresh environment — catching any drift (corpus, state files, source,
   pinned deps) since the package was built at milestone .320. A package that
   built green once but silently rotted is worse than no package, because it
   wastes the one external reproducer's goodwill.
2. **PREPARE THE LOWEST-FRICTION EXTERNAL ASK** — a public ``workflow_dispatch``
   reproduction workflow, a one-paragraph reproducer invite, and an operator
   checklist whose terminal step is a single click. The point is to reduce the
   operator's remaining G2 action to one button.

This module is the testable core behind ``scripts/experiment_3488_*``. It
deliberately does **NOT** push, does **NOT** trigger external CI, and **NEVER**
sets ``g2_met`` / ``g2_independent_reproducer`` true. Per the Operator-Only
External Publication discipline, only an actual external/CI run by a
non-operator may flip those. This module brings G2 to "one click from met".

Isolation strategy
-------------------
The regression run uses an environment isolated from the *working repo* so a
stale on-disk repo cannot mask package drift:

- **fresh_venv** (preferred when constructible + deps installable): a brand-new
  ``python -m venv`` that installs the package's pinned ``requirements.txt`` then
  the package itself, and runs the one command. Strongest isolation: fresh
  interpreter site-packages.
- **isolated_dir** (always-available fallback): the tarball is unpacked into a
  temp dir OUTSIDE the repo and the packaged harness is run there with the
  *packaged* ``carnot`` source on ``PYTHONPATH`` (never the working repo's import
  path). This exercises the packaged code + packaged corpus + packaged FR-11
  state from a directory the working tree cannot contaminate. Pinned deps are
  identical to the measuring environment, so the numbers are faithful.

Either way the regression gate is the same: the reproduced condition-A mean
AUROC must land inside the published CI ``[0.9027, 0.9235]``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from carnot.eval.fover_g2_package import (
    CONDITION_A_CI_HIGH,
    CONDITION_A_CI_LOW,
    LEARNING_CONTRIB_CI_HIGH,
    LEARNING_CONTRIB_CI_LOW,
    N_EXAMPLES,
    ONE_COMMAND_REPRO,
    PACKAGE_NAME,
    RANDOM_SEEDS,
    TARBALL_REL,
    _safe_extractall,
    classify_ci,
    maybe_ipfs_add,
    parse_harness_numbers,
    sha256_of_file,
)

# Where the upstream package builder (Exp 3476) recorded the canonical sha256 we
# regression-check against. A mismatch means the on-disk tarball drifted from the
# one whose numbers were verified at .320 — that is the integrity signal.
EXP3476_ARTIFACT_REL = (
    "results/experiment_3476_fover_g2_self_contained_external_package_v1.json"
)

HARNESS_REL_IN_PKG = "scripts/reproduce_fover_headline.py"

# The lowest-friction external-ask artifacts. All are written to the WORKING TREE
# only — never pushed, never triggered. The operator checklist's terminal step is
# the single external-ask action.
REPRO_WORKFLOW_REL = ".github/workflows/fover-g2-repro.yml"
REPRODUCER_INVITE_REL = "docs/g2-reproducer-invite.md"
OPERATOR_CHECKLIST_REL = "ops/g2-external-ask-operator-checklist.md"

RUNBOOK_REL = "ops/reproduction-runbook-fover-headline.md"


# ---------------------------------------------------------------------------
# Step 0 — preconditions (pure, easily testable)
# ---------------------------------------------------------------------------


def check_preconditions(repo_root: Path) -> dict[str, Any]:
    """Step 0 PRECONDITIONS for the regression verify.

    (a) The package tarball must be present OR rebuildable from Exp 3476's
        builder (the harness + corpus the builder needs are themselves the
        builder's preconditions, so "rebuildable" reduces to "those inputs
        exist"). (b) An isolated runner must be constructible — ``venv`` is in
        the stdlib so a fresh venv is always at least *attemptable*, and the
        ``isolated_dir`` fallback only needs ``tempfile`` + the current
        interpreter, which always exist. We surface both facts so the artifact
        records exactly which precondition gated a block.

    Principle: naming a missing resource up front pre-empts the fabrication mode
    where the agent silently lacks the package and synthesizes a passing number
    instead of emitting a ``blocked_*`` verdict.
    """
    tar_path = repo_root / TARBALL_REL
    harness = repo_root / HARNESS_REL_IN_PKG
    corpus = repo_root / "data" / "fover_corpus.jsonl"
    package_present = tar_path.exists()
    rebuildable = harness.exists() and corpus.exists()
    if not package_present and not rebuildable:
        return {
            "ok": False,
            "blocked_reason": "blocked_g2_package_unavailable",
            "package_present": package_present,
            "package_rebuildable": rebuildable,
        }
    # An isolated runner is constructible iff we can at least run the
    # always-available ``isolated_dir`` fallback (tempfile + current python).
    isolated_runner_ok = bool(sys.executable)
    if not isolated_runner_ok:
        return {
            "ok": False,
            "blocked_reason": "blocked_fresh_env_unavailable",
            "package_present": package_present,
            "package_rebuildable": rebuildable,
        }
    return {
        "ok": True,
        "package_present": package_present,
        "package_rebuildable": rebuildable,
        "tarball": str(tar_path),
    }


def read_recorded_sha256(repo_root: Path) -> str | None:
    """Return the package sha256 Exp 3476 recorded, or None if unreadable.

    This is the canonical checksum the regression run compares the on-disk
    tarball against. Reading it from the artifact (not hard-coding it) keeps the
    check honest if the upstream package is legitimately rebuilt and re-recorded.
    """
    artifact = repo_root / EXP3476_ARTIFACT_REL
    if not artifact.exists():
        return None
    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("package_sha256")
    return str(value) if value else None


def verify_sha256(tar_path: Path, recorded_sha256: str | None) -> dict[str, Any]:
    """Re-compute the tarball sha256 and compare to the recorded one.

    ``package_sha256_verified`` is true only on an exact match. A mismatch means
    the on-disk tarball drifted from the one whose numbers were verified — an
    external party fetching by checksum would get a different artifact, so this
    is a content-integrity gate for the external ask.
    """
    if not tar_path.exists():
        return {"computed": None, "recorded": recorded_sha256, "verified": False}
    computed = sha256_of_file(tar_path)
    verified = recorded_sha256 is not None and computed == recorded_sha256
    return {"computed": computed, "recorded": recorded_sha256, "verified": verified}


# ---------------------------------------------------------------------------
# Isolated-environment regression runs
# ---------------------------------------------------------------------------


def _extract_tarball(tar_path: Path, dest: Path) -> Path:
    """Safely extract the tarball into ``dest`` and return the package dir."""
    with tarfile.open(tar_path, "r:gz") as tar:
        _safe_extractall(tar, dest)
    return dest / PACKAGE_NAME


def run_package_in_isolated_dir(
    tar_path: Path,
    python_exe: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = 600,
) -> dict[str, Any]:
    """Unpack the tarball OUTSIDE the repo and run the packaged harness there.

    The packaged ``carnot`` source is put on ``PYTHONPATH`` (the package ships
    its source under ``python/``) and the harness is run with ``cwd`` inside the
    extracted dir — so it reads the *packaged* corpus + FR-11 state, never the
    working repo's. This is the always-available fallback: it needs no network
    and no new venv, and the pinned deps already match the measuring environment.
    Returns ``{method, exit_code, stdout, stderr, error?}``; never raises on a
    subprocess failure (captures it so the caller can classify honestly).
    """
    python_exe = python_exe or sys.executable
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_g2_regression_iso_"))
    try:
        extracted = _extract_tarball(tar_path, tmpdir)
        harness = extracted / HARNESS_REL_IN_PKG
        if not harness.exists():
            return {"method": "isolated_dir", "error": "package_missing_harness",
                    "exit_code": None, "stdout": "", "stderr": ""}
        env = {
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(extracted / "python"),
            "PATH": _system_path(),
        }
        proc = runner(
            [python_exe, HARNESS_REL_IN_PKG],
            cwd=str(extracted),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "method": "isolated_dir",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": (proc.stderr or "")[-2000:],
        }
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"method": "isolated_dir",
                "error": f"isolated_dir_exception_{type(exc).__name__}",
                "exit_code": None, "stdout": "", "stderr": ""}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_package_in_fresh_venv(
    tar_path: Path,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = 1200,
) -> dict[str, Any]:
    """Unpack the tarball, build a fresh venv, install pinned deps, run the cmd.

    Strongest isolation short of Docker: a brand-new interpreter site-packages
    built by ``python -m venv``, ``pip install -r requirements.txt`` (the package's
    exact pins), ``pip install --no-deps -e .``, then the reproducer harness. Any
    step failing (e.g. no network to fetch the pinned wheels) returns an
    ``error`` so the caller falls back to ``isolated_dir`` honestly rather than
    fabricating a number. Never raises on a subprocess failure.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_g2_regression_venv_"))
    try:
        extracted = _extract_tarball(tar_path, tmpdir)
        if not (extracted / "run.sh").exists():
            return {"method": "fresh_venv", "error": "package_missing_run_sh",
                    "exit_code": None, "stdout": "", "stderr": ""}
        venv_dir = tmpdir / "venv"
        steps = [
            ([sys.executable, "-m", "venv", str(venv_dir)], None),
        ]
        for cmd, _ in steps:
            proc = runner(cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                return {"method": "fresh_venv",
                        "error": "venv_create_failed",
                        "exit_code": proc.returncode,
                        "stdout": proc.stdout, "stderr": (proc.stderr or "")[-2000:]}
        vpy = venv_dir / "bin" / "python"
        env = {"JAX_PLATFORMS": "cpu", "PATH": _system_path()}
        install_reqs = runner(
            [str(vpy), "-m", "pip", "install", "--quiet", "--no-cache-dir",
             "-r", str(extracted / "requirements.txt")],
            cwd=str(extracted), env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        if install_reqs.returncode != 0:
            return {"method": "fresh_venv", "error": "pip_install_requirements_failed",
                    "exit_code": install_reqs.returncode, "stdout": install_reqs.stdout,
                    "stderr": (install_reqs.stderr or "")[-2000:]}
        install_pkg = runner(
            [str(vpy), "-m", "pip", "install", "--quiet", "--no-cache-dir",
             "--no-deps", "-e", "."],
            cwd=str(extracted), env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        if install_pkg.returncode != 0:
            return {"method": "fresh_venv", "error": "pip_install_package_failed",
                    "exit_code": install_pkg.returncode, "stdout": install_pkg.stdout,
                    "stderr": (install_pkg.stderr or "")[-2000:]}
        proc = runner(
            [str(vpy), HARNESS_REL_IN_PKG],
            cwd=str(extracted), env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        return {"method": "fresh_venv", "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": (proc.stderr or "")[-2000:]}
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"method": "fresh_venv",
                "error": f"fresh_venv_exception_{type(exc).__name__}",
                "exit_code": None, "stdout": "", "stderr": ""}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _system_path() -> str:
    """Return a PATH that includes the current interpreter's bin dir.

    The isolated runs pass a minimal ``env``; without a PATH that contains the
    interpreter's directory, subprocess resolution of ``python``/``pip`` shims
    can fail on some platforms. We keep it explicit + small for reproducibility.
    """
    import os
    bindir = str(Path(sys.executable).parent)
    base = os.environ.get("PATH", "/usr/bin:/bin")
    return f"{bindir}:{base}"


def regression_verify(
    tar_path: Path,
    prefer_fresh_venv: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run the package in an isolated env and classify the reproduced numbers.

    Tries ``fresh_venv`` first when ``prefer_fresh_venv`` (strongest isolation);
    if that hits an infrastructure error (no network for the pinned wheels, venv
    create failure) it falls back to the always-available ``isolated_dir`` run.
    Returns the reproduced numbers + CI classification + which method succeeded.
    """
    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    if prefer_fresh_venv:
        venv_run = run_package_in_fresh_venv(tar_path, runner=runner)
        attempts.append({"method": "fresh_venv", "error": venv_run.get("error"),
                         "exit_code": venv_run.get("exit_code")})
        if not venv_run.get("error"):
            result = venv_run
    if result is None:
        iso_run = run_package_in_isolated_dir(tar_path, runner=runner)
        attempts.append({"method": "isolated_dir", "error": iso_run.get("error"),
                         "exit_code": iso_run.get("exit_code")})
        result = iso_run

    stdout = result.get("stdout", "") or ""
    cond_a, lc = parse_harness_numbers(stdout)
    cond_a_in_ci, lc_in_ci, reproduced = classify_ci(cond_a, lc)
    isolated_checksum = _parse_isolated_checksum(stdout)
    return {
        "clean_env_method": result.get("method") if not result.get("error") else None,
        "exit_code": result.get("exit_code"),
        "error": result.get("error"),
        "condition_a_auroc": cond_a,
        "learning_contribution": lc,
        "condition_a_in_ci": cond_a_in_ci,
        "learning_contribution_in_ci": lc_in_ci,
        "reproduced": reproduced,
        "isolated_checksum": isolated_checksum,
        "attempts": attempts,
        "stderr_tail": (result.get("stderr") or "")[-500:],
    }


def _parse_isolated_checksum(stdout: str) -> str | None:
    """Pull the harness-printed ``reproducibility_checksum:`` from stdout."""
    import re
    m = re.search(r"reproducibility_checksum:\s*(\S+)", stdout)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Lowest-friction external-ask authoring (pure string builders)
# ---------------------------------------------------------------------------


def build_repro_workflow_yaml() -> str:
    """Return a public ``workflow_dispatch`` reproduction workflow.

    A one-click GitHub Actions workflow an external party (or the operator) runs
    from the Actions tab. It does a fresh ``pip install -e .`` on a clean
    ``ubuntu-latest`` runner — a machine that is not the operator's box and has no
    access to the operator's ``results/`` — then runs the reproducer harness,
    which exits non-zero unless both numbers land in their published CIs. A green
    run on GitHub-hosted infra IS the non-operator reproduction G2 needs. This
    file is committed to the working tree only; it is NEVER pushed or triggered by
    autonomous work (Operator-Only External Publication).
    """
    return (
        "# FoVer G2 one-click reproduction (publication gate G2).\n"
        "#\n"
        "# G2 (independent reproduction) is the SOLE remaining blocker to\n"
        "# paper_ready (ops/north-star.md §2). This workflow is the lowest-friction\n"
        "# external ask: a clean ubuntu-latest runner (NOT the operator's box) does\n"
        "# a fresh `pip install -e .` and recomputes the FoVer headline AUROC from\n"
        "# the committed corpus, asserting the numbers land inside their published\n"
        "# confidence intervals. A green run on GitHub-hosted infrastructure is\n"
        "# independent, non-operator evidence that the headline reproduces.\n"
        "#\n"
        "# To close G2: Actions tab -> 'FoVer G2 One-Click Reproduction' ->\n"
        "# 'Run workflow'. CPU-only and cheap (verifier ensemble scoring a labeled\n"
        "# corpus, NOT live LLM generation; no GPU, no 35B model, no HF creds).\n"
        "name: FoVer G2 One-Click Reproduction\n"
        "\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "\n"
        "jobs:\n"
        "  reproduce:\n"
        "    name: Recompute FoVer headline AUROC on a clean runner\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Checkout\n"
        "        uses: actions/checkout@v4\n"
        "\n"
        "      - name: Setup Python\n"
        "        uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: '3.12'\n"
        "\n"
        "      - name: Install Carnot-EBM (fresh, CPU-only)\n"
        "        run: pip install -e .\n"
        "\n"
        "      # Exits non-zero (failing the job) unless condition-A mean AUROC is in\n"
        "      # [0.9027, 0.9235] AND learning_contribution mean in [0.0125, 0.0245].\n"
        "      - name: Reproduce FoVer headline and assert published CIs\n"
        "        env:\n"
        "          JAX_PLATFORMS: cpu\n"
        "        run: python3 scripts/reproduce_fover_headline.py\n"
    )


def build_reproducer_invite(
    package_sha256: str | None,
    package_cid: str | None,
    one_command: str = ONE_COMMAND_REPRO,
) -> str:
    """Return the one-paragraph invite a non-operator reproducer reads.

    Plain-language, no Carnot internals: what they prove, the single command, and
    what to report back. Short on purpose — the lower the friction, the likelier
    the one external run that closes G2 actually happens.
    """
    cid_line = (
        f"- IPFS (content-addressed fetch): `ipfs get {package_cid}`\n"
        if package_cid else ""
    )
    sha_line = f"- Package sha256: `{package_sha256}`\n" if package_sha256 else ""
    return (
        "# Reproduce the FoVer headline in one command (help close gate G2)\n"
        "\n"
        "We are looking for **one person who is not the project operator** to "
        "independently reproduce the headline result of the Carnot-EBM verifier "
        "ensemble. It is **CPU-only**, needs **no GPU, no large model, and no API "
        "keys**, and takes a couple of minutes (mostly `pip install`). You do not "
        "need any prior knowledge of the project.\n"
        "\n"
        "**Option A — one-click (no checkout):** on the project's GitHub, open the "
        "Actions tab, pick **\"FoVer G2 One-Click Reproduction\"**, and press "
        "**Run workflow**. A green run is the reproduction.\n"
        "\n"
        "**Option B — self-contained tarball (one command):** download "
        "`g2-fover-repro.tar.gz` and run:\n"
        "\n"
        "```bash\n"
        f"{one_command}\n"
        "```\n"
        "\n"
        "A zero exit (`echo $?` -> `0`) is the pass: the harness exits non-zero "
        "unless condition-A mean AUROC lands in `[0.9027, 0.9235]` and the FR-11 "
        "learning contribution lands in `[0.0125, 0.0245]`, over n=1,000 and 5 "
        "seeds. **Please report back** the two printed numbers, your platform, and "
        "your Python/library versions — that report is what closes G2.\n"
        "\n"
        "## Integrity\n"
        "\n"
        f"{sha_line}"
        f"{cid_line}"
    )


def build_operator_checklist(
    *,
    package_path: str | None,
    package_sha256: str | None,
    package_sha256_verified: bool,
    package_cid: str | None,
    reproduced_auroc: float | None,
    auroc_within_ci: bool,
    clean_env_method: str | None,
    workflow_path: str,
    invite_path: str,
) -> str:
    """Return the operator checklist whose TERMINAL step is one external action.

    Everything above the terminal step is autonomous-verified state (the
    regression run, the integrity check, the prepared files). The terminal step
    is the single thing only the operator may do per Operator-Only External
    Publication: push the branch and click "Run workflow" (or send the invite).
    The checklist exists so the operator's remaining G2 work is one click, with
    every precondition already green.
    """
    chk = lambda ok: "[x]" if ok else "[ ]"  # noqa: E731 — terse on purpose
    auroc_str = f"{reproduced_auroc:.4f}" if isinstance(reproduced_auroc, float) else "n/a"
    cid_str = package_cid or "(no IPFS node at build time)"
    return (
        "# Operator checklist — close gate G2 (one external action)\n"
        "\n"
        "G2 (independent reproduction) is the SOLE remaining blocker to "
        "`paper_ready` (`ops/north-star.md` §2). Autonomous work has verified "
        "everything that can be verified without an external run; the terminal "
        "step below is reserved for you (Operator-Only External Publication).\n"
        "\n"
        "## Autonomous-verified preconditions (all green before the ask)\n"
        "\n"
        f"- {chk(bool(package_path))} Self-contained package present: "
        f"`{package_path or 'MISSING'}`\n"
        f"- {chk(package_sha256_verified)} Package sha256 re-verified against the "
        f"recorded checksum: `{package_sha256 or 'n/a'}`\n"
        f"- {chk(auroc_within_ci)} Package re-run in an isolated environment "
        f"(`{clean_env_method or 'unavailable'}`) reproduced condition-A AUROC "
        f"`{auroc_str}` within the published CI `[0.9027, 0.9235]`\n"
        f"- {chk(True)} Content-addressed fetch: `{cid_str}`\n"
        f"- {chk(True)} One-click workflow committed to the working tree: "
        f"`{workflow_path}`\n"
        f"- {chk(True)} Reproducer invite drafted: `{invite_path}`\n"
        "\n"
        "## TERMINAL STEP — the single external action (operator-only)\n"
        "\n"
        "1. Review and push the prepared files to the canonical remote "
        "(`github.com/Carnot-EBM/carnot-ebm`).\n"
        "2. **Send the invite** in `docs/g2-reproducer-invite.md` to one "
        "non-operator reproducer, **or** open the Actions tab and press "
        "**Run workflow** on **\"FoVer G2 One-Click Reproduction\"** yourself "
        "from a non-operator account.\n"
        "3. When the external/CI run lands condition-A AUROC in "
        "`[0.9027, 0.9235]`, record it per "
        "`ops/reproduction-runbook-fover-headline.md` and flip G2 to met. Only "
        "this confirmed non-operator run closes G2 — autonomous work never does.\n"
    )


def append_runbook(
    repo_root: Path,
    *,
    reproduced_auroc: float | None,
    auroc_within_ci: bool,
    clean_env_method: str | None,
    package_sha256: str | None,
    package_sha256_verified: bool,
    package_cid: str | None,
    exp_id: str = "exp3488",
    run_date: str = "2026-05-30",
    artifact_name: str = (
        "experiment_3488_fover_g2_clean_room_regression_verify_external_ask_v1"
    ),
) -> bool:
    """Append (never delete) the regression result to the runbook. Returns ok.

    ``exp_id``, ``run_date``, and ``artifact_name`` default to the Exp 3488 /
    .321 values for backward compatibility; callers running a later drift-check
    pass the correct identifiers so the runbook section is self-documenting.
    """
    runbook = repo_root / RUNBOOK_REL
    if not runbook.exists():
        return False
    auroc_str = f"{reproduced_auroc:.4f}" if isinstance(reproduced_auroc, float) else "n/a"
    cid_line = f"- IPFS CID: `{package_cid}`\n" if package_cid else ""
    section = (
        f"\n## Clean-room regression verify + external ask ({exp_id}, {run_date})\n\n"
        "The self-contained package was re-run from an environment isolated from "
        "the working repo to catch any drift since it was built (.320):\n\n"
        f"- Isolation method: `{clean_env_method or 'unavailable'}`\n"
        f"- Reproduced condition-A mean AUROC: `{auroc_str}` "
        f"(within published CI [0.9027, 0.9235]: {auroc_within_ci})\n"
        f"- Package sha256: `{package_sha256 or 'n/a'}` "
        f"(re-verified against recorded checksum: {package_sha256_verified})\n"
        f"{cid_line}"
        "- Lowest-friction external ask prepared (committed to the working tree, "
        "NOT pushed/triggered): `.github/workflows/fover-g2-repro.yml`, "
        "`docs/g2-reproducer-invite.md`, "
        "`ops/g2-external-ask-operator-checklist.md`.\n\n"
        "G2 remains UNMET. Closure requires a confirmed non-operator external/CI "
        "run (Operator-Only External Publication). Artifact: "
        f"`results/{artifact_name}.json`.\n"
    )
    with open(runbook, "a", encoding="utf-8") as f:
        f.write(section)
    return True


# ---------------------------------------------------------------------------
# Verdict / status mapping + artifact (pure)
# ---------------------------------------------------------------------------


def determine_verdict(
    *,
    package_available: bool,
    auroc_within_ci: bool,
    sha256_verified: bool,
    external_ask_ready: bool,
) -> str:
    """Map the regression + external-ask outcome to a terminal ``complete:`` verdict.

    All branches are terminal — the experiment reached a scientific conclusion in
    each. ``regression_clean`` requires BOTH the AUROC in CI AND the sha256
    verified AND the external ask authored; a drifted AUROC or sha is honestly
    reported as drift so the operator rebuilds rather than ships a rotten package.
    """
    if not package_available:
        return "complete: blocked_g2_package_unavailable"
    if auroc_within_ci and sha256_verified and external_ask_ready:
        return (
            "complete: "
            "fover_g2_package_regression_clean_external_ask_ready_g2_operator_gated"
        )
    return "complete: fover_g2_package_regression_drift_detected_needs_rebuild"


def build_artifact(
    *,
    start_time: float,
    preconditions: dict[str, Any],
    regression: dict[str, Any],
    sha_check: dict[str, Any],
    ipfs_result: dict[str, Any],
    workflow_path: str | None,
    invite_path: str | None,
    checklist_path: str | None,
    runbook_appended: bool,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Assemble the Exp 3488 artifact with principle annotations on every field."""
    duration_s = clock() - start_time
    package_available = bool(preconditions.get("ok"))
    auroc = regression.get("condition_a_auroc")
    auroc_within_ci = bool(regression.get("condition_a_in_ci"))
    sha256_verified = bool(sha_check.get("verified"))
    external_ask_ready = bool(workflow_path and checklist_path and invite_path)
    verdict = determine_verdict(
        package_available=package_available,
        auroc_within_ci=auroc_within_ci,
        sha256_verified=sha256_verified,
        external_ask_ready=external_ask_ready,
    )
    package_cid = ipfs_result.get("package_cid")
    checksum = (
        regression.get("isolated_checksum")
        or sha_check.get("computed")
        or "preconditions_blocked"
    )
    return {
        "artifact": (
            "experiment_3488_fover_g2_clean_room_regression_verify_external_ask_v1"
        ),
        "schema": "carnot.fover_g2_clean_room_regression_verify_external_ask_v1",
        "experiment": 3488,
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "package_path": preconditions.get("tarball") and TARBALL_REL,
        "package_reproduced_auroc": auroc,
        "package_auroc_within_ci": auroc_within_ci,
        "package_learning_contribution": regression.get("learning_contribution"),
        "package_sha256": sha_check.get("computed"),
        "package_sha256_verified": sha256_verified,
        "package_sha256_recorded": sha_check.get("recorded"),
        "package_cid": package_cid,
        "ipfs_available": ipfs_result.get("ipfs_available", False),
        "clean_env_method": regression.get("clean_env_method"),
        "regression_attempts": regression.get("attempts"),
        "one_command_repro": ONE_COMMAND_REPRO,
        "external_ask_workflow_path": workflow_path,
        "reproducer_invite_path": invite_path,
        "operator_checklist_path": checklist_path,
        "runbook_appended": runbook_appended,
        "g2_met": False,
        "g2_independent_reproducer": False,
        "external_run_pending": True,
        "operator_action_required": (
            "Closing G2 requires a person who is NOT the operator to run the "
            "one-click workflow or the self-contained package and report "
            "condition-A AUROC in [0.9027, 0.9235]. Per Operator-Only External "
            "Publication, autonomous work may regression-verify the package and "
            "prepare the ask but may never flip g2_met."
        ),
        "reproducibility_checksum": checksum,
        "random_seed": RANDOM_SEEDS,
        "duration_s": duration_s,
        "n_examples": N_EXAMPLES,
        "live_model_invoked": False,
        "preconditions_checked": preconditions,
        "condition_a_ci": [CONDITION_A_CI_LOW, CONDITION_A_CI_HIGH],
        "learning_contribution_ci": [LEARNING_CONTRIB_CI_LOW, LEARNING_CONTRIB_CI_HIGH],
        "field_principles": {
            "honest_verdict": (
                "complete:/success:/passed:/shipped_ prefix required by CLAUDE.md "
                "Verdict Terminal-Prefix Discipline."
            ),
            "inference_substrate": (
                "verifier_ensemble_against_cached_candidates: the package scores "
                "the verifier ensemble against the labeled FoVer corpus; no live "
                "LLM is loaded, so adversarial_verify.py applies the 1s floor."
            ),
            "package_reproduced_auroc": (
                "The AUROC the self-contained package produced in the fresh "
                "environment — proves it still works."
            ),
            "package_auroc_within_ci": (
                "Boolean: reproduced AUROC within the stated 0.9131 CI — the "
                "regression gate."
            ),
            "package_sha256_verified": (
                "Boolean: tarball SHA256 re-verified against Exp 3476's recorded "
                "checksum — content integrity for an external party."
            ),
            "package_cid": (
                "IPFS CID of the package (decentralization rule 3) — "
                "content-addressed external fetch."
            ),
            "external_ask_workflow_path": (
                "The public workflow_dispatch file an external party one-clicks "
                "(committed locally, not pushed/triggered)."
            ),
            "operator_checklist_path": (
                "The operator checklist whose terminal step is the single "
                "external-ask action."
            ),
            "g2_met": (
                "MUST be false — G2 flips only on a confirmed non-operator "
                "external run (Operator-Only External Publication)."
            ),
            "external_run_pending": (
                "Boolean true — the honest remaining state."
            ),
            "random_seed": "Determinism — the published seeds [42,137,271,314,1729].",
            "reproducibility_checksum": (
                "Content hash anchoring the run (isolated harness checksum when "
                "available, else the verified tarball sha256)."
            ),
            "duration_s": "Fresh-env install + cached scoring wall time; 1s floor.",
        },
    }
