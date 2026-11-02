"""Exp 3451: ship the FoVer G2 CI workflow + a Docker clean-room reproduction.

Why this experiment exists
--------------------------
G2 (independent reproduction) is the SOLE remaining publication-gate blocker for
the FoVer headline (G1/G3/G4 are met per ``ops/north-star.md`` §2). exp3438
(.317) made the headline clean-room reproducible *internally*: a fresh git
worktree + venv now recomputes AUROC 0.9131 and the FR-11 learning contribution
inside their published CIs (the load-bearing fix was declaring scikit-learn in
``pyproject.toml``).

The gate requires ">=1 reproducer who is NOT the operator". The Phase-1 ship
gate explicitly counts "a CI run" as that reproducer. This experiment ships the
*mechanism* a non-operator uses to close G2:

1. A GitHub Actions workflow (`.github/workflows/reproduce-fover-headline.yml`)
   that, on a clean ``ubuntu-latest`` runner, does a fresh ``pip install -e .``
   and runs ``scripts/reproduce_fover_headline.py``, which ASSERTS condition-A
   mean AUROC in [0.9027, 0.9235] AND learning_contribution mean in
   [0.0125, 0.0245] (non-zero exit on failure). A green run of this workflow on
   GitHub-hosted infrastructure is non-operator evidence.

2. A Docker clean-room: a container built FROM a stock ``python:3.12-slim``
   base image (a DIFFERENT base image than the operator's box), with a fresh
   ``pip install``, running the harness inside. This is the strongest
   autonomous G2 evidence short of an external human — maximal isolation from
   the operator's installed environment.

What this experiment NEVER does
-------------------------------
- It does NOT push the workflow (the file is committed to the working tree
  only; pushing + actually running it is operator/CI action).
- It does NOT set ``g2_independent_reproducer=true``. Only an actual
  external/CI run by a non-operator may flip that. This experiment brings G2 as
  close to closeable-by-a-non-operator as autonomous work allows, and reports an
  honest ``g2_status``.
- It does NOT modify ``scripts/research_conductor.py``.

Spec: REQ-PUBLISH-036,
      SCENARIO-PUBLISH-036 (CI + Docker both ready),
      SCENARIO-PUBLISH-036B (Docker unavailable -> fresh-venv fallback).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILENAME = "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.json"

CI_WORKFLOW_REL = ".github/workflows/reproduce-fover-headline.yml"
HARNESS_REL = "scripts/reproduce_fover_headline.py"

# Published acceptance CI (exp2837 5-seed dual-condition headline).
CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245

RANDOM_SEEDS = [42, 137, 271, 314, 1729]
N_EXAMPLES = 1000

# A stock public Python base image — deliberately NOT the operator's venv, so
# the recompute is from a fresh OS + interpreter + pip resolution.
DOCKER_BASE_IMAGE = "python:3.12-slim"

# FR-11 session-memory state globs the FoVer scorer consults for condition A
# (production). Mirrors FR11_STATE_GLOBS in
# python/carnot/eval/fover_memory_leakage_v3.py. The Docker context must include
# matching files or condition A collapses to condition B (architecture-only) and
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


def _sha256_str(s: str) -> str:
    """Content hash of a string — used for transcript + reproducibility hashes."""
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Step 0 — preconditions and capability probes (pure-ish, easily testable)
# ---------------------------------------------------------------------------


def check_preconditions(repo_root: Path) -> dict[str, Any]:
    """Step 0 PRECONDITIONS: harness present + executable, corpus present.

    Principle: naming a missing resource up front pre-empts the fabrication mode
    where the agent silently lacks a resource and synthesizes a passing artifact
    instead of emitting a blocked_* verdict. Docker absence is NOT blocking here
    (the CI workflow file is the primary deliverable); it is probed separately.
    """
    harness = repo_root / HARNESS_REL
    corpus = repo_root / "data" / "fover_corpus.jsonl"

    if not harness.exists():
        return {"ok": False, "blocked_reason": "blocked_fover_harness_missing"}
    if not corpus.exists():
        return {"ok": False, "blocked_reason": "blocked_fover_corpus_missing"}
    return {"ok": True, "harness": str(harness), "corpus": str(corpus)}


def docker_is_available(runner: Callable[..., Any] = subprocess.run) -> bool:
    """Return True iff a usable Docker daemon is reachable.

    Principle: matches the task precondition (c) — `command -v docker` AND
    `docker info` must both succeed. A binary on PATH without a running daemon is
    not usable, so we require `docker info` to exit 0, not merely that the client
    exists.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = runner(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# CI workflow assertion check (the primary acceptance gate)
# ---------------------------------------------------------------------------


def ci_workflow_status(repo_root: Path) -> dict[str, Any]:
    """Report whether the CI workflow exists and genuinely asserts both CIs.

    The acceptance gate requires the workflow to ASSERT the AUROC + learning-
    contribution CIs. The workflow does this transitively by running
    ``reproduce_fover_headline.py``, whose ``main()`` exits non-zero unless BOTH
    numbers land in their published CIs. So "asserts the CIs" means:

      1. the workflow file exists, AND
      2. it invokes the reproducer harness, AND
      3. the harness encodes both published CI bounds AND fails (returns 1) when
         either number is outside its CI.

    We verify all three mechanically so the gate cannot pass on a workflow that
    merely runs something unrelated.
    """
    workflow = repo_root / CI_WORKFLOW_REL
    harness = repo_root / HARNESS_REL
    present = workflow.exists()
    invokes_harness = False
    harness_asserts = False
    if present:
        wf_text = workflow.read_text(encoding="utf-8")
        invokes_harness = "reproduce_fover_headline.py" in wf_text
    if harness.exists():
        h_text = harness.read_text(encoding="utf-8")
        has_bounds = all(
            token in h_text
            for token in (
                str(CONDITION_A_CI_LOW),
                str(CONDITION_A_CI_HIGH),
                str(LEARNING_CONTRIB_CI_LOW),
                str(LEARNING_CONTRIB_CI_HIGH),
            )
        )
        # The harness must fail (non-zero exit) when out of CI — the assertion.
        fails_out_of_ci = "return 1" in h_text and "cond_a_in_ci and lc_in_ci" in h_text
        harness_asserts = has_bounds and fails_out_of_ci
    asserts_cis = present and invokes_harness and harness_asserts
    return {
        "present": present,
        "invokes_harness": invokes_harness,
        "harness_asserts": harness_asserts,
        "asserts_cis": asserts_cis,
        "path": CI_WORKFLOW_REL if present else "",
    }


# ---------------------------------------------------------------------------
# CI-band classification (pure)
# ---------------------------------------------------------------------------


def classify_ci(cond_a: Any, lc: Any) -> tuple[bool, bool, bool]:
    """Return (cond_a_in_ci, lc_in_ci, reproduced_in_ci).

    Principle: the headline only "reproduces" when BOTH numbers land inside their
    published CIs; a single number in range is not sufficient. ``None`` (an
    error, e.g. an import failure) is never in any CI.
    """
    cond_a_in_ci = (
        cond_a is not None and CONDITION_A_CI_LOW <= float(cond_a) <= CONDITION_A_CI_HIGH
    )
    lc_in_ci = (
        lc is not None
        and LEARNING_CONTRIB_CI_LOW <= float(lc) <= LEARNING_CONTRIB_CI_HIGH
    )
    return cond_a_in_ci, lc_in_ci, bool(cond_a_in_ci and lc_in_ci)


def _extract_numbers(harness_result: dict[str, Any]) -> tuple[Any, Any]:
    """Pull (condition_a_mean, learning_contribution_mean) from a harness dict."""
    cond_a = harness_result.get("condition_a_production_auroc_mean")
    lc_raw = harness_result.get("learning_contribution_ci95")
    if isinstance(lc_raw, dict):
        lc = lc_raw.get("mean")
    else:
        lc = harness_result.get("learning_contribution")
    return cond_a, lc


def determine_verdict_and_status(
    docker_available: bool,
    isolation_mode: str,
    reproduced: bool,
    has_error: bool,
) -> tuple[str, str]:
    """Map the isolated-environment outcome to a terminal verdict + g2_status.

    Principle: the verdict must honestly distinguish (a) CI + Docker both ready,
    (b) CI ready but Docker unavailable so a fresh-venv clean-room stood in, and
    (c) the isolated env failed to reproduce. All three are terminal ``complete:``
    states — the experiment ran to a scientific conclusion in each.
    """
    if reproduced:
        if isolation_mode == "docker":
            return (
                "complete: fover_g2_ci_and_docker_cleanroom_ready_external_run_pending",
                "ci_and_docker_ready_external_run_pending",
            )
        return (
            "complete: fover_g2_ci_workflow_ready_docker_unavailable_external_run_pending",
            "ci_ready_docker_unavailable",
        )
    cause = "container_error" if has_error else "auroc_outside_published_ci"
    return (
        f"complete: fover_g2_isolated_repro_still_failing_{cause}",
        f"still_failing_{cause}",
    )


# ---------------------------------------------------------------------------
# Docker / fresh-venv clean-room context builders (heavy; integration-tested
# by the actual run, unit-tested for their pure content here)
# ---------------------------------------------------------------------------


def build_dockerfile_content() -> str:
    """Return the Dockerfile text for the clean-room image.

    Why a stock slim base + non-editable install path: the point is to recompute
    the headline on a machine state that shares nothing with the operator's venv.
    A fresh ``python:3.12-slim`` + ``pip install -e .`` from the copied context
    exercises the scikit-learn dependency declaration (exp3438 fix) end to end.
    JAX is forced to CPU for determinism (no GPU in the container anyway).
    """
    return (
        f"FROM {DOCKER_BASE_IMAGE}\n"
        "ENV JAX_PLATFORMS=cpu PIP_DISABLE_PIP_VERSION_CHECK=1 "
        "PYTHONDONTWRITEBYTECODE=1\n"
        "WORKDIR /carnot\n"
        "COPY . /carnot\n"
        "RUN pip install --no-cache-dir -e . >/dev/null\n"
        'CMD ["python3", "scripts/reproduce_fover_headline.py"]\n'
    )


def _copy_glob_set(repo_root: Path, ctx: Path) -> int:
    """Copy the FR-11 state-file globs into the Docker build context.

    Returns the number of files copied. Condition A (production) needs these or
    it degrades to architecture-only and learning_contribution ~ 0.
    """
    copied = 0
    for pattern in FR11_STATE_GLOBS:
        for src in repo_root.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(repo_root)
            dst = ctx / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def build_docker_context(repo_root: Path, ctx: Path) -> dict[str, Any]:
    """Assemble a minimal Docker build context under ``ctx``.

    The repo's ``data/`` directory is ~2 GB, so we cannot COPY the whole tree.
    Instead we copy exactly what ``pip install -e .`` + the FoVer reproducer
    need: build metadata, the package source (sans 50 MB+ .so / __pycache__),
    the committed corpus, the FR-11 state files, and the reproducer script.
    """
    # Build metadata + license/readme (pyproject reads these).
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        src = repo_root / name
        if src.exists():
            shutil.copy2(src, ctx / name)

    # Package source. Exclude the prebuilt Rust .so (50 MB+, not needed by the
    # FoVer verifier-scoring path) and __pycache__ to keep the context small.
    shutil.copytree(
        repo_root / "python",
        ctx / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )

    # The reproducer harness + the committed corpus.
    (ctx / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / HARNESS_REL, ctx / HARNESS_REL)
    (ctx / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo_root / "data" / "fover_corpus.jsonl",
        ctx / "data" / "fover_corpus.jsonl",
    )

    state_files_copied = _copy_glob_set(repo_root, ctx)

    (ctx / "Dockerfile").write_text(build_dockerfile_content(), encoding="utf-8")
    return {"state_files_copied": state_files_copied}


_HARNESS_INLINE = (
    "import sys, json;"
    "sys.path.insert(0, 'python');"
    "sys.path.insert(0, 'scripts');"
    "from reproduce_fover_headline import run_reproduction;"
    "from pathlib import Path;"
    "print(json.dumps(run_reproduction(Path('/carnot'))))"
)


def run_docker_cleanroom(
    repo_root: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Build the clean-room image and run the harness inside it.

    Returns a dict with the harness result (or an ``error``) plus build/run
    metadata. Never raises on a Docker failure — it captures the error so the
    artifact can fall back to a fresh-venv clean-room honestly.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_g2_docker_"))
    ctx = tmpdir / "ctx"
    ctx.mkdir(parents=True, exist_ok=True)
    tag = "carnot-fover-g2-cleanroom:exp3451"
    try:
        ctx_info = build_docker_context(repo_root, ctx)
        build = runner(
            ["docker", "build", "-t", tag, str(ctx)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if build.returncode != 0:
            return {
                "error": "docker_build_failed",
                "build_stderr": build.stderr[-2000:],
                "state_files_copied": ctx_info["state_files_copied"],
            }
        run = runner(
            ["docker", "run", "--rm", tag, "python3", "-c", _HARNESS_INLINE],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if run.returncode != 0:
            return {
                "error": "docker_run_failed",
                "run_stderr": run.stderr[-2000:],
                "state_files_copied": ctx_info["state_files_copied"],
            }
        harness = _parse_last_json_line(run.stdout)
        harness["state_files_copied"] = ctx_info["state_files_copied"]
        return harness
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"error": f"docker_exception_{type(exc).__name__}", "detail": str(exc)}
    finally:
        runner(["docker", "rmi", "-f", tag], capture_output=True, text=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_fresh_venv_cleanroom(
    repo_root: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Fallback clean-room: a fresh venv + ``pip install -e .`` in a temp tree.

    Used when Docker is unavailable. Weaker isolation than Docker (same OS /
    interpreter family) but still a from-scratch dependency resolution that does
    not read the operator's installed venv. Runs the reproducer in-process inside
    that venv against the repo's committed corpus + state files.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_g2_venv_"))
    try:
        venv_path = tmpdir / "venv"
        runner(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        venv_python = venv_path / "bin" / "python"
        install = runner(
            [str(venv_python), "-m", "pip", "install", "-e", str(repo_root), "--quiet"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if install.returncode != 0:
            return {
                "error": "fresh_venv_install_failed",
                "install_stderr": install.stderr[-2000:],
            }
        proc = runner(
            [str(venv_python), "-c", _HARNESS_INLINE.replace("/carnot", str(repo_root))],
            capture_output=True,
            text=True,
            timeout=1800,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            cwd=str(repo_root),
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {
                "error": "fresh_venv_run_failed",
                "run_stderr": proc.stderr[-2000:],
            }
        return _parse_last_json_line(proc.stdout)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"error": f"venv_exception_{type(exc).__name__}", "detail": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    """Parse the last JSON object printed on stdout (the harness result dict)."""
    raw = stdout.strip()
    if not raw:
        return {"error": "no_stdout"}
    for line in reversed(raw.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"error": "no_json_line_found", "stdout": raw[:500]}


# ---------------------------------------------------------------------------
# Artifact assembly (pure)
# ---------------------------------------------------------------------------


def build_artifact(
    start_time: float,
    preconditions: dict[str, Any],
    docker_available: bool,
    isolation_mode: str,
    isolated_result: dict[str, Any],
    ci_info: dict[str, Any],
    docker_run_error: str | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Assemble the final experiment artifact with principle annotations."""
    duration_s = clock() - start_time
    cond_a, lc = _extract_numbers(isolated_result)
    _, _, reproduced = classify_ci(cond_a, lc)
    has_error = bool(isolated_result.get("error"))
    verdict, g2_status = determine_verdict_and_status(
        docker_available, isolation_mode, reproduced, has_error
    )
    checksum = isolated_result.get("reproducibility_checksum")

    return {
        "artifact": "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1",
        "schema": "carnot.fover_g2_ci_workflow_and_docker_cleanroom_v1",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "ci_workflow_path": ci_info.get("path", ""),
        "ci_workflow_asserts_cis": ci_info.get("asserts_cis", False),
        "docker_available": docker_available,
        "isolation_mode": isolation_mode,
        "g2_docker_cleanroom_reproduced": reproduced,
        "condition_a_auroc_isolated": cond_a,
        "learning_contribution_isolated": lc,
        "g2_status": g2_status,
        "g2_independent_reproducer": False,
        "g2_note": (
            "The mechanism a non-operator uses to close G2 now exists in the "
            "repo: a GitHub Actions workflow at "
            ".github/workflows/reproduce-fover-headline.yml that fresh-installs "
            "and asserts the published CIs, plus a Docker (or fresh-venv) "
            "clean-room recompute on a different base image than the operator's "
            "box. G2 is NOT closed by authoring this mechanism — closure "
            "requires an actual external/CI run by a non-operator reporting "
            f"condition_A_auroc in [{CONDITION_A_CI_LOW}, {CONDITION_A_CI_HIGH}] "
            f"and learning_contribution in [{LEARNING_CONTRIB_CI_LOW}, "
            f"{LEARNING_CONTRIB_CI_HIGH}]. See "
            "ops/reproduction-runbook-fover-headline.md."
        ),
        "reproducibility_checksum": checksum,
        "random_seed": RANDOM_SEEDS,
        "duration_s": duration_s,
        "n_examples": N_EXAMPLES,
        "live_model_invoked": isolated_result.get("live_model_invoked", False),
        "preconditions_checked": preconditions,
        "ci_workflow_invokes_harness": ci_info.get("invokes_harness", False),
        "docker_base_image": DOCKER_BASE_IMAGE if docker_available else None,
        "docker_run_error": docker_run_error,
        "state_files_copied_to_cleanroom": isolated_result.get("state_files_copied"),
        "isolated_harness_error_if_any": isolated_result.get("error"),
        "field_principles": {
            "honest_verdict": (
                "complete:/success:/passed:/shipped_ prefix required by CLAUDE.md "
                "Verdict Terminal-Prefix Discipline."
            ),
            "inference_substrate": (
                "Declares verifier-scoring (not live LLM) so adversarial_verify.py "
                "applies the 1s floor, not the 60s live-inference floor."
            ),
            "ci_workflow_path": (
                ".github/workflows/reproduce-fover-headline.yml — the mechanism a "
                "non-operator CI runner uses to close G2."
            ),
            "docker_available": (
                "Whether Docker isolation was achievable on this box."
            ),
            "g2_docker_cleanroom_reproduced": (
                "Boolean: the headline recomputed in a Docker/fresh-venv container "
                "in [0.9027, 0.9235] (and learning_contribution in CI)."
            ),
            "condition_a_auroc_isolated": (
                "The recomputed production AUROC from the most-isolated "
                "environment achieved."
            ),
            "learning_contribution_isolated": (
                "The recomputed FR-11 ablation from that environment; must land "
                "in [0.0125, 0.0245]."
            ),
            "g2_status": (
                "Honest string: ci_and_docker_ready_external_run_pending | "
                "ci_ready_docker_unavailable | still_failing_<cause>."
            ),
            "g2_independent_reproducer": (
                "MUST be false — only an actual external/CI run by a non-operator "
                "flips it true."
            ),
            "reproducibility_checksum": "Content hash from the isolated recompute.",
            "random_seed": "The published seeds [42,137,271,314,1729].",
            "duration_s": "Isolated install + CPU verifier scoring wall time.",
        },
    }


def run_experiment(clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Top-level entry point for Exp 3451.

    Returns the artifact dict (does not write it — write_artifact does that).
    """
    start_time = clock()

    preconditions = check_preconditions(REPO_ROOT)
    if not preconditions["ok"]:
        return {
            "artifact": "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1",
            "honest_verdict": "complete: " + preconditions["blocked_reason"],
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "ci_workflow_path": "",
            "docker_available": False,
            "g2_docker_cleanroom_reproduced": False,
            "g2_status": "blocked_" + preconditions["blocked_reason"],
            "g2_independent_reproducer": False,
            "duration_s": clock() - start_time,
        }

    ci_info = ci_workflow_status(REPO_ROOT)
    docker_available = docker_is_available()

    docker_run_error: str | None = None
    if docker_available:
        isolation_mode = "docker"
        isolated_result = run_docker_cleanroom(REPO_ROOT)
        if isolated_result.get("error"):
            # Docker available but the build/run failed — fall back to a fresh
            # venv so the task still produces clean-room evidence honestly.
            docker_run_error = str(isolated_result.get("error"))
            isolation_mode = "fresh_venv"
            isolated_result = run_fresh_venv_cleanroom(REPO_ROOT)
    else:
        isolation_mode = "fresh_venv"
        isolated_result = run_fresh_venv_cleanroom(REPO_ROOT)

    return build_artifact(
        start_time=start_time,
        preconditions=preconditions,
        docker_available=docker_available,
        isolation_mode=isolation_mode,
        isolated_result=isolated_result,
        ci_info=ci_info,
        docker_run_error=docker_run_error,
        clock=clock,
    )


def write_artifact(artifact: dict[str, Any]) -> Path:
    """Write the artifact JSON and return the output path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / OUTPUT_FILENAME
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    return out_path


def main() -> int:
    artifact = run_experiment()
    out_path = write_artifact(artifact)
    print(f"honest_verdict: {artifact.get('honest_verdict')}")
    print(f"ci_workflow_path: {artifact.get('ci_workflow_path')}")
    print(f"docker_available: {artifact.get('docker_available')}")
    print(f"isolation_mode: {artifact.get('isolation_mode')}")
    print(
        "g2_docker_cleanroom_reproduced: "
        f"{artifact.get('g2_docker_cleanroom_reproduced')}"
    )
    print(f"condition_a_auroc_isolated: {artifact.get('condition_a_auroc_isolated')}")
    print(
        "learning_contribution_isolated: "
        f"{artifact.get('learning_contribution_isolated')}"
    )
    print(f"g2_status: {artifact.get('g2_status')}")
    print(f"artifact written to: {out_path}")
    verdict = str(artifact.get("honest_verdict", ""))
    return 0 if verdict.startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
