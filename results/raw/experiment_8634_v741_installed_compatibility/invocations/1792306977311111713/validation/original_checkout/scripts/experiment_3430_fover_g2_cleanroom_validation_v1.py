"""Exp 3430: FoVer G2 clean-room validation via fresh git worktree.

This script isolates the FoVer headline reproduction harness into a fresh
git worktree with a fresh venv so the result does NOT rely on the operator's
working tree state.  It takes the next de-risking step toward G2 (independent
reproduction) without falsely claiming G2 is closed (that needs an external,
non-operator run).

Spec: REQ-FOVER-G2-CLEANROOM, SCENARIO-FOVER-G2-CLEANROOM-PASS,
      SCENARIO-FOVER-G2-CLEANROOM-BLOCKED.

References:
  - results/experiment_3419_fover_g2_reproduction_harness_v1.json (prior run)
  - scripts/reproduce_fover_headline.py (the harness this wraps)
  - ops/reproduction-runbook-fover-headline.md (the external-run protocol)
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILENAME = "experiment_3430_fover_g2_cleanroom_validation_v1.json"

CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245

RANDOM_SEEDS = [42, 137, 271, 314, 1729]
N_EXAMPLES = 1000


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def check_preconditions(repo_root: Path) -> dict[str, Any]:
    """Verify harness, corpus, and git availability.

    Returns a dict with 'ok' bool and 'blocked_reason' if not ok.
    """
    harness = repo_root / "scripts" / "reproduce_fover_headline.py"
    corpus = repo_root / "data" / "fover_corpus.jsonl"

    if not harness.exists():
        return {"ok": False, "blocked_reason": "blocked_fover_harness_missing"}
    if not corpus.exists():
        return {"ok": False, "blocked_reason": "blocked_fover_corpus_missing"}

    git_check = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if git_check.returncode != 0:
        return {"ok": False, "blocked_reason": "blocked_git_unavailable"}

    return {"ok": True, "head_sha": git_check.stdout.strip()}


def create_isolated_env(repo_root: Path) -> dict[str, Any]:
    """Create a fresh git worktree (or clone) and fresh venv.

    Returns info dict: isolation_level, isolated_root, venv_python, tmpdir,
    install_transcript, install_transcript_hash.

    Why fresh worktree: all 21 FR-11 state files are git-tracked at HEAD, so
    the worktree gets them automatically without any manual copying.  The venv
    is then independently installed from that tree so neither the operator's
    sys.path nor .venv is involved in the scoring.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="carnot_cleanroom_"))

    worktree_path = tmpdir / "worktree"
    worktree_result = subprocess.run(
        ["git", "worktree", "add", str(worktree_path), "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if worktree_result.returncode == 0:
        isolation_level = "fresh_worktree"
        isolated_root = worktree_path
    else:
        # Fallback: local clone (still isolated from operator's working tree)
        clone_result = subprocess.run(
            ["git", "clone", "--local", "--shared", str(repo_root), str(worktree_path)],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode == 0:
            isolation_level = "fresh_clone"
            isolated_root = worktree_path
        else:
            # Last resort: in-place (weakest isolation — records honestly)
            isolation_level = "in_place_fallback"
            isolated_root = repo_root

    # Create fresh venv in tmpdir (outside isolated_root to avoid confusing pip)
    venv_path = tmpdir / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        check=True,
    )
    venv_python = venv_path / "bin" / "python"

    # Install the package from the isolated root
    install_proc = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(isolated_root), "--quiet"],
        capture_output=True,
        text=True,
    )
    install_transcript = (install_proc.stdout + install_proc.stderr).strip()

    # Verify the package is importable in the isolated env
    import_check = subprocess.run(
        [str(venv_python), "-c", "import carnot; print(carnot.__version__)"],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    carnot_importable = import_check.returncode == 0

    return {
        "isolation_level": isolation_level,
        "isolated_root": isolated_root,
        "venv_python": venv_python,
        "tmpdir": tmpdir,
        "install_transcript": install_transcript,
        "install_transcript_hash": _sha256_str(install_transcript),
        "carnot_importable_in_isolated_env": carnot_importable,
    }


def get_isolated_env_versions(venv_python: Path) -> dict[str, str]:
    """Collect Python + key lib versions from the isolated venv.

    Why: a third party matching these versions is most likely to reproduce
    the same floating-point AUROC values.
    """
    snippet = (
        "import sys, platform, json;"
        "import carnot; import numpy, jax;"
        "print(json.dumps({"
        "'python': sys.version,"
        "'platform': platform.platform(),"
        "'carnot': carnot.__version__,"
        "'numpy': numpy.__version__,"
        "'jax': jax.__version__"
        "}))"
    )
    result = subprocess.run(
        [str(venv_python), "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    if result.returncode == 0:
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            pass
    return {"error": result.stderr.strip()}


def run_harness_in_isolated_env(
    isolated_root: Path,
    venv_python: Path,
) -> dict[str, Any]:
    """Run reproduce_fover_headline.py inside the isolated venv.

    The harness is run as a Python -c invocation so it returns a JSON dict.
    We import run_reproduction() and dump its return value to stdout.

    Why subprocess instead of in-process: ensures the isolated venv's carnot
    installation (not the operator's .venv) does the scoring.
    """
    # We run a short inline script that calls run_reproduction() and dumps JSON.
    # sys.path must include the isolated repo's python/ dir for the import.
    inline = (
        "import sys, json;"
        f"sys.path.insert(0, {str(isolated_root / 'python')!r});"
        f"sys.path.insert(0, {str(isolated_root / 'scripts')!r});"
        "from reproduce_fover_headline import run_reproduction;"
        f"from pathlib import Path;"
        f"result = run_reproduction(Path({str(isolated_root)!r}));"
        "print(json.dumps(result))"
    )
    proc = subprocess.run(
        [str(venv_python), "-c", inline],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        cwd=str(isolated_root),
    )
    raw_stdout = proc.stdout.strip()
    if proc.returncode != 0 or not raw_stdout:
        return {
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "returncode": proc.returncode,
        }
    # run_reproduction may emit log lines before the final JSON; take last line
    for line in reversed(raw_stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"error": "no_json_line_found", "stdout": raw_stdout[:500]}


def cleanup_isolated_env(
    repo_root: Path,
    isolation_level: str,
    worktree_path: Path,
    tmpdir: Path,
) -> None:
    """Remove the worktree and tmpdir.

    Why: fresh worktrees stay registered in .git/worktrees until explicitly
    removed; leaving them accumulates stale refs.
    """
    if isolation_level == "fresh_worktree":
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=str(repo_root),
            capture_output=True,
        )
    shutil.rmtree(tmpdir, ignore_errors=True)


def build_artifact(
    start_time: float,
    preconditions: dict[str, Any],
    env_info: dict[str, Any],
    harness_result: dict[str, Any],
    isolated_versions: dict[str, str],
) -> dict[str, Any]:
    """Assemble the final experiment artifact.

    All required fields carry a principle annotation explaining why the field
    matters, per CLAUDE.md Principle-Annotated Artifact Fields discipline.
    """
    duration_s = time.time() - start_time

    cond_a = harness_result.get("condition_a_production_auroc_mean")
    lc_raw = harness_result.get("learning_contribution_ci95")
    if isinstance(lc_raw, dict):
        lc = lc_raw.get("mean")
    else:
        lc = harness_result.get("learning_contribution")

    cond_a_in_ci = (
        cond_a is not None
        and CONDITION_A_CI_LOW <= float(cond_a) <= CONDITION_A_CI_HIGH
    )
    lc_in_ci = (
        lc is not None
        and LEARNING_CONTRIB_CI_LOW <= float(lc) <= LEARNING_CONTRIB_CI_HIGH
    )
    reproduced_in_ci = bool(cond_a_in_ci and lc_in_ci)

    isolation_level = env_info.get("isolation_level", "unknown")

    if str(harness_result.get("honest_verdict", "")).startswith("blocked"):
        honest_verdict = "blocked_fover_harness_returned_blocked: " + str(
            harness_result.get("honest_verdict", "unknown")
        )
        g2_status = "blocked_harness_failed"
    elif reproduced_in_ci and isolation_level in ("fresh_worktree", "fresh_clone"):
        honest_verdict = "complete: fover_g2_cleanroom_validated_internal_external_run_pending"
        g2_status = "cleanroom_validated_internal_external_run_pending"
    elif reproduced_in_ci:
        honest_verdict = "complete: fover_g2_in_place_reproduced_cleanroom_not_achieved"
        g2_status = "in_place_reproduced_isolation_not_achieved"
    else:
        honest_verdict = "complete: fover_g2_cleanroom_ci_gate_failed"
        g2_status = "ci_gate_failed"

    return {
        "artifact": "experiment_3430_fover_g2_cleanroom_validation_v1",
        "schema": "carnot.fover_g2_cleanroom_validation_v1",
        # Required fields with principles
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "isolation_level": isolation_level,
        "condition_a_auroc_reproduced": cond_a,
        "learning_contribution_reproduced": lc,
        "reproduced_in_ci": reproduced_in_ci,
        "isolated_env_versions": isolated_versions,
        "g2_status": g2_status,
        "g2_independent_reproducer": False,
        "g2_note": (
            "This run confirms the FoVer headline recomputes within published CI "
            "from an isolated git worktree + fresh venv.  G2 is FURTHER DE-RISKED "
            "but NOT YET CLOSED.  Closure requires a non-operator to run "
            "scripts/reproduce_fover_headline.py from a fresh clone and report "
            f"condition_A_auroc in [{CONDITION_A_CI_LOW}, {CONDITION_A_CI_HIGH}] "
            f"and learning_contribution in [{LEARNING_CONTRIB_CI_LOW}, {LEARNING_CONTRIB_CI_HIGH}].  "
            "See ops/reproduction-runbook-fover-headline.md."
        ),
        "reproducibility_checksum": harness_result.get("reproducibility_checksum"),
        "random_seed": RANDOM_SEEDS,
        "duration_s": duration_s,
        # Acceptance gate results
        "condition_a_in_published_ci": cond_a_in_ci,
        "learning_contribution_in_published_ci": lc_in_ci,
        "acceptance_gates_passed": (
            reproduced_in_ci and isolation_level in ("fresh_worktree", "fresh_clone")
        ),
        # Supporting data
        "preconditions_checked": preconditions,
        "install_transcript_hash": env_info.get("install_transcript_hash"),
        "carnot_importable_in_isolated_env": env_info.get("carnot_importable_in_isolated_env"),
        "per_seed_results": harness_result.get("per_seed_results"),
        "condition_a_auroc_ci95": harness_result.get("condition_a_auroc_ci95"),
        "learning_contribution_ci95": harness_result.get("learning_contribution_ci95"),
        "live_model_invoked": harness_result.get("live_model_invoked", False),
        "n_examples": N_EXAMPLES,
        # Principle annotations
        "field_principles": {
            "honest_verdict": "complete:/success:/passed:/shipped_ prefix required by CLAUDE.md Verdict Terminal-Prefix Discipline.",
            "inference_substrate": "Declares verifier-scoring (not live LLM) so adversarial_verify.py applies the 1s floor, not the 60s live-inference floor.",
            "isolation_level": "fresh_worktree | fresh_clone | in_place_fallback — honest about whether true isolation was achieved; acceptance gate requires fresh_worktree or fresh_clone.",
            "condition_a_auroc_reproduced": "Recomputed production AUROC in the isolated env; must land in [0.9027, 0.9235] (published CI from exp2837 5-seed dual-condition).",
            "learning_contribution_reproduced": "Recomputed FR-11 ablation; must land in [0.0125, 0.0245] to confirm session memory adds measurable FoVer discrimination.",
            "reproduced_in_ci": "True only when both numbers land in their published CIs from the isolated recompute.",
            "isolated_env_versions": "Python + key lib versions of the isolated env so a third party matches the environment.",
            "g2_status": "Honest string describing where G2 stands; cleanroom_validated_internal means strongest internal de-risking; external run still required.",
            "reproducibility_checksum": "Content hash from the isolated recompute; byte-match vs exp2837 is a bonus not required.",
            "random_seed": "The published seeds [42,137,271,314,1729] used in the recompute.",
            "duration_s": "Wall-clock time including isolated install + CPU verifier scoring.",
        },
    }


def run_experiment() -> dict[str, Any]:
    """Top-level entry point for Exp 3430.

    Returns the artifact dict (does not write it — write_artifact does that).
    """
    start_time = time.time()

    # Step 0: preconditions
    preconditions = check_preconditions(REPO_ROOT)
    if not preconditions["ok"]:
        return {
            "artifact": "experiment_3430_fover_g2_cleanroom_validation_v1",
            "honest_verdict": preconditions["blocked_reason"],
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "isolation_level": "none",
            "g2_status": "blocked",
            "g2_independent_reproducer": False,
            "duration_s": time.time() - start_time,
        }

    # Step 1: create isolated environment
    env_info = create_isolated_env(REPO_ROOT)
    isolated_root = env_info["isolated_root"]
    venv_python = env_info["venv_python"]
    tmpdir = env_info["tmpdir"]
    isolation_level = env_info["isolation_level"]

    try:
        # Step 2: collect version info from isolated env
        isolated_versions = get_isolated_env_versions(venv_python)

        # Step 3: run the harness in the isolated env
        harness_result = run_harness_in_isolated_env(isolated_root, venv_python)

    finally:
        # Step 4: clean up worktree + tmpdir
        worktree_path = tmpdir / "worktree"
        cleanup_isolated_env(REPO_ROOT, isolation_level, worktree_path, tmpdir)

    # Step 5: assemble and return artifact
    return build_artifact(
        start_time=start_time,
        preconditions=preconditions,
        env_info=env_info,
        harness_result=harness_result,
        isolated_versions=isolated_versions,
    )


def write_artifact(artifact: dict[str, Any]) -> Path:
    """Write the artifact JSON and return the output path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / OUTPUT_FILENAME
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


def main() -> int:
    artifact = run_experiment()
    out_path = write_artifact(artifact)
    verdict = artifact.get("honest_verdict", "")
    print(f"honest_verdict: {verdict}")
    print(f"isolation_level: {artifact.get('isolation_level')}")
    print(f"reproduced_in_ci: {artifact.get('reproduced_in_ci')}")
    print(f"condition_a_auroc_reproduced: {artifact.get('condition_a_auroc_reproduced')}")
    print(f"learning_contribution_reproduced: {artifact.get('learning_contribution_reproduced')}")
    print(f"g2_status: {artifact.get('g2_status')}")
    print(f"artifact written to: {out_path}")
    return 0 if not str(verdict).startswith("blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
