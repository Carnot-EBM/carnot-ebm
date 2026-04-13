"""Experiment 267 — HuggingFace per-token EBM README batch update.

Spec refs: REQ-HF-267-A (banner present), REQ-HF-267-B (append structure),
           REQ-HF-267-C (repo enumeration), REQ-HF-267-D (mock HF push),
           REQ-HF-267-E (credential blocker)
Story: HuggingFace Publishing Milestones #1 (research-program.md)
Run date: 20260413

What this script does
---------------------
1. Checks for HuggingFace authentication via huggingface_hub (Python API, not CLI
   subprocess). If credentials are absent, writes a clear blocker artifact and exits.
2. Enumerates all 16 per-token activation EBM model repos under Carnot-EBM.
3. For each repo:
   a. Fetches the current README.md from HuggingFace Hub.
   b. Checks whether the Phase 1 status banner and the "What's Proven to Work"
      section are already present — if so, marks the repo as skipped_already_current.
   c. Otherwise, prepends the status banner (without overwriting existing content)
      and appends the proven-work section.
   d. Pushes the updated README via the huggingface_hub Python API.
   e. Logs repo_id, hf_url, and push status (success / failed / skipped_already_current).
4. Writes results/experiment_267_results.json with the per-repo log and summary counts.

Usage
-----
    cd /path/to/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_267_hf_readme_update.py

To authenticate first (if not already done):
    huggingface-cli login
or set the HF_TOKEN environment variable.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "experiment_267_results.json"

HF_ORG = "Carnot-EBM"
RUN_DATE = "20260413"

# The 16 per-token activation EBM model repos published during Phase 1.
# These were trained to detect hallucination confidence signals from LLM
# hidden-state activations — they are NOT correctness verifiers.
MODEL_REPOS: list[str] = [
    "Carnot-EBM/per-token-ebm-bonsai-17b-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e2b-it-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e2b-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e4b-it-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e4b-nothink",
    "Carnot-EBM/per-token-ebm-gptoss-20b-nothink",
    "Carnot-EBM/per-token-ebm-lfm25-12b-nothink",
    "Carnot-EBM/per-token-ebm-lfm25-350m-nothink",
    "Carnot-EBM/per-token-ebm-qwen3-06b",
    "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-08b-think",
    "Carnot-EBM/per-token-ebm-qwen35-27b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-2b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-35b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-4b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-9b-nothink",
]

# ---------------------------------------------------------------------------
# Status banner — prepended to each README (Phase 1 research artifact notice).
# Explains that these models detect output confidence, not correctness, and
# points users to the production pipeline via `pip install carnot`.
# ---------------------------------------------------------------------------

STATUS_BANNER = textwrap.dedent("""\
    > ⚠️ **PHASE 1 RESEARCH ARTIFACT**
    >
    > This model detects **output confidence** (hallucination likelihood signals
    > from LLM hidden-state activations), **not correctness**. It cannot verify
    > whether a model's answer is right — it can only signal how uncertain the
    > model appears token-by-token.
    >
    > **For production use**, install the full Carnot pipeline which includes
    > FormalClaimVerifier (solver-routed formal claim verification), PBT code
    > verification (property-based testing on 164-problem HumanEval), process
    > integrity detection (right-for-wrong-reasons), and the Carnot MCP server:
    >
    > ```bash
    > pip install carnot
    > ```
    >
    > See [Carnot on GitHub](https://github.com/Carnot-EBM/carnot-ebm) for
    > documentation and the full production API.

""")

# ---------------------------------------------------------------------------
# Proven-work section — appended to each README.
# Lists capabilities that have been validated on real (non-simulated) data
# with live GPU inference as of 2026.
# ---------------------------------------------------------------------------

PROVEN_SECTION = textwrap.dedent("""\

    ## What's Proven to Work (2026)

    The following Carnot pipeline capabilities have been validated with live GPU
    inference (not simulation) as of April 2026. Install via `pip install carnot`.

    | Capability | What it does | Evidence |
    |---|---|---|
    | **FormalClaimVerifier** | Solver-routed formal claim verification: arithmetic, boolean-entailment, set-membership, execution-oracle, cardinality, comparison routes | 1,243 solver-routable rows from live GSM8K + HumanEval traces (Exp 244/246) |
    | **PBT code verification** | Property-based testing (Hypothesis) catches bugs that official test suites miss | +3.0pp on 164-problem HumanEval with Gemma4-E4B-it (Exp 226); 2 official-test misses caught on Qwen3.5-0.8B (Exp 227) |
    | **Process integrity detection** | Detects right-for-wrong-reasons answers where the output is correct but the reasoning process is invalid | 5 right-for-wrong-reasons cases caught across 30-case HumanEval cohort (Exp 251) |
    | **Carnot MCP server** | Exposes `verify_code_with_pbt` and 6 other tools to any MCP-compatible agent | 7 discoverable tools, 30s timeout, 10K input guard (VERIFY-031) |

    These results use instruction-tuned models (Gemma4-E4B-it, Qwen3.5-0.8B) on
    live CUDA hardware. All per-token EBM confidence results (this model family)
    are Phase 1 research artifacts and should not be interpreted as correctness scores.
""")

# ---------------------------------------------------------------------------
# Idempotency sentinel strings — detect whether a README already has our additions.
# We use a substring of each addition that is unique enough to serve as a marker.
# ---------------------------------------------------------------------------

_BANNER_SENTINEL = "PHASE 1 RESEARCH ARTIFACT"
_SECTION_SENTINEL = "What's Proven to Work (2026)"


# ---------------------------------------------------------------------------
# README content helpers
# ---------------------------------------------------------------------------


def is_already_current(readme_text: str) -> bool:
    """Return True if the README already contains both the banner and the proven section.

    This is the idempotency guard. If both sentinels are present we skip the repo
    so repeated runs don't accumulate duplicate content.
    """
    return _BANNER_SENTINEL in readme_text and _SECTION_SENTINEL in readme_text


def build_updated_readme(existing: str) -> str:
    """Prepend STATUS_BANNER and append PROVEN_SECTION to an existing README.

    If the banner is already present (idempotency check), do not insert a second copy.
    The existing content is preserved verbatim between the banner and the new section.

    YAML front-matter (--- ... --- block at the very top) is left untouched because
    we prepend after detecting it.  If there is no front-matter, the banner goes
    at the very top.
    """
    # Handle idempotency: if the banner sentinel is already present, don't double-add.
    if _BANNER_SENTINEL in existing:
        # Banner already there. Append proven section if missing.
        if _SECTION_SENTINEL not in existing:
            return existing + PROVEN_SECTION
        return existing

    # Check for YAML front-matter block (lines 0..N bounded by "---").
    # We want to insert the banner *after* the closing "---" so the YAML stays valid.
    lines = existing.split("\n")
    insert_at_char = 0

    if lines and lines[0].strip() == "---":
        # Find the closing ---
        closing_idx = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_idx = i
                break
        if closing_idx is not None:
            # Insert after the closing --- line
            before_fm = "\n".join(lines[: closing_idx + 1]) + "\n"
            after_fm = "\n".join(lines[closing_idx + 1 :])
            updated = before_fm + "\n" + STATUS_BANNER + after_fm
            if _SECTION_SENTINEL not in updated:
                updated = updated + PROVEN_SECTION
            return updated

    # No front-matter: prepend banner, then existing content, then proven section.
    updated = STATUS_BANNER + existing
    if _SECTION_SENTINEL not in updated:
        updated = updated + PROVEN_SECTION
    return updated


# ---------------------------------------------------------------------------
# HuggingFace API interaction helpers
# (separated into small functions so tests can mock them individually)
# ---------------------------------------------------------------------------


def _fetch_readme(api: object, repo_id: str) -> str:
    """Download the current README.md text from a HuggingFace model repo.

    Uses hf_hub_download to pull README.md to a local temp path and reads it.
    Raises any exception from huggingface_hub on failure (caller handles).
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename="README.md",
        repo_type="model",
    )
    return Path(local_path).read_text(encoding="utf-8")


def _push_readme(api: object, repo_id: str, content: str) -> None:
    """Upload the updated README.md text to a HuggingFace model repo.

    Uses the huggingface_hub HfApi.upload_file method (Python API, not subprocess).
    Raises any exception from huggingface_hub on failure (caller handles).
    """
    import io

    from huggingface_hub import CommitOperationAdd

    # upload_file accepts a bytes-like or path. We use bytes to avoid tmp file creation.
    encoded = content.encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(encoded),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Exp 267 ({RUN_DATE}): add Phase 1 banner + proven-work section",
    )


def update_model_readme(api: object, repo_id: str) -> dict[str, object]:
    """Fetch, update, and push the README for a single model repo.

    Returns a log dict with keys: repo_id, hf_url, status, and optionally error.
    status is one of: "success", "failed", "skipped_already_current".
    """
    hf_url = f"https://huggingface.co/{repo_id}"
    log: dict[str, object] = {"repo_id": repo_id, "hf_url": hf_url}

    try:
        existing = _fetch_readme(api, repo_id)
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = str(exc)
        print(f"  [FAIL] {repo_id}: fetch error — {exc}")
        return log

    if is_already_current(existing):
        log["status"] = "skipped_already_current"
        print(f"  [SKIP] {repo_id}: already current")
        return log

    updated = build_updated_readme(existing)

    try:
        _push_readme(api, repo_id, updated)
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = str(exc)
        print(f"  [FAIL] {repo_id}: push error — {exc}")
        return log

    log["status"] = "success"
    print(f"  [OK]   {repo_id} → {hf_url}")
    return log


# ---------------------------------------------------------------------------
# Authentication check
# ---------------------------------------------------------------------------


def check_authenticated() -> tuple[bool, str]:
    """Return (authenticated, username_or_reason).

    Uses huggingface_hub.whoami() which reads the cached token from
    ~/.cache/huggingface/token or the HF_TOKEN env var.
    Returns (True, username) on success, (False, reason) on failure.
    """
    import huggingface_hub

    try:
        info = huggingface_hub.whoami()
        username = info.get("name", "unknown")
        print(f"[HF] Authenticated as: {username}")
        return True, username
    except Exception as exc:
        reason = (
            f"Not authenticated: {exc}. "
            "Run `huggingface-cli login` or set the HF_TOKEN environment variable."
        )
        return False, reason


# ---------------------------------------------------------------------------
# Blocker artifact
# ---------------------------------------------------------------------------


def write_blocker_artifact(results_path: Path, message: str) -> None:
    """Write a clear blocker artifact when HF credentials are absent.

    The artifact lists all repos that would have been updated so the operator
    knows exactly what `huggingface-cli login` will unblock.
    """
    artifact: dict[str, object] = {
        "experiment": 267,
        "run_date": RUN_DATE,
        "status": "blocked",
        "blocker_message": message,
        "instructions": (
            "Run `huggingface-cli login` to authenticate, "
            "or set the HF_TOKEN environment variable, then re-run this script."
        ),
        "repos_to_update": MODEL_REPOS,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\n[BLOCKER] Results written to {results_path}")
    print(f"[BLOCKER] {message}")
    print("[BLOCKER] Run: huggingface-cli login")


# ---------------------------------------------------------------------------
# Results artifact builder
# ---------------------------------------------------------------------------


def build_results_artifact(repo_logs: list[dict[str, object]]) -> dict[str, object]:
    """Assemble the final experiment_267_results.json artifact from per-repo logs.

    Computes summary counts and determines overall status:
    - "complete"  — all repos updated or skipped (zero failures)
    - "partial"   — some repos failed
    - "blocked"   — should not happen here (handled by write_blocker_artifact)
    """
    success = sum(1 for r in repo_logs if r.get("status") == "success")
    skipped = sum(1 for r in repo_logs if r.get("status") == "skipped_already_current")
    failed = sum(1 for r in repo_logs if r.get("status") == "failed")
    total = len(repo_logs)

    overall_status = "complete" if failed == 0 else "partial"

    return {
        "experiment": 267,
        "run_date": RUN_DATE,
        "status": overall_status,
        "description": (
            "Batch update of 16 per-token EBM model READMEs on HuggingFace: "
            "prepend Phase 1 research artifact status banner, "
            "append What's Proven to Work (2026) section."
        ),
        "summary": {
            "total": total,
            "success": success,
            "skipped_already_current": skipped,
            "failed": failed,
        },
        "repo_logs": repo_logs,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the HuggingFace README batch update.

    Returns 0 on complete/partial success, 1 on blocked (no credentials).
    """
    print(f"Experiment 267 — HuggingFace README batch update (run_date={RUN_DATE})")
    print(f"Updating {len(MODEL_REPOS)} repos under {HF_ORG}/...\n")

    # Step 1: check credentials
    ok, reason = check_authenticated()
    if not ok:
        write_blocker_artifact(RESULTS_PATH, reason)
        return 1

    # Step 2: build HfApi instance (authenticated)
    from huggingface_hub import HfApi

    api = HfApi()

    # Step 3: update each repo
    repo_logs: list[dict[str, object]] = []
    for repo_id in MODEL_REPOS:
        print(f"Processing {repo_id} …")
        log = update_model_readme(api, repo_id)
        repo_logs.append(log)

    # Step 4: write results artifact
    artifact = build_results_artifact(repo_logs)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    summary = artifact["summary"]
    print(f"\nDone. {summary['success']} updated, {summary['skipped_already_current']} skipped, {summary['failed']} failed.")
    print(f"Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
