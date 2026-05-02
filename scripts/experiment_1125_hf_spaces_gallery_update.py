"""Experiment 1125 — HuggingFace Spaces gallery update: Hashi cartridge + benchmarks.

This is the .87 milestone gallery refresh.  The task this script discharges:

1. Confirm the Hashi cartridge (exp1124) is registered in the WOPR gallery
   (``spaces/wopr-games/games/__init__.py``).  exp1124 already shipped the
   cartridge source — this experiment is the gallery-side update that
   exposes it to end users.

2. Confirm the gallery README carries the latest benchmark numbers from
   the surrounding milestone:

   * ThinkPRM v2 AUROC=0.9946 (exp1111)
   * k=5 AND-composition deployed, max pairwise r=0.462 (exp1108/exp1121)
   * Energy verifier retrain post-inversion fix (exp1120)
   * 36 LLM failure exemplars, 100% mathematical-objective TP rate
     (exp1112)

3. Retrieve HF_TOKEN via SOPS and upload the updated space to HuggingFace
   so the cartridge and the new numbers are live for users.

4. Verify the deployed Space URL returns HTTP 200.

Why this exists as a standalone experiment rather than a hook on exp1124:
the .84 milestone retrospective recommended decoupling cartridge
authoring from gallery deployment so that a malformed deploy step does
not cause a perfectly good cartridge experiment to read as a failure.
exp1102 (N-Queens) was the first script to follow this discipline; this
one is its Hashi-equivalent.

Usage::

    JAX_PLATFORMS=cpu python scripts/experiment_1125_hf_spaces_gallery_update.py

The script writes its artifact to
``results/experiment_1125_hf_spaces_gallery_update.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_DIR = REPO_ROOT / "spaces" / "wopr-games"
GAMES_DIR = SPACES_DIR / "games"
README_PATH = SPACES_DIR / "README.md"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1125_hf_spaces_gallery_update.json"
HF_SPACE_REPO = "Carnot-EBM/wopr-games"
LIVE_URL = "https://huggingface.co/spaces/Carnot-EBM/wopr-games"

# These are the numbers the milestone's downstream artifacts agreed on.
# Substring presence in the gallery README is what counts as "the
# benchmark section was actually updated" — searching for the SOURCE
# experiment IDs is more robust than searching for the numbers, because
# the numbers may be re-rounded or re-formatted in a later edit.
BENCHMARK_MARKERS = [
    "0.9946",  # ThinkPRM v2 AUROC (exp1111)
    "exp1111",
    "0.462",  # k=5 AND-composition max pairwise r (exp1108/exp1121)
    "exp1121",
    "exp1120",  # Energy verifier retrain
    "exp1112",  # LLM failure exemplar corpus
    "36 exemplars",
]


def _retrieve_hf_token() -> tuple[str | None, str]:
    """Try SOPS first, then environment variable.

    Returns ``(token, source_description)``.  The source string is logged
    in the artifact so an auditor can tell whether the deploy ran with
    a credential pulled from SOPS or from a less-trustworthy ambient
    environment variable.
    """
    for cmd in [
        "sops -d secrets/hf_token.enc.yaml",
        "sops -d ops/secrets.yaml",
    ]:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=REPO_ROOT,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "HF_TOKEN" in line and ":" in line:
                        token = line.split(":", 1)[1].strip()
                        if token:
                            return token, f"sops:{cmd.split()[2]}"
        except Exception:  # noqa: BLE001 — sops is best-effort here
            pass

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token, "env:HF_TOKEN"

    return None, "not_found"


def _cartridge_registered(init_path: Path, class_name: str) -> bool:
    """Return True iff ``class_name`` is wired into ``ALL_GAMES``.

    Two checks: the import string and an instantiation line.  We don't
    parse Python — a substring scan is robust enough because
    ``games/__init__.py`` is a tiny hand-maintained file.
    """
    if not init_path.exists():
        return False
    content = init_path.read_text()
    return class_name in content and "ALL_GAMES" in content and f"{class_name}()" in content


def _count_cartridges(init_path: Path) -> int:
    """Count the number of ``XGame()`` instantiations inside ALL_GAMES.

    This mirrors the exp1102 helper: we walk the file linearly until we
    hit the ``[`` that opens ALL_GAMES, count ``Game()`` lines, and stop
    at the closing ``]``.
    """
    if not init_path.exists():
        return 0
    n = 0
    in_block = False
    for line in init_path.read_text().splitlines():
        if "ALL_GAMES" in line and "[" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if "Game()" in line:
            n += 1
        if "]" in line and "Game" not in line:
            break
    return n


def _benchmark_results_present(readme_path: Path, markers: list[str]) -> tuple[bool, list[str]]:
    """Verify the README contains every required benchmark marker.

    Returns ``(all_present, missing_markers)``.  We treat ANY missing
    marker as "results not updated" rather than partial — gallery
    visitors should never see a stale-and-mixed table.
    """
    if not readme_path.exists():
        return False, list(markers)
    content = readme_path.read_text()
    missing = [m for m in markers if m not in content]
    return (len(missing) == 0), missing


def _upload_space(token: str) -> tuple[bool, str | None]:
    """Upload ``spaces/wopr-games/`` to HuggingFace.

    Tries the ``huggingface_hub`` Python API first because it produces
    structured errors and is the documented surface; falls back to the
    ``hf upload`` CLI shell-out so a deploy still works on a host that
    is missing the Python package.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]

        api = HfApi(token=token)
        url = api.upload_folder(
            folder_path=str(SPACES_DIR),
            repo_id=HF_SPACE_REPO,
            repo_type="space",
        )
        return True, str(url) if url is not None else None
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — surface to caller as deploy_error
        # Fall through to CLI fallback only on truly unexpected errors —
        # we still want to capture the original message for the artifact.
        cli_ok, cli_msg = _upload_via_cli(token)
        if cli_ok:
            return True, cli_msg
        return False, f"hub_api_failed: {exc!s} | cli_fallback: {cli_msg}"

    return _upload_via_cli(token)


def _upload_via_cli(token: str) -> tuple[bool, str | None]:
    """Fallback shell-out to ``hf upload``.

    Kept separate so the python-API path can call it on its own
    exception handling.  Capped at 180s — Spaces are small (~MB) and a
    healthy hub responds in seconds.
    """
    try:
        env = {**os.environ, "HF_TOKEN": token}
        result = subprocess.run(
            [
                "hf",
                "upload",
                HF_SPACE_REPO,
                str(SPACES_DIR),
                ".",
                "--repo-type",
                "space",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode == 0:
            for line in (result.stdout + result.stderr).splitlines():
                if "huggingface.co" in line:
                    return True, line.strip()
            return True, None
        return False, (result.stderr or result.stdout)[:500]
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _http_status(url: str) -> int:
    """Return the HTTP status code for ``url``.

    HEAD first (cheap), then a curl fallback so the script does not
    silently report 0 because urllib was blocked by some odd proxy.
    """
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status)
    except Exception:
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "--max-time",
                    "30",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=35,
            )
            return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        except Exception:
            return 0


def _decide_verdict(
    *,
    hf_token_found: bool,
    hashi_cartridge_local: bool,
    benchmark_results_updated: bool,
    deploy_attempted: bool,
    gallery_updated: bool,
    live_http_status: int,
) -> str:
    """Map the boolean outcome matrix to one of the schema's verdicts.

    The schema enum (per the conductor task) is::

        deployed_live | local_only_hf_token_unavailable | partial | failed

    We err toward ``partial`` when something is half-done and ``failed``
    only when the local files themselves are wrong — because a missing
    HF token is a deploy-credentials issue, not a research-result one.
    """
    if not hashi_cartridge_local:
        return "failed"
    if not hf_token_found:
        # Local files OK but we cannot reach the hub — intentional
        # degraded mode the conductor schema explicitly accommodates.
        return "local_only_hf_token_unavailable"
    if gallery_updated and live_http_status == 200 and benchmark_results_updated:
        return "deployed_live"
    if deploy_attempted:
        return "partial"
    return "partial"


def main() -> None:
    started_at = time.time()

    # 1. HF token: SOPS preferred, env fallback, none-acceptable.
    hf_token, token_source = _retrieve_hf_token()
    hf_token_found = hf_token is not None

    # 2. Local-state checks: cartridge wired in, README updated.
    init_path = GAMES_DIR / "__init__.py"
    hashi_path = GAMES_DIR / "hashi.py"
    hashi_cartridge_local = hashi_path.exists() and _cartridge_registered(init_path, "HashiGame")
    n_cartridges_deployed = _count_cartridges(init_path)

    benchmark_results_updated, missing_markers = _benchmark_results_present(
        README_PATH, BENCHMARK_MARKERS
    )

    # 3. Deploy.  Skipped when local state is wrong — no point pushing
    # a broken gallery — and skipped when no token, since HF rejects
    # anonymous uploads on these repos.
    deploy_attempted = False
    gallery_updated = False
    deploy_error: str | None = None
    space_url: str | None = None

    if hf_token_found and hashi_cartridge_local:
        deploy_attempted = True
        ok, deploy_msg = _upload_space(hf_token)  # type: ignore[arg-type]
        gallery_updated = ok
        if ok:
            space_url = deploy_msg or LIVE_URL
        else:
            deploy_error = deploy_msg

    # 4. Liveness probe — only meaningful if we actually pushed.
    live_http_status = 0
    if deploy_attempted and gallery_updated:
        live_http_status = _http_status(LIVE_URL)

    honest_verdict = _decide_verdict(
        hf_token_found=hf_token_found,
        hashi_cartridge_local=hashi_cartridge_local,
        benchmark_results_updated=benchmark_results_updated,
        deploy_attempted=deploy_attempted,
        gallery_updated=gallery_updated,
        live_http_status=live_http_status,
    )

    # ``gallery_updated`` per the conductor schema is True when EITHER
    # the deploy succeeded OR local files are ready for a future deploy.
    # We surface both signals in the artifact so a downstream auditor
    # can disambiguate.
    gallery_updated_artifact = bool(gallery_updated or hashi_cartridge_local)
    hashi_cartridge_deployed = bool(
        gallery_updated and live_http_status == 200 and hashi_cartridge_local
    )

    duration_s = round(time.time() - started_at, 3)

    artifact = {
        "experiment": "exp1125_hf_spaces_gallery_update",
        "title": "HuggingFace Spaces gallery update — Hashi cartridge + benchmarks",
        "schema": "carnot.hf_spaces_gallery_update.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "duration_s": duration_s,
        "hashi_cartridge_local": hashi_cartridge_local,
        "hashi_cartridge_deployed": hashi_cartridge_deployed,
        "benchmark_results_updated": benchmark_results_updated,
        "missing_benchmark_markers": missing_markers,
        "n_cartridges_deployed": n_cartridges_deployed,
        "hf_token_found": hf_token_found,
        "hf_token_source": token_source,
        "deploy_attempted": deploy_attempted,
        "gallery_updated": gallery_updated_artifact,
        "deploy_succeeded": gallery_updated,
        "deploy_error": deploy_error,
        "live_http_status": live_http_status,
        "live_url": LIVE_URL,
        "space_url": space_url if space_url else (LIVE_URL if gallery_updated else None),
        "honest_verdict": honest_verdict,
        "cost_usd": 0.0,
        "decision_class": "deploy",
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
