#!/usr/bin/env python3
"""Generate a diagram/figure via paperbanana, backed by a local open-weight model.

**Researcher summary:**
    This is the entrypoint for "generate a diagram" going forward. As of
    2026-07-21, Gemini/Claude/Codex API tokens are no longer available for
    this project, so the default image-generation backend is a local,
    open-weight model (``baidu/ERNIE-Image``, Apache-2.0) served from this
    machine's own GPU instead of a paid closed-weight API.

**Detailed explanation for engineers:**
    Wires the vendored ``external/paperbanana`` CLI (see CLAUDE.md "Audit
    untrusted code") to ``python/carnot/imagegen/ernie_image_server.py``:
    a small local FastAPI process that speaks just enough of the OpenAI
    Images API for paperbanana's unmodified ``openai_imagen`` provider to
    treat it as a drop-in backend via ``OPENAI_BASE_URL``. No paperbanana
    source is patched.

    ``CARNOT_IMAGE_BACKEND`` (env var, or ``--backend``) selects the model
    backend: ``ernie-local`` (default), ``gemini``, ``openai``, or ``none``.
    Only ``ernie-local`` is local-first / decentralization-compliant per
    CLAUDE.md's Decentralization-Respecting Design Constraints rule 1; the
    others are kept only for completeness and require API keys this project
    no longer has.

    KNOWN GAP (documented, not silently hidden): this script only swaps the
    *image-generation* role. paperbanana's Retriever/Planner/Stylist/Critic
    agents ("VLM_PROVIDER") still default to Gemini and are NOT addressed
    here — see ops/known-issues.md 2026-07-21 entry for the local-VLM
    follow-up. Without a local VLM_PROVIDER, the full 7-agent pipeline still
    needs GOOGLE_API_KEY (or another VLM provider's key) even after this
    change; only the raster image-gen step becomes truly local.

Spec: REQ-PUBLISH-042, SCENARIO-PUBLISH-042

Usage:
    python scripts/generate_diagram.py --caption "Architecture overview" \\
        --input docs/research-notes/some-methodology.txt -- --iterations 3

    Everything after a bare "--" (or any unrecognized flag) is passed
    through verbatim to `paperbanana generate`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8990
PLACEHOLDER_API_KEY = "local-not-a-real-key"
SERVER_START_TIMEOUT_S = 30.0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _server_healthy(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if an ernie_image_server is already answering /healthz."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=timeout):
            return True
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def _launch_ernie_server(host: str, port: int, gpu: "int | None") -> subprocess.Popen:
    """Start the ERNIE-Image shim as a detached background process.

    Left running after this script exits (loading an 8B-parameter pipeline
    is expensive; subsequent invocations should reuse it via the healthz
    check above, mirroring how this project's llama-server processes are
    launched once and reused across experiments).
    """
    cmd = [
        sys.executable,
        "-m",
        "carnot.imagegen.ernie_image_server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    log_path = os.path.join(REPO_ROOT, "ops", ".ernie_image_server.log")
    log_file = open(log_path, "a")  # noqa: SIM115 - intentionally kept open by the child
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_for_health(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _server_healthy(host, port):
            return True
        time.sleep(1.0)
    return False


def _ensure_ernie_local_backend(host: str, port: int, gpu: "int | None") -> dict[str, str]:
    """PRECONDITIONS + server bring-up for the ernie-local backend.

    Returns the environment overlay to apply to the `paperbanana` subprocess
    call, or exits the process with an honest blocked_* message on failure
    (Pre-Launch Preconditions Discipline: never fabricate a diagram, never
    proceed past a missing resource).
    """
    from carnot.imagegen.ernie_image_server import HF_REPO_ID, ernie_image_cached

    if not ernie_image_cached():
        print(
            f"blocked_ernie_image_not_cached: {HF_REPO_ID} is not in the local "
            f"HuggingFace cache. Fetch it first:\n"
            f"  huggingface-cli download {HF_REPO_ID}",
            file=sys.stderr,
        )
        sys.exit(1)
    if shutil.which("paperbanana") is None:
        print(
            "blocked_paperbanana_not_installed: run "
            "`uv pip install -e external/paperbanana[openai]` first "
            "(see ops/known-issues.md 2026-07-21 entry).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _server_healthy(host, port):
        print(f"ernie_image_server not running at {host}:{port} -- launching it...")
        _launch_ernie_server(host, port, gpu)
        if not _wait_for_health(host, port, SERVER_START_TIMEOUT_S):
            print(
                f"blocked_ernie_server_failed_to_start: no healthz response within "
                f"{SERVER_START_TIMEOUT_S:.0f}s. Check ops/.ernie_image_server.log.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"ernie_image_server healthy at {host}:{port}")
    else:
        print(f"reusing already-running ernie_image_server at {host}:{port}")

    return {
        "IMAGE_PROVIDER": "openai_imagen",
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or PLACEHOLDER_API_KEY,
        "OPENAI_BASE_URL": f"http://{host}:{port}/v1",
        "OPENAI_IMAGE_MODEL": "ernie-image",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("CARNOT_IMAGE_BACKEND", "ernie-local"),
        choices=["ernie-local", "gemini", "openai", "none"],
        help="Image-gen backend. Default: ernie-local (local, open-weight, no API key).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--gpu", type=int, default=None, help="CUDA device index for the ernie-local server."
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.backend == "none":
        print(
            "backend=none: no diagram generation configured. Use matplotlib directly for statistical plots."
        )
        return

    env = dict(os.environ)
    if args.backend == "ernie-local":
        env.update(_ensure_ernie_local_backend(args.host, args.port, args.gpu))
        vlm_provider = env.get("VLM_PROVIDER", "gemini")
        if vlm_provider in {"gemini", "openai", "anthropic", "atlas", "bedrock"} and not (
            env.get("GOOGLE_API_KEY") or env.get(f"{vlm_provider.upper()}_API_KEY")
        ):
            print(
                "WARNING: VLM_PROVIDER (paperbanana's planner/critic/stylist role) "
                f"still defaults to '{vlm_provider}' and no matching API key is set. "
                "Only the image-generation step is local right now -- see "
                "ops/known-issues.md 2026-07-21 entry for the local-VLM follow-up. "
                "Pass --vlm-provider ollama or --vlm-provider openai_local through "
                "to paperbanana if you have a local VLM server running.",
                file=sys.stderr,
            )
    else:
        print(
            f"WARNING: backend={args.backend} requires a paid API key this project "
            "no longer has by default; this path is kept only for completeness.",
            file=sys.stderr,
        )

    cmd = ["paperbanana", "generate", *passthrough]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
