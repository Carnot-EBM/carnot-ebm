#!/usr/bin/env python3
"""Generate a diagram/figure via paperbanana, backed by a local open-weight model.

**Researcher summary:**
    This is the entrypoint for "generate a diagram" going forward. As of
    2026-07-21, Gemini/Claude/Codex API tokens are no longer available for
    this project, so the default image-generation backend is a local,
    open-weight model (``baidu/ERNIE-Image-Turbo``, Apache-2.0, 8 inference
    steps) served from this machine's own GPU instead of a paid closed-weight
    API.

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

    As of 2026-07-22, the VLM role (Retriever/Planner/Stylist/Critic —
    paperbanana's ``VLM_PROVIDER``) also defaults to a no-paid-API option:
    paperbanana's own ``claude_code`` provider, which shells out to the
    ``claude`` CLI already installed on this machine (uses the existing
    Claude Code subscription, no ``ANTHROPIC_API_KEY`` needed). Verified
    directly 2026-07-22: headless ``claude -p --output-format json --model
    sonnet`` calls succeed with no API key set, correctly read and describe
    a real generated image (the exact reference-image prompt pattern
    ``ClaudeCodeVLM._generate`` uses), and are cheap after the first call in
    a run (~$0.72 cold-start cache-creation cost for the first call in a
    fresh session; ~$0.04-$0.10 per subsequent resumed call in the same
    session). This closes the paid-API gap for the full 7-agent pipeline,
    not just the raster image-gen step. Override via ``--vlm-provider`` /
    ``--vlm-model`` passthrough flags, or the ``VLM_PROVIDER`` / ``VLM_MODEL``
    env vars, if a different provider is ever preferred.

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

# paperbanana's Settings.vlm_model defaults to "gemini-2.5-flash" -- if we
# only set VLM_PROVIDER=claude_code and leave VLM_MODEL unset, that bogus
# model name gets passed straight to `claude -p --model gemini-2.5-flash`
# (ClaudeCodeVLM(model=settings.vlm_model) in providers/registry.py) and the
# CLI call fails. Both must be set together.
DEFAULT_VLM_PROVIDER = "claude_code"
DEFAULT_VLM_MODEL = "sonnet"
# Providers that need a paid API key paperbanana does not have a local
# fallback for. claude_code is deliberately absent: it needs the `claude`
# CLI (checked separately below), not an API key.
_PAID_VLM_PROVIDERS = {"gemini", "openai", "anthropic", "atlas", "bedrock"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_paperbanana() -> "str | None":
    """Locate the paperbanana CLI, preferring the current interpreter's own venv.

    `shutil.which("paperbanana")` alone only checks $PATH, which is wrong
    when this script is invoked as `.venv/bin/python scripts/generate_diagram.py`
    without that venv's bin/ directory being on PATH (e.g. from a plain
    `python3 scripts/generate_diagram.py` shebang-driven call, or a subagent
    shell that never `source`d activate) -- paperbanana would then be
    genuinely installed (editable, into this exact interpreter's site-packages)
    but reported as "not installed", a misleading blocked_* verdict.
    """
    venv_candidate = os.path.join(os.path.dirname(sys.executable), "paperbanana")
    if os.path.isfile(venv_candidate) and os.access(venv_candidate, os.X_OK):
        return venv_candidate
    return shutil.which("paperbanana")


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
    paperbanana_path = _find_paperbanana()
    if paperbanana_path is None:
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
        # Default the VLM role to the local, no-paid-API claude_code provider
        # too -- only if the caller hasn't already picked something else via
        # env or a --vlm-provider passthrough flag.
        if "VLM_PROVIDER" not in env and "--vlm-provider" not in passthrough:
            env["VLM_PROVIDER"] = DEFAULT_VLM_PROVIDER
            env.setdefault("VLM_MODEL", DEFAULT_VLM_MODEL)
        vlm_provider = env.get("VLM_PROVIDER", "gemini")
        if vlm_provider == "claude_code" and shutil.which("claude") is None:
            print(
                "blocked_claude_cli_not_found: VLM_PROVIDER=claude_code needs the "
                "`claude` CLI on PATH (Pre-Launch Preconditions Discipline). Install "
                "Claude Code and sign in, or pass --vlm-provider/--vlm-model to pick "
                "a different provider.",
                file=sys.stderr,
            )
            sys.exit(1)
        if vlm_provider in _PAID_VLM_PROVIDERS and not (
            env.get("GOOGLE_API_KEY") or env.get(f"{vlm_provider.upper()}_API_KEY")
        ):
            print(
                "WARNING: VLM_PROVIDER (paperbanana's planner/critic/stylist role) "
                f"is set to '{vlm_provider}' and no matching API key is set. "
                "Pass --vlm-provider claude_code (default), ollama, or openai_local "
                "for a no-paid-API option.",
                file=sys.stderr,
            )
    else:
        print(
            f"WARNING: backend={args.backend} requires a paid API key this project "
            "no longer has by default; this path is kept only for completeness.",
            file=sys.stderr,
        )

    cmd = [_find_paperbanana() or "paperbanana", "generate", *passthrough]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
