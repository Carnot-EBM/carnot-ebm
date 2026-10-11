"""REQ-VERIFY-8402: pytest receives historical dependency roots explicitly."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from carnot.reporting import historical_consumer_roots_8402 as e


def bound_popen(manifest: Path, version: str, original: Any) -> Any:
    """Carry the same dependency root into original CLI children without changing their validators."""

    def start(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, (list, tuple)):
            scripts = [
                i
                for i, arg in enumerate(argv)
                if str(arg).startswith(str(e.ROOT / "scripts/experiments/"))
                and str(arg) != str(e.ROOT / e.CLI)
            ]
            if scripts:
                index = scripts[0]
                argv = [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--historical-exec",
                    str(manifest),
                    "--historical-version",
                    version,
                    "--historical-argv",
                    *argv[index:],
                ]
        return original(argv, *args, **kwargs)

    return start


def version_for(module: str) -> str | None:
    """Select a pinned epoch by consumer identity, never by today's active roadmap."""
    if "local_consumer_qualification_8347" in module:
        return "720"
    if "threshold_guard_8362" in module or "v721_capstone_" in module:
        return "721"
    if "direct_atomic_state_8376" in module:
        return "722"
    return None


def pytest_addoption(parser: Any) -> None:
    """Explicit command arguments make the dependency roots visible in each family receipt."""
    parser.addoption("--historical-fixture-manifest")
    parser.addoption("--historical-current-root")


def historical_dependency_root(request: Any, tmp_path_factory: Any) -> Iterator[None]:
    """Provision parent directories before entering inherited readers and leave assertions untouched."""
    manifest = request.config.getoption("--historical-fixture-manifest")
    version = version_for(request.module.__name__)
    if not manifest or version is None:
        yield
        return
    fixtures = json.loads(Path(manifest).read_bytes())
    if not e.check_fixtures(fixtures):
        raise ValueError("historical_fixture_custody")
    current = Path(request.config.getoption("--historical-current-root"))
    if not all((current / name).is_file() for name in [e.DESIGN, e.ACTIVE, e.STAGED]):
        raise ValueError("explicit_current_root_required")
    scratch = tmp_path_factory.mktemp("historical-writes")
    with (
        e.inject(fixtures[version], scratch),
        patch.object(subprocess, "Popen", bound_popen(Path(manifest), version, subprocess.Popen)),
    ):
        yield


historical_dependency_root = pytest.fixture(autouse=True)(historical_dependency_root)
