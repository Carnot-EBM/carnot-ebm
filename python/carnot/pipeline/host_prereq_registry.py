"""Host prerequisite registry for Carnot research experiments.

**Why this module exists:**
    RETRO-006 (2026.04.24 retrospective): AMD XDNA NPU experiments (Exps 292, 303,
    314, 335) each independently discovered the same two missing system packages
    (``ninja``, ``openblas``).  This wasted approximately 4 experiment slots — each
    experiment had to rediscover the same missing packages from scratch.

    This module provides a central registry (``ops/host-prereqs.md``) of host-level
    prerequisites that an experiment class requires before it will succeed.  The
    registry is loaded once at construction time; ``check_prereqs()`` runs each
    package's check command via subprocess and returns the list of missing packages.

    **Using the registry in an experiment:**

    ```python
    from carnot.pipeline.host_prereq_registry import HostPrereqRegistry

    registry = HostPrereqRegistry()
    missing = registry.check_prereqs(experiment_class="npu")
    if missing:
        # emit a blocked artifact rather than spinning for 20 minutes
        artifact = tmpl.build_result(
            {"missing_prereqs": missing, "next_action": "Install missing host packages"},
            status="blocked",
        )
    ```

Spec: REQ-INFRA-006, SCENARIO-INFRA-009, SCENARIO-INFRA-010
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default path to the registry markdown file
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "ops" / "host-prereqs.md"
"""Path to the canonical host prerequisite registry.

Resolves from ``python/carnot/pipeline/host_prereq_registry.py`` up 3 levels
(pipeline/ → carnot/ → python/ → repo_root), then into ``ops/host-prereqs.md``.
Can be overridden in tests by passing ``registry_path`` to ``HostPrereqRegistry.__init__``.
"""


# ---------------------------------------------------------------------------
# PrereqEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class PrereqEntry:
    """One row in the host-prereqs.md table.

    Fields
    ------
    package : str
        Human-readable package name (or env-var name like ``CARNOT_FORCE_LIVE``).
    check_command : str
        Shell command to run to verify the package is present.
        A zero exit code means present; non-zero or FileNotFoundError means missing.
        Use ``"env:VAR_NAME"`` prefix for environment-variable checks (no subprocess).
    install_arch : str
        ``pacman``-style install command for Arch Linux.
    install_debian : str
        ``apt``-style install command for Debian/Ubuntu.
    required_for : list[str]
        Experiment class tags this package is required for (e.g. ``["npu", "fpga"]``).
        The tag ``"all"`` means every experiment class.
    """

    package: str
    check_command: str
    install_arch: str
    install_debian: str
    required_for: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HostPrereqRegistry
# ---------------------------------------------------------------------------


class HostPrereqRegistry:
    """Loads ``ops/host-prereqs.md`` and checks host-level prerequisites.

    **Construction:** reads and parses the markdown table immediately.  Construction
    does not run any check commands; that is deferred to ``check_prereqs()``.

    **Thread safety:** all public methods are read-only after construction; safe
    to call from multiple threads.

    Parameters
    ----------
    registry_path : Path | None
        Override the default path to ``ops/host-prereqs.md``.  Used in tests.
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH
        self._entries: list[PrereqEntry] = _parse_registry(path)
        _log.debug(
            "HostPrereqRegistry: loaded %d entries from %s", len(self._entries), path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[PrereqEntry]:
        """All loaded prerequisite entries (read-only view)."""
        return list(self._entries)

    def check_prereqs(
        self,
        experiment_class: Optional[str] = None,
    ) -> list[str]:
        """Return a list of package names that are missing on this host.

        Runs each entry's check command (or env-var check) to determine whether
        the package is available.  A package is considered **missing** when:
        - Its check command exits with a non-zero return code, OR
        - Its check command binary is not found (``FileNotFoundError``), OR
        - ``subprocess.TimeoutExpired`` (5 s hard limit per check).

        Parameters
        ----------
        experiment_class : str | None
            If supplied, only entries whose ``required_for`` list contains
            ``experiment_class`` OR the special tag ``"all"`` are checked.
            If ``None``, all registered entries are checked.

        Returns
        -------
        list[str]
            Package names (as they appear in the registry) that are missing.
            Empty list means all required prerequisites are satisfied.
        """
        candidates = self._filter_entries(experiment_class)
        missing: list[str] = []
        for entry in candidates:
            if not self._is_present(entry):
                missing.append(entry.package)
        return missing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_entries(self, experiment_class: Optional[str]) -> list[PrereqEntry]:
        """Return entries relevant to *experiment_class* (or all if None)."""
        if experiment_class is None:
            return list(self._entries)
        return [
            e
            for e in self._entries
            if "all" in e.required_for or experiment_class in e.required_for
        ]

    def _is_present(self, entry: PrereqEntry) -> bool:
        """Run the check command for *entry* and return True if it is present.

        Environment-variable entries use the ``"env:VAR_NAME"`` prefix and check
        ``os.environ`` directly — no subprocess is spawned.

        All other entries run ``entry.check_command`` in a shell-free subprocess
        with a 5 s timeout.  Non-zero exit, FileNotFoundError, or TimeoutExpired
        all count as "missing" (returns ``False``).
        """
        cmd = entry.check_command.strip()

        # Environment-variable check — no subprocess needed
        if cmd.startswith("env:"):
            var_name = cmd[4:].strip()
            present = os.environ.get(var_name) == "1"
            if not present:
                _log.debug(
                    "HostPrereqRegistry: env var %s not set to '1' — marking missing",
                    var_name,
                )
            return present

        # Subprocess check
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except FileNotFoundError:
            _log.debug(
                "HostPrereqRegistry: check command not found for %s: %r",
                entry.package,
                cmd,
            )
            return False
        except subprocess.TimeoutExpired:
            _log.warning(
                "HostPrereqRegistry: check command timed out for %s: %r",
                entry.package,
                cmd,
            )
            return False
        except Exception as exc:  # pragma: no cover — unexpected errors are non-fatal
            _log.warning(
                "HostPrereqRegistry: unexpected error checking %s: %s",
                entry.package,
                exc,
            )
            return False


# ---------------------------------------------------------------------------
# Markdown table parser
# ---------------------------------------------------------------------------


def _parse_registry(path: Path) -> list[PrereqEntry]:
    """Parse the markdown table in *path* into a list of ``PrereqEntry`` objects.

    The table is expected to have the columns (in order):
        Package | Check Command | Install (Arch) | Install (Debian) | Required For

    Lines starting with ``|`` and not being the header or separator row are
    treated as data rows.  The ``Required For`` column is split on commas and
    each tag is stripped of whitespace.

    Parameters
    ----------
    path : Path
        Absolute path to the markdown file containing the table.

    Returns
    -------
    list[PrereqEntry]
        Parsed entries; empty if the file does not exist or has no data rows.
    """
    if not path.exists():
        _log.warning("HostPrereqRegistry: registry not found at %s", path)
        return []

    text = path.read_text(encoding="utf-8")
    entries: list[PrereqEntry] = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip the header and separator rows
        if re.match(r"^\|\s*[-:]+\s*\|", line) or re.match(
            r"^\|\s*Package\s*\|", line, re.IGNORECASE
        ):
            continue

        # Split cells and strip whitespace
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue

        package, check_cmd, install_arch, install_debian, required_for_raw = cells[:5]
        required_for = [t.strip() for t in required_for_raw.split(",") if t.strip()]

        if not package:
            continue

        entries.append(
            PrereqEntry(
                package=package,
                check_command=check_cmd,
                install_arch=install_arch,
                install_debian=install_debian,
                required_for=required_for,
            )
        )

    return entries
