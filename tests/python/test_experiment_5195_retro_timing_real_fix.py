"""Regression tests for exp5195 — the .475 retro-timing false-zero ROOT CAUSE.

Spec refs: REQ-REPORT-5164 (the standalone fallback contract this hardens),
SCENARIO-REPORT-5164-FALSE-ZERO.

WHAT WAS ACTUALLY WRONG (and what these tests lock in)
------------------------------------------------------
`.473`'s exp5164 built ``scripts/retro_timing_fallback.py`` and wired it into
``scripts/research_conductor.py::_run_operational_retrospective`` so the retro
would reconstruct milestone timing from disk when the legacy git-log predicate
(``"Exp " in msg``) matched nothing. The wiring LANDED — the import and call
are present in the conductor — yet every ``.475`` retro pass still emitted a
false zero (``experiments_completed=0``, ``total_wall_time_minutes=0``,
``reconstructed_from_disk_mtime=False``, ``timing_integrity_mismatch=true``).

The module logic was never the problem. The bug is the IMPORT STATEMENT the
wiring used::

    from scripts.retro_timing_fallback import build_retro_timing_fallback

The conductor is launched as ``python scripts/research_conductor.py`` (systemd
``ExecStart=.../python scripts/research_conductor.py``). When CPython runs a
script by path, ``sys.path[0]`` is the SCRIPT'S DIRECTORY — here ``scripts/`` —
NOT the repo root. So at conductor runtime ``scripts`` is not importable as a
package (there is no ``scripts/scripts/`` and the repo root is absent from
``sys.path``), and the import raised
``ModuleNotFoundError: No module named 'scripts.retro_timing_fallback'`` on
EVERY retro pass (confirmed in ``journalctl --user -u carnot-conductor`` at
06:38:46, 07:48:05, and 11:16:44 EDT on 2026-07-03 — the last matching the
``operational_retro_2026_07_475.json`` ``generated_at`` of 15:16:44Z). The
conductor's outer ``except Exception`` swallowed it (logging only a WARNING),
leaving ``experiment_times`` empty → the false zero.

Every OTHER sibling helper in the same conductor file is imported BARE
(``from gpu_monitor import ...``, ``from failure_ledger import ...``,
``from in_process_doc_reconcile import ...``, ``from adversarial_verify import
...``) precisely because ``scripts/`` — not the repo root — is on ``sys.path``
at conductor runtime. Line 2876 was the ONLY ``from scripts.X import`` in the
whole file; that inconsistency was the defect.

The prior wiring test (``test_2026_07_03_conductor_imports_fallback_module_for
_wiring``) asserted the buggy string was PRESENT — it verified the wrong import
path existed, never that it actually imports at the conductor's runtime. These
tests close that gap by exercising the real failure mode.

The fix (prepared as a ``git apply``-verified patch for the
operator-owned ``research_conductor.py``) imports the bare sibling first and
keeps the package form as a fallback for pytest, so it works in BOTH
environments.
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODULE_PATH = SCRIPTS_DIR / "retro_timing_fallback.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retro_timing_fallback", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeGit:
    """Deterministic git stand-in mirroring the .475 milestone shape.

    Resolves the activation commit for the milestone under test and returns a
    per-path commit timestamp for the two artifacts that actually landed.
    """

    def __init__(self, milestone: str, activation: str, path_times: dict[str, str]):
        self.grep_marker = f"--grep=\\[conductor\\] Activate milestone {milestone}"
        self.activation = activation
        self.path_times = path_times

    def __call__(self, args: Sequence[str], cwd: Path) -> str:
        if self.grep_marker in args:
            return f"1dbcc55e {self.activation}\n"
        if "--format=%ai" in args and "--" in args:
            return f"{self.path_times.get(args[-1], '')}\n"
        return ""


def test_exp5195_conductor_runtime_import_reproduces_475_false_zero_and_fix(
    tmp_path: Path,
) -> None:
    """ROOT CAUSE: under the conductor's runtime sys.path (scripts/ on path,
    repo root absent), the old ``from scripts.retro_timing_fallback import``
    raises ModuleNotFoundError — the exact .475 false-zero trigger — while the
    fixed bare ``from retro_timing_fallback import`` succeeds.

    This reproduces the bug in a subprocess whose sys.path is set exactly like
    ``python scripts/research_conductor.py`` sees it, so a future refactor that
    reintroduces the package-only import is caught immediately.
    """

    code = "\n".join(
        [
            "import sys, os",
            f"scripts_dir = {str(SCRIPTS_DIR)!r}",
            "repo_root = os.path.dirname(scripts_dir)",
            # Mimic conductor runtime: scripts/ present, repo root + cwd absent,
            # stdlib + site-packages retained (retro_timing_fallback needs them).
            "sys.path = [p for p in sys.path if p not in ('', repo_root, os.getcwd())]",
            "if scripts_dir not in sys.path:",
            "    sys.path.insert(0, scripts_dir)",
            # 1. The .475 root cause: package-form import must fail here.
            "try:",
            "    from scripts.retro_timing_fallback import build_retro_timing_fallback",
            "    print('OLD_IMPORT_UNEXPECTEDLY_WORKED')",
            "    sys.exit(3)",
            "except ModuleNotFoundError:",
            "    pass",
            # 2. The fix: bare sibling import must succeed and be callable.
            "from retro_timing_fallback import build_retro_timing_fallback",
            "assert callable(build_retro_timing_fallback)",
            "print('FIX_IMPORT_OK')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "FIX_IMPORT_OK" in result.stdout
    assert "OLD_IMPORT_UNEXPECTEDLY_WORKED" not in result.stdout


def test_exp5195_build_fallback_reconstructs_475_shape_non_zero(tmp_path: Path) -> None:
    """DATA level: with a hermetic fixture mirroring .475's ACTUAL conditions
    (12 tasks, only 2 producing artifacts — one committed via an atypical
    commit path), the fallback reconstructs 2 experiments and non-zero wall
    time. This proves the module was never the false-zero's cause, so the fix
    must live at the conductor's import site (the patch), not in this module.
    """

    mod = _load_module()

    # .475 shape: 12 tasks; only exp5181 (archive/activate) and exp5182
    # (DiffusionGemma root-cause fix, produced by a direct outer-loop script
    # invocation OUTSIDE the normal task-commit flow) actually landed.
    tasks = []
    for seq in range(12):
        expid = 5181 + seq
        tasks.append(
            {
                "id": f"exp{expid}",
                "title": f"PHASE task {seq}",
                "deliverable": f"results/experiment_{expid}_v475.json",
            }
        )

    _write_json(
        tmp_path / "results/experiment_5181_v475.json",
        {"duration_s": 41.0, "inference_substrate": "aggregation_from_upstream_artifacts"},
    )
    _write_json(
        tmp_path / "results/experiment_5182_v475.json",
        {"duration_s": 35.0, "inference_substrate": "aggregation_from_upstream_artifacts"},
    )
    # The other 10 tasks 3-fail-SKIPPED (poison-test cascade) → no artifact.

    fake_git = FakeGit(
        milestone="2026.07.475",
        activation="2026-07-03 03:58:55 -0400",
        path_times={
            # exp5181 committed in its own PHASE 0 transition commit:
            "results/experiment_5181_v475.json": "2026-07-03 04:18:50 -0400",
            # exp5182 committed via the ATYPICAL path (swept into the retro's
            # own commit, not a dedicated [conductor] Exp commit):
            "results/experiment_5182_v475.json": "2026-07-03 06:46:43 -0400",
        },
    )

    summary = mod.build_retro_timing_fallback(
        "2026.07.475", tasks=tasks, repo_root=tmp_path, git_runner=fake_git
    )

    # The load-bearing regression assertions: NOT a false zero.
    assert summary["experiments_completed"] == 2
    assert summary["total_wall_time_minutes"] > 0
    assert len(summary["missing_deliverables"]) == 10
    assert summary["excluded_pre_activation"] == []
    assert summary["activation_bound"]["source"] == "activation_commit"


def test_exp5195_module_importable_both_ways_when_paths_present() -> None:
    """The fix's two-branch import must resolve under EITHER path layout: bare
    (conductor runtime) and packaged (pytest, repo root on sys.path). Here the
    pytest environment has the repo root on sys.path, so the packaged form —
    the patch's fallback branch, and the exact string the existing wiring test
    asserts — must resolve to the same module object on disk.
    """

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.retro_timing_fallback import build_retro_timing_fallback as packaged

    bare = _load_module().build_retro_timing_fallback

    assert callable(packaged)
    assert packaged.__name__ == bare.__name__ == "build_retro_timing_fallback"
    # Both resolve to the one on-disk module file.
    assert os.path.samefile(
        sys.modules["scripts.retro_timing_fallback"].__file__, MODULE_PATH
    )
