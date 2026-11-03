"""REQ-REPORT-8722: bind V747 slots to their declared bytes and original terminal checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v746_capstone_inputs as previous
from carnot.reporting import v747_contract_methods as contract_source
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
__all__ = ["previous", "contract_source"]
ROOT = previous.ROOT
NAME, TASK, MILESTONE = "experiment_8722_v747_capstone", "exp8722-capstone", "2026.11.747"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v747_capstone_8722.py"]
OWNED = [
    f"python/carnot/reporting/v747_capstone{s}.py"
    for s in ("_inputs", "_measure", "_reduce", "_report", "")
] + [CLI]
MODEL_SPECS: list[Json] = []
DESIGN = contract_source.DESIGN
atomic_json, reference, load, verify, freeze, digest, sha256_file, observe = (
    previous.atomic_json,
    previous.reference,
    previous.load,
    previous.verify,
    previous.freeze,
    previous.digest,
    previous.sha256_file,
    previous.observe,
)
PINS = {
    8708: "26870ffbb0e2ab7c62f3d629802ff35bdbb52f0e02931398032d9d0d4ae52d38",
    8709: "076aeef271b1ac5da33adcc6dafa5038d287fe5c83126193e8d95e816c6ffecb",
    8712: "70d65744e10471aa67a77671dbb894b766c587bd59988e0600766cc43765829f",
    8715: "1e205986c906fecbb1578c0df30ea828a31d7c8b42c8cfd736868f98bd7965b2",
    8716: "7152eb60177c873d646ec02305b63819269e7c4410471f7ea5a00c28e7c77c13",
    8717: "938d2cd6a28f8b0ae1ee1a551199a8ab7c3b29de91dca12f2d83aa8d1379fef6",
    8719: "9d27b1f8ca99b50d0452d7e1d204c18223a16b7eec36208d8bc5c44830c6f1ac",
    8720: "bcef7715e5181dd49b89cf019a184465783803a6ff7e51b3d5a79bb182a308f7",
    8721: "b0ae72a7347ca763362f4cb80158326cf95a855781f04093696e0e3b693c8cd6",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase boundaries without creating artificial execution time."""
    print(f"[exp8722] phase={phase} completed={completed} pending={pending}", flush=True)


def contract(root: Path) -> Json:
    """Reuse the original full-object binder and retain identities even on external absence."""
    result = contract_source.authority(root)
    table, tasks = parse_design((ROOT / DESIGN).read_text(), milestone=MILESTONE)
    return dict(
        result, tasks=tasks, visible_table=table, canonical_tasks_sha256=contract_source.PIN
    )


# The retained driver expects a dependency with an authority(root) entry point.
authority = SimpleNamespace(authority=contract)


def slot(task: Json, raw: Path, lines: list[str], root: Path = ROOT) -> Json:
    """Fallback conductor records explain scheduling only; scientific operands stay exact."""
    row = previous.resolve(task, raw, lines, root)
    declared = row["declared_reference"]
    number = row["experiment_id"]
    if row["producer_executed"] and (
        row["primary_reference"]["path"] != declared["path"]
        or (number in PINS and declared["sha256"] != "sha256:" + PINS[number])
    ):
        row.update(
            producer_qualified=False,
            terminal_authenticated=False,
            scientific_result_measured=False,
            verdict_class="disqualified",
        )
        row["authentication_errors"].append("exact_declared_versioned_primary")
    return dict(row)
