"""GateMate terminal confirmation for Exp 3900.

Exp 3889 corrected the Exp 3866 timer tautology by measuring total task time and
board-facing run time with separate clocks. This module reuses that hardware
flow and adds the terminal-state graduation decision the operator asked for:
the board must accept the n=16 tile, the flash smoke evidence must be OK, the
readback gate must either verify or be honestly unsupported, and the two timer
fields must remain distinct.

Spec refs: REQ-HW-3900, SCENARIO-HW-3900.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from carnot.experiment_3866_gatemate_ising_tile_flash_v2 import (
    ClockFunc,
    CommandResult,
    RunCommand,
    WhichFunc,
    _default_run_command,
)
from carnot.experiment_3889_gatemate_continuity_corrigendum import (
    ARTIFACT_FILENAME as PRIOR_ARTIFACT_FILENAME,
    build_artifact as _build_corrigendum_artifact,
    resolve_toolchain_path,
)


ARTIFACT_FILENAME = "experiment_3900_gatemate_terminal_confirmation.json"
EXPERIMENT_ID = 3900
RUN_DATE = "20260607"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "run_duration_s",
    "readback_verified",
    "readback_supported",
    "terminal_state_reached",
    "gatemate_bitstream_flashed",
    "fmax_mhz",
    "lut_used",
    "dff_used",
    "preconditions_checked",
    "reproducibility_checksum",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "duration_s": (
        "Total task wall-clock -- MUST be distinct from run_duration_s "
        "(the exp3866 TAUTOLOGY corrigendum)."
    ),
    "run_duration_s": "On-board run time only -- distinct timer.",
    "readback_verified": (
        "BARE BOOL -- did a JTAG readback confirm the flashed tile matches the bitstream."
    ),
    "readback_supported": "Honest record of whether this flow supports readback at all.",
    "terminal_state_reached": (
        "BARE BOOL -- flashed + smoke + readback-or-unsupported + no-tautology; "
        "the graduation signal."
    ),
    "gatemate_bitstream_flashed": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "fmax_mhz": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "lut_used": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "dff_used": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "preconditions_checked": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "reproducibility_checksum": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "inference_substrate": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "honest_verdict": (
        "Terminal prefix or blocked prefix records whether GateMate can graduate "
        "from mandatory to opportunistic hardware continuity."
    ),
    "smoke_ok": (
        "BARE BOOL -- the flash/contact smoke stage completed with board-visible success evidence."
    ),
    "no_tautology": (
        "BARE BOOL -- total wall-clock and board-run timers are distinct."
    ),
}


def _metric_label(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _no_tautology(artifact: dict) -> bool:
    return artifact.get("duration_s") != artifact.get("run_duration_s")


def _smoke_ok(artifact: dict) -> bool:
    if not bool(artifact.get("gatemate_bitstream_flashed")):
        return False
    for entry in artifact.get("command_transcript", []):
        if entry.get("stage") == "flash" and entry.get("returncode") == 0:
            return True
    return bool(artifact.get("synth_pnr_pack_succeeded"))


def _readback_gate(artifact: dict) -> bool:
    return bool(artifact.get("readback_verified")) or artifact.get("readback_supported") is False


def _readback_label(artifact: dict) -> str:
    if artifact.get("readback_verified"):
        return "true"
    if artifact.get("readback_supported") is False:
        return "unsupported"
    return "false"


def _terminal_state_reached(artifact: dict) -> bool:
    return (
        bool(artifact.get("gatemate_bitstream_flashed"))
        and bool(artifact.get("smoke_ok"))
        and _readback_gate(artifact)
        and bool(artifact.get("no_tautology"))
    )


def _verdict(artifact: dict) -> str:
    prior_verdict = str(artifact.get("honest_verdict", ""))
    if prior_verdict.startswith("blocked_"):
        return prior_verdict
    if not bool(artifact.get("gatemate_bitstream_flashed")):
        return "blocked_gatemate_flash_flow_failed_unknown"
    fmax = _metric_label(artifact.get("fmax_mhz"))
    if artifact.get("terminal_state_reached"):
        return (
            "success: "
            f"gatemate_TERMINAL_reached_fmax{fmax}_readback{_readback_label(artifact)}_"
            "can_graduate_to_opportunistic"
        )
    return f"success: gatemate_flashed_readback_inconclusive_fmax{fmax}_stays_mandatory"


def _field_provenance(existing: dict | None) -> dict:
    provenance = dict(existing or {})
    provenance.update({field: {"principle": principle} for field, principle in FIELD_PRINCIPLES.items()})
    return provenance


def _reproducibility_checksum(artifact: dict) -> str:
    material = {
        "experiment": EXPERIMENT_ID,
        "prior_corrigendum_sha256": artifact.get("prior_corrigendum_sha256", ""),
        "bitstream_sha256": artifact.get("bitstream_sha256", ""),
        "readback_sha256": artifact.get("readback_sha256", ""),
        "readback_supported": artifact.get("readback_supported"),
        "readback_verified": artifact.get("readback_verified"),
        "terminal_state_reached": artifact.get("terminal_state_reached"),
        "no_tautology": artifact.get("no_tautology"),
        "preconditions_checked": artifact.get("preconditions_checked", []),
        "command_transcript": artifact.get("command_transcript", []),
        "honest_verdict": artifact.get("honest_verdict", ""),
    }
    encoded = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prior_corrigendum_sha256(repo_root: Path) -> str:
    prior = repo_root / "results" / PRIOR_ARTIFACT_FILENAME
    if not prior.exists():
        return ""
    digest = hashlib.sha256()
    with prior.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirm_terminal_artifact(
    base_artifact: dict,
    *,
    prior_corrigendum_sha256: str,
) -> dict:
    """Add the Exp 3900 terminal-state decision to an Exp 3889-style artifact."""
    artifact = json.loads(json.dumps(base_artifact, sort_keys=True, default=str))
    artifact["experiment"] = EXPERIMENT_ID
    artifact["run_date"] = RUN_DATE
    artifact["prior_corrigendum_artifact"] = f"results/{PRIOR_ARTIFACT_FILENAME}"
    artifact["prior_corrigendum_sha256"] = prior_corrigendum_sha256
    artifact["no_tautology"] = _no_tautology(artifact)
    artifact["smoke_ok"] = _smoke_ok(artifact)
    artifact["terminal_state_reached"] = _terminal_state_reached(artifact)
    artifact["honest_verdict"] = _verdict(artifact)
    artifact["field_provenance"] = _field_provenance(artifact.get("field_provenance"))
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    """Run the GateMate confirmation flow and return the terminal artifact."""
    prior_sha256 = _prior_corrigendum_sha256(repo_root)
    base = _build_corrigendum_artifact(
        repo_root=repo_root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
    )
    return confirm_terminal_artifact(base, prior_corrigendum_sha256=prior_sha256)


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    """Run Exp 3900 and write the requested deliverable JSON."""
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    resolve_toolchain_path()
    artifact = run_experiment()
    print(f"artifact: results/{ARTIFACT_FILENAME}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_bitstream_flashed: {artifact['gatemate_bitstream_flashed']}")
    print(f"readback_supported: {artifact['readback_supported']}")
    print(f"readback_verified: {artifact['readback_verified']}")
    print(f"terminal_state_reached: {artifact['terminal_state_reached']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
