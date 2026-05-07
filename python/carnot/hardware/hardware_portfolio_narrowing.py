"""Exp 1460 hardware portfolio narrowing artifact builder.

Carnot has accumulated more hardware branches than one milestone can advance
honestly. This module records the 20260507 scope decision as structured data:
only tracks with immediate research value and usable local readiness remain
active, while speculative or blocked hardware paths get explicit reopen gates.

Spec refs: REQ-HW-049, SCENARIO-HW-049.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260507"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1460_hardware_portfolio_narrowing.json"
)
DECISION_NOTE_REL = Path("docs/research-notes/hardware_portfolio_narrowing.md")
ARCHITECTURE_REL = Path("_bmad/architecture.md")
HARDWARE_WISHLIST_REL = Path("research-hardware-wishlist.md")

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "active_hardware_tracks",
    "active_hardware_track_count",
    "deferred_hardware_tracks",
    "architecture_updated",
    "hardware_wishlist_updated",
    "decision_note_path",
    "honest_verdict",
}

DOC_MARKERS = (
    "Active hardware tracks (Exp 1460)",
    "Deferred hardware tracks (Exp 1460)",
)

ACTIVE_HARDWARE_TRACKS: list[dict[str, Any]] = [
    {
        "track_id": "dual_rtx3090_live_sota_runtime",
        "label": "Dual RTX 3090 CUDA local SOTA runtime repair",
        "scope": (
            "Local GGUF/CUDA runtime repair for cached Qwen/Gemma flagship "
            "models on the two visible RTX 3090 GPUs."
        ),
        "evidence": [
            "Exp 1442 saw two NVIDIA RTX 3090 GPUs with roughly 24 GB free each.",
            "Exp 1442 found cached Qwen3.6-35B-A3B and Gemma4-31B GGUF models.",
            "Exp 1442 blocked on llama.cpp CUDA library loading, not on GPU visibility.",
        ],
        "immediate_research_value": (
            "Unblocks headline-eligible live repair and full-pipeline benchmark "
            "evidence without adding another speculative accelerator."
        ),
        "readiness": (
            "Hardware is locally visible; runtime is not ready because "
            "libcudart.so.12 was missing for llama_cpp in Exp 1442."
        ),
        "claim_boundary": (
            "No live SOTA inference claim until a smoke run records "
            "usable_response=true on the target model/runtime."
        ),
    },
    {
        "track_id": "kv260_discrete_sb_rtl_sim",
        "label": "KV260/FPGA Discrete SB RTL lint and simulation",
        "scope": (
            "Keep KV260 work active only at the source-level RTL, lint, and "
            "simulator-evidence layer."
        ),
        "evidence": [
            "Exp 1451 found discrete_sb_256.v and its testbench present.",
            "Exp 1451 completed Verilator lint and Icarus simulation.",
            "Exp 1451 recorded Vivado unavailable and hardware_execution_performed=false.",
        ],
        "immediate_research_value": (
            "Preserves the FPGA sampler path with local, repeatable HDL evidence "
            "while avoiding fabricated board-latency claims."
        ),
        "readiness": (
            "RTL lint/simulation is ready with local Verilator/Icarus/Yosys; "
            "board execution remains blocked until Vivado and a bitfile run exist."
        ),
        "claim_boundary": (
            "No KV260 board execution, bitfile, or latency claim until Vivado "
            "synthesis, bitfile flashing, and board commands are captured."
        ),
    },
    {
        "track_id": "thrml_tsu_compatibility_sim",
        "label": "THRML/Extropic TSU compatibility simulation",
        "scope": (
            "Keep the Extropic path active only as THRML/JAX compatibility, "
            "sampler-interface parity, and CPU simulation accounting."
        ),
        "evidence": [
            "Research references describe THRML as the near-term public software surface.",
            "Research references state Z1/XTR-0 remain strategic without local hardware access.",
            "Prior THRML audits require hardware_claim_allowed=false without TSU execution.",
        ],
        "immediate_research_value": (
            "Maintains a portable TSU-facing sampler abstraction without blocking "
            "the milestone on unavailable external hardware."
        ),
        "readiness": (
            "Documentation and compatibility-audit paths are available; Extropic "
            "hardware access is not locally evidenced."
        ),
        "claim_boundary": (
            "No Extropic hardware access, Z1/XTR-0 execution, or TSU latency claim "
            "until an authenticated hardware run is captured."
        ),
    },
]

DEFERRED_HARDWARE_TRACKS: list[dict[str, str]] = [
    {
        "track_id": "kv260_board_execution",
        "label": "KV260 board execution and latency claims",
        "reason": "Vivado is absent, no bitfile was produced, and no board commands ran.",
        "reopen_condition": (
            "Reopen when Vivado synthesis produces a bitfile, CARNOT_KV260_BITFILE "
            "points to it, and a KV260/PYNQ board run records real latency."
        ),
    },
    {
        "track_id": "amd_xdna_strix_npu",
        "label": "AMD Strix/XDNA NPU acceleration",
        "reason": (
            "VitisAI and IRON paths remain blocked by missing packages or wheels; "
            "no NPU kernel produced acceleration evidence."
        ),
        "reopen_condition": (
            "Reopen when mlir-aie or AMD's VitisAI onnxruntime wheel is installed "
            "and a local NPU benchmark reports a real speedup."
        ),
    },
    {
        "track_id": "extropic_z1_xtr0_hardware",
        "label": "Extropic Z1/XTR-0 hardware execution",
        "reason": "No local Extropic hardware or authenticated execution transcript exists.",
        "reopen_condition": (
            "Reopen when Carnot has early-access credentials or hardware and a THRML "
            "or SDK run records model, device, latency, and sample-quality evidence."
        ),
    },
    {
        "track_id": "photonic_ising_cluster",
        "label": "Photonic or optical Ising-machine substrates",
        "reason": "The path is architectural context only; Carnot has no local optical hardware.",
        "reopen_condition": (
            "Reopen when a concrete photonic provider, simulator-to-hardware API, "
            "or collaborator run can evaluate Carnot Ising cases."
        ),
    },
    {
        "track_id": "dwave_qpu_cloud",
        "label": "D-Wave QPU cloud experiments",
        "reason": (
            "Cloud QPU access is not the current blocker for repair/runtime evidence "
            "and would add another branch during scope reduction."
        ),
        "reopen_condition": (
            "Reopen when a specific Ising/QUBO benchmark cannot be answered by CPU, "
            "GPU, or KV260 simulation and a Leap token plus budget are available."
        ),
    },
    {
        "track_id": "large_fpga_alveo_agilex",
        "label": "Large production FPGA boards",
        "reason": "Production FPGA purchases do not help until the KV260 RTL path closes.",
        "reopen_condition": (
            "Reopen after KV260 lint, synthesis, and board execution produce a "
            "measured sampler result that justifies scaling fabric capacity."
        ),
    },
    {
        "track_id": "rx7900xtx_egpu",
        "label": "RX 7900 XTX Thunderbolt eGPU path",
        "reason": "The current local CUDA RTX 3090 pair is more ready for SOTA runtime repair.",
        "reopen_condition": (
            "Reopen when the RTX CUDA runtime path is exhausted or when ROCm/JAX on "
            "the eGPU is connected and verified with a real Carnot benchmark."
        ),
    },
]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """REQ-HW-049: write a schema-shaped startup marker before final scoring."""

    artifact: dict[str, Any] = {
        "status": "in_progress",
        "active_hardware_tracks": [],
        "active_hardware_track_count": 0,
        "deferred_hardware_tracks": [],
        "architecture_updated": False,
        "hardware_wishlist_updated": False,
        "decision_note_path": "",
        "honest_verdict": (
            "Portfolio narrowing is in progress; no active/deferred decision has "
            "been finalized yet."
        ),
    }
    return _write_json(Path(path), artifact)


def build_portfolio_decision(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 1460 decision from fixed evidence and doc markers.

    The active/deferred lists are deliberately conservative. They encode what
    the repository evidence can support today, not what the long-term hardware
    roadmap would ideally include.
    """

    root = Path(project_root)
    active_tracks = deepcopy(ACTIVE_HARDWARE_TRACKS)
    deferred_tracks = deepcopy(DEFERRED_HARDWARE_TRACKS)
    artifact: dict[str, Any] = {
        "status": "complete",
        "schema_version": 1,
        "experiment_id": 1460,
        "run_date": run_date,
        "decision_basis": (
            "Immediate research value plus current readiness from Exp 1442, "
            "Exp 1451, hardware wishlist status, and Extropic/THRML notes."
        ),
        "sources_read": [
            "CODEX.md",
            "CLAUDE.md",
            "_bmad/architecture.md",
            "research-hardware-wishlist.md",
            "research-references.md",
            "results/experiment_1442_live_sota_repair_runtime_preflight.json",
            "results/experiment_1451_discrete_sb_rtl_lint_sim_rerun.json",
            "ops/known-issues.md",
        ],
        "active_hardware_tracks": active_tracks,
        "active_hardware_track_count": len(active_tracks),
        "deferred_hardware_tracks": deferred_tracks,
        "architecture_updated": _doc_has_markers(root / ARCHITECTURE_REL),
        "hardware_wishlist_updated": _doc_has_markers(root / HARDWARE_WISHLIST_REL),
        "decision_note_path": DECISION_NOTE_REL.as_posix(),
        "honest_verdict": (
            "active_tracks_narrowed_to_3_with_no KV260 board, Extropic, NPU, "
            "or photonic execution claim"
        ),
    }
    return artifact


def run_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    run_date: str = DEFAULT_RUN_DATE,
) -> dict[str, Any]:
    """Write the in-progress marker, validate docs, and write the final artifact."""

    output = Path(output_path)
    write_in_progress_artifact(output)
    artifact = build_portfolio_decision(project_root=project_root, run_date=run_date)
    validate_artifact(artifact)
    return _write_json(output, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1460 schema and the conservative claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    active_tracks = artifact["active_hardware_tracks"]
    deferred_tracks = artifact["deferred_hardware_tracks"]
    if not isinstance(active_tracks, list):
        raise ValueError("active_hardware_tracks must be a list")
    if artifact["active_hardware_track_count"] != len(active_tracks):
        raise ValueError("active_hardware_track_count must equal active track count")
    if not 2 <= artifact["active_hardware_track_count"] <= 3:
        raise ValueError("active_hardware_track_count must be between 2 and 3")
    for track in active_tracks:
        _validate_active_track(track)
    if not isinstance(deferred_tracks, list) or not deferred_tracks:
        raise ValueError("deferred_hardware_tracks must be a non-empty list")
    for track in deferred_tracks:
        if not str(track.get("reopen_condition", "")).strip():
            raise ValueError(f"deferred track {track.get('track_id')} lacks reopen_condition")
    if artifact["architecture_updated"] is not True:
        raise ValueError("architecture_updated must be true before completion")
    if artifact["hardware_wishlist_updated"] is not True:
        raise ValueError("hardware_wishlist_updated must be true before completion")
    if artifact["decision_note_path"] != DECISION_NOTE_REL.as_posix():
        raise ValueError("decision_note_path must point to the Exp 1460 decision note")
    verdict = str(artifact["honest_verdict"])
    for token in ("no KV260 board", "Extropic", "NPU", "photonic execution claim"):
        if token not in verdict:
            raise ValueError(f"honest_verdict must preserve claim boundary: {token}")


def _validate_active_track(track: Mapping[str, Any]) -> None:
    required = {
        "track_id",
        "label",
        "scope",
        "evidence",
        "immediate_research_value",
        "readiness",
        "claim_boundary",
    }
    missing = required - set(track)
    if missing:
        raise ValueError(f"active track {track.get('track_id')} missing {sorted(missing)}")
    if not isinstance(track["evidence"], list) or not track["evidence"]:
        raise ValueError(f"active track {track['track_id']} must include evidence")
    if not str(track["claim_boundary"]).startswith("No "):
        raise ValueError(f"active track {track['track_id']} must start claim_boundary with No")


def _doc_has_markers(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in DOC_MARKERS)


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    """CLI entry point used for manual artifact refresh."""

    artifact = run_experiment()
    print(
        artifact["active_hardware_track_count"],
        [track["track_id"] for track in artifact["active_hardware_tracks"]],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
