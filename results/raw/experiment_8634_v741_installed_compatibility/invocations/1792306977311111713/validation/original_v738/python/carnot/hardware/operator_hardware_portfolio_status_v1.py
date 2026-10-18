"""Exp 2907 operator hardware portfolio status card.

Spec refs: REQ-HW-063, SCENARIO-HW-063.

This module is intentionally an aggregation layer. It does not touch any board,
toolchain, package manager, or bitstream because the operator needs a quick
portfolio readout from evidence that already exists. Each board gets one compact
row, and the citations preserve which upstream artifact supplied that row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2907_operator_hardware_portfolio_status_v1.json"
RUN_DATE = "20260523"
SCHEMA = "carnot.operator_hardware_portfolio_status.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2898_FILENAME = "experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
EXP2899_FILENAME = "experiment_2899_gatemate_a1_n16_ising_tile_bitstream_build_v1.json"
EXP2900_FILENAME = "experiment_2900_polarfire_carnot_dispatch_smoke_v1.json"
EXP2901_FILENAME = "experiment_2901_thrml_local_import_repair_v1.json"

BOARD_ORDER = ("kv260", "gatemate", "polarfire", "thrml")
REQUIRED_BOARD_FIELDS = {"state", "last_artifact", "next_step"}
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "per_board_status",
    "cited_upstream_artifacts",
    "duration_s",
}

KV260_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "board_transcript_path",
    "kv260_overlay_loaded",
    "bitstream_sha256",
    "per_seed_results",
)
GATEMATE_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "synth_succeeded",
    "place_and_route_succeeded",
    "bitstream_sha256",
)
POLARFIRE_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "polarfire_arch",
    "scorer_output_hash_verified",
    "no_fpga_fabric_claim",
    "duration_s",
)
THRML_FIELDS = (
    "honest_verdict",
    "thrml_import_succeeded",
    "thrml_version_installed",
    "parity_energy_delta",
    "metadata.field_principles.no_tsu_access_claim",
    "metadata.field_principles.no_hardware_acceleration_claim",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock for the portfolio aggregation.

    Tests point this at a temporary `results/` directory. In production the
    default reads the repository `results/` directory and writes the operator
    card beside the upstream artifacts.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def artifact_path(self, filename: str) -> Path:
        return self.output_dir() / filename

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class BoardDefinition:
    board: str
    experiment_id: str
    filename: str
    fields_imported: Sequence[str]
    ready_state: str
    ready_next_step: str
    ready_when: Callable[[Mapping[str, Any]], bool]

    def citation_path(self) -> str:
        return f"results/{self.filename}"


@dataclass(frozen=True)
class UpstreamArtifact:
    definition: BoardDefinition
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    missing: bool = False
    malformed: bool = False


def _kv260_ready(payload: Mapping[str, Any]) -> bool:
    return (
        _complete(payload)
        and payload.get("inference_substrate") == "hardware_smoke"
        and bool(payload.get("board_transcript_path"))
        and bool(payload.get("bitstream_sha256"))
        and len(_dict_rows(payload.get("per_seed_results"))) == 3
    )


def _gatemate_ready(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("bitstream_sha256")) and bool(payload.get("place_and_route_succeeded"))


def _polarfire_ready(payload: Mapping[str, Any]) -> bool:
    return (
        _complete(payload)
        and payload.get("inference_substrate") == "hardware_smoke"
        and payload.get("polarfire_arch") == "riscv64"
        and payload.get("scorer_output_hash_verified") is True
        and payload.get("no_fpga_fabric_claim") is True
    )


def _thrml_ready(payload: Mapping[str, Any]) -> bool:
    principles = payload.get("metadata", {}).get("field_principles", {})
    return (
        _complete(payload)
        and payload.get("thrml_import_succeeded") is True
        and payload.get("parity_energy_delta") is not None
        and principles.get("no_tsu_access_claim") is True
        and principles.get("no_hardware_acceleration_claim") is True
    )


BOARD_DEFINITIONS = {
    "kv260": BoardDefinition(
        board="kv260",
        experiment_id="exp2898",
        filename=EXP2898_FILENAME,
        fields_imported=KV260_FIELDS,
        ready_state="ready_live_latency_recorded",
        ready_next_step="Use as KV260 baseline; add same-basis CPU comparison before speedup claims.",
        ready_when=_kv260_ready,
    ),
    "gatemate": BoardDefinition(
        board="gatemate",
        experiment_id="exp2899",
        filename=EXP2899_FILENAME,
        fields_imported=GATEMATE_FIELDS,
        ready_state="ready_bitstream_built_pending_flash",
        ready_next_step="Stage operator review before flashing the generated bitstream.",
        ready_when=_gatemate_ready,
    ),
    "polarfire": BoardDefinition(
        board="polarfire",
        experiment_id="exp2900",
        filename=EXP2900_FILENAME,
        fields_imported=POLARFIRE_FIELDS,
        ready_state="ready_riscv64_cpu_dispatch_verified",
        ready_next_step="Treat as CPU-dispatch proof; FPGA fabric acceleration remains separate.",
        ready_when=_polarfire_ready,
    ),
    "thrml": BoardDefinition(
        board="thrml",
        experiment_id="exp2901",
        filename=EXP2901_FILENAME,
        fields_imported=THRML_FIELDS,
        ready_state="ready_software_parity_no_tsu_claim",
        ready_next_step="Use import/parity evidence; require TSU access before hardware claims.",
        ready_when=_thrml_ready,
    ),
}


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Build and optionally write the Exp 2907 operator status artifact."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    upstreams = _load_upstreams(active_config)
    failed_upstreams = _failed_upstreams(upstreams)
    honest_verdict = (
        "complete: operator_hardware_portfolio_status_aggregated"
        if not failed_upstreams
        else f"blocked_{failed_upstreams[0]}"
    )
    artifact = {
        "artifact": OUTPUT_FILENAME.removesuffix(".json"),
        "schema": SCHEMA,
        "spec": ["REQ-HW-063", "SCENARIO-HW-063"],
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_board_status": {
            board: _status_for_board(upstreams[board]) for board in BOARD_ORDER
        },
        "cited_upstream_artifacts": _citations(upstreams),
        "duration_s": _round_float(active_config.clock() - started_at),
        "no_new_board_execution": True,
        "no_new_hardware_claim": True,
        "claim_boundary": (
            "Aggregation-only operator card. Board execution, bitstream build, package "
            "repair, and TSU hardware claims remain exclusively in the cited upstream artifacts."
        ),
    }
    validate_artifact(artifact)
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def artifact_has_required_fields(artifact: Mapping[str, Any]) -> bool:
    """Return whether the card keeps the required schema and per-board shape."""

    per_board_status = artifact.get("per_board_status")
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and isinstance(artifact.get("honest_verdict"), str)
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and isinstance(per_board_status, dict)
        and tuple(per_board_status.keys()) == BOARD_ORDER
        and all(
            isinstance(row, dict) and set(row) == REQUIRED_BOARD_FIELDS
            for row in per_board_status.values()
        )
        and isinstance(artifact.get("cited_upstream_artifacts"), list)
        and isinstance(artifact.get("duration_s"), float)
        and artifact.get("duration_s", -1.0) >= 0.0
        and artifact.get("no_new_board_execution") is True
        and artifact.get("no_new_hardware_claim") is True
    )


def validate_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise a clear error before an incomplete operator card is written."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not artifact_has_required_fields(artifact):
        raise ValueError("Exp 2907 artifact failed required schema validation")
    return artifact


def _load_upstreams(config: ExperimentConfig) -> dict[str, UpstreamArtifact]:
    return {
        board: _read_upstream(definition, config.artifact_path(definition.filename))
        for board, definition in BOARD_DEFINITIONS.items()
    }


def _read_upstream(definition: BoardDefinition, path: Path) -> UpstreamArtifact:
    if not path.exists():
        return UpstreamArtifact(definition, path, {}, "", missing=True)
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return UpstreamArtifact(definition, path, {}, sha256, malformed=True)
    return UpstreamArtifact(definition, path, payload, sha256)


def _status_for_board(upstream: UpstreamArtifact) -> dict[str, str]:
    definition = upstream.definition
    if upstream.missing:
        return _board_row(
            "missing_upstream_artifact",
            definition,
            f"Produce {definition.experiment_id} before relying on the portfolio card.",
        )
    if upstream.malformed:
        return _board_row(
            "malformed_upstream_artifact",
            definition,
            f"Repair {definition.experiment_id} JSON before relying on the portfolio card.",
        )
    if definition.ready_when(upstream.payload):
        return _board_row(definition.ready_state, definition, definition.ready_next_step)
    blocked_state = _blocked_state(upstream.payload)
    if blocked_state:
        return _board_row(blocked_state, definition, _blocked_next_step(definition.board))
    return _board_row(
        "needs_operator_review",
        definition,
        "Read upstream verdict and decide whether to rerun or defer.",
    )


def _board_row(state: str, definition: BoardDefinition, next_step: str) -> dict[str, str]:
    return {
        "state": state,
        "last_artifact": definition.citation_path(),
        "next_step": next_step,
    }


def _blocked_state(payload: Mapping[str, Any]) -> str:
    verdict = str(payload.get("honest_verdict", ""))
    return verdict.split()[0].rstrip(":") if verdict.startswith("blocked") else ""


def _blocked_next_step(board: str) -> str:
    return {
        "gatemate": (
            "Provision nextpnr-gatemate, rerun n=16 build, and do not flash until "
            "a bitstream exists."
        ),
        "kv260": "Restore KV260 preconditions, rerun board harness, and preserve the transcript.",
        "polarfire": "Restore PolarFire SSH/riscv64/Python preconditions before dispatch.",
        "thrml": "Repair THRML import/parity evidence before making any TSU plan.",
    }[board]


def _failed_upstreams(upstreams: Mapping[str, UpstreamArtifact]) -> list[str]:
    failures: list[str] = []
    for board in BOARD_ORDER:
        item = upstreams[board]
        if item.missing:
            failures.append(f"missing_{item.definition.experiment_id}_artifact")
        if item.malformed:
            failures.append(f"malformed_{item.definition.experiment_id}_artifact")
    return failures


def _citations(upstreams: Mapping[str, UpstreamArtifact]) -> list[dict[str, Any]]:
    citations = []
    for board in BOARD_ORDER:
        item = upstreams[board]
        if item.missing or item.malformed:
            continue
        citations.append(
            {
                "experiment_id": item.definition.experiment_id,
                "path": item.definition.citation_path(),
                "fields_imported": list(item.definition.fields_imported),
                "sha256": item.sha256,
            }
        )
    return citations


def _complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("honest_verdict", "")).startswith(("complete:", "success:"))


def _dict_rows(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _round_float(value: float) -> float:
    return round(max(0.0, float(value)), 12)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI convenience wrapper.
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
