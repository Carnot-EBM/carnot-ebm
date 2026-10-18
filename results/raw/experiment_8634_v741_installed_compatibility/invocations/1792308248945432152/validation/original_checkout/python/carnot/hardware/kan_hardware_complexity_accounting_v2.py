"""Exp 2904 KV260-anchored KAN hardware complexity accounting.

Spec refs: REQ-KAN-2904, SCENARIO-KAN-2904.

This module aggregates evidence that already exists in the repository. It reads
the tiny KAN node count from Exp 2893 and the KV260 utilization report tied to
the Exp 2898 bitstream SHA. The output is an accounting artifact only: it does
not run Vivado, program the board, or claim that a KAN design was synthesized.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import hashlib
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.kan_hardware_complexity_accounting.v2"
ARTIFACT = "experiment_2904_kan_hardware_complexity_accounting_v2"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2893_REL_PATH = Path("results/experiment_2893_kan_hardware_complexity_accounting_v1.json")
EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
UTILIZATION_REPORT_REL_PATH = Path(
    "output/carnot_ising_v4_bd/project/carnot_ising_v4.runs/impl_1/"
    "carnot_ising_v4_bd_wrapper_utilization_placed.rpt"
)
BITSTREAM_REL_PATH = Path("output/carnot_ising_v4_bd/carnot_ising_v4.bit")
OUTPUT_REL_PATH = Path("results/experiment_2904_kan_hardware_complexity_accounting_v2.json")

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "kan_node_count",
    "kv260_lut_used",
    "kv260_bram_used",
    "kv260_dsp_used",
    "scaling_estimate_to_next_size",
    "cited_upstream_artifacts",
    "duration_s",
}


@dataclass(frozen=True)
class KV260Utilization:
    """The few Vivado resource counts needed for the Exp 2904 accounting row."""

    kv260_lut_used: int
    kv260_bram_used: int
    kv260_dsp_used: int
    kv260_lut_available: int
    kv260_bram_available: int
    kv260_dsp_available: int


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject missing, malformed, or array-valued evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected JSON object in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def parse_vivado_utilization_report(report_text: str) -> KV260Utilization:
    """Extract KV260 LUT, BRAM, and DSP used/available counts from a Vivado report."""

    lut_used, lut_available = _used_and_available(report_text, "CLB LUTs")
    bram_used, bram_available = _used_and_available(report_text, "Block RAM Tile")
    dsp_used, dsp_available = _used_and_available(report_text, "DSPs")
    return KV260Utilization(
        kv260_lut_used=lut_used,
        kv260_bram_used=bram_used,
        kv260_dsp_used=dsp_used,
        kv260_lut_available=lut_available,
        kv260_bram_available=bram_available,
        kv260_dsp_available=dsp_available,
    )


def extract_kan_node_count(exp2893: dict[str, Any]) -> int:
    """Read the current tiny-KAN node count from Exp 2893's PWA fixture summary."""

    for container_key in ("tiny_pwa_structure", "complexity_metrics"):
        container = exp2893.get(container_key)
        if isinstance(container, dict) and "unit_count" in container:
            node_count = int(container["unit_count"])
            if node_count <= 0:
                raise ValueError("Exp 2893 KAN node count must be positive")
            return node_count
    raise ValueError("missing Exp 2893 KAN node count")


def validate_exp2898_upstream(exp2898: dict[str, Any], expected_bitstream_sha256: str) -> None:
    """Ensure the latency artifact can be used as an upstream KV260 evidence anchor."""

    verdict = str(exp2898.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "success:")):
        raise ValueError("Exp 2898 honest_verdict is not terminal success evidence")
    if exp2898.get("inference_substrate") != "hardware_smoke":
        raise ValueError("Exp 2898 inference_substrate must be hardware_smoke")
    actual_sha = str(exp2898.get("bitstream_sha256", ""))
    if actual_sha != expected_bitstream_sha256:
        raise ValueError("Exp 2898 bitstream SHA does not match local bitstream evidence")


def build_scaling_estimate(
    *,
    kan_node_count: int,
    utilization: KV260Utilization,
    next_kan_node_count: int | None = None,
) -> dict[str, Any]:
    """Scale the upstream KV260 utilization linearly to the next KAN node size."""

    next_count = next_kan_node_count or kan_node_count * 2
    if kan_node_count <= 0 or next_count <= 0:
        raise ValueError("KAN node counts must be positive")
    lut_estimate = ceil(utilization.kv260_lut_used * next_count / kan_node_count)
    bram_estimate = ceil(utilization.kv260_bram_used * next_count / kan_node_count)
    dsp_estimate = ceil(utilization.kv260_dsp_used * next_count / kan_node_count)
    lut_pct = 100.0 * lut_estimate / utilization.kv260_lut_available
    bram_pct = 100.0 * bram_estimate / utilization.kv260_bram_available
    dsp_pct = 100.0 * dsp_estimate / utilization.kv260_dsp_available
    return {
        "current_kan_node_count": kan_node_count,
        "next_kan_node_count": next_count,
        "scaling_rule": "linear_double_from_current_node_count",
        "scaling_basis": (
            "full_exp2898_kv260_bitstream_utilization_divided_by_current_kan_node_count"
        ),
        "estimated_kv260_lut_used": lut_estimate,
        "estimated_kv260_bram_used": bram_estimate,
        "estimated_kv260_dsp_used": dsp_estimate,
        "kv260_lut_available": utilization.kv260_lut_available,
        "kv260_bram_available": utilization.kv260_bram_available,
        "kv260_dsp_available": utilization.kv260_dsp_available,
        "estimated_lut_utilization_pct": round(lut_pct, 6),
        "estimated_bram_utilization_pct": round(bram_pct, 6),
        "estimated_dsp_utilization_pct": round(dsp_pct, 6),
        "remaining_lut_after_next_size": utilization.kv260_lut_available - lut_estimate,
        "remaining_bram_after_next_size": utilization.kv260_bram_available - bram_estimate,
        "remaining_dsp_after_next_size": utilization.kv260_dsp_available - dsp_estimate,
        "fits_kv260_lut_budget": lut_estimate <= utilization.kv260_lut_available,
        "fits_kv260_bram_budget": bram_estimate <= utilization.kv260_bram_available,
        "fits_kv260_dsp_budget": dsp_estimate <= utilization.kv260_dsp_available,
        "claim_boundary": (
            "Conservative aggregation from upstream KV260 bitstream utilization; "
            "not a KAN synthesis, timing-closure result, or new board execution."
        ),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 2904 JSON artifact without writing it."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    exp2893_path = root_path / EXP2893_REL_PATH
    exp2898_path = root_path / EXP2898_REL_PATH
    report_path = root_path / UTILIZATION_REPORT_REL_PATH
    bitstream_path = root_path / BITSTREAM_REL_PATH

    exp2893 = load_json(exp2893_path)
    exp2898 = load_json(exp2898_path)
    bitstream_sha = _sha256(bitstream_path)
    validate_exp2898_upstream(exp2898, bitstream_sha)

    kan_node_count = extract_kan_node_count(exp2893)
    utilization = parse_vivado_utilization_report(report_path.read_text(encoding="utf-8"))
    duration_s = (time.perf_counter() if now_s is None else now_s) - started
    artifact = {
        "experiment": 2904,
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "status": "complete",
        "spec": ["REQ-KAN-2904", "SCENARIO-KAN-2904"],
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "honest_verdict": (
            "complete: KAN hardware complexity accounting v2 aggregated from Exp 2893 "
            "KAN shape and Exp 2898 KV260 bitstream utilization; no KAN synthesis claim"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kan_node_count": kan_node_count,
        "kv260_lut_used": utilization.kv260_lut_used,
        "kv260_bram_used": utilization.kv260_bram_used,
        "kv260_dsp_used": utilization.kv260_dsp_used,
        "kv260_utilization_source": UTILIZATION_REPORT_REL_PATH.as_posix(),
        "exp2898_bitstream_sha256": bitstream_sha,
        "exp2898_bitstream_sha256_source": str(exp2898.get("bitstream_sha256_source", "")),
        "scaling_estimate_to_next_size": build_scaling_estimate(
            kan_node_count=kan_node_count,
            utilization=utilization,
        ),
        "cited_upstream_artifacts": [
            _citation(
                "exp2893",
                exp2893_path,
                EXP2893_REL_PATH,
                ["tiny_pwa_structure.unit_count"],
            ),
            _citation(
                "exp2898",
                exp2898_path,
                EXP2898_REL_PATH,
                [
                    "honest_verdict",
                    "inference_substrate",
                    "bitstream_sha256",
                    "bitstream_sha256_source",
                    "kv260_overlay_loaded",
                ],
            ),
            _citation(
                "exp2898",
                report_path,
                UTILIZATION_REPORT_REL_PATH,
                ["CLB LUTs", "Block RAM Tile", "DSPs"],
            ),
            _citation("exp2898", bitstream_path, BITSTREAM_REL_PATH, ["sha256"]),
        ],
        "kan_synthesis_claim_made": False,
        "new_board_execution_claim_made": False,
        "claim_boundary": (
            "This is a deterministic aggregation from upstream artifacts. The KV260 counts "
            "come from the Exp 2898 bitstream metadata/report path, while the KAN node count "
            "comes from Exp 2893's tiny PWA fixture."
        ),
    }
    return validate_artifact(artifact)


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the Exp 2904 deliverable and return its path."""

    root_path = Path(root)
    out_path = _resolve(root_path, Path(output_path))
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether the artifact satisfies the required Exp 2904 safe schema."""

    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and isinstance(artifact.get("honest_verdict"), str)
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and isinstance(artifact.get("kan_node_count"), int)
        and artifact.get("kan_node_count", 0) > 0
        and isinstance(artifact.get("kv260_lut_used"), int)
        and isinstance(artifact.get("kv260_bram_used"), int)
        and isinstance(artifact.get("kv260_dsp_used"), int)
        and isinstance(artifact.get("scaling_estimate_to_next_size"), dict)
        and isinstance(artifact.get("cited_upstream_artifacts"), list)
        and bool(artifact.get("cited_upstream_artifacts"))
        and isinstance(artifact.get("duration_s"), float)
        and artifact.get("kan_synthesis_claim_made") is False
        and artifact.get("new_board_execution_claim_made") is False
    )


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Raise a clear error if a caller tries to write an incomplete artifact."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not artifact_has_required_fields(artifact):
        raise ValueError("Exp 2904 artifact failed required schema validation")
    return artifact


def _used_and_available(report_text: str, label: str) -> tuple[int, int]:
    """Read one Vivado utilization table row by label."""

    for line in report_text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip().replace("*", "") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0] == label:
            return _int_cell(cells[1]), _int_cell(cells[4])
    raise ValueError(f"missing utilization row: {label}")


def _int_cell(value: str) -> int:
    """Parse the integer fields used by Vivado utilization tables."""

    return int(value.replace(",", ""))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local upstream artifact file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _citation(
    experiment_id: str,
    absolute_path: Path,
    relative_path: Path,
    fields_imported: list[str],
) -> dict[str, Any]:
    """Build a stable citation entry for one upstream file."""

    return {
        "experiment_id": experiment_id,
        "artifact_path": relative_path.as_posix(),
        "fields_imported": fields_imported,
        "sha256": _sha256(absolute_path),
    }


def _resolve(root: Path, path: Path) -> Path:
    """Resolve a repository-relative path against the chosen root."""

    return path if path.is_absolute() else root / path


def main() -> None:  # pragma: no cover - CLI convenience wrapper.
    print(write_artifact())


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper.
    main()
