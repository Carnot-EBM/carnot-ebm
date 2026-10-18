"""Exp 2906 FR-11 KV260 replay dispatch pilot.

This module does not run a new model or a new board benchmark. It builds the
pilot artifact by reading the CPU replay artifacts and the live KV260 board
artifact already produced this milestone. The goal is deliberately narrow:
show that the replay side has clean inputs and that a KV260 UIO dispatch path
has live-board evidence. Performance comparison is left for a later experiment
with a same-basis CPU baseline.

Spec: REQ-LEARN-2906,
      SCENARIO-LEARN-2906,
      SCENARIO-LEARN-2906-BLOCKED.
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
OUTPUT_FILENAME = "experiment_2906_fr11_hardware_accelerated_replay_pilot_v1.json"
EXP2882_FILENAME = "experiment_2882_fr11_recmem_replay_scaleup_v1.json"
EXP2887_FILENAME = "experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json"
EXP2898_FILENAME = "experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "dispatch_path_validated",
    "cited_upstream_artifacts",
    "duration_s",
}

CPU_FIELDS = [
    "honest_verdict",
    "recmem_replay_scaleup_ready",
    "n_examples",
    "target_examples_met",
    "selected_example_ids",
    "live_llm_called",
    "model_weights_mutated",
    "run_date",
]
CORRIGENDUM_FIELDS = [
    "honest_verdict",
    "fr11_scaleup_clean",
    "best_policy",
    "n_examples",
    "live_llm_called",
    "model_weights_mutated",
]
KV260_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_overlay_loaded",
    "kv260_uio_devices_present",
    "bitstream_sha256",
    "board_harness_summary",
    "ising_problem_spec",
    "per_seed_results",
    "board_transcript_path",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict for the pilot aggregation.",
    "inference_substrate": "Fixed to aggregation_from_upstream_artifacts; no model or board command is run here.",
    "dispatch_path_validated": "True only when the replay and live KV260 dispatch gates all pass.",
    "cited_upstream_artifacts": "SHA256 citations for each upstream artifact and imported field list.",
    "duration_s": "Real wall-clock duration for the aggregation; no sleep padding.",
    "pilot_only": "True because the artifact validates dispatch plumbing only.",
    "no_hardware_performance_claim": "True because no same-basis CPU comparison is made in this experiment.",
}

VERDICT_BY_GATE = {
    "missing_exp2882_artifact": "blocked_missing_exp2882_artifact",
    "missing_exp2887_artifact": "blocked_missing_exp2887_artifact",
    "missing_exp2898_artifact": "blocked_missing_exp2898_artifact",
    "exp2882_artifact_malformed": "blocked_malformed_exp2882_artifact",
    "exp2887_artifact_malformed": "blocked_malformed_exp2887_artifact",
    "exp2898_artifact_malformed": "blocked_malformed_exp2898_artifact",
    "exp2882_replay_not_ready": "blocked_exp2882_replay_not_ready",
    "exp2882_target_too_small": "blocked_exp2882_target_too_small",
    "exp2882_live_llm_called": "blocked_exp2882_live_llm_called",
    "exp2882_model_weights_mutated": "blocked_exp2882_model_weights_mutated",
    "exp2887_corrigendum_not_clean": "blocked_exp2887_corrigendum_not_clean",
    "exp2887_live_llm_called": "blocked_exp2887_live_llm_called",
    "exp2887_model_weights_mutated": "blocked_exp2887_model_weights_mutated",
    "exp2898_not_complete": "blocked_exp2898_not_complete",
    "exp2898_not_hardware_smoke": "blocked_exp2898_not_hardware_smoke",
    "kv260_precondition_failed": "blocked_kv260_precondition_failed",
    "kv260_overlay_missing": "blocked_kv260_overlay_missing",
    "kv260_uio_devices_missing": "blocked_kv260_uio_devices_missing",
    "kv260_uio_dispatch_missing": "blocked_kv260_uio_dispatch_missing",
    "kv260_bitstream_sha_missing": "blocked_kv260_bitstream_sha_missing",
    "kv260_seed_results_incomplete": "blocked_kv260_seed_results_incomplete",
    "kv260_seed_result_nonpositive": "blocked_kv260_seed_result_nonpositive",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock for the Exp 2906 aggregation.

    The caller can point ``results_dir`` at a fixture directory in tests. In
    production it defaults to the repository `results/` directory so the module
    reads the same upstream artifacts that the conductor writes.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self, filename: str) -> Path:
        return self.output_dir() / filename


@dataclass(frozen=True)
class UpstreamArtifact:
    experiment_id: str
    path: Path
    fields_imported: Sequence[str]
    payload: Mapping[str, Any]
    sha256: str
    missing: bool = False
    malformed: bool = False


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Build the pilot artifact from upstream replay and KV260 board evidence."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    upstreams = _load_upstreams(active_config)
    gate_results = _gate_results(upstreams)
    failed_gates = [gate["name"] for gate in gate_results if not gate["passed"]]
    dispatch_path_validated = not failed_gates
    if dispatch_path_validated:
        honest_verdict = "complete: kv260_replay_dispatch_path_validated_pilot_only"
    else:
        honest_verdict = _blocked_verdict(failed_gates)

    cpu = upstreams["exp2882"].payload
    corrigendum = upstreams["exp2887"].payload
    kv260 = upstreams["exp2898"].payload
    artifact: dict[str, Any] = {
        "artifact": "experiment_2906_fr11_hardware_accelerated_replay_pilot_v1",
        "schema": "carnot.fr11.hardware_accelerated_replay_pilot.v1",
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dispatch_path_validated": dispatch_path_validated,
        "cited_upstream_artifacts": _citations(active_config, upstreams),
        "duration_s": _round_float(active_config.clock() - started_at),
        "run_date": RUN_DATE,
        "pilot_only": True,
        "no_hardware_performance_claim": True,
        "failed_gates": failed_gates,
        "gate_results": gate_results,
        "dispatch_path_summary": {
            "cpu_replay_ready": _cpu_replay_ready(cpu),
            "corrigendum_clean": _corrigendum_clean(corrigendum),
            "kv260_live_board_dispatch_ready": _kv260_ready(kv260),
        },
        "cpu_replay_summary": _cpu_replay_summary(cpu),
        "corrigendum_summary": _corrigendum_summary(corrigendum),
        "kv260_board_summary": _kv260_board_summary(kv260),
        "field_principles": FIELD_PRINCIPLES,
    }
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def _load_upstreams(config: ExperimentConfig) -> dict[str, UpstreamArtifact]:
    return {
        "exp2882": _read_upstream(
            "exp2882",
            config.artifact_path(EXP2882_FILENAME),
            CPU_FIELDS,
        ),
        "exp2887": _read_upstream(
            "exp2887",
            config.artifact_path(EXP2887_FILENAME),
            CORRIGENDUM_FIELDS,
        ),
        "exp2898": _read_upstream(
            "exp2898",
            config.artifact_path(EXP2898_FILENAME),
            KV260_FIELDS,
        ),
    }


def _read_upstream(
    experiment_id: str,
    path: Path,
    fields_imported: Sequence[str],
) -> UpstreamArtifact:
    if not path.exists():
        return UpstreamArtifact(
            experiment_id=experiment_id,
            path=path,
            fields_imported=fields_imported,
            payload={},
            sha256="",
            missing=True,
        )
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return UpstreamArtifact(
            experiment_id=experiment_id,
            path=path,
            fields_imported=fields_imported,
            payload={},
            sha256=sha256,
            malformed=True,
        )
    if not isinstance(payload, dict):
        return UpstreamArtifact(
            experiment_id=experiment_id,
            path=path,
            fields_imported=fields_imported,
            payload={},
            sha256=sha256,
            malformed=True,
        )
    return UpstreamArtifact(
        experiment_id=experiment_id,
        path=path,
        fields_imported=fields_imported,
        payload=payload,
        sha256=sha256,
    )


def _gate_results(upstreams: Mapping[str, UpstreamArtifact]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for key in ("exp2882", "exp2887", "exp2898"):
        item = upstreams[key]
        if item.missing:
            gates.append(_gate(f"missing_{key}_artifact", False, str(item.path)))
        elif item.malformed:
            gates.append(_gate(f"{key}_artifact_malformed", False, str(item.path)))
    if any(not gate["passed"] for gate in gates):
        return gates

    exp2882 = upstreams["exp2882"].payload
    exp2887 = upstreams["exp2887"].payload
    exp2898 = upstreams["exp2898"].payload
    gates.extend(
        [
            _gate(
                "exp2882_replay_not_ready",
                bool(exp2882.get("recmem_replay_scaleup_ready")),
                exp2882.get("honest_verdict", ""),
            ),
            _gate("exp2882_target_too_small", _n_examples(exp2882) >= 50, _n_examples(exp2882)),
            _gate("exp2882_live_llm_called", exp2882.get("live_llm_called") is False, ""),
            _gate(
                "exp2882_model_weights_mutated",
                exp2882.get("model_weights_mutated") is False,
                "",
            ),
            _gate(
                "exp2887_corrigendum_not_clean",
                bool(exp2887.get("fr11_scaleup_clean")),
                exp2887.get("honest_verdict", ""),
            ),
            _gate("exp2887_live_llm_called", exp2887.get("live_llm_called") is False, ""),
            _gate(
                "exp2887_model_weights_mutated",
                exp2887.get("model_weights_mutated") is False,
                "",
            ),
            _gate(
                "exp2898_not_complete",
                str(exp2898.get("honest_verdict", "")).startswith(("complete:", "success:")),
                exp2898.get("honest_verdict", ""),
            ),
            _gate(
                "exp2898_not_hardware_smoke",
                exp2898.get("inference_substrate") == "hardware_smoke",
                exp2898.get("inference_substrate", ""),
            ),
            _gate(
                "kv260_precondition_failed",
                _preconditions_available(exp2898.get("preconditions_checked")),
                "",
            ),
            _gate("kv260_overlay_missing", bool(exp2898.get("kv260_overlay_loaded")), ""),
            _gate(
                "kv260_uio_devices_missing",
                bool(exp2898.get("kv260_uio_devices_present")),
                "",
            ),
            _gate(
                "kv260_uio_dispatch_missing",
                bool(_board_harness(exp2898).get("selected_uio")),
                _board_harness(exp2898).get("selected_uio", ""),
            ),
            _gate("kv260_bitstream_sha_missing", bool(exp2898.get("bitstream_sha256")), ""),
            _gate(
                "kv260_seed_results_incomplete",
                len(_per_seed_results(exp2898)) == 3,
                len(_per_seed_results(exp2898)),
            ),
            _gate(
                "kv260_seed_result_nonpositive",
                _seed_result_medians_positive(exp2898),
                "",
            ),
        ]
    )
    return gates


def _gate(name: str, passed: bool, observed: object) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed}


def _blocked_verdict(failed_gates: Sequence[str]) -> str:
    first = failed_gates[0] if failed_gates else "unknown_gate"
    return VERDICT_BY_GATE.get(first, f"blocked_{first}")


def _cpu_replay_ready(payload: Mapping[str, Any]) -> bool:
    return (
        bool(payload.get("recmem_replay_scaleup_ready"))
        and _n_examples(payload) >= 50
        and payload.get("live_llm_called") is False
        and payload.get("model_weights_mutated") is False
    )


def _corrigendum_clean(payload: Mapping[str, Any]) -> bool:
    return (
        bool(payload.get("fr11_scaleup_clean"))
        and payload.get("live_llm_called") is False
        and payload.get("model_weights_mutated") is False
    )


def _kv260_ready(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("honest_verdict", "")).startswith(("complete:", "success:"))
        and payload.get("inference_substrate") == "hardware_smoke"
        and _preconditions_available(payload.get("preconditions_checked"))
        and bool(payload.get("kv260_overlay_loaded"))
        and bool(payload.get("kv260_uio_devices_present"))
        and bool(_board_harness(payload).get("selected_uio"))
        and bool(payload.get("bitstream_sha256"))
        and len(_per_seed_results(payload)) == 3
        and _seed_result_medians_positive(payload)
    )


def _cpu_replay_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected_example_ids")
    return {
        "honest_verdict": payload.get("honest_verdict", ""),
        "recmem_replay_scaleup_ready": bool(payload.get("recmem_replay_scaleup_ready")),
        "n_examples": _n_examples(payload),
        "selected_example_count": len(selected) if isinstance(selected, list) else 0,
        "live_llm_called": bool(payload.get("live_llm_called", False)),
        "model_weights_mutated": bool(payload.get("model_weights_mutated", False)),
    }


def _corrigendum_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "honest_verdict": payload.get("honest_verdict", ""),
        "fr11_scaleup_clean": bool(payload.get("fr11_scaleup_clean")),
        "best_policy": payload.get("best_policy", ""),
        "n_examples": _n_examples(payload),
    }


def _kv260_board_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    harness = _board_harness(payload)
    problem = payload.get("ising_problem_spec")
    return {
        "honest_verdict": payload.get("honest_verdict", ""),
        "kv260_overlay_loaded": payload.get("kv260_overlay_loaded", ""),
        "uio_device_count": len(payload.get("kv260_uio_devices_present", []) or []),
        "selected_uio": harness.get("selected_uio", ""),
        "uio0_mmap_checked": bool(harness.get("uio0_mmap_checked", False)),
        "bitstream_sha256": payload.get("bitstream_sha256", ""),
        "n_spins": problem.get("n_spins") if isinstance(problem, dict) else None,
        "per_seed_result_count": len(_per_seed_results(payload)),
        "board_transcript_path": payload.get("board_transcript_path", ""),
    }


def _n_examples(payload: Mapping[str, Any]) -> int:
    try:
        return int(payload.get("n_examples", 0))
    except (TypeError, ValueError):
        return 0


def _preconditions_available(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if not bool(item.get("available", item.get("passed", False))):
            return False
    return True


def _board_harness(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("board_harness_summary")
    return value if isinstance(value, dict) else {}


def _per_seed_results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("per_seed_results")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _seed_result_medians_positive(payload: Mapping[str, Any]) -> bool:
    rows = _per_seed_results(payload)
    if len(rows) != 3:
        return False
    try:
        return all(float(row.get("per_sample_wall_clock_us_median", 0.0)) > 0.0 for row in rows)
    except (TypeError, ValueError):
        return False


def _citations(
    config: ExperimentConfig,
    upstreams: Mapping[str, UpstreamArtifact],
) -> list[dict[str, Any]]:
    citations = []
    for key in ("exp2882", "exp2887", "exp2898"):
        item = upstreams[key]
        if item.missing or item.malformed:
            continue
        citations.append(
            {
                "experiment_id": item.experiment_id,
                "path": _relative_path(item.path, config.repo_root),
                "fields_imported": list(item.fields_imported),
                "sha256": item.sha256,
            }
        )
    return citations


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _round_float(value: float) -> float:
    return round(float(value), 12)


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
