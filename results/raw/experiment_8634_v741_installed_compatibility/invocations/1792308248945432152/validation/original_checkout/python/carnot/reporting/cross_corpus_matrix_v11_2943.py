"""Build the Exp 2943 cross-corpus matrix v11 artifact.

Spec refs: REQ-REPORT-2943, SCENARIO-REPORT-2943.

This module is an aggregation layer for paper-boundary evidence. It reads the
already checked-in matrix and corrigenda artifacts, records where each imported
field came from, and writes a v11 artifact without rerunning model generation,
verifier scoring, sampler comparisons, or hardware measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.cross_corpus_matrix.v11_corrigenda_auprc.v1"
ARTIFACT = "experiment_2943_cross_corpus_matrix_v11"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2943_cross_corpus_matrix_v11.json")

MATRIX_V10_REL_PATH = Path(
    "results/experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1.json"
)
EXP2938_REL_PATH = Path("results/experiment_2938_kv260_mmd_vs_cpu_sequential_gibbs_v1.json")
EXP2939_REL_PATH = Path(
    "results/experiment_2939_cpu_synchronous_parallel_same_schedule_baseline_v1.json"
)
EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2942_REL_PATH = Path("results/experiment_2942_kv260_continuation_n_scaling_v1.json")

CORRIGENDUM_CLEAN_ROWS = [
    "exp2938_kv260_mmd_corrigendum",
    "exp2939_same_schedule_speedup_corrigendum",
    "exp2940_code_corpus_auprc_corrigendum",
    "exp2942_kv260_n_scaling_corrigendum",
]


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    fields_imported: tuple[str, ...]
    required_fields: tuple[str, ...]


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp2935",
        MATRIX_V10_REL_PATH,
        (
            "clean_rows",
            "flagged_rows",
            "blocked_rows",
            "rows_clean",
            "rows_flagged",
            "rows_blocked",
            "matrix_rows",
        ),
        ("clean_rows", "flagged_rows", "blocked_rows"),
    ),
    SourceSpec(
        "exp2938",
        EXP2938_REL_PATH,
        (
            "distributions_distinguishable",
            "per_seed_mmd_squared",
            "per_seed_mmd_pvalue",
            "paper_v6_recommendation",
        ),
        ("honest_verdict", "distributions_distinguishable", "per_seed_mmd_squared"),
    ),
    SourceSpec(
        "exp2939",
        EXP2939_REL_PATH,
        ("kv260_speedup_vs_same_schedule_cpu.value", "paper_v6_recommendation"),
        ("honest_verdict", "kv260_speedup_vs_same_schedule_cpu.value"),
    ),
    SourceSpec(
        "exp2940",
        EXP2940_REL_PATH,
        (
            "code_corpus_auprc",
            "code_corpus_baseline_random_auprc.value",
            "fover_corpus_auprc.value",
            "paper_v6_recommendation",
        ),
        ("honest_verdict", "code_corpus_auprc", "fover_corpus_auprc.value"),
    ),
    SourceSpec(
        "exp2942",
        EXP2942_REL_PATH,
        ("bitstream_supports_variable_n", "measured_crossover_n", "per_n_results"),
        ("honest_verdict", "bitstream_supports_variable_n", "per_n_results"),
    ),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for malformed inputs."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2943: build matrix v11 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)
    payloads = {
        spec.experiment_id: read_json_object(root_path / spec.path) for spec in SOURCE_SPECS
    }
    cited = _cited_upstream_artifacts(root_path)
    errors = _required_upstream_errors(payloads)
    v10 = payloads.get("exp2935", {})

    if errors:
        return _blocked_artifact(v10, cited, errors, duration_s)

    exp2938 = payloads["exp2938"]
    exp2939 = payloads["exp2939"]
    exp2940 = payloads["exp2940"]
    exp2942 = payloads["exp2942"]
    rows_clean = _unique_strings([*_v10_bucket(v10, "clean"), *CORRIGENDUM_CLEAN_ROWS])
    rows_flagged = _unique_strings(_v10_bucket(v10, "flagged"))
    rows_blocked = _unique_strings(_v10_bucket(v10, "blocked"))
    per_corpus_auprc = _per_corpus_auprc(exp2940)
    speedup = _kv260_same_schedule_speedup(exp2939)
    crossover = _kv260_n_crossover_measured(exp2942)
    outcomes = _deep_think_corrigenda_outcomes(exp2938, exp2939, exp2940, exp2942)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _complete_verdict(per_corpus_auprc, speedup, crossover),
        "matrix_v11_ready": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "rows_clean": rows_clean,
        "rows_flagged": rows_flagged,
        "rows_blocked": rows_blocked,
        "per_corpus_auprc": per_corpus_auprc,
        "kv260_same_schedule_speedup_recorded": speedup,
        "kv260_n_crossover_measured": crossover,
        "deep_think_corrigenda_outcomes": outcomes,
        "matrix_rows": _matrix_rows(v10, outcomes),
        "cited_upstream_artifacts": cited,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_sampler_run": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    _write_json(out_path, artifact)
    return out_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cited_upstream_artifacts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "artifact_path": spec.path.as_posix(),
            "fields_imported": list(spec.fields_imported),
            "sha256": sha256_file(root / spec.path),
        }
        for spec in SOURCE_SPECS
    ]


def _required_upstream_errors(
    payloads: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        payload = payloads.get(spec.experiment_id, {})
        if not payload:
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "artifact_path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                    "missing_fields": list(spec.required_fields),
                }
            )
            continue
        missing = [field for field in spec.required_fields if _get_path(payload, field) is None]
        if missing:
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "artifact_path": spec.path.as_posix(),
                    "reason": "missing_required_field",
                    "missing_fields": missing,
                }
            )
    return errors


def _blocked_artifact(
    v10: Mapping[str, Any],
    cited: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": "blocked_required_upstream_missing",
        "matrix_v11_ready": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "rows_clean": _unique_strings(_v10_bucket(v10, "clean")),
        "rows_flagged": _unique_strings(_v10_bucket(v10, "flagged")),
        "rows_blocked": _unique_strings(_v10_bucket(v10, "blocked")),
        "per_corpus_auprc": {},
        "kv260_same_schedule_speedup_recorded": 0.0,
        "kv260_n_crossover_measured": 0,
        "deep_think_corrigenda_outcomes": {},
        "matrix_rows": list(v10.get("matrix_rows", []))
        if isinstance(v10.get("matrix_rows"), list)
        else [],
        "required_upstream_errors": errors,
        "cited_upstream_artifacts": cited,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_sampler_run": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }


def _per_corpus_auprc(exp2940: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "FoVer": {
            "source_experiment_id": "exp2940",
            "source_field": "fover_corpus_auprc.value",
            "value": _as_float(_get_path(exp2940, "fover_corpus_auprc.value"), "FoVer AUPRC"),
        },
        "code_corpora": {
            "baseline_random_auprc": _as_float(
                _get_path(exp2940, "code_corpus_baseline_random_auprc.value"),
                "code baseline AUPRC",
            ),
            "source_experiment_id": "exp2940",
            "source_field": "code_corpus_auprc",
            "value": _as_float(exp2940.get("code_corpus_auprc"), "code corpus AUPRC"),
        },
    }


def _kv260_same_schedule_speedup(exp2939: Mapping[str, Any]) -> float:
    return _as_float(
        _get_path(exp2939, "kv260_speedup_vs_same_schedule_cpu.value"),
        "same-schedule speedup",
    )


def _kv260_n_crossover_measured(exp2942: Mapping[str, Any]) -> int:
    direct = exp2942.get("kv260_n_crossover_measured")
    if isinstance(direct, int | float) and direct > 0:
        return int(direct)
    measured = exp2942.get("measured_crossover_n")
    if isinstance(measured, int | float) and measured > 0:
        return int(measured)
    return 0


def _deep_think_corrigenda_outcomes(
    exp2938: Mapping[str, Any],
    exp2939: Mapping[str, Any],
    exp2940: Mapping[str, Any],
    exp2942: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "exp2938": {
            "distributions_distinguishable": bool(exp2938["distributions_distinguishable"]),
            "per_seed_mmd_squared": list(exp2938.get("per_seed_mmd_squared", [])),
            "per_seed_mmd_pvalue": list(exp2938.get("per_seed_mmd_pvalue", [])),
            "paper_v6_recommendation": str(exp2938.get("paper_v6_recommendation", "")),
        },
        "exp2939": {
            "same_schedule_speedup": _kv260_same_schedule_speedup(exp2939),
            "paper_v6_recommendation": str(exp2939.get("paper_v6_recommendation", "")),
        },
        "exp2940": {
            "code_corpus_auprc": _as_float(exp2940.get("code_corpus_auprc"), "code corpus AUPRC"),
            "fover_corpus_auprc": _as_float(
                _get_path(exp2940, "fover_corpus_auprc.value"),
                "FoVer AUPRC",
            ),
            "paper_v6_recommendation": _get_path(exp2940, "paper_v6_recommendation.value"),
        },
        "exp2942": {
            "bitstream_supports_variable_n": bool(exp2942["bitstream_supports_variable_n"]),
            "measured_crossover_n": _kv260_n_crossover_measured(exp2942),
            "per_n_results": list(exp2942.get("per_n_results", [])),
        },
    }


def _matrix_rows(v10: Mapping[str, Any], outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(v10.get("matrix_rows", [])) if isinstance(v10.get("matrix_rows"), list) else []
    rows.extend(
        [
            {
                "row_id": "exp2938_kv260_mmd_corrigendum",
                "row_class": "clean",
                "headline_eligible": False,
                "paper_claim_eligible": True,
                "claim_boundary": "Retracts exact FPGA sampling; KV260 row is fixed-schedule heuristic evidence.",
                "summary": outcomes["exp2938"],
            },
            {
                "row_id": "exp2939_same_schedule_speedup_corrigendum",
                "row_class": "clean",
                "headline_eligible": False,
                "paper_claim_eligible": True,
                "claim_boundary": "Same-schedule CPU comparison supersedes the earlier sequential-Gibbs speedup framing.",
                "summary": outcomes["exp2939"],
            },
            {
                "row_id": "exp2940_code_corpus_auprc_corrigendum",
                "row_class": "clean",
                "headline_eligible": True,
                "paper_claim_eligible": True,
                "claim_boundary": "Adds code-corpus AUPRC/base-rate columns from Exp 2940.",
                "summary": outcomes["exp2940"],
            },
            {
                "row_id": "exp2942_kv260_n_scaling_corrigendum",
                "row_class": "clean",
                "headline_eligible": False,
                "paper_claim_eligible": True,
                "claim_boundary": "Records measured n-scaling rows only; zero crossover means not measured.",
                "summary": outcomes["exp2942"],
            },
        ]
    )
    return rows


def _complete_verdict(
    per_corpus_auprc: Mapping[str, Mapping[str, Any]],
    speedup: float,
    crossover: int,
) -> str:
    code_auprc = _as_float(per_corpus_auprc["code_corpora"]["value"], "code corpus AUPRC")
    return (
        "complete: matrix_v11_ready=true; "
        f"code_corpus_auprc={code_auprc:.6g}; "
        f"kv260_same_schedule_speedup_recorded={speedup:.6g}; "
        f"kv260_n_crossover_measured={crossover}"
    )


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _v10_bucket(v10: Mapping[str, Any], bucket: str) -> list[Any]:
    legacy = v10.get(f"rows_{bucket}")
    if isinstance(legacy, list):
        return legacy
    current = v10.get(f"{bucket}_rows")
    return current if isinstance(current, list) else []


def _get_path(payload: Mapping[str, Any], dotted_field: str) -> Any:
    current: Any = payload
    for part in dotted_field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _as_float(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)
