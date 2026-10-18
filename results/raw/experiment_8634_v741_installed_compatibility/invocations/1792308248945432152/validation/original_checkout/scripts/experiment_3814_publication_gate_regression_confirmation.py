#!/usr/bin/env python3
"""Exp 3814: record the publication-gate regression check.

This runner does not define publication readiness itself. It executes the
existing fixed G1-G4 gate, records the booleans it returns, and checks that the
frozen FoVer headline source still rounds to 0.9131.

Spec refs: REQ-PUBLISH-3814, SCENARIO-PUBLISH-3814.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_REL_PATH = Path("results/experiment_3814_publication_gate_regression_confirmation.json")
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_REL_PATH
GATE_REL_PATH = Path("scripts/publication_gate.py")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
HEADLINE_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")

RANDOM_SEED = 3814
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: publication_gate_regression_confirmed_g1_g2_g3_g4_pass_"
    "paper_ready_true_frozen_fover_0.9131_unchanged_no_gate_redefined"
)
BLOCKED_INTERPRETER_VERDICT = "blocked_interpreter_or_publication_gate_unavailable"
BLOCKED_GATE_VERDICT = "blocked_publication_gate_json_unavailable"
BLOCKED_HEADLINE_VERDICT = "blocked_frozen_fover_headline_source_missing"

SOURCE_REL_PATHS: Mapping[str, Path] = {
    "gate_runner": GATE_REL_PATH,
    "gate_definition": NORTH_STAR_REL_PATH,
    "g1_g4_headline_source": HEADLINE_REL_PATH,
    "g2_manual_state": Path("ops/publication_gate_state.json"),
    "g2_reproduction_runbook": Path("ops/reproduction-runbook-fover-headline.md"),
    "g3_technical_report_source": Path("docs/technical-report.md"),
    "g3_paper_source": Path("docs/arxiv-paper/main.tex"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "g1_pass",
    "g2_pass",
    "g3_pass",
    "g4_pass",
    "paper_ready",
    "frozen_fover_auroc_unchanged",
    "any_gate_regressed",
    "gate_definitions_unchanged",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES: Mapping[str, str] = {
    "honest_verdict": "Terminal prefix; the regression-check outcome.",
    "inference_substrate": "Runs the gate script and records booleans; no live model.",
    "g1_pass": (
        "BARE bool -- headline measured (FoVer 0.9131, 5-seed, CI, "
        "adversarial-clean) still passes."
    ),
    "g2_pass": (
        "BARE bool -- the independent CI reproducer still attests the headline within CI."
    ),
    "g3_pass": "BARE bool -- prose stays narrowing-clean.",
    "g4_pass": (
        "BARE bool -- headline numbers still trace to a primary artifact with "
        "random_seed + reproducibility_checksum."
    ),
    "paper_ready": "BARE bool -- G1 and G2 and G3 and G4.",
    "frozen_fover_auroc_unchanged": (
        "BARE bool -- the frozen 0.9131 headline is unchanged."
    ),
    "any_gate_regressed": (
        "BARE bool -- surfaces any regression honestly for the operator."
    ),
    "gate_definitions_unchanged": (
        "BARE bool -- no gate was redefined to show progress."
    ),
    "cited_upstream_artifacts": "Provenance for the gate sources.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def _absolute(root: Path, relative_path: Path) -> Path:
    return (root / relative_path).resolve()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - malformed fixtures are not useful here
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_entry(root: Path, role: str, relative_path: Path) -> JsonDict:
    path = _absolute(root, relative_path)
    payload = _load_json(path) if path.suffix == ".json" else {}
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256_file(path),
        "random_seed": payload.get("random_seed"),
        "random_seeds_used": payload.get("random_seeds_used"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
    }


def _interpreter_preconditions(root: Path, executable: str | Path | None = None) -> JsonDict:
    exe = Path(executable or sys.executable)
    resolved_exe = exe.resolve()
    gate_path = _absolute(root, GATE_REL_PATH)
    venv_python = ".venv" in exe.parts or ".venv" in resolved_exe.parts
    return {
        "executable": str(exe),
        "resolved_executable": str(resolved_exe),
        "venv_python": venv_python,
        "publication_gate_script_exists": gate_path.exists(),
        "publication_gate_script": str(gate_path),
        "ok": bool(venv_python and gate_path.exists()),
    }


def _run_publication_gate_json(root: Path, executable: str | Path | None = None) -> JsonDict:
    exe = str(Path(executable or sys.executable))
    gate_path = _absolute(root, GATE_REL_PATH)
    command = [exe, str(gate_path), "--json"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host failure path
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "data": {},
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "data": {},
        }
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {}
        parsed_ok = False
    else:
        parsed_ok = isinstance(parsed, dict)
    return {
        "ok": parsed_ok,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "data": parsed if isinstance(parsed, dict) else {},
    }


def _gate_bool(gate_data: Mapping[str, Any], gate_name: str) -> bool:
    gates = gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return False
    gate = gates.get(gate_name)
    return bool(gate.get("pass", False)) if isinstance(gate, Mapping) else False


def _frozen_fover_auroc(root: Path) -> float | None:
    payload = _load_json(_absolute(root, HEADLINE_REL_PATH))
    value = payload.get("condition_a_production_auroc_mean")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 4)


def _duration(started: float, now_s: float | None) -> float:
    ended = time.perf_counter() if now_s is None else float(now_s)
    return max(round(ended - started, 6), 0.0001)


def _verdict(
    *,
    preconditions_ok: bool,
    gate_ok: bool,
    frozen_fover_auroc_unchanged: bool,
    any_gate_regressed: bool,
    gate_definitions_unchanged: bool,
) -> str:
    if not preconditions_ok:
        return BLOCKED_INTERPRETER_VERDICT
    if not gate_ok:
        return BLOCKED_GATE_VERDICT
    if not frozen_fover_auroc_unchanged:
        return BLOCKED_HEADLINE_VERDICT
    if any_gate_regressed:
        return "complete: publication_gate_regression_detected_operator_review_required"
    if not gate_definitions_unchanged:
        return "complete: publication_gate_definition_changed_operator_review_required"
    return TERMINAL_VERDICT


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = PROJECT_ROOT,
    *,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    executable: str | Path | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    north_star_path = _absolute(root_path, NORTH_STAR_REL_PATH)
    north_star_sha_before = _sha256_file(north_star_path)
    preconditions = _interpreter_preconditions(root_path, executable)

    if publication_gate_data is None:
        gate_run = _run_publication_gate_json(root_path, executable)
        gate_data = gate_run["data"] if gate_run["ok"] else {}
    else:
        gate_data = dict(publication_gate_data)
        gate_run = {
            "ok": True,
            "command": ["injected_publication_gate_data"],
            "returncode": 0,
            "stdout": json.dumps(gate_data, sort_keys=True),
            "stderr": "",
            "data": gate_data,
        }

    g1 = _gate_bool(gate_data, "G1")
    g2 = _gate_bool(gate_data, "G2")
    g3 = _gate_bool(gate_data, "G3")
    g4 = _gate_bool(gate_data, "G4")
    paper_ready = bool(gate_data.get("paper_ready", False))
    any_gate_regressed = not (g1 and g2 and g3 and g4 and paper_ready)
    frozen_fover_auroc = _frozen_fover_auroc(root_path)
    frozen_fover_auroc_unchanged = frozen_fover_auroc == FROZEN_FOVER_AUROC
    north_star_sha_after = _sha256_file(north_star_path)
    gate_definitions_unchanged = (
        north_star_sha_before is not None and north_star_sha_before == north_star_sha_after
    )
    duration_s = _duration(started, now_s)

    artifact = {
        "experiment": 3814,
        "schema": "carnot.publication_gate_regression_confirmation.v1",
        "spec_refs": ["REQ-PUBLISH-3814", "SCENARIO-PUBLISH-3814"],
        "honest_verdict": _verdict(
            preconditions_ok=bool(preconditions["ok"]),
            gate_ok=bool(gate_run["ok"]),
            frozen_fover_auroc_unchanged=frozen_fover_auroc_unchanged,
            any_gate_regressed=any_gate_regressed,
            gate_definitions_unchanged=gate_definitions_unchanged,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "publication_gate_json": gate_data,
        "publication_gate_command": {
            "command": gate_run["command"],
            "returncode": gate_run["returncode"],
            "stdout": gate_run["stdout"],
            "stderr": gate_run["stderr"],
        },
        "g1_pass": g1,
        "g2_pass": g2,
        "g3_pass": g3,
        "g4_pass": g4,
        "paper_ready": paper_ready,
        "frozen_fover_auroc": frozen_fover_auroc,
        "frozen_fover_auroc_unchanged": frozen_fover_auroc_unchanged,
        "any_gate_regressed": any_gate_regressed,
        "gate_definitions_unchanged": gate_definitions_unchanged,
        "gate_definition_source": str(north_star_path),
        "gate_definition_sha256_before": north_star_sha_before,
        "gate_definition_sha256_after": north_star_sha_after,
        "headline_source_artifact": str(_absolute(root_path, HEADLINE_REL_PATH)),
        "preconditions": preconditions,
        "cited_upstream_artifacts": [
            _source_entry(root_path, role, relative_path)
            for role, relative_path in SOURCE_REL_PATHS.items()
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)  # pragma: no cover


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact), "missing required fields")
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]), "missing principles")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong substrate")
    for key in (
        "g1_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "paper_ready",
        "frozen_fover_auroc_unchanged",
        "any_gate_regressed",
        "gate_definitions_unchanged",
    ):
        _require(isinstance(artifact[key], bool), f"{key} must be a bare bool")
    _require(artifact["random_seed"] == RANDOM_SEED, "wrong random seed")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")
    encoded = json.dumps(artifact, sort_keys=True)
    _require("GGUF" not in encoded and "CUDA" not in encoded, "forbidden substrate marker")
    _require("live_llm_inference" not in encoded, "forbidden live-model marker")
    if artifact["honest_verdict"] == TERMINAL_VERDICT:
        _require(artifact["paper_ready"] is True, "terminal verdict requires paper_ready")
        _require(artifact["any_gate_regressed"] is False, "terminal verdict requires no regression")
        _require(artifact["frozen_fover_auroc_unchanged"] is True, "terminal verdict requires frozen AUROC")
        _require(artifact["gate_definitions_unchanged"] is True, "terminal verdict requires fixed gate")
    else:
        _require(str(artifact["honest_verdict"]).startswith(("blocked_", "complete:")), "bad verdict")


def run(
    root: Path | str = PROJECT_ROOT,
    *,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    executable: str | Path | None = None,
) -> Path:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        publication_gate_data=publication_gate_data,
        started_s=started_s,
        now_s=now_s,
        executable=executable,
    )
    validate_artifact(artifact)
    output_path = root_path / OUTPUT_REL_PATH
    write_json(output_path, artifact)
    return output_path


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(PROJECT_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
