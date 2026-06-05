"""Exp 3840 publication-gate regression confirmation.

This module is aggregation-only. It runs the fixed G1-G4 publication gate,
reads the frozen FoVer headline source, and spot-checks the v353 artifacts with
the disciplined artifact summarizer.

Spec refs: REQ-PUBLISH-3840, SCENARIO-PUBLISH-3840.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3840_publication_gate_regression_confirmation.json")
GATE_REL_PATH = Path("scripts/publication_gate.py")
SUMMARIZER_REL_PATH = Path("scripts/summarize_artifact.py")
HEADLINE_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")

RANDOM_SEED = 3840
FROZEN_FOVER_AUROC = 0.9131
SPOT_CHECK_EXPERIMENTS = (3835, 3836, 3837, 3838)
INFERENCE_SUBSTRATE = "aggregation_from_publication_gate_and_artifact_summaries"
TERMINAL_VERDICT = (
    "complete: publication_gate_regression_confirmed_g1_g2_g3_g4_pass_"
    "paper_ready_true_frozen_fover_0.9131_unchanged"
)

REQUIRED_ARTIFACT_FIELDS = (
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "unmet_gates",
    "frozen_fover_auroc",
    "honest_verdict",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)

FIELD_PROVENANCE: Mapping[str, Mapping[str, str]] = {
    "g1": {
        "source": "publication_gate.py --json gates.G1.pass",
        "principle": "each gate boolean \u2014 the four conditions for paper_ready",
    },
    "g2": {
        "source": "publication_gate.py --json gates.G2.pass",
        "principle": "each gate boolean \u2014 the four conditions for paper_ready",
    },
    "g3": {
        "source": "publication_gate.py --json gates.G3.pass",
        "principle": "each gate boolean \u2014 the four conditions for paper_ready",
    },
    "g4": {
        "source": "publication_gate.py --json gates.G4.pass",
        "principle": "each gate boolean \u2014 the four conditions for paper_ready",
    },
    "paper_ready": {
        "source": "publication_gate.py --json paper_ready",
        "principle": "G1^G2^G3^G4 \u2014 the standing convergence invariant, MUST be true",
    },
    "unmet_gates": {
        "source": "publication_gate.py --json unmet_gates",
        "principle": "empty list expected; any entry is a regression to surface",
    },
    "frozen_fover_auroc": {
        "source": str(HEADLINE_REL_PATH),
        "principle": "MUST read 0.9131 \u2014 the headline must not have moved",
    },
    "honest_verdict": {
        "source": "closed-set Exp 3840 verdict synthesis",
        "principle": "terminal complete: prefix or blocked_<resource>",
    },
    "random_seed": {
        "source": "Exp 3840 deterministic aggregation constant",
        "principle": "stable reproducibility marker for this aggregation run",
    },
    "reproducibility_checksum": {
        "source": "sha256 over artifact payload with this field blanked",
        "principle": "detects accidental artifact drift",
    },
    "duration_s": {
        "source": "time.perf_counter wall-clock measurement",
        "principle": "plausibility floor for an aggregation-only run",
    },
    "inference_substrate": {
        "source": "Exp 3840 aggregation design",
        "principle": "must not claim live model execution",
    },
}


def _absolute(root: Path, relative_path: Path) -> Path:
    return (root / relative_path).resolve()


def _load_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - host or malformed fixture fault
        return {}
    return payload if isinstance(payload, dict) else {}


def _duration(started_s: float, now_s: float | None) -> float:
    ended = time.perf_counter() if now_s is None else float(now_s)
    return max(round(ended - started_s, 6), 0.0001)


def _blocked_result(
    resource: str,
    *,
    command: list[str],
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> JsonDict:
    return {
        "ok": False,
        "blocked_resource": resource,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "data": {},
    }


def run_publication_gate_json(
    root: Path | str = PROJECT_ROOT,
    *,
    executable: str | Path | None = None,
) -> JsonDict:
    """Run the fixed publication gate and return parsed JSON plus command metadata."""

    root_path = Path(root)
    exe = Path(executable) if executable is not None else root_path / ".venv" / "bin" / "python"
    gate_path = _absolute(root_path, GATE_REL_PATH)
    command = [str(exe), str(gate_path), "--json"]
    if not exe.exists():
        return _blocked_result("publication_gate_python", command=command)
    if not gate_path.exists():
        return _blocked_result("publication_gate_script", command=command)
    try:
        completed = subprocess.run(
            command,
            cwd=root_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host failure
        return _blocked_result("publication_gate_json", command=command, stderr=str(exc))
    if completed.returncode != 0:
        return _blocked_result(
            "publication_gate_json",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if not parsed:
        return _blocked_result(
            "publication_gate_json",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return {
        "ok": True,
        "blocked_resource": "",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "data": parsed,
    }


def _gate_bool(publication_gate_data: Mapping[str, Any], gate_name: str) -> bool:
    gates = publication_gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return False
    gate = gates.get(gate_name)
    return bool(gate.get("pass", False)) if isinstance(gate, Mapping) else False


def _unmet_gates(publication_gate_data: Mapping[str, Any]) -> list[str]:
    raw = publication_gate_data.get("unmet_gates")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [name for name in ("G1", "G2", "G3", "G4") if not _gate_bool(publication_gate_data, name)]


def _gate_source_names(publication_gate_data: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    gates = publication_gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return names
    for gate in gates.values():
        if isinstance(gate, Mapping) and isinstance(gate.get("source"), str):
            names.add(Path(str(gate["source"])).name)
    return names


def _frozen_fover_auroc(root: Path) -> float | None:
    payload = _load_json(_absolute(root, HEADLINE_REL_PATH))
    value = payload.get("condition_a_production_auroc_mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 4)


def _artifact_paths(root: Path, exp_id: int) -> list[Path]:
    results = root / "results"
    return sorted(results.glob(f"experiment_{exp_id}_*.json")) + sorted(
        results.glob(f"experiment_{exp_id}.json")
    )


def _live_recheck(stdout: str) -> str:
    lower = stdout.lower()
    for marker, status in (
        ("live re-check: critical", "critical"),
        ("live re-check: warn", "warn"),
        ("live re-check: clean", "clean"),
    ):
        if marker in lower:
            return status
    return "unknown"  # pragma: no cover - summarizer always prints this field


def _run_summarizer(root: Path, exp_id: int, executable: str | Path | None) -> JsonDict:
    exe = Path(executable) if executable is not None else root / ".venv" / "bin" / "python"
    summarizer = _absolute(root, SUMMARIZER_REL_PATH)
    command = [str(exe), str(summarizer), str(exp_id)]
    if not exe.exists():
        return _blocked_result("summarizer_python", command=command)
    if not summarizer.exists():
        return _blocked_result("summarize_artifact", command=command)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host failure
        return _blocked_result("summarize_artifact", command=command, stderr=str(exc))
    ok = completed.returncode in (0, 1, 2)
    return {
        "ok": ok,
        "blocked_resource": "" if ok else f"summarize_artifact_exp{exp_id}",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "live_recheck": _live_recheck(completed.stdout),
    }


def spot_check_353_artifacts(
    root: Path | str = PROJECT_ROOT,
    *,
    executable: str | Path | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run summarize_artifact.py over Exp 3835-3838 and classify gate feeders."""

    root_path = Path(root)
    gate_sources = _gate_source_names(publication_gate_data or {})
    summaries: dict[str, JsonDict] = {}
    artifacts: dict[str, list[JsonDict]] = {}
    flagged_gate_feeders: list[JsonDict] = []
    live_critical: list[str] = []

    for exp_id in SPOT_CHECK_EXPERIMENTS:
        exp_key = str(exp_id)
        summary = _run_summarizer(root_path, exp_id, executable)
        summaries[exp_key] = summary
        if not summary["ok"]:
            return {
                "ok": False,
                "blocked_resource": summary["blocked_resource"],
                "checked_experiments": list(SPOT_CHECK_EXPERIMENTS),
                "summaries": summaries,
                "artifacts": artifacts,
                "flagged_adversarial_true_gate_feeders": flagged_gate_feeders,
                "live_critical_artifacts": live_critical,
            }
        if summary["live_recheck"] == "critical":
            live_critical.append(exp_key)

        paths = _artifact_paths(root_path, exp_id)
        if not paths:  # pragma: no cover - fixture and repo both carry these artifacts
            return {
                "ok": False,
                "blocked_resource": f"experiment_{exp_id}_artifact",
                "checked_experiments": list(SPOT_CHECK_EXPERIMENTS),
                "summaries": summaries,
                "artifacts": artifacts,
                "flagged_adversarial_true_gate_feeders": flagged_gate_feeders,
                "live_critical_artifacts": live_critical,
            }
        entries: list[JsonDict] = []
        for path in paths:
            payload = _load_json(path)
            stamped = payload.get("flagged_adversarial") is True
            feeds_gate = path.name in gate_sources
            entry = {
                "path": str(path),
                "stamped_flagged_adversarial": stamped,
                "feeds_publication_gate": feeds_gate,
                "honest_verdict": payload.get("honest_verdict"),
            }
            entries.append(entry)
            if stamped and feeds_gate:
                flagged_gate_feeders.append({"experiment": exp_id, "path": str(path)})
        artifacts[exp_key] = entries

    return {
        "ok": True,
        "blocked_resource": "",
        "checked_experiments": list(SPOT_CHECK_EXPERIMENTS),
        "summaries": summaries,
        "artifacts": artifacts,
        "flagged_adversarial_true_gate_feeders": flagged_gate_feeders,
        "live_critical_artifacts": live_critical,
    }


def _blocked_verdict(resource: str) -> str:
    return resource if resource.startswith("blocked_") else f"blocked_{resource}"


def _regression_verdict(unmet_gates: list[str], *, paper_ready: bool) -> str:
    if unmet_gates:
        suffix = "_".join(unmet_gates)
    else:
        suffix = "paper_ready" if not paper_ready else "unknown"
    return f"complete: publication_gate_REGRESSION_DETECTED_unmet_{suffix}"


def _verdict(
    *,
    blocked_resource: str,
    frozen_fover_auroc: float | None,
    frozen_fover_auroc_unchanged: bool,
    flagged_gate_feeders: list[JsonDict],
    unmet_gates: list[str],
    paper_ready: bool,
) -> str:
    if blocked_resource:
        return _blocked_verdict(blocked_resource)
    if frozen_fover_auroc is None:
        return "blocked_frozen_fover_headline_source"
    if not frozen_fover_auroc_unchanged:
        return "complete: publication_gate_REGRESSION_DETECTED_unmet_frozen_fover_auroc"
    if flagged_gate_feeders:
        return "complete: publication_gate_REGRESSION_DETECTED_unmet_flagged_adversarial_gate_feeder"
    if unmet_gates or not paper_ready:
        return _regression_verdict(unmet_gates, paper_ready=paper_ready)
    return TERMINAL_VERDICT


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = PROJECT_ROOT,
    *,
    publication_gate_run: Mapping[str, Any] | None = None,
    spot_check_result: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    executable: str | Path | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    gate_run = (
        dict(publication_gate_run)
        if publication_gate_run is not None
        else run_publication_gate_json(root_path, executable=executable)
    )
    gate_data = gate_run["data"] if gate_run.get("ok") else {}

    g1 = _gate_bool(gate_data, "G1")
    g2 = _gate_bool(gate_data, "G2")
    g3 = _gate_bool(gate_data, "G3")
    g4 = _gate_bool(gate_data, "G4")
    paper_ready = bool(gate_data.get("paper_ready", False))
    unmet = _unmet_gates(gate_data) if gate_run.get("ok") else []
    frozen_fover = _frozen_fover_auroc(root_path)
    frozen_unchanged = frozen_fover == FROZEN_FOVER_AUROC

    if spot_check_result is not None:
        spot_checks = dict(spot_check_result)
    elif gate_run.get("ok"):
        spot_checks = spot_check_353_artifacts(
            root_path,
            executable=executable,
            publication_gate_data=gate_data,
        )
    else:
        spot_checks = {
            "ok": True,
            "blocked_resource": "",
            "checked_experiments": list(SPOT_CHECK_EXPERIMENTS),
            "summaries": {},
            "artifacts": {},
            "flagged_adversarial_true_gate_feeders": [],
            "live_critical_artifacts": [],
        }

    blocked_resource = ""
    if not gate_run.get("ok"):
        blocked_resource = str(gate_run.get("blocked_resource", "publication_gate_json"))
    elif not spot_checks.get("ok"):
        blocked_resource = str(spot_checks.get("blocked_resource", "summarize_artifact"))

    flagged_gate_feeders = list(spot_checks.get("flagged_adversarial_true_gate_feeders", []))
    duration_s = _duration(started, now_s)
    artifact: JsonDict = {
        "experiment": RANDOM_SEED,
        "schema": "carnot.publication_gate_regression_3840.v1",
        "spec_refs": ["REQ-PUBLISH-3840", "SCENARIO-PUBLISH-3840"],
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "unmet_gates": unmet,
        "frozen_fover_auroc": frozen_fover,
        "frozen_fover_auroc_unchanged": frozen_unchanged,
        "publication_gate_json": gate_data,
        "publication_gate_command": {
            "command": gate_run.get("command", []),
            "returncode": gate_run.get("returncode"),
            "stdout": gate_run.get("stdout", ""),
            "stderr": gate_run.get("stderr", ""),
        },
        "spot_check_experiment_ids": list(SPOT_CHECK_EXPERIMENTS),
        "spot_check_artifacts": spot_checks.get("artifacts", {}),
        "spot_check_summaries": spot_checks.get("summaries", {}),
        "flagged_adversarial_true_gate_feeders": flagged_gate_feeders,
        "live_critical_spot_check_experiments": spot_checks.get("live_critical_artifacts", []),
        "no_flagged_adversarial_true_gate_feeder": not flagged_gate_feeders,
        "honest_verdict": _verdict(
            blocked_resource=blocked_resource,
            frozen_fover_auroc=frozen_fover,
            frozen_fover_auroc_unchanged=frozen_unchanged,
            flagged_gate_feeders=flagged_gate_feeders,
            unmet_gates=unmet,
            paper_ready=paper_ready,
        ),
        "field_provenance": {
            key: dict(value)
            for key, value in FIELD_PROVENANCE.items()
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
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
    provenance = artifact.get("field_provenance")
    _require(isinstance(provenance, Mapping), "field_provenance must be a mapping")
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(provenance), "missing field provenance")
    for key in ("g1", "g2", "g3", "g4", "paper_ready"):
        _require(isinstance(artifact[key], bool), f"{key} must be bool")
    _require(isinstance(artifact["unmet_gates"], list), "unmet_gates must be list")
    _require(
        artifact["frozen_fover_auroc"] is None
        or isinstance(artifact["frozen_fover_auroc"], (int, float)),
        "unexpected FoVer AUROC",
    )
    _require(artifact["random_seed"] == RANDOM_SEED, "wrong random_seed")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong inference_substrate")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")
    verdict = str(artifact["honest_verdict"])
    _require(verdict.startswith(("complete:", "blocked_")), "verdict must be terminal or blocked")
    if verdict == TERMINAL_VERDICT:
        _require(artifact["g1"] and artifact["g2"] and artifact["g3"] and artifact["g4"], "gate failed")
        _require(artifact["paper_ready"] is True, "paper_ready must be true")
        _require(artifact["unmet_gates"] == [], "terminal verdict requires no unmet gates")
        _require(artifact["frozen_fover_auroc"] == FROZEN_FOVER_AUROC, "FoVer moved")
        _require(
            artifact["flagged_adversarial_true_gate_feeders"] == [],
            "terminal verdict requires no flagged gate feeder",
        )


def run(
    root: Path | str = PROJECT_ROOT,
    *,
    publication_gate_run: Mapping[str, Any] | None = None,
    spot_check_result: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    executable: str | Path | None = None,
) -> Path:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        publication_gate_run=publication_gate_run,
        spot_check_result=spot_check_result,
        started_s=started_s,
        now_s=now_s,
        executable=executable,
    )
    validate_artifact(artifact)
    output_path = root_path / OUTPUT_REL_PATH
    write_json(output_path, artifact)
    return output_path
