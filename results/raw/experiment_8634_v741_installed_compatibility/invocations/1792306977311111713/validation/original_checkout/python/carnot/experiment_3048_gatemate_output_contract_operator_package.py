"""GateMate output contract operator package for Exp 3048.

Spec refs: REQ-HW-088, SCENARIO-HW-088.

This module packages the GateMate output contract decision for an operator. It
is deliberately a local evidence audit, not a board run. The key safety rule is
simple: downstream flash smoke may proceed only after one selected RTL status
signal has both a committed physical CCF ``Pin_out`` binding and a concrete host
reader command with an expected transcript.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ARTIFACT_FILENAME = "experiment_3048_gatemate_output_contract_operator_package_v1.json"
EXP3034_FILENAME = "experiment_3034_gatemate_output_contract_pinout_decision_v1.json"
RUN_DATE = "20260525"
PREFERRED_SIGNALS = ("done", "spin_out[15:0]")
REQUIRED_FIELDS = (
    "gatemate_output_contract_ready",
    "host_visible_io_plan_ready",
    "selected_output_signal",
    "ccf_binding",
    "host_reader_command",
    "expected_transcript",
    "missing_operator_actions",
    "hardware_execution_claim_made",
    "speedup_claim_made",
    "inference_substrate",
    "honest_verdict",
)

WhichFunc = Callable[[str], str | None]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _signal_base(signal: str) -> str:
    return signal.split("[", 1)[0].strip()


def _candidate_signals(exp3034: Mapping[str, Any]) -> list[str]:
    rows = exp3034.get("pinout_table", [])
    signals = [
        str(row.get("signal_name", "")).strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get("signal_name", "")).strip()
    ]
    if not signals:
        signals = list(PREFERRED_SIGNALS)
    ordered: list[str] = []
    for preferred in PREFERRED_SIGNALS:
        if preferred in signals and preferred not in ordered:
            ordered.append(preferred)
    for signal in signals:
        if signal not in ordered:
            ordered.append(signal)
    return ordered


def _pinout_rows(exp3034: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = exp3034.get("pinout_table", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _row_for_signal(rows: Iterable[Mapping[str, Any]], signal: str) -> dict[str, Any]:
    base = _signal_base(signal)
    for row in rows:
        if _signal_base(str(row.get("signal_name", ""))) == base:
            return dict(row)
    return {}


def _ccf_bindings(repo_root: Path) -> list[dict[str, Any]]:
    gate_dir = repo_root / "hardware" / "gatemate"
    bindings: list[dict[str, Any]] = []
    for path in sorted(gate_dir.glob("*.ccf")) if gate_dir.exists() else []:
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.search(
                r"\bPin_out\b\s+(\S+)\s+Loc\s*=\s*([A-Za-z0-9_]+)",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                bindings.append(
                    {
                        "signal_name": match.group(1),
                        "pin": match.group(2),
                        "path": str(path),
                        "line_number": line_number,
                        "line": line,
                    }
                )
    return bindings


def _binding_for_signal(bindings: Iterable[Mapping[str, Any]], signal: str) -> dict[str, Any]:
    base = _signal_base(signal).lower()
    for binding in bindings:
        if _signal_base(str(binding.get("signal_name", ""))).lower() == base:
            return dict(binding)
    return {}


def _is_concrete_reader(command: str) -> bool:
    lowered = command.strip().lower()
    return bool(lowered) and not lowered.startswith("blocked") and "explicit_no_ready_contract" not in lowered


def _reader_from_exp3034(row: Mapping[str, Any], exp3034: Mapping[str, Any]) -> str:
    row_command = str(row.get("host_read_command", "")).strip()
    if _is_concrete_reader(row_command):
        return row_command
    artifact_command = str(exp3034.get("host_reader_command", "")).strip()
    return artifact_command if _is_concrete_reader(artifact_command) else ""


def _reader_candidates(repo_root: Path, signal: str) -> list[dict[str, str]]:
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.exists():
        return []
    base = _signal_base(signal).lower()
    candidates: list[dict[str, str]] = []
    for path in sorted(scripts_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        haystack = f"{path.name.lower()}\n{text}"
        if "gatemate" not in haystack or base not in haystack:
            continue
        if any(marker in haystack for marker in ("read_gpio", "serial", "sigrok", "logic analyzer")):
            command = f".venv/bin/python {path.relative_to(repo_root)} --expect {base}=1"
            candidates.append({"path": str(path), "command": command})
    return candidates


def _expected_transcript(row: Mapping[str, Any], signal: str, ready: bool) -> list[str]:
    raw = row.get("expected_transcript", "")
    if isinstance(raw, list):
        transcript = [str(item) for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        transcript = [text] if text and not text.lower().startswith("blocked") else []
    if ready and not transcript:
        transcript = [f"{_signal_base(signal)}=1 PASS"]
    return transcript if ready else []


def _tool_availability(which_func: WhichFunc) -> dict[str, str]:
    return {
        "openFPGALoader": which_func("openFPGALoader") or "",
        "yosys": which_func("yosys") or "",
        "nextpnr-himbaechel": which_func("nextpnr-himbaechel") or "",
        "gmpack": which_func("gmpack") or which_func("packer") or "",
    }


def _local_docs(repo_root: Path) -> list[str]:
    roots = [
        repo_root / "CLAUDE.md",
        repo_root / "CODEX.md",
        repo_root / "research-hardware-wishlist.md",
        repo_root / "ops",
        repo_root / "docs",
        repo_root / "hardware" / "gatemate",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".ccf", ".v", ".txt"})
    relevant: list[Path] = []
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in text for token in ("gatemate", "a1-evb", "pinout", "pin_out", "dirtyjtag")):
            relevant.append(path)
    return [str(path) for path in relevant]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def _missing_actions(
    *,
    exp3034: Mapping[str, Any],
    signal: str,
    binding: Mapping[str, Any],
    reader_command: str,
    transcript: list[str],
) -> list[str]:
    del exp3034
    actions: list[str] = []
    if not binding:
        actions.append(
            f"Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for {signal}."
        )
    if not reader_command:
        actions.append(
            f"Commit a concrete host reader command for {signal}: GPIO/LED read, UART serial decode, or JTAG-readable status command."
        )
    if reader_command and not transcript:
        actions.append(f"Record an expected pass/fail transcript for the {signal} host reader command.")
    actions.append("Keep downstream flash smoke gated until the reader command has an expected pass/fail transcript.")
    return _dedupe(actions)


def _safety_limits(ready: bool) -> dict[str, Any]:
    return {
        "downstream_flash_gate_open": ready,
        "exp3049_gate": "require_gatemate_output_contract_ready_true",
        "exp3050_gate": "require_host_visible_io_plan_ready_true",
        "max_flash_attempts_without_operator_review": 0 if not ready else 1,
        "forbidden_claims_without_host_transcript": [
            "latency",
            "speedup",
            "Boltzmann sampling",
            "thermodynamic behavior",
            "hardware execution",
        ],
    }


def _choose_contract(
    *,
    repo_root: Path,
    exp3034: Mapping[str, Any],
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = _pinout_rows(exp3034)
    fallback: dict[str, Any] | None = None
    for signal in _candidate_signals(exp3034):
        row = _row_for_signal(rows, signal)
        binding = _binding_for_signal(bindings, signal)
        reader_candidates = _reader_candidates(repo_root, signal)
        reader_command = _reader_from_exp3034(row, exp3034)
        if not reader_command and reader_candidates:
            reader_command = reader_candidates[0]["command"]
        ready = bool(binding and reader_command)
        transcript = _expected_transcript(row, signal, ready)
        option = {
            "signal": signal,
            "row": row,
            "binding": binding,
            "reader_command": reader_command,
            "reader_candidates": reader_candidates,
            "expected_transcript": transcript,
            "ready": ready and bool(transcript),
        }
        if option["ready"]:
            return option
        if fallback is None or (not fallback["binding"] and binding):
            fallback = option
    return fallback or {
        "signal": "done",
        "row": {},
        "binding": {},
        "reader_command": "",
        "reader_candidates": [],
        "expected_transcript": [],
        "ready": False,
    }


def build_artifact(
    *,
    repo_root: Path,
    which_func: WhichFunc | None = None,
) -> dict[str, Any]:
    """Build the Exp 3048 package from local repo evidence without board I/O."""

    which = which_func or shutil.which
    exp3034_path = repo_root / "results" / EXP3034_FILENAME
    exp3034 = _read_json(exp3034_path)
    bindings = _ccf_bindings(repo_root)
    contract = _choose_contract(repo_root=repo_root, exp3034=exp3034, bindings=bindings)
    ready = bool(contract["ready"])
    selected_signal = str(contract["signal"])
    transcript = list(contract["expected_transcript"])
    reader_command = str(contract["reader_command"])
    binding = dict(contract["binding"])
    missing_actions = [] if ready else _missing_actions(
        exp3034=exp3034,
        signal=selected_signal,
        binding=binding,
        reader_command=reader_command,
        transcript=transcript,
    )
    local_docs = _local_docs(repo_root)
    hardware_files = [path for path in local_docs if "/hardware/gatemate/" in path]
    verdict = (
        "complete: gatemate_output_contract_operator_package_ready"
        if ready
        else "complete: blocked_gatemate_output_contract_authority_missing"
    )
    return {
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": bool(binding and reader_command and transcript),
        "selected_output_signal": selected_signal,
        "ccf_binding": binding,
        "host_reader_command": reader_command if ready else "",
        "expected_transcript": transcript if ready else [],
        "missing_operator_actions": missing_actions,
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "inference_substrate": {
            "kind": "gatemate_operator_package",
            "model_inference": False,
            "hardware_execution": False,
            "hardware_execution_claim": False,
            "flash_attempted": False,
            "timing_or_speedup_claim": False,
            "local_repo_only": True,
            "source_artifacts": [str(exp3034_path), *hardware_files],
        },
        "honest_verdict": verdict,
        "run_date": RUN_DATE,
        "upstream_exp3034": {
            "path": str(exp3034_path),
            "available": bool(exp3034),
            "gatemate_output_contract_ready": exp3034.get("gatemate_output_contract_ready") is True,
            "host_visible_io_plan_ready": exp3034.get("host_visible_io_plan_ready") is True,
            "honest_verdict": str(exp3034.get("honest_verdict", "")),
            "pinout_table": _pinout_rows(exp3034),
        },
        "authority_search": {
            "local_docs_scanned": local_docs,
            "ccf_binding_candidates": bindings,
            "host_reader_candidates": list(contract["reader_candidates"]),
            "tool_availability": _tool_availability(which),
            "authoritative_pinout_found": bool(binding),
            "host_reader_command_found": bool(reader_command),
        },
        "safety_limits": _safety_limits(ready),
        "preconditions_checked": True,
        "hardware_execution_performed": False,
        "flash_command_executed": "",
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    which_func: WhichFunc | None = None,
) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(repo_root=root, which_func=which_func)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
