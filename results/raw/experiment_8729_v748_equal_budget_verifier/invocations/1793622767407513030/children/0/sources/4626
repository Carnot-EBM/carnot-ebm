"""Package the Exp 3815 EDLM operator seed decision surface.

Spec refs: REQ-REPORT-3815, SCENARIO-REPORT-3815,
SCENARIO-REPORT-3815-MISSING-PREFLIGHT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting.archive_v345_activate_v346_3776 import (
    JsonDict,
    compact_verify_report,
    duration_from,
    is_sha256,
    no_forbidden_markers,
    payload_checksum,
    read_json_object,
    report_is_clean,
    sha256_path,
    write_payload,
    _ensure,
)
from scripts import adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3815_edlm_operator_seed_staging_package.json")
STAGING_NOTE_REL_PATH = Path("docs/research-notes/edlm-operator-seed-staging-20260604.md")
EXP3793_REL_PATH = Path("results/experiment_3793_edlm_no_train_preflight_readiness.json")
EXP3781_REL_PATH = Path("results/experiment_3781_edlm_next_thesis_feasibility_scoping.json")
MENU_REL_PATH = Path("docs/research-notes/phase3-alternative-thesis-menu.md")
RANDOM_SEED = 3815
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a documentation package "
    "over existing artifacts, no live model, no clone, no train)."
)
TERMINAL_VERDICT = (
    "complete: "
    "edlm_operator_seed_staged_one_command_seed_packaged_kill_gate_design_"
    "documented_loop_does_not_seed_operator_gated_operator_curated_doc_unedited"
)
BLOCKED_PREFLIGHT_VERDICT = "blocked_edlm_preflight_missing: exp3793 absolute artifact absent"
BLOCKED_INTERPRETER_VERDICT = "blocked_edlm_interpreter_not_venv_python: rerun under .venv/bin/python"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "staging_note_written",
    "operator_seed_command",
    "kill_gate_design_documented",
    "loop_does_not_seed",
    "edlm_remains_operator_gated",
    "operator_curated_doc_unedited",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the staging outcome.",
    "inference_substrate": (
        "Documentation package over existing artifacts; no clone, no train, "
        "no fresh model run."
    ),
    "staging_note_written": (
        "BARE bool -- the operator-ready staging note exists and is the core deliverable."
    ),
    "operator_seed_command": (
        "The exact one-command seed from Exp 3793 so the operator can seed .350 "
        "in one decision."
    ),
    "kill_gate_design_documented": (
        "BARE bool -- the note documents tiny-scale stability, equal-compute "
        "comparison, and honest-negative exit."
    ),
    "loop_does_not_seed": (
        "BARE bool, true -- no clone, no train, no model; seeding remains the "
        "operator's call."
    ),
    "edlm_remains_operator_gated": (
        "BARE bool, true -- the EDLM seed-vs-freeze decision stays an operator surface."
    ),
    "operator_curated_doc_unedited": (
        "BARE bool, true -- operator-curated documents were not edited."
    ),
    "cited_upstream_artifacts": "Provenance for Exp 3793 and Exp 3781.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def interpreter_is_venv_python(executable: str | Path) -> bool:
    """Return true when the workflow is pinned to `.venv/bin/python`."""

    path = Path(executable)
    return path.name == "python" and path.parent.name == "bin" and path.parent.parent.name == ".venv"


def build_staging_note(
    *,
    operator_seed_command: str,
    exp3793_path: Path,
    exp3781_path: Path,
    menu_path: Path,
) -> str:
    """Return the operator-facing EDLM seed staging note."""

    return (
        "# EDLM Operator Seed Staging - 2026-06-04\n\n"
        "**Status:** OPERATOR-GATED staging package. The loop does NOT seed EDLM: "
        "it clones nothing, trains nothing, and runs no model.\n\n"
        "## One-Command Seed\n\n"
        "```bash\n"
        f"{operator_seed_command}\n"
        "```\n\n"
        "This is the complete operator action needed to seed `.350`. Running it is "
        "the operator's call; this staging note does not run it.\n\n"
        "## Tiny-Scale Kill-Gate Design\n\n"
        "Part (a) stability mirrors Thesis-A `.341`: run only a tiny EDLM fit "
        "smoke under `.venv/bin/python` on the internal 3090, with a hard "
        "cuda-block before training. The first question is whether the tiny "
        "EDLM can train stably inside a bounded budget without Diverges/NaN "
        "events or obvious memory no-headroom.\n\n"
        "Part (b) is the matched-COMPUTE comparison. Compare residual EDLM "
        "generation against an autoregressive baseline at equal total inference "
        "FLOPs. EDLM's iterative diffusion plus sequence-level energy correction "
        "gets no free pass for extra decoding passes. A win counts only at equal "
        "compute, with the same P0.1/Thesis-A trap explicitly closed.\n\n"
        "Honest-negative exit: if tiny training diverges, produces NaN, cannot fit "
        "inside the bounded internal 3090 budget, or the AR/corpus setup has no "
        "headroom, EDLM is bounded at small scale for this route and STOP. Do not "
        "scale, do not reinterpret an invalid no-headroom comparison as a result.\n\n"
        "## Decision Readiness\n\n"
        "The operator can seed `.350` by running the one command above. Everything "
        "after that is the `.350 roadmap`: vendor+audit, tiny-EDLM fit smoke, "
        "matched-compute harness, and kill-gate verdict.\n\n"
        "## Boundary\n\n"
        "EDLM tests a different mechanism from both bounded routes: discrete "
        "diffusion with a sequence-level energy correction, not energy selection "
        "and not Thesis-A energy-as-sole-generator. This note does not claim EDLM "
        "sidesteps the bounds; it only stages the falsifiable operator seed.\n\n"
        "## Provenance\n\n"
        f"- Exp 3793 preflight: `{exp3793_path}`\n"
        f"- Exp 3781 feasibility scoping: `{exp3781_path}`\n"
        f"- Phase-3 menu framing: `{menu_path}`\n"
    )


def upstream_citation(experiment_id: int, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Return compact provenance for a cited upstream artifact."""

    return {
        "experiment_id": experiment_id,
        "absolute_path": str(path.resolve()),
        "sha256": sha256_path(path),
        "honest_verdict": payload.get("honest_verdict"),
        "random_seed": payload.get("random_seed"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
    }


def build_blocked_artifact(
    *,
    honest_verdict: str,
    duration_s: float,
    exp3793_path: Path,
    verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a blocked artifact without fabricating seed readiness."""

    payload: JsonDict = {
        "schema": "carnot.edlm_operator_seed_staging_3815.v1",
        "experiment_id": "exp3815",
        "task_id": "exp3815-edlm-operator-seed-staging",
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "staging_note_written": False,
        "operator_seed_command": None,
        "kill_gate_design_documented": False,
        "loop_does_not_seed": True,
        "edlm_remains_operator_gated": True,
        "operator_curated_doc_unedited": True,
        "cited_upstream_artifacts": [],
        "precondition_status": {
            "exp3793_absolute_path": str(exp3793_path.resolve()),
            "exp3793_present": exp3793_path.exists(),
        },
        "adversarial_verify_clean": report_is_clean(verify_report),
        "adversarial_verify_report": compact_verify_report(verify_report or {"flags": []}),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload, complete=False)
    return payload


def build_complete_artifact(
    *,
    root: Path,
    exp3793: Mapping[str, Any],
    exp3781: Mapping[str, Any],
    operator_seed_command: str,
    note_path: Path,
    duration_s: float,
    verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the complete Exp 3815 terminal artifact."""

    exp3793_path = root / EXP3793_REL_PATH
    exp3781_path = root / EXP3781_REL_PATH
    menu_path = root / MENU_REL_PATH
    payload: JsonDict = {
        "schema": "carnot.edlm_operator_seed_staging_3815.v1",
        "experiment_id": "exp3815",
        "task_id": "exp3815-edlm-operator-seed-staging",
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "staging_note_written": True,
        "operator_seed_command": operator_seed_command,
        "kill_gate_design_documented": True,
        "loop_does_not_seed": True,
        "edlm_remains_operator_gated": True,
        "operator_curated_doc_unedited": True,
        "cited_upstream_artifacts": [
            upstream_citation(3793, exp3793_path, exp3793),
            upstream_citation(3781, exp3781_path, exp3781),
        ],
        "staging_note_path": str(STAGING_NOTE_REL_PATH),
        "staging_note_sha256": sha256_path(note_path),
        "menu_framing_path": str(MENU_REL_PATH),
        "adversarial_verify_clean": report_is_clean(verify_report),
        "adversarial_verify_report": compact_verify_report(verify_report or {"flags": []}),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload, complete=True)
    return payload


def validate_artifact(artifact: Mapping[str, Any], *, complete: bool) -> None:
    """Validate the Exp 3815 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    _ensure(
        not [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles],
        "field_principles must cover every required field",
    )
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    _ensure(artifact.get("loop_does_not_seed") is True, "loop must not seed")
    _ensure(artifact.get("edlm_remains_operator_gated") is True, "EDLM must remain operator-gated")
    _ensure(artifact.get("operator_curated_doc_unedited") is True, "operator-curated docs must stay untouched")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    duration_s = artifact.get("duration_s")
    _ensure(isinstance(duration_s, int | float) and not isinstance(duration_s, bool), "duration_s numeric")
    _ensure(float(duration_s) >= 0.0001, "duration_s below plausibility floor")
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be sha256")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum mismatch")
    if complete:
        _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "complete verdict mismatch")
        _ensure(artifact.get("staging_note_written") is True, "staging note must be written")
        _ensure(artifact.get("kill_gate_design_documented") is True, "kill-gate design required")
        _ensure(isinstance(artifact.get("operator_seed_command"), str), "seed command required")
        _ensure(report_is_clean(artifact.get("adversarial_verify_report")), "critical verifier flag present")
    else:
        _ensure(str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked verdict required")
        _ensure(artifact.get("staging_note_written") is False, "blocked path must not write note")
        _ensure(artifact.get("operator_seed_command") is None, "blocked path must not fabricate seed")


def run(
    root: Path | str = REPO_ROOT,
    *,
    executable: str | None = None,
    output_path: Path | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    verify_runner: Callable[[Path], Mapping[str, Any]] | None = None,
) -> Path:
    """Write the Exp 3815 note and terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output = output_path or root_path / OUTPUT_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    verify = verify_runner or adversarial_verify.verify_artifact
    exp3793_path = (root_path / EXP3793_REL_PATH).resolve()
    duration_s = duration_from(start, now_s)

    if not interpreter_is_venv_python(executable or sys.executable):
        payload = build_blocked_artifact(
            honest_verdict=BLOCKED_INTERPRETER_VERDICT,
            duration_s=duration_s,
            exp3793_path=exp3793_path,
        )
        write_payload(output, payload)
        report = verify(output)
        payload["adversarial_verify_report"] = compact_verify_report(report)
        payload["adversarial_verify_clean"] = report_is_clean(report)
        payload["reproducibility_checksum"] = payload_checksum(payload)
        validate_artifact(payload, complete=False)
        write_payload(output, payload)
        return output

    if not exp3793_path.exists():
        payload = build_blocked_artifact(
            honest_verdict=BLOCKED_PREFLIGHT_VERDICT,
            duration_s=duration_s,
            exp3793_path=exp3793_path,
        )
        write_payload(output, payload)
        report = verify(output)
        payload["adversarial_verify_report"] = compact_verify_report(report)
        payload["adversarial_verify_clean"] = report_is_clean(report)
        payload["reproducibility_checksum"] = payload_checksum(payload)
        validate_artifact(payload, complete=False)
        write_payload(output, payload)
        return output

    exp3793 = read_json_object(exp3793_path)
    exp3781_path = (root_path / EXP3781_REL_PATH).resolve()
    exp3781 = read_json_object(exp3781_path)
    menu_path = (root_path / MENU_REL_PATH).resolve()
    menu_text = menu_path.read_text(encoding="utf-8")
    operator_seed_command = str(exp3793.get("operator_seed_command", ""))
    readiness_go = (
        exp3793.get("readiness_verdict") == "go"
        or "edlm_no_train_preflight_go" in str(exp3793.get("honest_verdict", ""))
    )
    _ensure(readiness_go, "Exp 3793 must record readiness GO")
    _ensure(exp3793.get("minimal_kill_gate_sound") is True, "Exp 3793 kill gate must be sound")
    _ensure(operator_seed_command.startswith("git clone "), "Exp 3793 seed command missing")
    _ensure("EDLM" in menu_text or "discrete diffusion" in menu_text, "Phase-3 menu framing missing")

    note_path = root_path / STAGING_NOTE_REL_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        build_staging_note(
            operator_seed_command=operator_seed_command,
            exp3793_path=exp3793_path,
            exp3781_path=exp3781_path,
            menu_path=menu_path,
        ),
        encoding="utf-8",
    )

    payload = build_complete_artifact(
        root=root_path,
        exp3793=exp3793,
        exp3781=exp3781,
        operator_seed_command=operator_seed_command,
        note_path=note_path,
        duration_s=duration_s,
    )
    write_payload(output, payload)
    report = verify(output)
    payload["adversarial_verify_report"] = compact_verify_report(report)
    payload["adversarial_verify_clean"] = report_is_clean(report)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload, complete=True)
    write_payload(output, payload)
    return output
