#!/usr/bin/env python3
"""Update the E2E plan for EBRM scorer and SMGI update verification.

The plan is the operator-facing checklist for full-system behavior, so this
script keeps the update deterministic: it appends the missing EBRM and SMGI
sections once, validates that the source experiment artifacts support those
checks, and writes a terminal JSON artifact that the conductor can inspect.

Spec: REQ-LEARN-1664, SCENARIO-LEARN-1664, SCENARIO-LEARN-1665.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = 1664
RUN_DATE = "20260510"
PLAN_LAST_UPDATED = "2026-05-10"
SCHEMA = "carnot.e2e_plan_update.v1"
SPEC_TRACES = ("REQ-LEARN-1664", "SCENARIO-LEARN-1664")

DEFAULT_PLAN_PATH = REPO_ROOT / "ops" / "e2e-test-plan.md"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1664_e2e_plan.json"
DEFAULT_EBRM_SCORER_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1656_ebrm_trace_scorer.json"
DEFAULT_KV260_BINDING_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_1657_kv260_ebrm_binding.json"
)
DEFAULT_HW_EVAL_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1658_hw_eval.json"
DEFAULT_SMGI_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1659_smgi_certified_updates.json"
SCORE_DELTA_TOLERANCE = 1e-6
ACCURACY_GATE = 0.8

REQUIRED_SECTION_IDS = ("E2E-006", "E2E-007")

PLAN_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "E2E-006",
        "EBRM Trace Scorer CPU/KV260 Verification",
        """
### E2E-006: EBRM Trace Scorer CPU/KV260 Verification

**Objective:** Verify that extracted logical traces are scored by the CPU EBRM
scorer and the KV260 q=3 Potts backend with matching energy results and
auditable per-case provenance.

**Spec refs:** `REQ-VERIFY-1656`, `SCENARIO-VERIFY-1656`,
`REQ-VERIFY-1657`, `SCENARIO-VERIFY-1657`, `REQ-VERIFY-1658`,
`SCENARIO-VERIFY-1658`.

**Source artifacts:** `results/experiment_1656_ebrm_trace_scorer.json`,
`results/experiment_1657_kv260_ebrm_binding.json`,
`results/experiment_1658_hw_eval.json`.

**Steps:**
1. Confirm the Exp 1656 CPU scorer artifact is complete, uses continuous
   energy, and reports `score_accuracy >= 0.8`.
2. Confirm the Exp 1657 KV260 binding artifact is complete, uses q=3 Potts
   states, and records whether hardware execution or software fallback was
   used.
3. Run or inspect Exp 1658 on bounded local SOTA output rows and compare CPU
   and KV260 energies over the same trace batch.
4. Verify every case score includes CPU energy, KV260 energy, absolute score
   delta, backend provenance, and Potts state metadata.

**Pass criteria:** Exp 1656, Exp 1657, and Exp 1658 artifacts are complete;
CPU/KV260 `max_score_delta <= 1e-6`; CPU and KV260 scoring accuracy match;
`scoring_delta_within_tolerance=true`; and no hardware execution claim is made
unless authenticated hardware evidence is present.
""",
    ),
    (
        "E2E-007",
        "SMGI Certified Update Verification",
        """
### E2E-007: SMGI Certified Update Verification

**Objective:** Verify that SMGI policy and memory updates become reusable only
when CerCE certificate evidence, replay retention, SessionMemory hash changes,
and model-weight immutability gates all pass.

**Spec refs:** `REQ-LEARN-1659`, `SCENARIO-LEARN-1659`,
`SCENARIO-LEARN-1660`.

**Source artifacts:** `results/experiment_1659_smgi_certified_updates.json`.

**Steps:**
1. Confirm the Exp 1659 artifact is complete and
   `continuous_self_learning_task=true`.
2. Verify the CerCE ledger gates report `accepted_violation_count=0`,
   `false_accept_delta <= 0`, `soundness_mistakes=0`, and
   `nonforgetting_certificate_rate=1.0`.
3. Inspect every certified update for matching certificate ID, present and
   changed SessionMemory hashes, full replay retention, zero replay failures,
   provenance, and `no_model_weight_mutation=true`.
4. Verify unsafe candidates remain in `rejected_updates` and never contribute
   to `certified_update_success=true`.

**Pass criteria:** `smgi_certified_update_ready=true`,
`certified_update_success=true`, at least one certified update is present,
all certified updates pass replay and hash gates, and no update mutates model
weights.
""",
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "run_date",
    "plan_path",
    "e2e_sections_added",
    "e2e_section_ids",
    "ebrm_e2e_ready",
    "smgi_e2e_ready",
    "source_artifacts",
    "plan_hash_before",
    "plan_hash_after",
    "plan_updated",
    "spec_traces",
    "tests_run",
    "blockers",
    "honest_verdict",
)


def stable_hash(text: str) -> str:
    """Return a stable hash for plan text or JSON evidence."""

    return sha256(text.encode("utf-8")).hexdigest()


def update_plan_text(plan_text: str) -> tuple[str, list[str]]:
    """Append missing Exp 1664 E2E sections without duplicating existing ones."""

    updated = plan_text.rstrip() + "\n"
    added: list[str] = []
    for section_id, title, section_text in PLAN_SECTIONS:
        heading = f"### {section_id}: {title}"
        if heading not in updated:
            updated = f"{updated}\n{section_text.strip()}\n"
            added.append(section_id)
    if added:
        updated = _update_last_updated(updated)
    return updated, added


def run_experiment(
    *,
    plan_path: Path | str = DEFAULT_PLAN_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    ebrm_scorer_artifact_path: Path | str = DEFAULT_EBRM_SCORER_ARTIFACT_PATH,
    kv260_binding_artifact_path: Path | str = DEFAULT_KV260_BINDING_ARTIFACT_PATH,
    hw_eval_artifact_path: Path | str = DEFAULT_HW_EVAL_ARTIFACT_PATH,
    smgi_artifact_path: Path | str = DEFAULT_SMGI_ARTIFACT_PATH,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Update the E2E plan and write the Exp 1664 terminal artifact."""

    plan_file = Path(plan_path)
    before_text = plan_file.read_text(encoding="utf-8")
    after_text, sections_added = update_plan_text(before_text)
    if after_text != before_text:
        plan_file.write_text(after_text, encoding="utf-8")

    ebrm_ready, ebrm_sources, ebrm_blockers = evaluate_ebrm_sources(
        ebrm_scorer_artifact_path,
        kv260_binding_artifact_path,
        hw_eval_artifact_path,
    )
    smgi_ready, smgi_sources, smgi_blockers = evaluate_smgi_source(smgi_artifact_path)
    source_artifacts = {**ebrm_sources, **smgi_sources}
    blockers = ebrm_blockers + smgi_blockers
    complete = bool(ebrm_ready and smgi_ready and not blockers)

    artifact: JsonDict = {
        "status": "complete" if complete else "blocked",
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "plan_path": str(plan_file),
        "e2e_sections_added": sections_added,
        "e2e_section_ids": list(REQUIRED_SECTION_IDS),
        "ebrm_e2e_ready": ebrm_ready,
        "smgi_e2e_ready": smgi_ready,
        "source_artifacts": source_artifacts,
        "plan_hash_before": stable_hash(before_text),
        "plan_hash_after": stable_hash(after_text),
        "plan_updated": after_text != before_text,
        "spec_traces": list(SPEC_TRACES),
        "tests_run": list(tests_run),
        "blockers": blockers,
        "honest_verdict": (
            "complete: e2e_plan_updated_for_ebrm_smgi"
            if complete
            else "blocked: e2e_plan_source_evidence_incomplete"
        ),
    }
    validate_artifact(artifact)
    return _write_json(Path(output_path), artifact)


def evaluate_ebrm_sources(
    ebrm_scorer_artifact_path: Path | str,
    kv260_binding_artifact_path: Path | str,
    hw_eval_artifact_path: Path | str,
) -> tuple[bool, dict[str, JsonDict], list[str]]:
    """Evaluate source evidence for the EBRM scorer/hardware E2E scenario."""

    exp1656 = _read_json(Path(ebrm_scorer_artifact_path))
    exp1657 = _read_json(Path(kv260_binding_artifact_path))
    exp1658 = _read_json(Path(hw_eval_artifact_path))
    sources = {
        "exp1656": _source_report(
            "Exp 1656",
            ebrm_scorer_artifact_path,
            exp1656,
            _exp1656_ready(exp1656),
        ),
        "exp1657": _source_report(
            "Exp 1657",
            kv260_binding_artifact_path,
            exp1657,
            _exp1657_ready(exp1657),
        ),
        "exp1658": _source_report(
            "Exp 1658",
            hw_eval_artifact_path,
            exp1658,
            _exp1658_ready(exp1658),
        ),
    }
    blockers = [
        f"{source['label']} source artifact is missing or incomplete"
        for source in sources.values()
        if not source["ready"]
    ]
    return not blockers, sources, blockers


def evaluate_smgi_source(
    smgi_artifact_path: Path | str,
) -> tuple[bool, dict[str, JsonDict], list[str]]:
    """Evaluate source evidence for the SMGI certified-update E2E scenario."""

    exp1659 = _read_json(Path(smgi_artifact_path))
    ready = _exp1659_ready(exp1659)
    sources = {
        "exp1659": _source_report("Exp 1659", smgi_artifact_path, exp1659, ready),
    }
    blockers = [] if ready else ["Exp 1659 source artifact is missing or incomplete"]
    return ready, sources, blockers


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal JSON artifact consumed by the conductor."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"schema mismatch: {artifact['schema']}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment_id mismatch")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["e2e_section_ids"] != list(REQUIRED_SECTION_IDS):
        raise AssertionError("e2e_section_ids mismatch")
    if artifact["spec_traces"] != list(SPEC_TRACES):
        raise AssertionError("spec_traces mismatch")
    if artifact["status"] == "complete":
        if artifact["ebrm_e2e_ready"] is not True or artifact["smgi_e2e_ready"] is not True:
            raise AssertionError("complete artifact requires EBRM and SMGI source evidence")
        if artifact["blockers"]:
            raise AssertionError("complete artifact cannot contain blockers")
    if artifact["status"] == "blocked" and not artifact["blockers"]:
        raise AssertionError("blocked artifact requires blockers")


def _exp1656_ready(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == "complete"
        and payload.get("ebrm_trace_scorer_ready") is True
        and payload.get("continuous_energy_used") is True
        and _float(payload.get("score_accuracy")) >= ACCURACY_GATE
    )


def _exp1657_ready(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == "complete"
        and payload.get("kv260_ebrm_binding_ready") is True
        and payload.get("continuous_energy_used") is True
        and _int(payload.get("potts_q_states")) == 3
        and _float(payload.get("score_accuracy")) >= ACCURACY_GATE
    )


def _exp1658_ready(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == "complete"
        and _int(payload.get("cases_total")) > 0
        and payload.get("scoring_delta_within_tolerance") is True
        and _float(payload.get("max_score_delta")) <= SCORE_DELTA_TOLERANCE
        and _float(payload.get("cpu_score_accuracy")) == _float(payload.get("kv260_score_accuracy"))
    )


def _exp1659_ready(payload: Mapping[str, Any]) -> bool:
    updates = payload.get("certified_updates")
    return bool(
        payload.get("status") == "complete"
        and payload.get("continuous_self_learning_task") is True
        and payload.get("smgi_certified_update_ready") is True
        and payload.get("certified_update_success") is True
        and payload.get("cerce_ledger_ready") is True
        and _int(payload.get("accepted_violation_count")) == 0
        and _int(payload.get("false_accept_delta")) <= 0
        and _int(payload.get("soundness_mistakes")) == 0
        and _float(payload.get("nonforgetting_certificate_rate")) == 1.0
        and isinstance(updates, list)
        and bool(updates)
        and all(_certified_update_ready(update) for update in updates)
    )


def _certified_update_ready(update: Any) -> bool:
    if not isinstance(update, Mapping):
        return False
    gates = update.get("gates")
    if not isinstance(gates, Mapping):
        return False
    return bool(
        update.get("no_model_weight_mutation") is True
        and update.get("prior_memory_hash")
        and update.get("updated_memory_hash")
        and update.get("prior_memory_hash") != update.get("updated_memory_hash")
        and _int(update.get("replay_case_count")) >= 1
        and _int(update.get("retained_case_count")) >= _int(update.get("replay_case_count"))
        and _int(update.get("replay_failure_count")) == 0
        and bool(update.get("provenance"))
        and gates.get("cerce_certificate_match") is True
        and gates.get("memory_hashes_present") is True
        and gates.get("memory_hash_changed") is True
        and gates.get("retention_replay_passed") is True
        and gates.get("no_model_weight_mutation") is True
    )


def _source_report(
    label: str,
    path: Path | str,
    payload: Mapping[str, Any],
    ready: bool,
) -> JsonDict:
    source_path = Path(path)
    return {
        "label": label,
        "path": str(source_path),
        "exists": source_path.exists(),
        "status": str(payload.get("status", "missing")),
        "ready": ready,
        "experiment_id": payload.get("experiment_id") or payload.get("experiment"),
        "spec_traces": list(payload.get("spec_traces", [])),
        "honest_verdict": str(payload.get("honest_verdict", "")),
    }


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _update_last_updated(text: str) -> str:
    replacement = f"**Last Updated:** {PLAN_LAST_UPDATED}"
    if re.search(r"\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}", text):
        return re.sub(r"\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}", replacement, text, count=1)
    return f"{replacement}\n\n{text}"  # pragma: no cover - current plan has the date line.


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    artifact = run_experiment(plan_path=args.plan_path, output_path=args.output_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
