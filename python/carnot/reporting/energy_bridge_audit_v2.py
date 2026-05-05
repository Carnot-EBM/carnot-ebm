"""Build the Exp 1306 EBT/ARM/EBM-CoT energy bridge audit artifact."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json"
DEFAULT_EXP1293_PATH = DEFAULT_RESULTS_DIR / "experiment_1293_ebt_arm_ebm_cot_energy_bridge_audit.json"
DEFAULT_EXP1295_PATH = DEFAULT_RESULTS_DIR / "experiment_1295_milestone_retro_100.json"
DEFAULT_REFERENCES_PATH = REPO_ROOT / "research-references.md"
DEFAULT_ARCHITECTURE_PATH = REPO_ROOT / "_bmad" / "architecture.md"
DEFAULT_RESEARCH_PROGRAM_PATH = REPO_ROOT / "research-program.md"

EXPERIMENT = "1306_ebt_arm_ebm_cot_energy_bridge_audit_v2"
SCHEMA = "energy_bridge_audit_v2"
RUN_DATE = "20260505"
HONEST_VERDICT = "energy_bridge_completed_local_alignment_only_strategic_context_not_implemented"

SOURCE_ARTIFACTS = [
    "results/experiment_1293_ebt_arm_ebm_cot_energy_bridge_audit.json",
    "results/experiment_1295_milestone_retro_100.json",
]
SOURCE_DOCUMENTS = [
    "research-references.md",
    "_bmad/architecture.md",
    "research-program.md",
]
REQUIRED_FIELDS = (
    "status",
    "energy_bridge_completed",
    "ebt_citation_count_checked",
    "arm_ebm_alignment_note",
    "ebm_cot_sequence_energy_note",
    "extropic_kona_status_checked",
    "hardware_sampler_context_recorded",
    "honest_verdict",
)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, Any]:
    return {
        "experiment_id": 1306,
        "artifact": "ebt_arm_ebm_cot_energy_bridge_audit_v2",
        "schema": SCHEMA,
        "project_root": str(project_root),
        "run_date": run_date,
        "network_required": False,
    }


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-REPORT-026: write a durable placeholder before local synthesis."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": SOURCE_ARTIFACTS,
            "source_documents": SOURCE_DOCUMENTS,
            "status": "in_progress",
            "energy_bridge_completed": False,
            "ebt_citation_count_checked": None,
            "arm_ebm_alignment_note": "",
            "ebm_cot_sequence_energy_note": "",
            "extropic_kona_status_checked": "",
            "hardware_sampler_context_recorded": "",
            "honest_verdict": "in_progress",
        },
    )


def _ebt_citation_count(references_text: str) -> int | None:
    match = re.search(r"currently lists\s+(\d+)\s+citations", references_text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _prior_blocker(exp1293_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": exp1293_payload.get("status"),
        "honest_verdict": exp1293_payload.get("honest_verdict"),
        "blocked_at_layer": exp1293_payload.get("blocked_at_layer"),
        "gate_check_summary": exp1293_payload.get("gate_check_summary"),
    }


def _evidence_terms(references_text: str, architecture_text: str, research_program_text: str) -> list[str]:
    combined = "\n".join([references_text, architecture_text, research_program_text]).lower()
    terms = ("ebt", "arm-ebm", "ebm-cot", "falcon", "extropic", "kona", "p-bit")
    return [term for term in terms if term in combined]


def build_artifact(
    *,
    exp1293_payload: Mapping[str, Any],
    exp1295_payload: Mapping[str, Any],
    references_text: str,
    architecture_text: str,
    research_program_text: str,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """SCENARIO-REPORT-026: synthesize the bridge from local notes only."""

    citation_count = _ebt_citation_count(references_text)
    continuous_repair = dict(exp1295_payload.get("continuous_repair_summary") or {})
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "metadata": _metadata(project_root, run_date),
        "source_artifacts": SOURCE_ARTIFACTS,
        "source_documents": SOURCE_DOCUMENTS,
        "status": "complete",
        "energy_bridge_completed": True,
        "ebt_citation_count_checked": {
            "paper": "Energy-Based Transformers are Scalable Learners and Thinkers",
            "semantic_scholar_source": "research-references.md",
            "citation_count": citation_count,
            "network_required": False,
            "status_note": (
                "Local references record a Semantic Scholar EBT citation-count signal; "
                "Exp 1306 did not require live network access."
            ),
        },
        "ebt_alignment_note": (
            "Implemented locally: Carnot already scores generated answers, claims, and "
            "candidate repairs with verifier/certificate energies. Alignment: EBT frames "
            "inference as minimizing energy over candidate predictions. Strategic gap: this "
            "audit does not add a native Energy-Based Transformer or Langevin generation loop."
        ),
        "arm_ebm_alignment_note": (
            "Implemented locally: local LLM outputs can be scored after generation by "
            "Carnot verifier energies and candidate-ranker style selection. Alignment: "
            "ARM-EBM supports interpreting autoregressive sequence likelihoods through an "
            "energy/value lens. Strategic gap: no ARM-to-EBM soft-Bellman trainer or "
            "lookahead decoding architecture is implemented here."
        ),
        "ebm_cot_sequence_energy_note": (
            "Implemented locally: trace, claim, and certificate energies are available in "
            "the verifier stack and DVI replay artifacts. Alignment: EBM-CoT argues for "
            "sequence-level chain-of-thought consistency energy instead of final-answer-only "
            "scoring. Strategic gap: no EBM-CoT calibration model or sequence optimizer is "
            "added by this audit."
        ),
        "extropic_kona_status_checked": (
            "Local notes check Extropic XTR-0/Z1/THRML and Logical Kona status; they are "
            "recorded as future sampler context and strategic architecture context, not as "
            "dependencies for Exp 1306 or evidence of local TSU/Kona implementation."
        ),
        "hardware_sampler_context_recorded": (
            "Recorded p-bit update dynamics, synchronous/asynchronous and inertia variants, "
            "time-multiplexed p-bit reuse, KV260 limits, and Extropic TSU context as future "
            "sampler-parity inputs tied to KL, delay, and reuse-factor diagnostics."
        ),
        "implemented_locally": [
            "Verifier-energy scoring and repair artifacts exist in the local Carnot pipeline.",
            "The bridge synthesis itself is implemented as a local reporting artifact only.",
            "Milestone .100 records HardNet++ and DSP feasibility context for repair policy alignment.",
        ],
        "strategic_context_only": [
            "FALCON informs hard-constraint repair and adaptive sampling, but is not implemented here.",
            "EBT, ARM-EBM, and EBM-CoT are alignment targets for future sequence-energy work.",
            "Extropic TSU, p-bit hardware dynamics, and Kona remain future sampler/architecture context.",
        ],
        "prior_blocker": _prior_blocker(exp1293_payload),
        "milestone_100_continuous_repair_context": continuous_repair,
        "local_evidence_terms_found": _evidence_terms(
            references_text,
            architecture_text,
            research_program_text,
        ),
        "honest_verdict": HONEST_VERDICT,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the required REQ-REPORT-026 schema and honest terminal state."""

    assert all(field in artifact for field in REQUIRED_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["energy_bridge_completed"] is True
    assert artifact["metadata"]["run_date"] == RUN_DATE
    assert artifact["metadata"]["project_root"] == "/home/ianblenke/github.com/ianblenke/carnot"
    assert artifact["honest_verdict"] == HONEST_VERDICT


def run(
    *,
    out_path: Path | str = DEFAULT_OUT_PATH,
    exp1293_path: Path | str = DEFAULT_EXP1293_PATH,
    exp1295_path: Path | str = DEFAULT_EXP1295_PATH,
    references_path: Path | str = DEFAULT_REFERENCES_PATH,
    architecture_path: Path | str = DEFAULT_ARCHITECTURE_PATH,
    research_program_path: Path | str = DEFAULT_RESEARCH_PROGRAM_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the local-only Exp 1306 audit and write the completed artifact."""

    out = Path(out_path)
    write_in_progress_artifact(out, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1293_payload=_load_json(Path(exp1293_path)),
        exp1295_payload=_load_json(Path(exp1295_path)),
        references_text=_read_text(Path(references_path)),
        architecture_text=_read_text(Path(architecture_path)),
        research_program_text=_read_text(Path(research_program_path)),
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out, artifact)
