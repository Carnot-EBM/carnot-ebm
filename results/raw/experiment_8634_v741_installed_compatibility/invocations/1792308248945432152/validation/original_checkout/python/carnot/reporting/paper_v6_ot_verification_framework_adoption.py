"""Build the Exp 1579 paper-v6 OT verification adoption artifact.

The ICLR 2026 OT verification paper gives Carnot cleaner words for verifier
geometry, but it does not prove Carnot's sampler or cascade claims. This module
keeps the adoption mechanical: write the startup artifact, render the mapping
note, record the conflicts that soften paper-v6 claims, and leave publication
actions disabled.

Spec refs: REQ-PUBLISH-023, SCENARIO-PUBLISH-025.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260508"
SCHEMA = "paper_v6_ot_verification_framework_adoption_v1"
EXPERIMENT = "1579_iclr26_ot_verification_framework_paper_v6_adoption"
DEFAULT_OUT_REL = Path(
    "results/experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json"
)
DEFAULT_NOTE_REL = Path("docs/research-notes/paper-v6-ot-verification-framework-adoption.md")
REQUESTED_PAPER_REL = Path("docs/papers/paper-v6/main.tex")
ARXIV_SOURCE = "https://arxiv.org/abs/2510.18982"
HONEST_VERDICT = "ot_framework_adopted_with_conflict_ledger_no_publication_trigger"

DEFAULT_OUT_PATH = REPO_ROOT / DEFAULT_OUT_REL
DEFAULT_NOTE_PATH = REPO_ROOT / DEFAULT_NOTE_REL

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "adoption_note_path",
    "ot_framework_adopted",
    "claim_conflict_count",
    "paper_patch_applied",
    "no_publication_trigger",
    "honest_verdict",
}

_OT_MAPPINGS: dict[str, dict[str, str]] = {
    "coverage": {
        "paper_term": (
            "Coverage is the constraint that the verifier-induced target policy "
            "remain supported by the generator proposal distribution."
        ),
        "carnot_mapping": (
            "Carnot's coverage is the generator proposal mass available to the "
            "candidate verifier cascade: local SOTA outputs, candidate warm-start "
            "states, and THRML/Soft-Gibbs finite-K neighborhoods."
        ),
        "boundary": (
            "Do not identify Carnot's Soft-Gibbs residual beta with the OT coverage "
            "budget beta; they govern different objects."
        ),
    },
    "ROC": {
        "paper_term": (
            "ROC is the verifier acceptance region relative to ground truth, tracked "
            "through TPR, FPR, and Youden's index J = TPR - FPR."
        ),
        "carnot_mapping": (
            "Carnot's cascade has an effective ROC composed from thresholded "
            "deterministic validators, energy verifiers, and short-circuit exits."
        ),
        "boundary": (
            "AUROC on one corpus is not the same as an operating ROC for a deployment "
            "threshold on SOTA outputs."
        ),
    },
    "sub-optimality": {
        "paper_term": (
            "Sub-optimality is the reward gap between the ideal verifier-induced "
            "target distribution and the distribution induced by a sampling algorithm."
        ),
        "carnot_mapping": (
            "Carnot's sub-optimality is the finite-K gap between the ideal validated "
            "acceptance set and the outputs reachable by the candidate-warm-start "
            "THRML plus Soft-Gibbs cascade."
        ),
        "boundary": (
            "Finite-K sampling, finite batch size, and imperfect ROC leave residual "
            "gap; paper-v6 should not state zero sub-optimality."
        ),
    },
}

_CLAIM_CONFLICTS: list[dict[str, str]] = [
    {
        "claim_id": "CONFLICT-1",
        "claim": "AND-composing more verifiers eliminates reward hacking.",
        "reason": (
            "Verifier ROC and measured verifier correlation control the effective "
            "acceptance region; exp1256 already shows k_eff is much smaller than "
            "nominal k."
        ),
        "softened_boundary": (
            "Paper-v6 should say the measured k=5 stack narrows the acceptance "
            "region, not that arbitrary k-composition eliminates gaming."
        ),
    },
    {
        "claim_id": "CONFLICT-2",
        "claim": "The finite-K sampler draws from the verifier target distribution.",
        "reason": (
            "The OT framework treats algorithm-induced distributions separately from "
            "the target; finite-K THRML, warm-start, BRS, and Soft-Gibbs runs still "
            "leave sub-optimality."
        ),
        "softened_boundary": (
            "Paper-v6 should describe candidate warm-start and THRML vendoring as "
            "finite-K implementation choices, not exact sampling from the OT target."
        ),
    },
    {
        "claim_id": "CONFLICT-3",
        "claim": "The Soft-Gibbs Residual coverage bound is an OT coverage theorem.",
        "reason": (
            "The residual beta is an inverse-temperature on verifier failures, while "
            "OT coverage beta constrains proposal-policy support."
        ),
        "softened_boundary": (
            "Paper-v6 should keep the Jensen acceptance bound as a Carnot residual "
            "result and reserve OT coverage language for proposal support."
        ),
    },
    {
        "claim_id": "CONFLICT-4",
        "claim": "High verifier AUROC implies robust deployment verification.",
        "reason": (
            "Verifier ROC is distribution- and threshold-dependent; exp1100/1120 "
            "show SOTA-output inversion and a corpus-bounded retrain fix."
        ),
        "softened_boundary": (
            "Paper-v6 should report FoVer and SOTA-inclusive calibration as local "
            "evidence, not universal verifier dominance."
        ),
    },
    {
        "claim_id": "CONFLICT-5",
        "claim": "More sampling compute monotonically improves verified outputs.",
        "reason": (
            "The OT framework splits transport, policy-improvement, and saturation "
            "regimes; verifier ROC and coverage determine whether extra samples "
            "reduce sub-optimality."
        ),
        "softened_boundary": (
            "Paper-v6 should say extra samples help only in the measured regime and "
            "under the measured verifier ROC."
        ),
    },
    {
        "claim_id": "CONFLICT-6",
        "claim": "THRML vendoring supplies sampler security or hardware execution.",
        "reason": (
            "OT verifier geometry is not a hardware-security proof, and exp1561 "
            "falsified THRML kinetic-security parity on the zero-coupling fixture."
        ),
        "softened_boundary": (
            "Paper-v6 should treat THRML as software sampler alignment while keeping "
            "kinetic security and Extropic hardware execution as open or absent."
        ),
    },
]


def default_ot_mappings() -> dict[str, dict[str, str]]:
    """Return the conservative OT-to-Carnot vocabulary mapping."""

    return deepcopy(_OT_MAPPINGS)


def default_claim_conflicts() -> list[dict[str, str]]:
    """Return the paper-v6 claims that need softer wording under OT geometry."""

    return deepcopy(_CLAIM_CONFLICTS)


def write_in_progress_artifact(out_path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-PUBLISH-023: write a schema-shaped startup marker before analysis."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-PUBLISH-023", "SCENARIO-PUBLISH-025"],
        "arxiv_source": ARXIV_SOURCE,
        "status": "in_progress",
        "adoption_note_path": "",
        "ot_framework_adopted": False,
        "claim_conflict_count": 0,
        "paper_patch_applied": False,
        "no_publication_trigger": True,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def build_artifact(
    *,
    adoption_note_path: str,
    claim_conflicts: Sequence[Mapping[str, str]],
    paper_patch_applied: bool,
) -> dict[str, Any]:
    """Build the terminal artifact after the adoption note is rendered."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-PUBLISH-023", "SCENARIO-PUBLISH-025"],
        "arxiv_source": ARXIV_SOURCE,
        "status": "complete",
        "adoption_note_path": adoption_note_path,
        "ot_framework_adopted": True,
        "claim_conflict_count": len(claim_conflicts),
        "paper_patch_applied": paper_patch_applied,
        "no_publication_trigger": True,
        "honest_verdict": HONEST_VERDICT,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that omit the ledger or accidentally trigger publication."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if artifact["ot_framework_adopted"] is not True:
        raise ValueError("ot_framework_adopted must be true")
    if artifact["claim_conflict_count"] < 4:
        raise ValueError("claim_conflict_count must record at least four conflicts")
    if artifact["no_publication_trigger"] is not True:
        raise ValueError("no_publication_trigger must remain true")
    if artifact["honest_verdict"] != HONEST_VERDICT:
        raise ValueError("honest_verdict must preserve the OT conflict-ledger boundary")


def render_note(
    *,
    mappings: Mapping[str, Mapping[str, str]],
    claim_conflicts: Sequence[Mapping[str, str]],
    paper_patch_applied: bool,
    requested_source_exists: bool,
) -> str:
    """Render the reviewer-facing adoption note."""

    source_state = "present" if requested_source_exists else "absent"
    patch_sentence = (
        "No patch was applied because docs/papers/paper-v6/main.tex is absent."
        if not requested_source_exists
        else "No patch was applied because no isolated insertion point was selected."
    )
    lines = [
        "# Paper v6 OT Verification Framework Adoption",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Source paper: arXiv:2510.18982 ({ARXIV_SOURCE})",
        f"Requested paper source state: `{source_state}`",
        "",
        "This note adopts the OT verification vocabulary for paper-v6 without "
        "turning it into a Carnot performance theorem. The safe reading is: "
        "Carnot has a verifier cascade whose proposal support, operating ROC, "
        "and finite-K sampler gap can be described with the framework's words.",
    ]
    for heading, key in [
        ("## Coverage Mapping", "coverage"),
        ("## ROC Mapping", "ROC"),
        ("## Sub-optimality Mapping", "sub-optimality"),
    ]:
        row = mappings[key]
        lines.extend(
            [
                "",
                heading,
                "",
                f"- OT term: {row['paper_term']}",
                f"- Carnot mapping: {row['carnot_mapping']}",
                f"- Boundary: {row['boundary']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Conflict Ledger",
            "",
            "These conflicts are wording constraints for paper-v6. Each one should "
            "soften an existing or tempting claim before any publication action.",
        ]
    )
    for conflict in claim_conflicts:
        lines.extend(
            [
                "",
                f"### {conflict['claim_id']}: {conflict['claim']}",
                "",
                f"- Reason: {conflict['reason']}",
                f"- Softened boundary: {conflict['softened_boundary']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Patch Plan",
            "",
            patch_sentence,
            "Use `docs/arxiv-paper/main.tex` only as a later integration target if "
            "the active paper-v6 source remains absent:",
            "",
            "- Related Work after the current verifier-stack comparator paragraphs: "
            "cite arXiv:2510.18982 as vocabulary for coverage, ROC, and "
            "sub-optimality.",
            "- Section 3 / framework near the k=5 cascade: add one paragraph that "
            "maps coverage to generator proposal support and ROC to the cascade's "
            "thresholded operating point.",
            "- Hardware and sampling limits: add one sentence that finite-K THRML "
            "and Soft-Gibbs Residual runs leave nonzero sub-optimality.",
            "",
            "This note does not trigger publication, arXiv submission, release, or push.",
            f"Paper patch applied: `{str(paper_patch_applied).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    note_path: str | Path = DEFAULT_NOTE_PATH,
    write_observer: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the local Exp 1579 adoption workflow."""

    root_path = Path(root)
    output = Path(out_path)
    note = Path(note_path)
    startup = write_in_progress_artifact(output)
    if write_observer is not None:
        write_observer(output, startup)
    requested_source_exists = (root_path / REQUESTED_PAPER_REL).is_file()
    mappings = default_ot_mappings()
    conflicts = default_claim_conflicts()
    note_text = render_note(
        mappings=mappings,
        claim_conflicts=conflicts,
        paper_patch_applied=False,
        requested_source_exists=requested_source_exists,
    )
    _write_text(note, note_text)
    artifact = build_artifact(
        adoption_note_path=_relative_to_root(note, root_path),
        claim_conflicts=conflicts,
        paper_patch_applied=False,
    )
    _write_json(output, artifact)
    if write_observer is not None:
        write_observer(output, artifact)
    return artifact


def _relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data
