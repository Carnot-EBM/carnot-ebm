"""Build the Exp 3256 p-dit/Potts partial-credit diagnostic manifest.

The p-dit and Potts references are useful here because Carnot already has
verifier rows whose states are not naturally binary.  This module does not
sample, synthesize hardware, or score any live model output.  It records a
CPU/simulation-only mapping from row-state labels to q-state variables, then
keeps exact deterministic verifier checks as the only correctness authority.

Spec refs: REQ-POTTS-009, SCENARIO-POTTS-009.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3256_pdit_potts_multistate_sampler_diagnostic_v1.json")
DELIVERABLE_PATH = REPO_ROOT / OUTPUT_REL_PATH
EXP1361_REL_PATH = Path("results/experiment_1361_pdit_certificate_state_hardware_mapping.json")
EXP1901_REL_PATH = Path("results/experiment_1901_pbit_pdit_ising_sampler_accounting.json")

EXPERIMENT_ID = "exp3256"
TASK_ID = "exp3256-pdit-potts-multistate-sampler-diagnostic-v1"
MILESTONE = "2026.05.301"
SCHEMA_VERSION = "carnot.pdit_potts_multistate_sampler_diagnostic.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 3256
HONEST_VERDICT = (
    "complete: CPU/simulation-only p-dit/Potts partial-credit mapping ready; "
    "exact fallback remains authority; no retired scope reopened"
)
VERDICT_DENIED_TERMS = ("live hardware", "thrml", "kona", "speedup")

REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "pdit_potts_mapping_ready",
    "candidate_verifier_row_types",
    "q_state_energy_mapping",
    "exact_fallback_preserved",
    "hardware_speedup_claim_allowed",
    "retired_pimi_scope_reopened",
    "thrml_scaling_sweep_reopened",
    "future_gated_experiment_contract",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
}


def build_candidate_verifier_row_types() -> list[JsonDict]:
    """REQ-POTTS-009-2: identify partial-credit rows that are q-state by nature."""

    return [
        {
            "row_type": "opencomputer_state_verifier_partial_credit",
            "source_artifact": "results/experiment_2920_opencomputer_style_state_verifier_harness_v1.json",
            "q": 4,
            "state_labels": [
                "exact_pass",
                "partial_credit_progress",
                "ambiguous_or_missing_observable",
                "exact_fail",
            ],
            "why_q_state_is_natural": (
                "The row already awards per-check points and localizes missing state, so "
                "collapsing it to one satisfied/violated spin loses useful verifier structure."
            ),
            "exact_fallback_authority": (
                "rerun deterministic state checks over the JSON, SQLite, filesystem, or JSONL "
                "observable touched by the task"
            ),
        },
        {
            "row_type": "logitext_partial_smt_context_row",
            "source_artifact": "results/experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.json",
            "q": 4,
            "state_labels": [
                "exact_smt_satisfied",
                "partial_smt_covered",
                "underconstrained_needs_extraction",
                "violated_by_exact_solver",
            ],
            "why_q_state_is_natural": (
                "Partial SMT coverage distinguishes solved formal fragments from rows that "
                "still need extraction or violate the exact fixture objective."
            ),
            "exact_fallback_authority": (
                "replay the exact SMT/string/arithmetic/objective checker for the formalized "
                "row fragment before accepting any candidate"
            ),
        },
    ]


def build_q_state_energy_mapping(
    row_types: Sequence[Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    """REQ-POTTS-009-2: turn selected row states into deterministic q-state energies."""

    rows = list(row_types or build_candidate_verifier_row_types())
    mappings: list[JsonDict] = []
    for row in rows:
        labels = [str(label) for label in row["state_labels"]]
        q = int(row["q"])
        energy_table = {label: float(index) for index, label in enumerate(labels)}
        row_type = str(row["row_type"])
        mappings.append(
            {
                "row_type": row_type,
                "q": q,
                "state_labels": labels,
                "potts_variable": {
                    "name": f"{row_type}.potts_state",
                    "state_count": q,
                    "domain": labels,
                    "update_rule": "categorical_cpu_simulation_only",
                },
                "pdit_variable": {
                    "name": f"{row_type}.pdit_state",
                    "alphabet_size": q,
                    "codes": {label: index for index, label in enumerate(labels)},
                },
                "energy_table": energy_table,
                "energy_definition": (
                    "E(row, state) = deterministic_table[state]; lower energy means closer "
                    "to exact verifier acceptance, but never replaces the exact fallback."
                ),
                "binary_one_hot_spin_count": q,
                "invalid_binary_one_hot_state_count": (2**q) - q,
                "exact_fallback_check": {
                    "preserved": True,
                    "authority": row["exact_fallback_authority"],
                    "gate": "exact_fallback_must_accept_before_certification",
                },
            }
        )
    return mappings


def principle_annotations() -> list[JsonDict]:
    """Record why this diagnostic is a mapping exercise, not evidence of acceleration."""

    return [
        {
            "principle": "pdit_multi_state_variables",
            "source": 'research-references.md: "Probabilistic Computing with P-Dit Units"',
            "annotation": "Use p-dits for categorical partial-credit states instead of binary one-hot bookkeeping.",
        },
        {
            "principle": "potts_q_state_partial_credit",
            "source": 'research-references.md: "Potts Machine -- Multi-Value Constraint States via Mean-Field"',
            "annotation": "Use Potts-style q-state labels for correct, partial, ambiguous, and violated row states.",
        },
        {
            "principle": "exact_fallback_authority",
            "source": "CODEX.md required workflow and Exp 3224 exact-checker discipline",
            "annotation": "Sampler energies can triage rows only after exact deterministic fallback remains available.",
        },
        {
            "principle": "retired_scope_boundary",
            "source": "ops/exclusion_manifest.yaml and research-references.md planning notes",
            "annotation": "This diagnostic must not reopen retired PIMI work or the retired scaling sweep line.",
        },
    ]


def upstream_artifact_summary(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Summarize prior p-dit accounting artifacts used as context."""

    root_path = Path(root)
    exp1361_path = root_path / EXP1361_REL_PATH
    exp1901_path = root_path / EXP1901_REL_PATH
    exp1361 = json.loads(exp1361_path.read_text(encoding="utf-8")) if exp1361_path.exists() else {}
    exp1901 = json.loads(exp1901_path.read_text(encoding="utf-8")) if exp1901_path.exists() else {}
    return [
        {
            "path": EXP1361_REL_PATH.as_posix(),
            "present": bool(exp1361),
            "role": "prior CPU-only q=4 p-dit certificate-state mapping",
            "selected_fields": {
                "pdit_variable_count": exp1361.get("pdit_variable_count"),
                "energy_equivalence_error": exp1361.get("energy_equivalence_error"),
                "hardware_claim_allowed": exp1361.get("hardware_claim_allowed"),
                "honest_verdict": exp1361.get("honest_verdict"),
            },
        },
        {
            "path": EXP1901_REL_PATH.as_posix(),
            "present": bool(exp1901),
            "role": "prior p-bit/p-dit Ising sampler accounting gate result",
            "selected_fields": {
                "status": exp1901.get("status"),
                "blocked_at_layer": exp1901.get("blocked_at_layer"),
                "honest_verdict": exp1901.get("honest_verdict"),
            },
        },
    ]


def future_gated_experiment_contract() -> JsonDict:
    """REQ-POTTS-009-4: define a follow-up only behind exact and scope gates."""

    return {
        "contract_allowed": True,
        "experiment_kind": "cpu_simulation_sampler_trial_only",
        "preconditions": [
            {"gate": "exact_fallback_preserved", "required": True},
            {"gate": "hardware_speedup_claim_allowed", "required": False},
            {"gate": "retired_pimi_scope_reopened", "required": False},
            {"gate": "thrml_scaling_sweep_reopened", "required": False},
        ],
        "allowed_measurements": [
            "q_state_energy_table_stability",
            "categorical_state_histogram_under_cpu_simulation",
            "ranking_agreement_with_exact_fallback_labels",
        ],
        "blocked_claims": [
            {
                "claim": "hardware_speedup",
                "allowed": False,
                "reason": "no authenticated hardware transcript in this diagnostic",
            },
            {
                "claim": "retired_pimi_scope",
                "allowed": False,
                "reason": "the mapping is p-dit/Potts bookkeeping, not a PIMI retry",
            },
            {
                "claim": "thrml_scaling_sweep",
                "allowed": False,
                "reason": "no scaling sweep is part of this manifest",
            },
        ],
    }


def checksum_for(artifact: Mapping[str, Any]) -> str:
    """Return the stable checksum for the artifact excluding its checksum field."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-POTTS-009: build the complete deterministic diagnostic manifest."""

    candidate_rows = build_candidate_verifier_row_types()
    q_state_mapping = build_q_state_energy_mapping(candidate_rows)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": principle_annotations(),
        "source_artifacts": upstream_artifact_summary(root),
        "pdit_potts_mapping_ready": True,
        "candidate_verifier_row_types": candidate_rows,
        "q_state_energy_mapping": q_state_mapping,
        "exact_fallback_preserved": all(
            mapping["exact_fallback_check"]["preserved"] for mapping in q_state_mapping
        ),
        "hardware_speedup_claim_allowed": False,
        "retired_pimi_scope_reopened": False,
        "thrml_scaling_sweep_reopened": False,
        "future_gated_experiment_contract": future_gated_experiment_contract(),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "honest_verdict": HONEST_VERDICT,
    }
    artifact["reproducibility_checksum"] = checksum_for(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3256 manifest violates schema or honesty gates."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["hardware_speedup_claim_allowed"] is not False:
        raise ValueError("hardware_speedup_claim_allowed must be false")
    if artifact["retired_pimi_scope_reopened"] is not False:
        raise ValueError("retired_pimi_scope_reopened must be false")
    if artifact["thrml_scaling_sweep_reopened"] is not False:
        raise ValueError("thrml_scaling_sweep_reopened must be false")
    if artifact["exact_fallback_preserved"] is not True:
        raise ValueError("exact_fallback_preserved must be true")

    for mapping in artifact["q_state_energy_mapping"]:
        if int(mapping["q"]) <= 2:
            raise ValueError("q_state_energy_mapping rows must use q > 2")
        if mapping["exact_fallback_check"]["preserved"] is not True:
            raise ValueError("each q_state_energy_mapping row must preserve exact fallback")

    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith("complete:") or any(
        term in verdict.lower() for term in VERDICT_DENIED_TERMS
    ):
        raise ValueError("honest_verdict must be complete and avoid unsupported evidence claims")
    if checksum_for(artifact) != artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(
    path: Path | str = DELIVERABLE_PATH, artifact: Mapping[str, Any] | None = None
) -> JsonDict:
    """Build, validate, and write the Exp 3256 JSON artifact."""

    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(write_artifact(), indent=2, sort_keys=True))
