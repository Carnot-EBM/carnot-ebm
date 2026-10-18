"""Build the Exp 3170 counterexample-certificate repair pilot v2 artifact.

Spec refs: REQ-VERIFY-3170, SCENARIO-VERIFY-3170.

WHY THIS EXISTS
---------------
Before a model repair call can be trusted, a repair must start from *exact*
counterexamples and bounded frontier certificates.  The prior repair gate
(exp3168/3169) is blocked because the verifier pipeline has outstanding
adversarial flags and a missing clean live rerun.  Rather than waiting for
those gates to clear before *preparing* the repair inputs, this module
assembles the certificate package now so repair can be invoked as a single
bounded call once the gate opens.

WHAT IT DOES
------------
1. Reads exact rows from prior experiments that have:
   - A known counterexample or solver certificate (false-accept, satisfiable-
     drift, or fragment-code category).
   - A specific verifier to rerun (python_ast_runtime_execution, z3_solver,
     python_json_parser).
2. Builds a ``certificate_record`` per row:  violated constraint, minimal
   failing assignment or trace, expected corrected invariant, solver authority,
   and MCS/unsat-core if available.
3. Scores any prior repair candidates from exp3169 against the certificates
   (there are none because the gate was blocked, so scored=0).
4. Records BEAVER-style bounded-frontier fields from exp3125 (bound width,
   explored mass, viable prefix count) per fixture, labelling unavailable
   fields honestly.
5. Writes a machine-readable pilot artifact with all required schema fields so
   future model repair calls can take the certificate package as input.

NO LIVE MODEL INFERENCE
-----------------------
This task is CPU / exact-authority only.  No GGUF model is loaded, no llama.cpp
subprocess is spawned, no repair runner is invoked.  The inference_substrate
field records this explicitly.  Duration should be sub-second.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3170_counterexample_certificate_repair_pilot_v2"
SCHEMA = "carnot.counterexample_certificate_repair_pilot.v2"
OUTPUT_REL_PATH = Path("results/experiment_3170_counterexample_certificate_repair_pilot_v2.json")

# ---------------------------------------------------------------------------
# Source artifact paths
# ---------------------------------------------------------------------------
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
EXP3125_REL_PATH = Path("results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3169_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")

# ---------------------------------------------------------------------------
# Required fields validated at artifact-write time
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = frozenset(
    {
        "counterexample_certificate_repair_pilot_v2_ready",
        "exact_row_count",
        "counterexample_count",
        "certificate_records",
        "bounded_frontier_records",
        "unavailable_certificate_fields",
        "prior_repair_candidates_scored",
        "exact_accept_count",
        "repair_call_required_for_next_step",
        "source_artifacts",
        "inference_substrate",
        "honest_verdict",
    }
)


# ---------------------------------------------------------------------------
# Row type constants
# ---------------------------------------------------------------------------
ROW_TYPE_FALSE_ACCEPT = "false_accept"
ROW_TYPE_SATISFIABLE_DRIFT = "satisfiable_drift"
ROW_TYPE_FRAGMENT_CODE = "fragment_code"

# Which row types we select in this pilot
PILOT_ROW_TYPES = (ROW_TYPE_FALSE_ACCEPT, ROW_TYPE_SATISFIABLE_DRIFT, ROW_TYPE_FRAGMENT_CODE)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or None if the file is missing.

    WHY: Source traceability requires content hashing so downstream auditors
    can confirm that the certificates derive from the exact same artifact
    version.  Missing files are None rather than an error because upstream
    experiments may have been run in a different environment.
    """
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_json(path: Path) -> JsonDict | None:
    """Load a JSON file from disk, returning None on any failure.

    WHY: Certificate construction must fail gracefully when an upstream
    artifact is absent.  We record missing sources in unavailable_certificate_fields
    rather than crashing.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Certificate record builder
# ---------------------------------------------------------------------------


@dataclass
class CertificateRecord:
    """All exact evidence needed to guide a model repair call for one row.

    WHY: A repair call that starts from vague instructions ("this row is
    wrong") is hard to bound and easy to game.  A certificate record gives
    the repair model the *exact* violated constraint, the minimal failing
    assignment or trace that proves the violation, the expected corrected
    invariant, and the verifier to rerun for acceptance.  Together these
    define a bounded repair specification that the adversarial verifier can
    check deterministically.
    """

    row_id: str
    row_type: str  # false_accept | satisfiable_drift | fragment_code
    exact_label: str
    violated_constraint: str
    minimal_failing_assignment: dict[str, Any]
    expected_corrected_invariant: str
    verifier_to_rerun: str
    solver_authority: str
    certificate_type: str  # solver_mcs | ast_execution | z3_unsat_core
    mcs: dict[str, Any] = field(default_factory=dict)
    unsat_core: list[str] = field(default_factory=list)
    prior_repair_candidate_available: bool = False
    prior_repair_scored: bool = False
    prior_repair_score: dict[str, Any] | None = None
    notes: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "row_id": self.row_id,
            "row_type": self.row_type,
            "exact_label": self.exact_label,
            "violated_constraint": self.violated_constraint,
            "minimal_failing_assignment": self.minimal_failing_assignment,
            "expected_corrected_invariant": self.expected_corrected_invariant,
            "verifier_to_rerun": self.verifier_to_rerun,
            "solver_authority": self.solver_authority,
            "certificate_type": self.certificate_type,
            "mcs": self.mcs,
            "unsat_core": self.unsat_core,
            "prior_repair_candidate_available": self.prior_repair_candidate_available,
            "prior_repair_scored": self.prior_repair_scored,
            "prior_repair_score": self.prior_repair_score,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# BEAVER-style bounded frontier record builder
# ---------------------------------------------------------------------------


@dataclass
class BoundedFrontierRecord:
    """Bounded-frontier evidence from the prefix-closed deterministic pilot.

    WHY: BEAVER (Bounded Exact Accept VErifieR) requires that every accept
    decision be grounded in a finite explored frontier with a known probability
    mass and bound width.  This record carries the frontier summary from
    exp3125 per fixture so the repair pilot can claim bounded coverage only
    within the explored region.
    """

    fixture_id: str
    exact_label: str
    bound_width: float
    explored_mass: float
    viable_prefix_count: int
    pruned_prefix_count: int
    constraint_families: list[str]
    frontier_coverage_note: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "fixture_id": self.fixture_id,
            "exact_label": self.exact_label,
            "bound_width": self.bound_width,
            "explored_mass": self.explored_mass,
            "viable_prefix_count": self.viable_prefix_count,
            "pruned_prefix_count": self.pruned_prefix_count,
            "constraint_families": self.constraint_families,
            "frontier_coverage_note": self.frontier_coverage_note,
        }


# ---------------------------------------------------------------------------
# Row selection: hardcoded pilot set sourced from exp3111 / exp3136
# ---------------------------------------------------------------------------
#
# WHY HARDCODED: This is a pilot over a *tiny* set of exact rows with known
# counterexamples.  The set was identified by exp3136 (false-accept autopsy)
# and exp3111 (certified coherence / Z3 MCS).  Adding a dynamic row-selection
# pass would couple this module to runtime experiment outputs in ways that are
# harder to audit.  The hardcoded set is explicit, replayable, and stable.


def _exp3111_certificate_map(exp3111_data: JsonDict | None) -> dict[str, JsonDict]:
    """Return Exp 3111 certificate rows keyed by fixture id.

    WHY: The pilot row set is intentionally fixed, but its exact certificate
    payload should still come from the source artifact whenever possible.  This
    keeps the repair package traceable instead of relying only on copied values
    in this module.
    """
    if exp3111_data is None:
        return {}
    certificates = exp3111_data.get("certificates")
    if not isinstance(certificates, list):
        return {}
    result: dict[str, JsonDict] = {}
    for row in certificates:
        if not isinstance(row, dict):
            continue
        fixture_id = row.get("fixture_id")
        if isinstance(fixture_id, str) and fixture_id:
            result[fixture_id] = row
    return result


def _enrich_from_exp3111(record: CertificateRecord, source: JsonDict | None) -> CertificateRecord:
    """Overlay source-certificate fields onto a pilot certificate record.

    WHY: Tests mutate the source certificate to prove that the builder follows
    exact evidence rather than stale copied constants.  Missing source fields
    leave the explicit pilot defaults in place so the unavailable-field list can
    explain the gap without dropping the row.
    """
    if not source:
        return record

    exact_label = source.get("exact_label")
    if isinstance(exact_label, str) and exact_label:
        record.exact_label = exact_label

    solver_authority = source.get("solver_authority")
    if isinstance(solver_authority, str) and solver_authority:
        record.solver_authority = solver_authority

    mcs = source.get("minimal_correction_set")
    if isinstance(mcs, dict):
        record.mcs = dict(mcs)

    unsat_core = source.get("unsat_core")
    if isinstance(unsat_core, list):
        record.unsat_core = list(unsat_core)

    diagnostics = source.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    if record.row_id == "resyn-3084-arith-003":
        claimed = diagnostics.get(
            "claimed_value", record.minimal_failing_assignment.get("claimed_value")
        )
        computed = diagnostics.get(
            "computed_value", record.minimal_failing_assignment.get("computed_value")
        )
        if isinstance(claimed, int) and isinstance(computed, int):
            gap = abs(claimed - computed)
            record.violated_constraint = (
                "arithmetic_equality: assert computed == claimed; "
                f"claimed={claimed}, computed={computed}, gap={gap}"
            )
            record.minimal_failing_assignment = {
                "claimed_value": claimed,
                "computed_value": computed,
                "gap": gap,
                "assertion_result": "FAILS",
            }
            record.expected_corrected_invariant = (
                f"claimed_value == computed_value (replace claimed_value {claimed} -> {computed})"
            )

    elif record.row_id == "resyn-3084-smt-000":
        constraints = diagnostics.get("constraints")
        if isinstance(constraints, list):
            record.minimal_failing_assignment = {
                "unsat_reason": "no assignment satisfies the listed constraint conjunction",
                "failure_type": "sat_validity_token_confusion",
                "model_emitted": "VALID",
                "correct_label": record.exact_label,
                "constraints": list(constraints),
            }

    elif record.row_type == ROW_TYPE_FRAGMENT_CODE:
        json_error = diagnostics.get("json_error")
        if isinstance(json_error, str) and json_error:
            record.minimal_failing_assignment = {
                "parse_error_type": "JSONDecodeError",
                "parse_error_message": json_error,
                "missing_token": ",",
                "insert_position": "after_value_before_next_key",
            }

    return record


def _build_certificate_records(
    exp3111_data: JsonDict | None,
    exp3136_data: JsonDict | None,
) -> list[CertificateRecord]:
    """Build certificate records for all pilot rows.

    WHY: Each record must trace to a specific solver certificate (exp3111
    certified coherence) or a false-accept mechanism (exp3136 autopsy).
    We combine both sources to cover the three row-type categories.
    """
    records: list[CertificateRecord] = []
    source_by_id = _exp3111_certificate_map(exp3111_data)

    # -----------------------------------------------------------------------
    # FALSE-ACCEPT ROWS (from exp3136 / exp3138 / exp3111)
    # -----------------------------------------------------------------------
    # Row 1: arithmetic false accept
    # Model said VALID, exact label is INVALID.
    # Certificate: claimed_value=47 != computed_value=43.
    # MCS from exp3111 resyn-3084-arith-003: replace claimed_value 47 → 43.
    records.append(
        CertificateRecord(
            row_id="resyn-3084-arith-003",
            row_type=ROW_TYPE_FALSE_ACCEPT,
            exact_label="INVALID",
            violated_constraint=(
                "arithmetic_equality: assert computed == claimed; "
                "claimed=47, computed=43, gap=4"
            ),
            minimal_failing_assignment={
                "claimed_value": 47,
                "computed_value": 43,
                "gap": 4,
                "assertion_result": "FAILS",
            },
            expected_corrected_invariant=(
                "claimed_value == computed_value (replace claimed_value 47 → 43)"
            ),
            verifier_to_rerun="python_ast_runtime_execution",
            solver_authority="python_ast_runtime_execution",
            certificate_type="ast_execution",
            mcs={"from": 47, "kind": "replace_claimed_value", "to": 43},
            unsat_core=["computed_value", "claimed_value"],
            notes=(
                "Contradiction miss: model returned VALID on an assertion that "
                "explicitly fails at runtime.  The MCS is a single integer "
                "replacement: change the claimed answer from 47 to 43."
            ),
        )
    )

    # Row 2: SMT false accept
    # Model said VALID (wrong token family), exact label is UNSAT.
    # Certificate: Z3 UNSAT core = [constraint_0, constraint_1].
    # MCS from exp3111 resyn-3084-smt-000: relax constraint_0.
    records.append(
        CertificateRecord(
            row_id="resyn-3084-smt-000",
            row_type=ROW_TYPE_FALSE_ACCEPT,
            exact_label="UNSAT",
            violated_constraint=(
                "z3_satisfiability: conjunction of constraint_0 and constraint_1 "
                "is UNSAT; model emitted VALID instead of UNSAT"
            ),
            minimal_failing_assignment={
                "unsat_reason": "no assignment satisfies constraint_0 AND constraint_1",
                "failure_type": "sat_validity_token_confusion",
                "model_emitted": "VALID",
                "correct_label": "UNSAT",
            },
            expected_corrected_invariant="model emits 'UNSAT' when Z3 confirms unsatisfiability",
            verifier_to_rerun="z3_solver",
            solver_authority="z3_solver",
            certificate_type="z3_unsat_core",
            mcs={
                "constraints_to_relax": ["constraint_0"],
                "kind": "remove_conflicting_constraint",
            },
            unsat_core=["constraint_0", "constraint_1"],
            notes=(
                "SAT/validity-token confusion: the model answered in the wrong "
                "token family (VALID instead of UNSAT).  The Z3 unsat core is "
                "[constraint_0, constraint_1]; relaxing constraint_0 would make "
                "the conjunction satisfiable."
            ),
        )
    )

    # -----------------------------------------------------------------------
    # SATISFIABLE-DRIFT ROW (from exp3111)
    # -----------------------------------------------------------------------
    # Row: resyn-3084-smt-001, SAT/coherent.
    # Satisfiable-drift means a candidate claims SAT and the solver confirms it.
    # We include this as a positive anchor: repair should NOT change this row.
    records.append(
        CertificateRecord(
            row_id="resyn-3084-smt-001",
            row_type=ROW_TYPE_SATISFIABLE_DRIFT,
            exact_label="SAT",
            violated_constraint=(
                "none: z3 confirms SAT; no violation; row is a "
                "satisfiable-drift anchor verifying correct positive accept"
            ),
            minimal_failing_assignment={},
            expected_corrected_invariant="model correctly emits 'SAT' — no repair needed",
            verifier_to_rerun="z3_solver",
            solver_authority="z3_solver",
            certificate_type="solver_mcs",
            mcs={},
            unsat_core=[],
            notes=(
                "Satisfiable-drift anchor: Z3 confirms the constraint system has "
                "a satisfying assignment.  A repair call MUST NOT change this row's "
                "outcome (it is already correct).  Used to detect repair over-reach."
            ),
        )
    )

    # -----------------------------------------------------------------------
    # FRAGMENT-CODE ROWS (from exp3111 repairable_invalid_candidates)
    # -----------------------------------------------------------------------
    # Row 1: resyn-3084-repair-json-000, REPAIRABLE, MCS = insert_delimiter ','
    records.append(
        CertificateRecord(
            row_id="resyn-3084-repair-json-000",
            row_type=ROW_TYPE_FRAGMENT_CODE,
            exact_label="REPAIRABLE",
            violated_constraint=(
                "json_well_formedness: JSON fragment is missing a comma delimiter; "
                "python json.loads raises json.JSONDecodeError"
            ),
            minimal_failing_assignment={
                "parse_error_type": "JSONDecodeError",
                "missing_token": ",",
                "insert_position": "after_value_before_next_key",
            },
            expected_corrected_invariant="JSON fragment parses without error after inserting ','",
            verifier_to_rerun="python_json_parser",
            solver_authority="python_json_parser",
            certificate_type="solver_mcs",
            mcs={"edits": [{"operation": "insert_delimiter", "token": ","}], "kind": "json_token_edit"},
            unsat_core=[],
            notes=(
                "Fragment-code repair: the JSON fragment is missing a comma between "
                "key-value pairs.  The MCS is a single token insertion.  A repair "
                "call should insert exactly one ',' and re-verify with the Python "
                "JSON parser."
            ),
        )
    )

    # Row 2: resyn-3084-repair-json-003, REPAIRABLE, MCS = insert_delimiter ','
    records.append(
        CertificateRecord(
            row_id="resyn-3084-repair-json-003",
            row_type=ROW_TYPE_FRAGMENT_CODE,
            exact_label="REPAIRABLE",
            violated_constraint=(
                "json_well_formedness: JSON fragment is missing a comma delimiter; "
                "python json.loads raises json.JSONDecodeError"
            ),
            minimal_failing_assignment={
                "parse_error_type": "JSONDecodeError",
                "missing_token": ",",
                "insert_position": "after_value_before_next_key",
            },
            expected_corrected_invariant="JSON fragment parses without error after inserting ','",
            verifier_to_rerun="python_json_parser",
            solver_authority="python_json_parser",
            certificate_type="solver_mcs",
            mcs={"edits": [{"operation": "insert_delimiter", "token": ","}], "kind": "json_token_edit"},
            unsat_core=[],
            notes=(
                "Fragment-code repair (second instance): same pattern as "
                "resyn-3084-repair-json-000.  Including both because the pilot "
                "must demonstrate that the certificate package covers multiple "
                "instances of the same repair class, not just one."
            ),
        )
    )

    return [_enrich_from_exp3111(record, source_by_id.get(record.row_id)) for record in records]


def _build_bounded_frontier_records(
    exp3125_data: JsonDict | None,
) -> tuple[list[BoundedFrontierRecord], list[str]]:
    """Extract BEAVER-style bounded frontier records from exp3125.

    Returns (records, unavailable_fields) where unavailable_fields lists
    any frontier fields that could not be populated because exp3125 was
    absent or missing required keys.

    WHY: The bounded frontier provides the coverage claim that limits
    how far any acceptance can extend.  Without it, a repair can only be
    offered as 'uncovered' (the honest default when the bounded pilot is
    missing).
    """
    unavailable: list[str] = []

    if exp3125_data is None:
        unavailable.append("bounded_frontier_records_from_exp3125: artifact absent")
        return [], unavailable

    constraint_families: list[str] = exp3125_data.get("constraint_families", [])
    frontier_rows: list[dict[str, Any]] = exp3125_data.get("frontier_rows", [])
    bound_width: float = float(exp3125_data.get("bound_width", 0.0))
    explored_mass: float = float(exp3125_data.get("explored_mass", 0.0))
    fixture_details: list[dict[str, Any]] = exp3125_data.get("fixture_details", [])

    if not fixture_details:
        unavailable.append("bounded_frontier_records_from_exp3125: fixture_details absent")
        return [], unavailable

    records: list[BoundedFrontierRecord] = []
    for fixture in fixture_details:
        fid = fixture.get("fixture_id", "unknown")
        expected_answer = fixture.get("expected_answer", "?")

        # Count viable and pruned prefixes for this fixture
        viable = sum(
            1 for r in frontier_rows
            if r.get("fixture_id") == fid and r.get("status") == "viable"
        )
        pruned = sum(
            1 for r in frontier_rows
            if r.get("fixture_id") == fid and r.get("status") == "pruned"
        )

        records.append(
            BoundedFrontierRecord(
                fixture_id=fid,
                exact_label=expected_answer,
                bound_width=bound_width,
                explored_mass=explored_mass,
                viable_prefix_count=viable,
                pruned_prefix_count=pruned,
                constraint_families=list(constraint_families),
                frontier_coverage_note=(
                    f"Coverage claim bounded to explored_mass={explored_mass:.4f} "
                    f"with bound_width={bound_width:.8f}; "
                    f"{viable} viable prefixes, {pruned} pruned prefixes."
                ),
            )
        )

    return records, unavailable


# ---------------------------------------------------------------------------
# Prior repair candidate scoring
# ---------------------------------------------------------------------------


def _score_prior_repair_candidates(
    exp3169_data: JsonDict | None,
    certificate_records: list[CertificateRecord],
) -> tuple[int, list[CertificateRecord]]:
    """Score any prior repair candidates from exp3169 against certificates.

    Returns (scored_count, updated_records).

    WHY: Distinguishing 'certificate prepared but no repair attempted' from
    'repair was attempted and assessed' is the load-bearing signal for
    downstream pipeline decisions.  scored=0 means the repair ladder was
    not materialised (gate was blocked), not that repair succeeded.
    """
    if exp3169_data is None:
        return 0, certificate_records

    repair_attempts: list[dict[str, Any]] = exp3169_data.get("repair_attempts", [])
    if not repair_attempts:
        # Gate was blocked; no repair candidates to score.
        return 0, certificate_records

    # Build lookup by row_id for certificate records
    cert_by_id = {c.row_id: c for c in certificate_records}
    scored = 0

    for attempt in repair_attempts:
        row_id = attempt.get("row_id", "")
        if row_id not in cert_by_id:
            continue
        cert = cert_by_id[row_id]
        cert.prior_repair_candidate_available = True
        cert.prior_repair_scored = True
        cert.prior_repair_score = {
            "attempt_verdict": attempt.get("verdict"),
            "exact_label_matches": (
                attempt.get("repaired_label") == cert.exact_label
            ),
            "repair_accepted": attempt.get("accepted", False),
        }
        scored += 1

    return scored, list(cert_by_id.values())


# ---------------------------------------------------------------------------
# Source artifact manifest builder
# ---------------------------------------------------------------------------


def _build_source_artifacts(repo_root: Path) -> list[JsonDict]:
    """Build the source_artifacts manifest for this experiment.

    WHY: Certificate claims must trace to specific files so any auditor
    can reproduce the certificate construction from first principles.
    """
    sources = [
        ("AGENTS.md", False, "agents_repo_instructions"),
        ("CODEX.md", False, "codex_repo_workflow"),
        ("CLAUDE.md", False, "claude_authenticity_rules"),
        ("openspec/capabilities/verification/spec.md", True, "verification_openspec"),
        (str(EXP3111_REL_PATH), True, "exp3111_certified_coherence_mcs"),
        (str(EXP3125_REL_PATH), True, "exp3125_prefix_closed_deterministic_bound"),
        (str(EXP3136_REL_PATH), True, "exp3136_false_accept_autopsy"),
        (str(EXP3137_REL_PATH), True, "exp3137_exact_safe_accept_contract"),
        (str(EXP3138_REL_PATH), True, "exp3138_canonical_grounding_pilot"),
        (str(EXP3168_REL_PATH), False, "exp3168_repair_gate_decision_v3"),
        (str(EXP3169_REL_PATH), False, "exp3169_repair_ladder_materializer_v4"),
        (
            "python/carnot/verify/counterexample_certificate_repair_pilot_v2.py",
            False,
            "exp3170_module",
        ),
        (
            "tests/python/test_experiment_3170_counterexample_certificate_repair_pilot_v2.py",
            True,
            "exp3170_tests",
        ),
    ]
    result = []
    for rel, required, role in sources:
        path = repo_root / rel
        present = path.exists()
        sha = _sha256_file(path)
        entry: JsonDict = {
            "path": rel,
            "present": present,
            "required": required,
            "role": role,
            "sha256": sha,
        }
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Unavailable certificate fields enumeration
# ---------------------------------------------------------------------------


def _enumerate_unavailable_fields(
    exp3111_data: JsonDict | None,
    exp3125_data: JsonDict | None,
    exp3169_data: JsonDict | None,
) -> list[str]:
    """List certificate fields that could not be populated and why.

    WHY: Missing exact evidence must stay visible — silently omitting a field
    would let downstream consumers assume the evidence is 'not needed' rather
    than 'not yet available'.  The list is consumed by adversarial-verify and
    by the conductor's repair-gate decision logic.
    """
    unavailable: list[str] = []

    if exp3111_data is None:
        unavailable.append(
            "solver_mcs_records: exp3111 artifact absent — MCS for Z3 rows unavailable"
        )

    if exp3125_data is None:
        unavailable.append(
            "bounded_frontier_records: exp3125 artifact absent — "
            "BEAVER bound_width and explored_mass unavailable"
        )

    if exp3169_data is None:
        unavailable.append(
            "prior_repair_candidates: exp3169 artifact absent — "
            "cannot score prior repair attempts"
        )
    elif not exp3169_data.get("repair_attempts"):
        # Gate was blocked so no repair attempts happened — this is not an error
        unavailable.append(
            "prior_repair_candidates: exp3169 gate was blocked (gated_skip=true); "
            "no repair attempts to score — expected, not an error"
        )

    # Live model logprobs are always unavailable in this CPU-only task
    unavailable.append(
        "live_model_logprobs: this task is CPU/exact-authority only — "
        "no live LLM inference; first-token entropy and calibration curves absent"
    )

    # Satisfiable-drift rows lack a minimal_failing_assignment because there
    # is no failure — the assignment is satisfiable.
    unavailable.append(
        "minimal_failing_assignment_for_satisfiable_drift: "
        "satisfiable-drift rows have no failing assignment by definition; "
        "the field is intentionally empty for row_type=satisfiable_drift"
    )

    return unavailable


# ---------------------------------------------------------------------------
# Main artifact builder
# ---------------------------------------------------------------------------


def build_artifact(
    repo_root: Path | None = None,
    output_path: Path | None = None,
) -> JsonDict:
    """Build and return the pilot artifact dict.

    This function is the single entry point for constructing the artifact.
    It reads upstream sources, builds certificate and frontier records,
    scores any prior repair candidates, and assembles the required fields.

    WHY: Centralising construction in one function makes the artifact
    deterministic and testable — tests can inject a tmp_path as repo_root
    and verify field completeness without touching the real results/ tree.
    """
    t0 = time.monotonic()

    if repo_root is None:
        repo_root = REPO_ROOT

    # ------------------------------------------------------------------
    # Load upstream artifacts (all optional — missing handled gracefully)
    # ------------------------------------------------------------------
    exp3111_data = _load_json(repo_root / EXP3111_REL_PATH)
    exp3125_data = _load_json(repo_root / EXP3125_REL_PATH)
    exp3136_data = _load_json(repo_root / EXP3136_REL_PATH)
    exp3169_data = _load_json(repo_root / EXP3169_REL_PATH)

    # ------------------------------------------------------------------
    # Build certificate records for each pilot row
    # ------------------------------------------------------------------
    cert_records = _build_certificate_records(exp3111_data, exp3136_data)

    # ------------------------------------------------------------------
    # Score any prior repair candidates from exp3169
    # ------------------------------------------------------------------
    scored_count, cert_records = _score_prior_repair_candidates(exp3169_data, cert_records)

    # ------------------------------------------------------------------
    # Build bounded frontier records from exp3125
    # ------------------------------------------------------------------
    frontier_records, frontier_unavailable = _build_bounded_frontier_records(exp3125_data)

    # ------------------------------------------------------------------
    # Enumerate unavailable certificate fields
    # ------------------------------------------------------------------
    unavailable = _enumerate_unavailable_fields(exp3111_data, exp3125_data, exp3169_data)
    unavailable.extend(frontier_unavailable)

    # ------------------------------------------------------------------
    # Derive summary statistics
    # ------------------------------------------------------------------
    exact_row_count = len(cert_records)
    counterexample_count = sum(
        1 for c in cert_records
        if c.minimal_failing_assignment  # non-empty = concrete counterexample
    )
    # Exact accept count: how many certificates confirm a repair was accepted
    # AND the repaired label exactly matches the row's ground-truth label.
    # A repair that was "accepted" but produced the wrong label is NOT an
    # exact accept — counting it would let a label-flipping repair masquerade
    # as a correct one (REQ-VERIFY-3170).
    exact_accept_count = sum(
        1 for c in cert_records
        if c.prior_repair_scored and c.prior_repair_score is not None
        and c.prior_repair_score.get("repair_accepted") is True
        and c.prior_repair_score.get("exact_label_matches") is True
    )

    # Repair is still needed: if no repairs have been accepted and the gate
    # is blocked (or not unblocked), a live model repair call is required.
    repair_gate_state = "blocked"
    if exp3169_data is not None:
        repair_gate_state = exp3169_data.get("gate_state", "blocked")
    repair_call_required = exact_accept_count < exact_row_count

    # ------------------------------------------------------------------
    # Assemble the artifact
    # ------------------------------------------------------------------
    source_artifacts = _build_source_artifacts(repo_root)

    duration_s = time.monotonic() - t0

    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "duration_s": round(duration_s, 6),
        # Required schema fields
        "counterexample_certificate_repair_pilot_v2_ready": True,
        "exact_row_count": exact_row_count,
        "counterexample_count": counterexample_count,
        "certificate_records": [c.to_dict() for c in cert_records],
        "bounded_frontier_records": [r.to_dict() for r in frontier_records],
        "unavailable_certificate_fields": unavailable,
        "prior_repair_candidates_scored": scored_count,
        "exact_accept_count": exact_accept_count,
        "repair_call_required_for_next_step": repair_call_required,
        "source_artifacts": source_artifacts,
        "inference_substrate": {
            "kind": "cpu_exact_authority_certificate_builder",
            "no_live_llm_inference": True,
            "executes_models": False,
            "executes_repairs": False,
            "executes_solvers": False,
            "live_model_calls": 0,
            "repair_calls": 0,
            "description": (
                "CPU-only deterministic certificate construction. "
                "Reads prior solver results and builds replayable certificate packages. "
                "No GGUF model, no llama.cpp subprocess, no repair runner invoked."
            ),
        },
        "honest_verdict": _build_verdict(
            exact_row_count, counterexample_count, scored_count, exact_accept_count
        ),
    }

    # Validate required fields are present
    missing = REQUIRED_FIELDS - set(artifact.keys())
    if missing:  # pragma: no cover
        raise RuntimeError(f"Artifact missing required fields: {sorted(missing)}")

    return artifact


def _build_verdict(
    exact_row_count: int,
    counterexample_count: int,
    scored_count: int,
    exact_accept_count: int,
) -> str:
    """Return a terminal verdict string starting with the required prefix.

    WHY: Conductor reconciler substring-matches partial tokens like 'blocked',
    'no_improvement', 'marginal' against verdicts.  The terminal-prefix
    discipline requires verdicts start with complete:/success:/passed:/shipped_.
    """
    # Recomputed from the same inputs the build_artifact field uses
    # (exact_accept_count < exact_row_count) so the verdict string and the
    # artifact's repair_call_required_for_next_step field can never disagree.
    repair_call_required = exact_accept_count < exact_row_count
    return (
        f"complete: counterexample_certificate_repair_pilot_v2_ready=true; "
        f"exact_row_count={exact_row_count}; "
        f"counterexample_count={counterexample_count}; "
        f"prior_repair_candidates_scored={scored_count}; "
        f"exact_accept_count={exact_accept_count}; "
        f"repair_call_required_for_next_step={str(repair_call_required).lower()}"
    )


def write_artifact(
    repo_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build the artifact and write it to disk.

    WHY: Separating build_artifact (returns dict) from write_artifact (writes
    file) makes the module testable without file I/O.
    """
    if repo_root is None:  # pragma: no cover
        repo_root = REPO_ROOT
    if output_path is None:
        output_path = repo_root / OUTPUT_REL_PATH

    artifact = build_artifact(repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    path = write_artifact()
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}")
    print(f"honest_verdict: {data['honest_verdict']}")
    print(f"exact_row_count: {data['exact_row_count']}")
    print(f"counterexample_count: {data['counterexample_count']}")
    print(f"duration_s: {data['duration_s']}")
