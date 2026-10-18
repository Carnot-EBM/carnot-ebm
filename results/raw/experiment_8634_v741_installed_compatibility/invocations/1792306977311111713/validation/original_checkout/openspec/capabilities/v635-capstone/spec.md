# V635 Capstone Evidence Matrix

### REQ-REPORT-7218: V635 Capstone SHALL Preserve Fourteen Independent Outcomes

Exp7218 SHALL write `results/experiment_7218_v635_capstone.json`. It SHALL
read the frozen Exp7205 V635 YAML and Markdown design bytes. It SHALL verify
their hashes and exact fourteen-row agreement. It SHALL use the matching active
sources only when the frozen sources are absent. The advisory Exp7205 verdict
SHALL remain visible and SHALL not replace the frozen contract rows.

For each earlier task, Exp7218 SHALL read the declared deliverable first. It
MAY use only the conductor fallback derived from the full task ID when the
declared file is absent. It SHALL not substitute a similar filename. A missing
output SHALL remain one explicit blocked matrix row. Quarantine checks SHALL
run separately from structured field gates. The reader SHALL unwrap only a
mapping that contains both `principle` and `value`.

The evidence matrix SHALL contain exactly fourteen ordered task rows, including
Exp7218. Each row SHALL retain the status, verdict, substrate class, venue,
source hash, raw-row count, quarantine state, readiness fields, value fields,
and limits. `rows` and `recomputed_claim_rows` SHALL contain numeric per-unit
claim rows. Each claim row SHALL contain `unit_id`, `arm`, `seed`, `metric`,
`error`, and `abstention`. Every promoted number SHALL equal a recomputation
from original producer rows or a named retained receipt.

The cumulative ARC summary SHALL deduplicate authentic Exp7193 and V635
induction receipts by unique induction ID. It SHALL report volume, distinct
sessions, distinct seeds, and model-validity counts. The target of ten SHALL
remain an operational collection target. It SHALL not become evidence of no
demand. It SHALL not claim a new solve for an already reproduced game. Tool
engagement SHALL not imply useful world-model reasoning.

The artifact SHALL report source-span fidelity and exact-execution value as
separate PRD facts. It SHALL separately report refinement primary value,
template-deletion causality, and the version-space comparator. It SHALL
separately report sampler stationarity, mixing quality, native ABI usability,
and NFR-01. Every scientific claim SHALL retain `verifier_is_oracle` and its
circularity class. Failed gates and external blocks SHALL remain visible.

Each contracted mechanism SHALL receive exactly one `continue`, `retire`, or
`needs_changed_prerequisite` decision with a specific reason. Repeated failures
SHALL retain the exact prior ID, exact prior verdict, and
`retire_if_same_verdict`. The V634 atomic prompt, queue-priority policy,
missing-tool explanation, and 10x production claim SHALL remain retired.
Exp7218 SHALL not edit the exclusion manifest or protected QA code.

Exp7218 SHALL run `scripts/publication_gate.py --json` unchanged and retain its
G1-G4 output. It SHALL not publish or submit externally. Its own computation
SHALL use `MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate=aggregation_from_upstream_artifacts`,
`inference_substrate_class=aggregation`, and `execution_venue=host` with a
separate hostname. It SHALL measure monotonic duration without padding.

The terminal artifact SHALL contain every required field and exact principle
from the Exp7218 task contract. It SHALL set `capstone_complete_score=1` when
all fourteen slots and internal checks are complete. If a gated or absent
external upstream prevents a scientific conclusion, the completed aggregation
SHALL use `verdict_class: blocked`. Its `gate_check_summary` SHALL name the
failed check, upstream, field, expected value, and observed value. It SHALL not
use `partial` for unchanged external incompleteness.

#### SCENARIO-REPORT-7218-CONTRACT: Frozen Sources Preserve Fourteen Slots

**Given** the frozen Exp7205 YAML and Markdown design bytes
**When** Exp7218 parses and compares their task tables
**Then** exactly fourteen ordered full task IDs and deliverables agree
**And** the last row identifies Exp7218 as self.

#### SCENARIO-REPORT-7218-INTAKE: Declared Paths And Quarantine Fail Closed

**Given** declared results, conductor blocks, and quarantined artifacts
**When** Exp7218 loads task evidence
**Then** it prefers each declared path and uses only the full-ID fallback
**And** quarantine rejection remains independent of a passing field gate.

#### SCENARIO-REPORT-7218-CLAIMS: Numeric Rows Recompute Every Promotion

**Given** authentic producer rows and declared numeric headline fields
**When** Exp7218 builds its claim ledger
**Then** each promoted value equals its independent row recomputation
**And** unavailable or quarantined evidence produces abstention, not invention.

#### SCENARIO-REPORT-7218-ARC: Unique Receipts Bound Live Evidence

**Given** overlapping ARC receipts from Exp7193, Exp7206, and Exp7207
**When** Exp7218 constructs the cumulative summary
**Then** each unique induction ID counts once and sessions remain distinct
**And** collection volume does not become efficacy or missing-tool proof.

#### SCENARIO-REPORT-7218-BOUNDARIES: PRD Questions Remain Separate

**Given** source, learning, sampler, ABI, and board evidence
**When** Exp7218 reports findings and limits
**Then** fidelity, value, stationarity, mixing, ABI, and NFR-01 stay separate
**And** readiness or circular checks do not become scientific value.

#### SCENARIO-REPORT-7218-DECISIONS: V634 Retirements Remain Closed

**Given** V634 branch decisions and V635 outcomes
**When** Exp7218 issues one decision per mechanism
**Then** repeated failures retain exact prior-failure signals
**And** runtime repairs do not reopen retired prompts, policies, explanations,
or the 10x claim.

#### SCENARIO-REPORT-7218-BLOCKED: Complete Matrix Reports External Blocks

**Given** all fourteen matrix slots and a failed upstream scientific gate
**When** Exp7218 completes its own aggregation
**Then** its status is complete and its capstone score is one
**And** its verdict is blocked with the exact external gate diagnosis.

#### SCENARIO-REPORT-7218-ARTIFACT: File Parser And Gate Detect Mutation

**Given** a terminal Exp7218 artifact
**When** the CLI reloads and validates its hashes, rows, claims, decisions,
publication receipt, and checksum
**Then** unchanged evidence passes
**And** a mutated roster, claim, gate, or checksum fails.

## Implementation Status (REQ-REPORT-7218)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-REPORT-7218 and SCENARIO-REPORT-7218-* | Implemented: Exp7218 aggregation module, executable entrypoint, and terminal artifact builder | Verified: focused RED/green tests and 100% changed-module coverage; remaining repository validation is tracked by the task run |
