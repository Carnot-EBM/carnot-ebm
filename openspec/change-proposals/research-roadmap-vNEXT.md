# Research Roadmap vNEXT: V476 -- Poison-Test Hardening + Retro-Timing Real Fix, DiffusionGemma vLLM-Native
Retry, MAP Landmark Retry, Hidden-State Verifier V2, QA/Verifier-Authenticity Follow-Through

**Milestone:** `2026.07.476`
**Status:** Planner-authored (fresh verification pass over an existing outer-loop draft of the same
milestone number -- see "Provenance" below). `research-roadmap-next.yaml` is now written (this planning
pass's own contribution -- the prior draft diagnosed everything but left the execution manifest unwritten;
see "Second verification pass" immediately below).
**Prepared:** 2026-07-03
**Predecessor:** `2026.07.475`
**Execution manifest:** `research-roadmap-next.yaml`

## Second verification pass (this planning invocation, 2026-07-03, later same day)

This invocation is the conductor's own automated planning-agent call (`research_conductor.py` logged
"No research-roadmap-next.yaml -- launching planning agent" at 08:30:26 EDT / 12:30:26 UTC, immediately
before this session's own turn-start timestamp of 12:30:33 UTC -- these are the same event). Rather than
re-deriving the plan below from scratch, this pass independently re-verified the prior draft's load-bearing
claims against primary sources a second time and found the design sound, with three corrections:

1. **The lp85 registry-inconsistency claim in "Current registry / gate state" below is INACCURATE as
   written.** Direct read of `ops/arc_solve_registry.yaml`'s `lp85` entry (mtime 2026-06-30, unchanged
   since before this draft was written) shows `levels_reproduced: 5` -- there is no "L3 canonical" value
   anywhere in the registry. The quarantined artifact's own claimed "prior level 5, new level 6" is
   actually CONSISTENT with the registry's real prior value (5); the only genuine discrepancy is the
   quarantined artifact's rejected claim of a new L6, which the mechanical layer already correctly
   flagged and excluded. There are TWO numbers in tension here (registry=5, quarantined claim=6), not
   three. `exp5206`'s capstone task (in the execution manifest) is scoped to confirm this precisely and
   correct any OTHER stale document that might still say "L3" (none found in this pass's search), rather
   than presupposing a three-way conflict that the primary source does not show.
2. **`exp5195`'s retro-timing investigation now has a sharp, code-grounded lead**, found by reading
   `scripts/research_conductor.py` lines 2768-2996 directly (not just the artifact's zero values): the
   `retro_timing_fallback` import IS present and IS called (confirming the prior draft's "confirmed wired"
   finding), but the call sits inside a bare `except Exception: logger.warning(...)` block. The `.475`
   retro's `experiments_completed=0` / `reconstructed_from_disk_mtime=False` combination is consistent with
   `build_retro_timing_fallback()` itself raising internally for this specific milestone (plausible given
   `.475`'s unusual execution history -- only 2 of 12 tasks produced artifacts, and one of those,
   `exp5182`, was produced by a direct outer-loop script invocation outside the conductor's normal
   task-commit flow, which may not fit whatever commit-message or activation-boundary pattern
   `_activation_bound()` expects). This is a *more specific, faster-to-test* hypothesis than "the LLM
   retro-writer may not transcribe faithfully" -- `exp5195`'s manifest task now leads with calling
   `build_retro_timing_fallback('2026.07.475', ...)` directly and grepping the daemon's `journalctl --user
   -u carnot-conductor` output around 2026-07-03 10:46 EDT for the swallowed-exception warning line, before
   falling back to the LLM-transcription theory.
3. **The ARC Prize deadline correction is now firmly sourced, not a lower-confidence snippet.** A direct
   `WebSearch` this session (not just a search snippet as the prior draft flagged) independently returned
   multiple results describing arcprize.org's own 2026 competition structure: **Milestone #1 was 2026-06-30
   (already passed, the trigger this project already acted on); Milestone #2 is 2026-09-30**, each carrying
   its own prize tier for open-sourcing by that date. This replaces the prior draft's vaguer "may run
   through November 2, 2026" framing with a firmer, near-term second date. Still an operator-attention item
   only (see below) -- this does not change this milestone's task allocation.

Everything else in this document (the `.475` post-mortem, the DiffusionGemma root-cause diagnosis, the
GAP-4891 ladder, the literature citations, the phase design and dependency graph) was independently
re-checked against the same primary sources this pass and held up; it is preserved below unchanged.

## Provenance -- this plan supersedes an unfinished on-disk draft of the same milestone

A prior outer-loop Claude session (earlier on 2026-07-03) diagnosed `.475`'s execution stall, performed
direct remediation (ran `exp5182` live, verified `retro_timing_fallback.py`'s wiring, restarted
`carnot-conductor.service`), and wrote a `research-roadmap-vNEXT.md` draft for `.476` -- but
`research-roadmap-next.yaml` was never written (confirmed: the file does not exist on disk), and the
milestone's *own* subsequent close (the `.475` operational retrospective, commit `958859613`,
2026-07-03 06:46:43 -0400) happened after that draft was written. This planning pass:

1. **Independently re-verified every load-bearing claim in that draft** against primary sources (the
   real `exp5182` artifact, `conductor-log.md`'s full `.475` timeline, the actual `.475` operational
   retrospective JSON, the two `.474` GGUF/vLLM probe artifacts, both audit reports, the exclusion
   manifest, `verifier_gaps.md`) rather than re-deriving the diagnosis from scratch or trusting the prior
   draft's prose uncritically.
2. **Found the prior draft's DiffusionGemma framing incomplete**: it treated the GGUF/vLLM loader stack
   as "never followed up," but both probes ran in `.474` and both are conclusively blocked
   (`blocked_gguf_load_failed`, `blocked_vllm_load_failed`). Fresh research this session found the *real*
   actionable lead: **vLLM shipped native DiffusionGemma support the same day the model released**
   (2026-06-10 vLLM blog post), and the `.474` probe's traceback shows it fell back to the generic
   Transformers backend inside vLLM (`"TransformersMultiModalMoEForCausalLM has no vLLM implementation,
   falling back to Transformers implementation"`) rather than exercising the native path -- almost
   certainly a stale vLLM version or wrong invocation, not a fundamental block. See "What this session's
   research found" below.
3. **Found a live, still-present bug the prior draft assumed was fixed**: `retro_timing_fallback.py` is
   confirmed wired into `research_conductor.py` (commit `75bc15756`) and the daemon was restarted, but
   the *actual* `.475` operational retrospective (`results/operational_retro_2026_07_475.json`, generated
   AFTER both the fix and the restart) still shows `experiments_completed=0`, `total_wall_time_minutes=0`,
   `reconstructed_from_disk_mtime=False` -- the exact false-zero pattern the fix was built to eliminate.
   This is read directly from the artifact, not inferred. `.476` reopens this as a live bug, not a
   verification formality.
4. **Found a second, unrelated, live doc-corruption bug**: `ops/known-issues.md` (10,651 lines) contains
   **187 byte-identical copies** of a `### NEW Phase 4 Canonical Metric MANDATORY` section, almost all
   concentrated in the file's final ~450 lines. Root-caused to `scripts/experiments/run_experiment_1911.py`
   lines 62-68, which unconditionally `open(path, "a").write(...)` the same fixed string with **no
   idempotency check**. Not mentioned in the prior draft.
5. **Found exp5181's `DURATION_TOO_SHORT` flag is very likely a false positive**, not a genuine
   fabrication signal: its `inference_substrate` is correctly declared
   `{"principle": "...", "value": "aggregation_from_upstream_artifacts"}` (2.49s is far above that
   substrate's 0.0001s floor), so the flag is almost certainly firing on GGUF/CUDA substring mentions
   inside *cited upstream* fields (e.g. quoting exp5182's mitigation log) rather than on a live-inference
   claim this artifact itself makes. This looks like the same unwrap/context-blindness bug class the
   QA-Layer Authenticity Discipline exists to catch, just in `adversarial_verify.py` rather than
   `exclusion_manifest_lint.py`.
6. **Verified the two new literature citations the prior draft added** (Radial Consensus Score,
   AutoPyVerifier) via direct WebFetch -- both are real, both check out topically. Logged formally to
   `research-references.md` (see below) since the prior draft cited them without logging them.
7. **Surfaced one governance-relevant correction for the operator, not for this plan to act on**: the
   ARC Prize deadline this project has been treating as final (`2026-06-30`, the trigger for retiring the
   ARC Submission Sprint Forcing Function) is, per a fresh WebFetch of arcprize.org, actually **"Milestone
   #1"** for early open-source prize eligibility -- the overall 2026 competition reportedly runs through
   **November 2, 2026** (this end date via search snippet only, not independently WebFetched -- flagged at
   lower confidence). This does not change this plan's task allocation (the self-solve audit's
   zero-advance finding and the "do not silently re-expand ARC's share" stance both still apply), but it
   is exactly the kind of fact that should inform the operator's pending "post-PHASE-D strategic direction"
   decision, so it is surfaced explicitly rather than left buried in a sub-agent transcript.

Where the prior draft's diagnosis held up against fresh verification (the poison-test-cascade root cause,
the MAP/hidden-state-verifier task designs, the two audit-report action items), this plan keeps it and
does not re-litigate it. Nothing here re-does verification work that was already correct.

## What actually happened in `.475` (verified directly against `results/*.json` and `conductor-log.md`,
not paraphrased from any prior doc)

`.475` activation was refused **51 times** between 03:15 and 05:25 UTC on `SCOPE_MATCHED_PRIOR_FAILURE`
exclusion-manifest violations, then the automated planner itself failed 3 times (a 50-turn cap, then two
1201s wall-clock+idle timeouts). The milestone was hand-activated at 07:59 UTC with 12 tasks queued.

Of those 12 tasks, **exactly two produced real artifacts** (confirmed: `results/experiment_5183*.json`
through `experiment_5192*.json` do not exist anywhere on disk):

| Task | Result |
|---|---|
| `exp5181` (archive .474->.475) | Ran, produced a real artifact, `flagged_adversarial: true` (`DURATION_TOO_SHORT` -- see finding #5 above, likely false positive) |
| `exp5182` (DiffusionGemma root-cause fix) | The CONDUCTOR's own attempt hit `Wall-clock+idle timeout after 1201s` and never ran `main()`. An outer-loop session ran the script directly against idle GPU 0 afterward, producing the real 630.3s artifact this plan reads (see below). The conductor's own retry at 10:35 UTC found "Deliverable already exists in repo" and marked it `OK`. |
| `exp5183`-`exp5192` (10 tasks) | **Never executed.** `exp5182`'s conductor-side timeout left `test_ondisk_deliverable_is_valid` red (a correct test whose one precondition -- the artifact -- didn't exist yet), which poisoned the SHARED pretest gate: every subsequent task either `SKIP`'d 3x on that shared failure or `GATE_BLOCK`'d on an upstream the conductor had started treating as retired. This is the **4th occurrence** of the "agent-shipped-incomplete-artifact poisons the pretest gate" incident class (prior: `exp3521`/`.325`, `exp3544`/`.326`, `exp3612`/`.332`). |

The `.475` operational retrospective ran at 10:46:43 EDT (commit `958859613`) once the task queue reached
a terminal state (SKIP/GATE_BLOCK both count as terminal for retro-triggering purposes) -- **not** because
substantive research happened in 10 of 12 slots. `.476`'s job is genuinely a recovery milestone: two
infra fixes (one of which the prior draft believed was already done and is not), plus re-attempting the
10 tasks that never got a chance to run, now informed by real new ground truth from `exp5182` and this
session's research.

## What `exp5182` actually found, live (verified directly against `results/experiment_5182_*.json`)

All four device-placement mitigations were attempted on an idle GPU 0 (confirmed via `nvidia-smi` before
starting; 630.3s total, genuinely GPU-bound):

| Mitigation | Description | Outcome | Duration |
|---|---|---|---|
| m1 | `device_map={"":0}`, 4-bit NF4, single GPU | `load_failed` -- **CUDA OOM** (need ~22.6 GiB+ of 23.56 GiB total) | 188.6s |
| m2 | `device_map="auto"` + explicit `_no_split_modules` correction, 4-bit NF4 | `load_failed` -- **meta-tensor / CPU-disk dispatch error** (the original bug) | 137.1s |
| m3 | `device_map={"":0}`, 4-bit NF4, `low_cpu_mem_usage=False` | `load_failed` -- **CUDA OOM** | 149.8s |
| m4 | `device_map={"":0}`, int8 | `load_failed` -- **CUDA OOM** | 149.0s |

**Root cause, precisely diagnosed** (quoted from the artifact): DiffusionGemma's encoder is a weight-tied
mirror of its decoder (`DiffusionGemmaModel._tied_weights_keys` ties `encoder.language_model.layers...` to
`decoder.layers...`); the checkpoint stores one physical copy. `device_map="auto"` splits encoder and
decoder across the two GPUs, breaking the shared-storage tie, so the encoder's tied weights are never
materialized (stay on the meta device) -> `Tensor.item()` fails at forward. Single-device placement
(`device_map={"":0}`) correctly co-locates the tied weights and *resolves the meta-tensor bug specifically*
-- but then the model needs more than 23.56 GiB even at 4-bit/int8 quantization on ONE GPU, so all three
single-device variants OOM instead. **Both failure modes are now root-caused and neither is a mystery**;
the remaining question is a memory-budget one, not a correctness one.

This directly narrows what `.476`'s DiffusionGemma task should try -- see "What this session's research
found" immediately below.

## What this session's research found (verified via direct WebFetch; logged to `research-references.md`
under a new `## V476 Planner References` section alongside this plan's commit)

**1. vLLM ships NATIVE DiffusionGemma support, as of the model's own release day.** The official vLLM blog
(`https://vllm.ai/blog/2026-06-10-diffusion-gemma`, fetched directly) is titled "DiffusionGemma: The First
Diffusion LLM (dLLM) Natively Supported in vLLM," dated 2026-06-10 -- the SAME day DiffusionGemma released
-- and demonstrates it on single H100/H200 via a new "model runner v2 `ModelState` abstraction." Our own
`.474` vLLM probe (`results/diffusiongemma_energy_prior_vllm.log`, dated 2026-06-14) shows
`"TransformersMultiModalMoEForCausalLM has no vLLM implementation, falling back to Transformers
implementation"` -- i.e. **that probe did NOT exercise the native path** and fell back to the same
device-placement-fragile Transformers backend `exp5182` just exhausted. This is a genuinely different,
untried mitigation, not a re-run of anything: `exp5196` should re-test vLLM using the exact recipe from the
blog post (checking vLLM version -- the native support needs a version that postdates 2026-06-10 -- and the
specific invocation flags/model-runner path the blog demonstrates), rather than the generic
`quantization="bitsandbytes"` + default routing our stale probe used.

**2. If vLLM-native also fails on 2x24GB (the blog only demonstrates H100/H200, not our VRAM budget), there
is a concretely-cited HF/accelerate mitigation we have not tried.** `huggingface/transformers#22018`
(fetched directly) describes the exact interaction class: `device_map="auto"` + `llm_int8_enable_fp32_cpu_offload=True`
failing because CPU-offloaded modules are excluded from the quantization module list at the wrong point in
`replace_8bit_linear`. The reporter's workaround, confirmed in the issue thread: **pass a manually
pre-configured `device_map` dict (not `"auto"`)** that explicitly co-locates the tied encoder+decoder
modules on GPU 0 and offloads other named (non-tied) modules to `"cpu"`, combined with
`llm_int8_enable_fp32_cpu_offload=True`. This is qualitatively different from all four of `exp5182`'s
mitigations (which were either fully-auto or fully-single-device with no partial-offload option) and
directly exploits the root cause `exp5182` diagnosed (tied weights must co-locate; only non-tied parts may
offload).

**3. GGUF / llama.cpp support is genuinely not ready -- confirmed, not a bug on our side.**
`ggml-org/llama.cpp#24427` ("Add diffusion-gemma block-diffusion support," fetched directly) is **Draft,
open, not merged**. The companion `ollama/ollama#16664` ("unknown model architecture: 'diffusion-gemma'")
is open with no linked fix. Our `.474` `blocked_gguf_load_failed` probe was correct; there is nothing to
retry here until the upstream PR lands. Do not propose a GGUF retry in `.477+` without checking whether
`#24427` has merged.

**4. Two literature citations verified real** (both WebFetch-confirmed, both check out against their
described use):
- **Radial Consensus Score** -- arXiv:2604.12196, "Beyond Majority Voting: Efficient Best-Of-N with Radial
  Consensus Score" (Nguyen, Gupta, Le). Computes a weighted Frechet mean of answer embeddings and ranks by
  radial distance to that center -- a training-free, black-box, embedding-geometry replacement for majority
  voting, validated across 7 benchmarks.
- **AutoPyVerifier** -- arXiv:2604.22937, "AutoPyVerifier: Learning Compact Executable Verifiers for Large
  Language Model Outputs" (Pezeshkpour, Hruschka). An LLM synthesizes candidate Python verifier functions;
  a DAG search identifies a compact/minimal SET whose combined satisfaction best predicts correctness
  (up to +55 F1 over the initial set). The paper's own core finding -- search a *set* of cheap
  discriminators, not one hand-crafted invariant -- is precisely what distinguishes `exp5205`'s pilot from
  the already-refuted single hand-invariant attempt (see `exp5205`'s `prior_failures` block below).

**5. Fresh 48-hour sweep: honestly nothing new.** A direct fetch of arXiv's cs.LG recent listing (2026-07-01
through 2026-07-03) and targeted searches for energy-based verifiers, hidden-state verification, and
ARC-AGI-3 results found nothing dated in that window. Reported as a clean negative rather than padded with
older papers restated as new.

## Current registry / gate state (read directly, not inferred)

- `ops/arc_solve_registry.yaml`: `reproducible_total_levels=69`, flat for 5+ consecutive milestones
  (`.471`-`.475`). `ops/arc_self_solve_audit_report.md` (2026-07-03, read directly this session) confirms
  **zero `SELF_DISCOVERY_ADVANCE`** in the most recent window: 19/20 recent solve artifacts are benign
  `development_proxy` re-reproductions of already-banked levels (regression re-fires, not progress); the
  1 outlier (`experiment_headway_lp85_capture.json`, a claimed lp85 L6) is a confirmed `OUTER_LOOP_RE`
  violation (`used_env_source: true` under a `development_proxy` stamp) that the mechanical layer already
  caught and quarantined (`flagged_adversarial: true`). **CORRECTED in this pass's second verification
  (see top of document):** the registry's actual `levels_reproduced` for lp85 is **5**, not "L3" as
  originally written here -- the quarantined artifact's own "prior level 5" claim already matches the real
  registry value; only its rejected "new level 6" is in tension with anything. `.476`'s capstone
  (`exp5206`) confirms this reading directly against the registry rather than reconciling a three-way
  conflict that the primary source does not actually show.
- `ops/verifier_gaps.md`: GAP-4891 (ARC trajectory-enumeration wall) diagnosed but not closed through four
  ladder stages (counts -> richer-scalar -> relational target-match [separates 3/4] -> stage-2 goal-energy
  guidance [separates but does not guide search] -> stage-3 relational-mask pruning [prunes edges, banks
  nothing]). The next lever, per the gap's own log, is a MAP-style pre-search stage. GAP-1
  (transpose/orientation discrimination) has one hand-invariant candidate TESTED AND REFUTED (2026-06-09):
  it degraded HYBRID rerank pass@2 0.452->0.419 and captured 0/2 transpose mis-votes. `exp5205` targets
  this gap with a structurally different approach (verified real per the literature check above).
- `ops/exclusion_manifest.yaml`: PHASE D external-text-scorer construction is terminally retired
  (`phase_d_external_text_scorer_retired_exp5163_v474`, 2026-07-02) after 27 artifacts across 7 milestones;
  hidden-state/internal-representation verifiers, ARC oracle-distinct verifier work, and the FoVer
  production ensemble are explicitly named as outside the retired scope. `gap3_trained_content_energy_selector_retired_stage2v2`
  and `generation_axis_exploration_signal_retired_exp5154_v473` remain retired and untouched by this plan.
- `ops/north-star.md`: the FoVer headline (AUROC 0.9131) and G1-G4 publication gate remain fixed and MET
  (`paper_ready: true` per `scripts/publication_gate.py`). ARC-AGI-3 remains the stated destination (S0),
  hardware focus is narrowed to KV260-as-sovereignty-story with GateMate/PolarFire opportunistic only (S3).
- `ops/verifier_authenticity_audit_report.md` (2026-07-01, read directly): 11 `AUTHENTIC`, 6
  `HONEST_HEURISTIC`, **2 `DISHONEST_NAMING`** -- `and_composition_verifier.py` (claims a production k=5
  AND-composition ensemble with paper-backed exponential null-space shrinkage; the default
  `SOSKANEnergyV3Adapter` member is untrained and returns a hardcoded neutral `0.5` for every input, plus
  score-capping and exception-swallowing that silently converts a broken verifier into a clean pass) and
  `claim_isolation_uncertainty_router.py` (implies routed cases get real isolated-claim verification; the
  implementation only copies manifest booleans and assigns fixed scores -- no model call). Never actioned;
  carried forward as `exp5203`.
- `ops/qa_layer_authenticity_audit_report.md` (2026-07-02, read directly): 1 unit scanned, **1 `REAL_BUG`**
  in `scripts/exclusion_manifest_lint.py` with a concrete counterexample: a task titled "Guard against FoVer
  premise reuse" whose prompt explicitly says "do not reuse the FoVer... premise" gets mis-classified
  `BLOCKED_PATTERN_MATCHED` because the check does raw substring matching with no word boundaries and no
  negation awareness. Never actioned; carried forward as `exp5204`.
- `ops/docs_audit_report.md` (2026-07-03, read directly): confirms the license contradiction (hero says
  MIT-0, footer says Apache 2.0) the prior draft flagged, **plus additional findings the prior draft did not
  mention**: a structural render risk (the "Recent progress" card is shoehorned into `.stats-bar` using an
  undefined `.r-desc` CSS class with a stray unmatched `</div>`; the "TTC & PREM" card sits after an extra
  `</div>` that may close the features grid early), an undefined-and-self-contradictory "FoVer" usage
  (0.9131 headline win vs. "0.125 FoVer baseline" in the same page), and a cluster of suspiciously-perfect
  result-card numbers (1.0, 60/60, "identical losses") with no visible n/CI. **All operator-curated content
  per Public Documentation Discipline -- no task in this plan touches `docs/index.html`**; flagged in
  "Recommended operator attention" below because the CSS/div-nesting issue is a rendering bug, not just a
  style judgment call.

## Phase design

### Phase 0 -- Transition
`exp5193`: routine `.475`->`.476` archive/activation. Reconciles `exp5182`'s live-remediated result into
the archive record precisely (produced outside the normal task-execution path, by direct outer-loop
intervention -- the archive record must say so, not imply a conductor task produced it), and verifies
(rather than trusts) exp5181's `DURATION_TOO_SHORT` flag against the evidence this plan already gathered
(the substrate is correctly declared `aggregation_from_upstream_artifacts`; the flag likely fired on
GGUF/CUDA substrings inside cited upstream fields -- confirm and document, feeding the finding toward a
future `adversarial_verify.py` fix pass rather than re-diagnosing it from zero next milestone).

### Phase INFRA-CRITICAL -- close the poison-test-cascade gap; fix retro-timing for real; stop the known-issues.md corruption
Positioned second, immediately after archive/activate, on purpose -- `.475`'s own reserved-infra slots were
placed *last* and were exactly what starved when the cascade hit early. A repeat cascade this milestone
would not erase these two tasks if they run early.
- `exp5194`: build the pretest-triage module (mirrors `retro_timing_fallback.py`'s pattern: new standalone
  module, cannot edit `scripts/research_conductor.py` directly) that detects the specific signature -- a
  just-added test whose only failure references a `results/*.json` path a sibling module's own `main()`
  would produce -- and prepares a scoped, auditable `xfail` remediation (never `skip`, per Tests-Must-
  Run-And-Assert) with a tracking note and expiry condition.
- `exp5195`: **re-open, not just verify**, the retro-timing bug. `retro_timing_fallback.py` is confirmed
  wired (commit `75bc15756`) and the daemon was restarted, but `results/operational_retro_2026_07_475.json`
  -- generated AFTER both -- still shows `experiments_completed=0`, `total_wall_time_minutes=0`,
  `reconstructed_from_disk_mtime=False`. Trace exactly which code path produced that specific artifact
  (is `timing_summary`'s assembled text actually reaching the LLM agent that writes the final retro JSON?
  did this retro get generated by a process that predates the fix landing in memory despite the restart?),
  fix it for real, add a regression test that asserts the END-TO-END retro JSON is correct for a milestone
  with known real work (not just that the wiring exists), and backfill `.469`/`.473`/`.474`/`.475`. Same
  task also fixes the unrelated `known-issues.md` duplication bug: `scripts/experiments/run_experiment_1911.py`
  lines 62-68 append the same fixed string with no idempotency check (187 duplicate copies confirmed on
  disk); add the idempotency guard and deduplicate the file down to one copy of that section (content-
  preserving cleanup, not a content removal -- consistent with Documentation Update Rules).

### Phase A -- Verifier-moat continuation: DiffusionGemma and GAP-4
- `exp5196`: reads `exp5182`'s real artifact first (root cause + all 4 exhausted mitigations), then tries,
  in order: (1) vLLM using the exact native-support recipe from the 2026-06-10 vLLM blog post (a genuinely
  untried path -- our `.474` probe fell back to the generic Transformers backend); (2) if that also fails
  or cannot fit the 2x24GB budget, the HF/accelerate custom-`device_map`-dict +
  `llm_int8_enable_fp32_cpu_offload=True` mitigation grounded in `transformers#22018` (co-locate tied
  modules on GPU 0, offload named non-tied modules to CPU); (3) if BOTH fail, retire the DiffusionGemma
  live-loading thread with a clear, evidence-backed write-up -- six well-motivated mitigations across two
  serving stacks is a thorough good-faith effort, and continuing to guess a seventh has strongly
  diminishing returns per the Failed-Experiment Rerun Discipline. Does NOT re-attempt GGUF (confirmed
  not-yet-supported upstream, `llama.cpp#24427` still Draft).
- `exp5197`: continue the GAP-4 forward-protocol scale-up from n=62 toward the ~180-sample significance
  floor, with genuine atomic checkpoint/resume this time (`exp5177`'s gap: a `checkpoint_path` field was
  declared but no file was ever written).

### Phase B -- Trajectory-enumeration wall: MAP landmark pre-stage (re-attempt; design unchanged, never ran)
- `exp5198`: the falsifiable 3-arm MAP gate `exp5172` specified and `.475` was never able to run --
  pruner-only vs. map-only vs. map-plus-pruner on CD82/SK48/SP80, CN04 negative control, same
  4000-expansion reproduction-gated budget as the existing relational-mask-pruner result it must beat.
- `exp5199`: gated on `exp5198` validating a lever. Satisfies the ARC Level-Up Attempt Guarantee's
  mandatory >=1-attempt floor for this roadmap (the gated-task pattern is the established precedent for
  satisfying this floor without forcing a doomed attempt on a null `exp5198`).

### Phase C -- Hidden-state verifier v2 (re-attempt, sharpened) and hardware continuity
- `exp5200`: PHSV-style trained probe (arXiv:2504.05419 -- chunk-level reasoning boundaries, last-token/
  last-layer hidden state, 2-layer MLP) on the MMLU-Pro headroom-confirmed pool (oracle_at_k=0.350 vs.
  sc_vote=0.075, CI95=[0.150, 0.425] excludes 0 -- the 2026-07-01 headroom check), replacing `exp5178`'s
  naive centroid probe on an underpowered n=6. Three mandatory zero-training baselines: self-certainty
  (arXiv:2502.18581), CLUE (arXiv:2510.01591), and **Radial Consensus Score** (arXiv:2604.12196,
  WebFetch-verified real this session) -- a trained probe must clear all three before any beats-SC claim is
  credible. This is `.476`'s designated continuous-self-learning experiment (JEPA-style Tier 3 predictive
  verification per research-program.md: a probe trained on the model's own accumulated correct/incorrect
  hidden-state experience).
- `exp5201`: hardware continuity -- KV260 + PolarFire SSH-reachability + hash-verified workload (routine,
  per Hardware-Task Continuity Discipline), and a genuine third-consecutive-milestone attempt to resolve
  the GateMate DirtyJTAG IDCODE regression (enumerates at USB level, IDCODE read fails; worked in May,
  regressed since).

### Phase D -- Literature-informed QA and verifier-authenticity follow-through
- `exp5202`: `_bmad/architecture.md` reconciliation -- 48+ days stale (Last Reconciled 2026-05-16),
  never addressed across `.475` (never ran) and before. Document the ARC-AGI-3 pivot, the PHASE D
  lifecycle (committed -> executed -> retired), the hidden-state-verifier program, and the current
  verification-tier table's drift since May.
- `exp5203`: prepare operator-facing remediation options (RENAME_TO_REFLECT_REALITY / RETIRE /
  REIMPLEMENT_PROPERLY, per the Verifier Authenticity Discipline's own decision categories) for
  `and_composition_verifier.py` and `claim_isolation_uncertainty_router.py`. The audit never edits
  verifiers and the operator decides; this task makes that decision cheap (read each verifier's real
  behavior, draft the three options with a recommendation and rationale, rename/retire/reimplement
  nothing silently).
- `exp5204`: fix `scripts/exclusion_manifest_lint.py`'s documented `REAL_BUG` (raw substring matching
  without word boundaries; `id`/`requires`/`operator_override`/`title`/`prompt` treated as bare values
  when they may be principle-wrapped; `_is_negated_context` only covers the KV260 `/dev/mmcblk` check, not
  `blocked_patterns`). Write the regression test reproducing the audit's own counterexample first (the
  "Guard against FoVer premise reuse" false-positive), fix, run the full `adversarial_verify.py` test
  suite, then a corpus-wide `--backfill` dry-run sanity check before committing, exactly as the QA-Layer
  Authenticity Discipline's "How to apply (operator)" section specifies.
- `exp5205`: an AutoPyVerifier-inspired (arXiv:2604.22937, WebFetch-verified real this session) pilot
  targeting GAP-1 (transpose/orientation discrimination). The paper's own core method -- search a compact
  SET of cheap candidate discriminators via LLM-synthesis + DAG search for joint satisfaction, not one
  hand-crafted invariant -- is precisely what distinguishes this from the already-refuted single
  directional-adjacency hand-invariant. Evaluated against the same square-transpose distractor subset
  (239 tasks, `results/arc_grid_verifier_invariants_v2.json`) the original refutation used, so the result
  is directly comparable.

### Phase Z -- Capstone
- `exp5206`: milestone capstone, reconciling all of the above honestly. Also absorbs the numeric-only
  `docs/technical-report.md` sync `.475` never ran (small, mechanical, folded in rather than kept as a
  separate low-priority task), and reconciles the lp85 registry inconsistency the ARC self-solve audit
  surfaced (L3 canonical vs. L5-claimed vs. L6-quarantined -- determine ground truth from
  `ops/arc_solve_registry.yaml` + the standing `arc_loop_solve_lp85.json`, fix the registry, and either
  restamp or drop the quarantined L6 claim per the audit's own recommendation).

## Dependency graph

```
exp5193 (archive/activate)
   |
   +-- exp5194 (poison-test-cascade triage module)                       [independent, EARLY]
   +-- exp5195 (retro-timing REAL fix + known-issues.md dedup)           [independent, EARLY]
   |
   +-- exp5196 (DiffusionGemma: vLLM-native retry -> custom device_map -> retire)  [independent]
   +-- exp5197 (GAP-4 scale-up, real checkpoint/resume)                  [independent]
   |
   +-- exp5198 (MAP landmark pre-stage A/B/C) --gated_on(lever validated)--> exp5199 (level-up attempt)
   |
   +-- exp5200 (hidden-state verifier v2 + RCS baseline)                 [independent]
   +-- exp5201 (hardware continuity: KV260/PolarFire/GateMate)           [independent]
   |
   +-- exp5202 (architecture.md reconciliation)                         [independent]
   +-- exp5203 (verifier-authenticity DISHONEST_NAMING remediation)     [independent]
   +-- exp5204 (exclusion_manifest_lint.py REAL_BUG fix)                [independent]
   +-- exp5205 (AutoPyVerifier-inspired GAP-1 pilot)                    [independent]
   |
   +-- exp5206 (capstone, reads all of the above; absorbs docs sync + lp85 registry reconciliation)
```

## Hardware requirements

| Task | Hardware | Notes |
|---|---|---|
| `exp5196` | 1x RTX 3090 (CUDA), vLLM stack | `exp5182` already confirmed GPU 0 idle-availability; vLLM-native path is the primary attempt this time |
| `exp5197` | ARC live-submission stack / cached candidate pool, per `exp5161`/`exp5177`'s established methodology | Continue, do not re-derive |
| `exp5198`/`exp5199` | CPU (offline ARC arcade simulation) | No GPU required |
| `exp5200` | 1x RTX 3090 or iGPU, GGUF-cached `gemma-4-26B-A4B-it-GGUF` | Matches `exp5178`'s target model for continuity |
| `exp5201` | KV260 (SSH), PolarFire (SSH), GateMate (USB DirtyJTAG) | Continuity + one genuine GateMate regression-diagnosis attempt |
| `exp5193`, `exp5194`, `exp5195`, `exp5202`, `exp5203`, `exp5204`, `exp5205`, `exp5206` | None (CPU, aggregation/doc/lint work) | `exp5205` evaluates cheap discriminators against a cached distractor pool, no LLM call |

## Risk notes

- **`exp5194` cannot itself apply its fix** -- same standing sandbox constraint as `retro_timing_fallback.py`'s
  own patch-prep pattern: it produces a ready patch + regression test, not a live edit to
  `scripts/research_conductor.py`. Known, accepted limitation, flagged explicitly.
- **`exp5196` may exhaust all known DiffusionGemma loading mitigations a second time.** If vLLM-native
  also fails (plausible -- the blog only demonstrates H100/H200, not a 2x24GB budget) and the custom-
  device_map mitigation also fails, per the Failed-Experiment Rerun Discipline this specific pilot should
  retire pending either an upstream fix or direct operator investigation. Do not propose a further HF/
  accelerate mitigation-variant task in `.477` without a genuinely new theory; GGUF stays blocked until
  `llama.cpp#24427` merges (check before re-proposing).
- **MAP may still not close the enumeration wall.** The 3-arm falsifiable gate is designed so a null result
  is exactly as reportable as a positive one; `exp5199` is gated so a null `exp5198` does not force a
  doomed level-up attempt.
- **The hidden-state verifier v2 may lose to all three free baselines, including Radial Consensus Score.**
  A clean loss to RCS specifically (a method this project had not previously benchmarked against) would be
  a materially informative negative, not a repeat of `exp5178`'s finding -- `retire_if_same_verdict: true`
  is set for exactly this reason.
- **`exp5195`'s retro-timing bug may have a deeper root cause than a simple wiring gap** (e.g. the LLM
  retro-writing agent not faithfully transcribing the assembled `timing_summary` text into its JSON
  schema, independent of whether the Python-side data is correct). If the first diagnosis pass doesn't
  find a clean fix, report the narrowed root cause honestly rather than declaring victory on a
  partial fix -- this bug has now survived one full "fix" cycle already.
- **This plan does not resolve the post-PHASE-D strategic gap or the ARC-deadline correction.** Both are
  named explicitly in "Recommended operator attention" below rather than decided unilaterally; this
  planning pass deliberately does not expand ARC's task-slot share by default, consistent with the
  self-solve audit's recommendation, but also does not treat the matter as closed.

## Recommended operator attention (not autonomous-loop actions)

1. **The `docs/index.html` MIT-0/Apache-2.0 license contradiction, PLUS a structural CSS/div-nesting risk**
   (`ops/docs_audit_report.md`, 2026-07-03) -- the "Recent progress" card is shoehorned into `.stats-bar`
   with an undefined `.r-desc` class and a stray unmatched `</div>`; the "TTC & PREM" card sits after
   another possibly-early-closing `</div>`. This is a rendering-correctness bug, not only a style judgment
   call. No task here touches `docs/index.html` per Public Documentation Discipline.
2. **The ARC Prize deadline correction (firmed up in this pass's second verification).** The `2026-06-30`
   date this project has treated as the final deadline (the trigger for retiring the ARC-AGI-3 Submission
   Sprint Forcing Function) is "Milestone #1" for early open-source prize eligibility per arcprize.org;
   **Milestone #2 is 2026-09-30**, each carrying its own open-source prize tier -- a firmer, nearer-term
   date than the original draft's "may run through November 2, 2026" snippet-sourced guess. Combined with
   the still-unresolved
   "post-PHASE-D strategic direction" question (CLAUDE.md's designated ARC-deadline successor track is
   itself now retired, and the self-solve audit reports zero net live-capability advance in the most recent
   window) -- an explicit operator decision on both facts together would remove ambiguity the next several
   planning cycles would otherwise re-adjudicate independently.
3. **Confirm `carnot-conductor.service`'s restart state and the exp5182 commit.** A prior outer-loop
   session restarted the daemon and left `exp5182`'s live-produced result file for the operator to review;
   confirm it has been committed on a natural commit point if not already done.

## Cross-references

- `ops/conductor-log.md` (2026-07-03 03:15-10:35 UTC) -- the full `.475` timeline this plan verified directly
- `results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json` -- the real mitigation-ladder
  result, read directly (not paraphrased) for this plan
- `results/operational_retro_2026_07_475.json` -- read directly; confirms the retro-timing bug is still live
- `results/diffusiongemma_energy_prior_gguf.json`, `results/diffusiongemma_energy_prior_vllm.json` and their
  `.log` files -- the `.474` probe artifacts read directly to ground `exp5196`'s design
- `research-references.md` `## V476 Planner References` (new section, added alongside this plan) -- this
  session's WebFetch-verified citations (vLLM native support, `transformers#22018`, `llama.cpp#24427`,
  Radial Consensus Score, AutoPyVerifier)
- `ops/verifier_gaps.md` GAP-4891, GAP-1 -- the open gaps `exp5198`/`exp5205` address
- `ops/verifier_authenticity_audit_report.md`, `ops/qa_layer_authenticity_audit_report.md`,
  `ops/arc_self_solve_audit_report.md`, `ops/docs_audit_report.md` -- all four read directly this session;
  two acted on (`exp5203`, `exp5204`), two flagged for operator attention or capstone reconciliation
- `ops/north-star.md` SS0, S3, S5 -- ARC-AGI-3 destination framing, hardware focus, verifier-moat reframe
- `incident_agent_shipped_test_cascade` (memory) -- the 4-occurrence incident class `exp5194` addresses
- CLAUDE.md "Failed-Experiment Rerun Discipline", "Exclusion-Manifest Cross-Check Before Planning",
  "Circularity / Oracle-Distinctness Discipline", "ARC Live-Path Reachability Discipline",
  "Verifier Authenticity Discipline", "QA-Layer Authenticity Discipline", "Tests Must Run and Assert",
  "Inference-Substrate Declaration Discipline", "Overdue-Priority Forcing Function",
  "ARC Level-Up Attempt Guarantee", "SOTA-Ingestion Cycle Discipline"
