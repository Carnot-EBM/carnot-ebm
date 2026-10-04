# V702 numerical protocol sealed by Exp8111

REQ-VERIFY-8111 / REQ-REPORT-8111. Frozen 20261004 before new V702 outcomes.
This exposed-development protocol cannot establish independent generalization.
Original Exp8098 slots and roles are immutable; labels never select replacements.
The source-trained radial branch has no qualified execution. FR-11 is independent.

The JSON below is the machine-readable numerical contract. Costs are losses;
positive paired gain means control loss minus treatment loss. Altered sources
have no inherited human label. Source interventions are diagnostics only.

```json
{
  "roles": {"fit":128,"tune":64,"evaluation":128,"stream":256,"retention":64},
  "exposure_scope":"exposed_development_within_run_disjoint",
  "source_interventions": {
    "panel":"first32 original fit slots, answer bytes fixed",
    "arms":["full_source","source_removed","cyclic_mismatch"],
    "duplicate":"one identical full_source repeat per panel slot",
    "mismatch":"next panel source modulo32; no replacement",
    "labels":"original-source evaluator labels never transfer to altered sources",
    "metrics":["absolute paired probability difference","parse rate","effect minus duplicate variation"],
    "science_gate":"nonzero source sensitivity is not transport readiness"
  },
  "fitted_head_grid": {
    "arms":["scalar_qwen","linear9","additive_cubic","radial16","equivalent_logistic","always_escalate"],
    "features":"historical Qwen logit clipped to [1e-6,1-1e-6] plus eight public lexical features",
    "geometry":"fit-only mean/std, zero std=1; recomputed within each fold",
    "folds":4,
    "ridge":[0.0001,0.001,0.01,0.1,1],
    "selection":"mean fit-fold log loss; largest ridge on exact ties",
    "intercept_penalized":false,
    "initial_centers":16,
    "reserved_centers":12,
    "center_selection":"public farthest-first, SHA256 source ties",
    "width":"median positive scaled fit pair distance, fallback1",
    "optimizer_max_iterations":256,
    "optimizer_max_seconds":600,
    "gradient_infinity_limit":1e-7,
    "calibration":"tune-only affine logit, identical procedure for every fitted arm",
    "additive":"Exp8098 sealed degree3, quantiles .25/.5/.75, endpoint multiplicity4, padding1e-8, clipping,63 columns",
    "equivalent_probability_tolerance":1e-10
  },
  "typed_costs":{"accept_y1":5,"accept_y0":0,"reject_y0":1,"reject_y1":0,"escalate":0.5,"ties":"escalate"},
  "delayed_memory": {
    "prerequisites":["methods_ready_score","stream_input_ready_score"],
    "fitted_head_required":false,
    "fit_capture_required":false,
    "start":"historical Qwen logit offset plus exactly zero residual",
    "warmup_slots":[1,64],
    "evaluation_slots":[65,256],
    "arms":["error_center","fixed_public_center","random_past_center","frozen_qwen_offset"],
    "geometry":"first64 original public stream slots only; available public values, no labels, no fit/tune data",
    "minimum_warmup_public_sources":28,
    "initial_centers":16,
    "maximum_centers":28,
    "opportunities":[64,128,192],
    "centers_per_opportunity":4,
    "width":"median positive scaled warmup pair distance, fallback1",
    "fixed_centers":"reserve next12 from warmup public farthest-first traversal",
    "random_centers":"seeded label-blind released past update rows",
    "error_centers":"released update rows with immutable frozen-offset issued errors",
    "duplicate_centers":"report duplicates/effective rank, never substitute",
    "shared_eligibility":"newest64 released update rows, at least16 and at least4 frozen-offset issued errors",
    "sgd_steps_per_update":4,
    "learning_rate":0.05,
    "ridge":0.01,
    "intercept_penalized":false,
    "compute_budget_seconds":1200,
    "feedback_delay_slots":20,
    "pending_capacity":32,
    "overflow":"oldest evicted, feedback permanently lost, lossless claim disqualified",
    "roles":"SHA256(source_id) modulo4: bucket0 admission-only, all others update-only",
    "admission":"commit candidate first, then next12 unused admission rows released strictly later; at least2/class",
    "admission_deadline":"before next opportunity and slot256; defer whole block if incomplete",
    "step_grid":[1,0.5,0.25,0.125],
    "admission_guard":"largest passing step: no extra false accepts; cost/Brier no worse than incumbent; cost<=frozen offset+.02, Brier<=frozen offset+.01",
    "rejected_update":"retain old coefficients, keep installed zero-weight centers",
    "event_order":["issue_prediction","durable_commit","commit_candidate","select_future_admission","release_feedback","admit_once","durable_update"],
    "durable_state":["pending_events","issuing_state_hashes","centers","coefficients","optimizer_step","rng_state","consumed_update_ids","used_admission_ids","baseline_hashes","original_slot_mask"],
    "retention":"commit all final heads and retention predictions before evaluator labels open"
  },
  "statistical_plan": {
    "hypotheses":["H1 radial versus additive on128 reserved evaluation slots","H2 error-center versus fixed-public-center on192 later stream slots"],
    "unit":"original source_cluster; seeds and duplicates add zero independent units",
    "multiplicity":"Bonferroni family alpha=.05 across exactly2 hypotheses: one-sided alpha=.025 each",
    "draws":10000,
    "valid_minimum":9500,
    "one_sided_confidence":0.975,
    "gain_lower_bound_minimum":0.02,
    "H1_support":{"complete":96,"per_class":12},
    "H2_support":{"complete":128,"per_class":8,"nonoverlapping_blocks":8},
    "beneficial_sources_minimum":5,
    "extra_false_accepts_maximum":0,
    "brier_increase_maximum":0.01,
    "other_control_cost_increase_maximum":0.02,
    "H1_resampling":"paired source bootstrap",
    "H2_resampling":"average seeds per source, moving original-slot blocks with missing masks",
    "primary_block":16,
    "sensitivity_blocks":[8,32],
    "seeds":[101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120],
    "retention":{"complete":48,"per_class":8,"cost_increase":0.02,"brier_increase":0.01,"extra_false_accepts":0},
    "precision":"descriptive finite-source intervals; no rare-event or lifelong-safety claim"
  }
}
```

Methods readiness authenticates design and original source custody. Stream
readiness additionally authenticates historical completions/features/masks with
at least224 usable stream and48 retention sources. Missing stream evidence
cannot veto sealed methods. No current model loads, generations or head training
occur in Exp8111. Exp8116 must commit real optimizer/RNG/pending states when run;
this seal records state requirements and zero-residual genesis, not learned states.
