with open("openspec/capabilities/research-reporting/spec.md", "a") as f:
    f.write("""
### REQ-REPORT-3763: Next-Phase-3-thesis decision menu

The Exp 3763 workflow shall produce a ranked menu of genuinely-different, UNTESTED routes that SUPERSEDES the .340 exp3722 menu, each with: the route, the ONE paper anchor + its matched-compute claim, WHY it sidesteps BOTH bounded negatives (selection AND generation), the cheapest possible kill-gate to test it, and the honest risk. Rank by (sidesteps-both-negatives) x (matched-compute evidence already exists) x (cheap-to-kill-gate).

The workflow SHALL NOT pick or commit to a route. It hands the operator a RANKED DECISION MENU.

The terminal artifact SHALL include bare top-level values for `honest_verdict`, `inference_substrate`, `ranked_thesis_menu`, `top_recommended_route`, `each_route_sidesteps_both_negatives`, `cheapest_kill_gate_per_route`, `loop_will_not_self_seed`, `supersedes_340_menu`, `random_seed`, `reproducibility_checksum`, and `duration_s`, plus `field_principles` documenting why each required value exists.

`inference_substrate` SHALL be exactly `aggregation_from_upstream_artifacts (principle: a literature/menu synthesis, no live model).`
`loop_will_not_self_seed` SHALL be true.
`supersedes_340_menu` SHALL be true.
`honest_verdict` SHALL equal `complete: next_phase3_thesis_menu_ranked_top_edlm_residual_corrector_supersedes_340_menu_all_routes_sidestep_both_negatives_for_operator_seeding`.

#### SCENARIO-REPORT-3763: Produces Ranked Decision Menu

**Given** the previous .340 menu and bounded Thesis A results
**When** the Exp 3763 workflow runs
**Then** it writes the required terminal artifact with the ranked decision menu of untested routes, marks `loop_will_not_self_seed=true` and `supersedes_340_menu=true`, passes adversarial verification without a critical flag, and leaves `scripts/research_conductor.py` unchanged.

## Implementation Status (REQ-REPORT-3763)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3763 | Planned (`scripts/experiment_3763_next_phase3_thesis_decision_menu.py`) | Planned (`tests/python/test_experiment_3763_next_phase3_thesis_decision_menu.py`) |
""")
