# Milestone .112 Scope-Reduction Manifest

Prior milestone: `2026.04.111`
Target milestone: `2026.04.112`
Run date: `20260507`

| requirement | mapped task id | deliverable path | acceptance field | retire/block rule |
|---|---|---|---|---|
| Activation / compliance manifest | exp1453 | results/experiment_1453_112_scope_reduction_activation_manifest.json; ops/milestone_112_scope_reduction_manifest.md | exp1453.scope_reduction_manifest_complete | Block .112 scope-compliance claims if any mandatory scope item lacks a mapped task, deliverable, acceptance field, or retire/block rule. |
| Experiment artifact classifier | exp1454 | results/experiment_1454_scope_artifact_classifier.json | exp1454.classification_table_written | Classify SIGNAL / NOISE / AMBIGUOUS before adding more experiment variants. |
| known-issues.md MANDATORY priority audit | exp1455 | results/experiment_1455_known_issues_priority_audit.json | exp1455.active_priority_count <= 10 | Block new mandatory-priority expansion until active priorities are trimmed by at least 40%. |
| GRPO/VPRM lineage consolidation and retirement | exp1456 | results/experiment_1456_grpo_vprm_lineage_retirement.json | exp1456.grpo_lineage_retired | Block planner proposals for GRPO v15 unless a human override names a new root cause and falsifiable gate. |
| WOPR puzzle cartridge retirement | exp1457 | results/experiment_1457_wopr_puzzle_cartridge_retirement.json | exp1457.wopr_puzzle_lineage_retired | Block future puzzle cartridges that do not connect to the verify-repair thesis or Phase-3 substrate trajectory. |
| HardNet++/DSP repair stack consolidation | exp1458 | results/experiment_1458_hardnet_dsp_repair_stack_retirement.json | exp1458.hardnet_dsp_lineage_retired | Block new HardNet++/DSP variants; preserve the conservative-replay lesson in one consolidation artifact. |
| Self-learning `_improved_non_headline` lineage decision | exp1459 | results/experiment_1459_self_learning_non_headline_decision.json | exp1459.self_learning_headline_pivot_selected or exp1459.self_learning_lineage_retired | Block another improved-non-headline variant unless it selects a headline pivot or retires the lineage. |
| Hardware portfolio narrowing | exp1460 | results/experiment_1460_hardware_portfolio_narrowing.json | exp1460.active_hardware_track_count <= 3 | Block broad new hardware branches until active hardware tracks are capped at three and out-of-scope tracks are documented. |
| Comparator-integration audit | exp1461 | results/experiment_1461_comparator_integration_audit.json | exp1461.comparator_decision_count >= 6 | Block broad new comparator branches until Abstract-CoT, Meta-Harness, Autodata, LARQL, Skillify, and GStack each receive cite/retire decisions. |
| Paper-v6 anchored-claims narrowing | exp1462 | results/experiment_1462_paper_v6_anchored_claims_narrowing.json | 3 <= exp1462.anchored_claim_count <= 5 | Block paper claim expansion until each retained claim has artifact evidence and unsupported territory is moved to appendix or future work. |

## Live-SOTA Runtime Carry-Forward Rules

- live_sota_runtime_repair_gate: Do not launch repair v3, energy reranking, or 100-case scale-up until a mandated local SOTA GGUF model completes live inference. Prior failures: exp1442=blocked_no_live_sota_runtime. retire_if_same_verdict=True.
- repair_v3_and_prescale_gated_missing: A .112 repair scale task must name the live-runtime fix and cannot reuse the same gate-blocked path as success evidence. Prior failures: exp1443=missing_artifact_gate_blocked_by_exp1442, exp1445=missing_artifact_gate_blocked_by_exp1443_exp1444. retire_if_same_verdict=True.

## Forbidden Exact Expansions

- grpo_v15: new GRPO v15 or GRPO/VPRM variant expansion during .112. Blocked until: exp1456.grpo_lineage_retired=true or explicit human override. Rule: Block as exact noise-line expansion.
- wopr_puzzle_cartridges: new WOPR puzzle cartridges during .112. Blocked until: exp1457.wopr_puzzle_lineage_retired=true. Rule: Block future cartridges unless they name a thesis link.
- hardnet_dsp_variants: new HardNet++/DSP variants during .112. Blocked until: exp1458.hardnet_dsp_lineage_retired=true. Rule: Block variant expansion and consolidate the lesson first.
- broad_comparator_hardware_branches: broad new comparator or hardware branches during .112. Blocked until: exp1460 and exp1461 narrow active hardware and comparator scope. Rule: Block branch expansion until cite/retire and active-track decisions land.
- self_learning_non_headline_variants: new self-learning `_improved_non_headline` variants during .112. Blocked until: exp1459 chooses a headline pivot or retires the lineage. Rule: Block more non-headline suffix churn without a decision.

## No-Change Confirmation

- scripts/research_conductor.py: unchanged_by_exp1453_activation_workflow
- research-roadmap.yaml: unchanged_by_exp1453_activation_workflow
