# Paper v6 Real-Data Results Table

Run date: `20260518`

## Summary

- Paper-ready local results: `1`
- Missing expected source artifacts: `11`
- Missing `.232` source artifacts: `4`
- Best Carnot Tier 0 AUROC achieved: `0.6852`
- HalluScan gap (`0.88 - best_auroc_achieved`): `0.1948`

## Results Table

| metric_name | value | n_examples | paper_ready | external_baseline | gap_to_baseline |
|---|---:|---:|---|---|---:|
| Tier 0g SemanticEnergy AUROC | 0.6852 | 36 | true | HalluScan AUROC 0.88 | 0.1948 |
| Tier 0h LaaB AUROC | n/a | n/a | false | HalluScan AUROC 0.88 | n/a |
| Tier 0i SpilledEnergy k=18 AUROC | n/a | n/a | false | HalluScan AUROC 0.88 | n/a |
| Tier 0j HALT AUROC | n/a | n/a | false | HalluScan AUROC 0.88 | n/a |
| HIVE 4-verifier ensemble AUROC | n/a | n/a | false | HIVE external AUROC 0.9236 | n/a |
| NSVIF verification pass rate | n/a | n/a | false | none | n/a |
| VERGE repair success rate | n/a | n/a | false | none | n/a |
| KV260 RTL lint errors | n/a | n/a | false | none | n/a |
| KV260 Yosys synthesis succeeded | n/a | n/a | false | none | n/a |
| KAN-CL hard-domain forgetting reduction pct | n/a | n/a | false | none | n/a |
| FST cached-telemetry validation | n/a | n/a | false | none | n/a |
| FST live inference completed | n/a | n/a | false | none | n/a |
| FR-11 cross-domain retention rate | n/a | n/a | false | none | n/a |
| HalluScan external baseline AUROC | 0.88 | n/a | false | external baseline | n/a |
| HIVE external baseline AUROC | 0.9236 | n/a | false | external baseline | n/a |

## Methodology Notes

| metric_name | source_artifact | source_status | methodology_note | adversarial_cleared |
|---|---|---|---|---|
| Tier 0g SemanticEnergy AUROC | results/experiment_2351_semantic_energy_real.json | available | 36 usable cached live Qwen3.6-35B-A3B telemetry rows; AUROC computed from cached-gguf-top-logprobs. | true |
| Tier 0h LaaB AUROC | results/experiment_2368_laab_k17.json | missing | missing source artifact; planned method: LaaB assertion-alignment NLI on real model outputs | false |
| Tier 0i SpilledEnergy k=18 AUROC | results/experiment_2369_spilled_energy_k18.json | missing | missing source artifact; planned method: SpilledEnergy k=18 logprob-compatible verifier on real outputs | false |
| Tier 0j HALT AUROC | results/experiment_2379_halt_tier0j.json | missing | missing source artifact; planned method: HALT latent-probe verifier on real telemetry outputs | false |
| HIVE 4-verifier ensemble AUROC | results/experiment_2380_hive_ensemble.json | missing | missing source artifact; planned method: 4-verifier ensemble over Tier 0g, 0h, 0i, and 0j | false |
| NSVIF verification pass rate | results/experiment_2366_nsvif_verge_real.json | missing | missing source artifact; planned method: NSVIF Z3 verification over real Qwen3.6-35B outputs | false |
| VERGE repair success rate | results/experiment_2366_nsvif_verge_real.json | missing | missing source artifact; planned method: VERGE repair over NSVIF-detected real-output violations | false |
| KV260 RTL lint errors | results/experiment_2372_kv260_rtl_fix.json | missing | missing source artifact; planned method: Verilator lint on KV260 Ising RTL after reserved-keyword fix | false |
| KV260 Yosys synthesis succeeded | results/experiment_2384_kv260_yosys.json | missing | missing source artifact; planned method: Yosys generic or Xilinx synthesis of lint-clean KV260 RTL | false |
| KAN-CL hard-domain forgetting reduction pct | results/experiment_2374_kancl_hard_domains.json | missing | missing source artifact; planned method: KAN-CL B-spline continual-learning stress test on hard domains | false |
| FST cached-telemetry validation | results/experiment_2365_fst_live_gen.json | missing | missing source artifact; planned method: FST PATH C cached telemetry validation on real model outputs | false |
| FST live inference completed | results/experiment_2382_fst_live_path_ab.json | missing | missing source artifact; planned method: FST PATH A/B live llama.cpp or transformers inference attempt | false |
| FR-11 cross-domain retention rate | results/experiment_2375_fr11_real_csl.json | missing | missing source artifact; planned method: FR-11 fast/slow cross-domain retention on real GGUF outputs | false |
| HalluScan external baseline AUROC | arXiv:2605.02443 | external | Published external HalluScan baseline from arXiv:2605.02443; not re-evaluated locally. | true |
| HIVE external baseline AUROC | arXiv:2604.26139 | external | Published external HIVE baseline from arXiv:2604.26139; not re-evaluated locally. | true |

## Source Artifact Availability

| source_id | path | milestone | status |
|---|---|---|---|
| exp2351 | results/experiment_2351_semantic_energy_real.json | 2026.05.230 carry-forward | available |
| exp2365 | results/experiment_2365_fst_live_gen.json | 2026.05.231 | missing |
| exp2366 | results/experiment_2366_nsvif_verge_real.json | 2026.05.231 | missing |
| exp2368 | results/experiment_2368_laab_k17.json | 2026.05.231 | missing |
| exp2369 | results/experiment_2369_spilled_energy_k18.json | 2026.05.231 | missing |
| exp2372 | results/experiment_2372_kv260_rtl_fix.json | 2026.05.231 | missing |
| exp2374 | results/experiment_2374_kancl_hard_domains.json | 2026.05.231 | missing |
| exp2375 | results/experiment_2375_fr11_real_csl.json | 2026.05.231 | missing |
| exp2379 | results/experiment_2379_halt_tier0j.json | 2026.05.232 | missing |
| exp2380 | results/experiment_2380_hive_ensemble.json | 2026.05.232 | missing |
| exp2382 | results/experiment_2382_fst_live_path_ab.json | 2026.05.232 | missing |
| exp2384 | results/experiment_2384_kv260_yosys.json | 2026.05.232 | missing |

## Milestone 2026.05.237 Headline Results

| metric_name | value | source | external_baseline | gap_to_baseline |
|---|---:|---|---|---:|
| Conformal ensemble AUROC (8 verifiers) | 0.9167 | exp2448 | HIVE 0.9236 | -0.0069 (gap remains) |
| Phase 1 ship gate met | true | exp2441 | n/a | n/a |
| FR-11 soundness/completeness tracking | true | exp2451 | n/a | n/a |
| KV260 synthesis state | missing | exp2452 | n/a | n/a |
| GateMate Ising synthesis | bitstream_flashed | exp2453 | n/a | n/a |
| PolarFire SoC smoke | ssh_reachable_install_failed | exp2454 | n/a | n/a |
| ODAR free-energy routing enabled | true | exp2455 | n/a | n/a |
