with open('openspec/capabilities/verifiable-reasoning/spec.md', 'a') as f:
    f.write('\n### REQ-EORM-1811: Track Langevin Energy Gradients for Early Exit\n')
    f.write('**Requirement:** The system MUST run Experiment 1811 to track Langevin energy gradients across EORM transformer layers and identify early-exit thresholds.\n')
    f.write('- Measure `optimal_exit_layer` distribution across a synthetic dataset.\n')
    f.write('- Output metrics to `results/experiment_1811_early_exit.json`.\n\n')
    f.write('### SCENARIO-EORM-1811: Identify Early Exit Layers\n')
    f.write('**When** Experiment 1811 is executed,\n')
    f.write('**Then** it writes a JSON artifact containing `optimal_exit_layer_distribution` and `mean_optimal_layer`.\n')
