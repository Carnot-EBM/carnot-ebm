import sys

with open('openspec/capabilities/self-learning/spec.md', 'a') as f:
    f.write('\n## REQ-LEARN-030: Latent Energy Spills as Reward Signal\n\n')
    f.write('**Given** the FR-11 continuous self-learning loop\n')
    f.write('**When** inference is run using `MODEL_SPECS = ["unsloth/Qwen3.6-35B-A3B-GGUF"]`\n')
    f.write('**Then** the spill detection algorithm identifies when the model relies on language priors\n')
    f.write('**And** failed examples are replayed and constraint templates updated using Latent Energy Spill values as priority weights\n')
    f.write('**And** the script `scripts/experiment_3410_fr11_updates_spills.py` outputs a retention and adaptation score to `results/experiment_3410_fr11_updates_spills.json`.\n\n')
    
    f.write('### SCENARIO-LEARN-030: Spills Driven Constraint Update\n\n')
    f.write('**Given** a set of model outputs\n')
    f.write('**When** latent energy spills are detected\n')
    f.write('**Then** the constraint templates are updated weighted by the spill values, correctly computing adaptation and retention scores.\n')
