import sys
from carnot.reporting.experiment_1853_retro import generate_retro

input_paths = [
    "results/experiment_1849_cocom_pruning.json",
    "results/experiment_1850_thrml_parity_n128.json",
    "results/experiment_1851_nla_probe.json",
    "results/experiment_1852_findings_audit.json"
]

output_path = "results/experiment_1853_retro.json"

generate_retro(input_paths, output_path)
print(f"Generated {output_path}")
