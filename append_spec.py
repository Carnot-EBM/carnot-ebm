import sys

with open('openspec/capabilities/research-reporting/spec.md', 'a') as f:
    f.write('\n### REQ-REPORT-3413: Exp 3413 Telemetry Aggregation v39\n\n')
    f.write('The Exp 3413 telemetry aggregation workflow shall read outputs from exp3405 to exp3412 and generate `results/experiment_3413_telemetry_aggregation_v39.json`.\n')
    f.write('The artifact shall include:\n')
    f.write('- `matrix_v39_ready` set to true\n')
    f.write('- `tallies` counting blocked, complete, flagged, and missing metrics\n\n')
    f.write('### SCENARIO-REPORT-3413: Exp 3413 Matrix Generated\n\n')
    f.write('Given outputs for exp3405 to exp3412, when Exp 3413 runs, then it writes all required REQ-REPORT-3413 fields, setting `matrix_v39_ready` to true.\n')
