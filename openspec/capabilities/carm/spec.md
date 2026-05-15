# Constraint-Aware Retrieval Module (CARM)

## Overview
The Constraint-Aware Retrieval Module (CARM) bridges the extraction gap using CARM principles. It extracts constraints from natural language instructions.

## Requirements

- **REQ-CARM-1772-1**: The module MUST extract tool-use constraints.
- **REQ-CARM-1772-2**: The module MUST evaluate extraction accuracy on the experiment 1771 CARE test suite.
- **REQ-CARM-1773-1**: The module MUST evaluate extraction recall and false accept rate on the CARE test suite using `unsloth/gemma-4-31B-it-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF` models.

## Scenarios

- **SCENARIO-CARM-1772-1**: Given an instruction requiring a tool, the prototype extracts the `tools_required` constraint.
- **SCENARIO-CARM-1773-1**: The dual-model evaluation successfully calculates recall rate and false accept rate for both models.

## Implementation Status

| Requirement / Scenario | Status | Notes |
| ---------------------- | ------ | ----- |
| REQ-CARM-1772-1        | Implemented | |
| REQ-CARM-1772-2        | Implemented | |
| REQ-CARM-1773-1        | Implemented | |
| SCENARIO-CARM-1772-1   | Implemented | |
| SCENARIO-CARM-1773-1   | Implemented | |
