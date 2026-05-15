# Constraint-Aware Retrieval Module (CARM)

## Overview
The Constraint-Aware Retrieval Module (CARM) bridges the extraction gap using CARM principles. It extracts constraints from natural language instructions.

## Requirements

- **REQ-CARM-1772-1**: The module MUST extract tool-use constraints.
- **REQ-CARM-1772-2**: The module MUST evaluate extraction accuracy on the experiment 1771 CARE test suite.

## Scenarios

- **SCENARIO-CARM-1772-1**: Given an instruction requiring a tool, the prototype extracts the `tools_required` constraint.

## Implementation Status

| Requirement / Scenario | Status | Notes |
| ---------------------- | ------ | ----- |
| REQ-CARM-1772-1        | Implemented | |
| REQ-CARM-1772-2        | Implemented | |
| SCENARIO-CARM-1772-1   | Implemented | |
