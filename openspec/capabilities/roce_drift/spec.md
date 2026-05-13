# ROCE Residual Drift Ledger

## Overview
Reasoning-Time Open Constraint Elicitation (ROCE) validator trees track constraints. The Residual Drift Ledger tracks multi-turn metrics across these compiled trees.

## Requirements
- REQ-ROCE-001: Extract constraints using the prototype ROCE layers.
- REQ-ROCE-002: Record multi-turn tracking metrics with explicit drift case counts.
- REQ-ROCE-003: Enforce `zero_false_accepts=true` logic.

## Scenarios
- SCENARIO-ROCE-1: The ledger parses compiled validator trees and extracts constraints without false accepts.
