import re

path = 'openspec/capabilities/verifiable-reasoning/spec.md'
with open(path, 'r') as f:
    content = f.read()

new_req = """
### REQ-DYNAMIC-EIDOKU-001: Dynamic Eidoku Gate Compiler
- REQ-DYNAMIC-EIDOKU-001-1: `DynamicEidokuCompiler` shall be implemented in `python/carnot/pipeline/dynamic_eidoku.py`.
- REQ-DYNAMIC-EIDOKU-001-2: It shall convert extracted `DynamicConstraint` instances into a `CompiledEidokuGate`.
- REQ-DYNAMIC-EIDOKU-001-3: `CompiledEidokuGate.compute_cost` shall calculate violation cost based on constraint evaluation.

#### SCENARIO-DYNAMIC-EIDOKU-001: Synthesize dynamic gate
**Given** a set of `DynamicConstraint` objects
**When** the `DynamicEidokuCompiler.compile()` is called with these constraints
**Then** it returns an executable `CompiledEidokuGate` instance that can evaluate responses.
"""

if 'REQ-DYNAMIC-EIDOKU-001' not in content:
    content = content.replace('### REQ-VERIFY-1500:', new_req + '\n### REQ-VERIFY-1500:')
    with open(path, 'w') as f:
        f.write(content)
