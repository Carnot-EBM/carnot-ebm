import sys

spec_content = """
## REQ-LEARN-1774: Differentiable Constraint Memory Bank

**Given** a need for multi-session continual learning without forgetting
**When** the DifferentiableMemoryBank is instantiated
**Then** it MUST provide differentiable read, write, and update operations using attention mechanisms.
**And** it MUST support gradients flowing through the memory read and write operations.

### REQ-LEARN-1774 Sub-requirements

- REQ-LEARN-1774-1: `DifferentiableMemoryBank` SHALL be initialised with `memory_size` and `vector_dim`.
- REQ-LEARN-1774-2: `read(query)` SHALL use softmax attention to return a weighted sum of memory vectors.
- REQ-LEARN-1774-3: `write(key, value)` SHALL write new information into the memory bank differentially.
- REQ-LEARN-1774-4: `update(query, value)` SHALL update existing memory slots based on attention weights.

### SCENARIO-LEARN-1774: Differentiable Memory Operations
**Given** an initialized memory bank
**When** a key-value pair is written, and a similar query is read
**Then** the retrieved value MUST be close to the written value
**And** backpropagation MUST successfully compute gradients for the query.
"""

with open('openspec/capabilities/pipeline/spec.md', 'a') as f:
    f.write(spec_content)
