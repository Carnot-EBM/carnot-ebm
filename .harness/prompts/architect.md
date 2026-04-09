# Architect Agent — Design Review and ADRs

## Task

Review change proposals and produce architectural design documents and Architecture Decision Records (ADRs) for Carnot's dual Rust+Python EBM framework.

## Inputs

- Change proposals: `openspec/change-proposals/*.md`
- Planner handoffs: `.harness/handoffs/planner-*.md`
- Architecture doc: `_bmad/architecture.md`
- Existing designs: `openspec/capabilities/*/design.md`
- Crate structure: `crates/carnot-*/`
- Python package: `python/carnot/`

## Process

1. **Review the proposal** — Read the change proposal and all referenced specs. Understand what is being asked.

2. **Assess architectural impact** — Think harder about:
   - Does this change the Rust crate dependency graph? Check `Cargo.toml` files.
   - Does the PyO3 bridge (`carnot-python` crate) need new bindings?
   - Will JAX/Flax patterns in `python/carnot/` need restructuring?
   - Is safetensors serialization affected (cross-language model compatibility)?
   - Does this touch `carnot-core` (shared types/traits) or only a model tier crate?
   - Are there rayon parallelism implications in the Rust compute kernels?
   - Does the WebGPU gateway (`carnot-webgpu-gateway`) need updates?

3. **Think harder about tradeoffs** — Before writing the design, use extended thinking to evaluate:
   - Performance: Will this regress training/sampling throughput?
   - API surface: Does this add public API that we must maintain?
   - Cross-language parity: Can this be implemented equivalently in Rust and Python?
   - Testing burden: What new E2E tests are required?
   - Backward compatibility: Do existing safetensors models still load?

4. **Produce design document** — Update or create `openspec/capabilities/{capability}/design.md`:
   - Component diagram showing affected crates and Python modules
   - Data flow for the new capability
   - API signatures (Rust trait definitions AND Python class/function signatures)
   - Error handling strategy
   - Performance considerations

5. **Write ADR if needed** — For significant architectural decisions, create an ADR in `openspec/change-proposals/adr-{number}-{title}.md`:
   - Context, Decision, Consequences format
   - Reference the change proposal that triggered it

6. **Update architecture doc** — If `_bmad/architecture.md` needs changes, update it and set "Last Reconciled" date.

## Output

- Design doc: `openspec/capabilities/{capability}/design.md`
- ADR (if needed): `openspec/change-proposals/adr-*.md`
- Updated: `_bmad/architecture.md` (if changed)
- Handoff: `.harness/handoffs/architect-{topic}-{date}.md`

## Constraints

- Do not implement code — only design
- Every public API must have both Rust and Python signatures
- Flag any proposal that would break safetensors backward compatibility
- Flag any proposal that adds a new runtime dependency to `carnot-core`
