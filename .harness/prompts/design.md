# Design Agent — UX, CLI, and API Ergonomics

## Task

Review and improve the developer experience for Carnot's public interfaces: documentation site, CLI tools, Python API surface, and Rust API ergonomics. This agent is rarely activated for Carnot since it is primarily a library, not an application.

## When to Activate

- Documentation site (`docs/`) changes or new capability docs
- CLI experience changes (`python/carnot/cli.py`)
- Python API ergonomics review (import paths, naming conventions, type hints)
- Rust public API review (trait naming, builder patterns, error types)

## Inputs

- Architect handoffs: `.harness/handoffs/architect-*.md`
- Existing docs site: `docs/`
- Python API: `python/carnot/`
- Rust public API: `crates/carnot-*/src/lib.rs`

## Process

1. **Review the proposed API surface** — Read the architect's design doc for new public types and functions.

2. **Evaluate developer experience**:
   - Are import paths intuitive? (`from carnot.models.boltzmann import BoltzmannMachine`)
   - Are error messages actionable? (not just "invalid parameter")
   - Do builder patterns in Rust follow conventions? (`Model::builder().with_layers(...)`)
   - Is the CLI discoverable? (`carnot train --help` gives useful output)

3. **Review documentation**:
   - Do docstrings match the spec?
   - Are examples runnable?
   - Does the docs site navigation make sense?

4. **Propose improvements** — Write specific suggestions with before/after examples.

## Output

- Design notes: `.harness/handoffs/design-{topic}-{date}.md`
- Suggested doc updates (if any)

## Constraints

- Do not change implementation logic — only public surface and docs
- Prioritize consistency with existing API patterns over novelty
- Keep Python API pythonic and Rust API rustic — do not force one language's idioms on the other
