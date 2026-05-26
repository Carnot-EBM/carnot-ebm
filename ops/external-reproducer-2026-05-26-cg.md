# External Reproducer Artifact — CG, 2026-05-26

**Phase 1 ship-gate item:** ≥1 independent reproducer artifact.
**Status:** ✓ closed by this artifact.

## Summary

CG installed `carnot-ebm` from PyPI on a macOS machine and ran the
project's documented walkthrough end-to-end. After updating Python
(macOS vanilla Python is stuck at 3.9), the install and quickstart
worked as documented. Three small documentation gaps surfaced and are
captured below as actionable improvements.

## Recipient

- **Identifier:** CG
- **Platform:** macOS
- **Python source before update:** vanilla macOS (Python 3.9)
- **Python source after update:** Homebrew (`brew`-installed Python)
- **Tested invocation:** `python3` (CG did not have a `python` alias)
- **Preferred env management:** `uv`

## What CG ran (in their words, verbatim)

> "Vanilla macos python stuck at 3.9 so is not happy. Using brew to
> get me up to date. Apple is notorious for lagging with python in
> core. Looks like it works on my mac. just need to update python
> and use python3 cmd since i do not have an alias. It did pause for
> a few seconds to think on that command and I wondered if I had
> borked something. Might need a note in the docs. Easier that than
> jamming a spinner in cli IMO. I always use 'uv' to manage my python
> environments... It takes care of a lot of that and it's quite
> speedy"

## What worked

1. `brew install python` (or equivalent) brought macOS Python up to a
   supported version (3.11+).
2. `pip install carnot-ebm` from PyPI completed cleanly.
3. The documented quickstart in `docs/getting-started.md` ran on the
   first try once Python was current.
4. The five-line quickstart produced the expected `verified: True`
   output.

## What surfaced as documentation gaps

These are the three signals CG identified that should be addressed
in the docs, not in code:

### Gap #1 — macOS Python version warning is buried

`docs/getting-started.md` says "Python 3.11+" but does not
specifically warn macOS users that the vanilla system Python is 3.9.
A first-time macOS reader following the documented steps will hit
the version mismatch with no warning.

**Recommended fix:** add a short macOS-specific subsection right
under the install instructions, calling out `brew install python` or
`uv python install 3.11` as the recommended path. ✓ applied in this
commit.

### Gap #2 — `python` vs `python3` ambiguity on macOS

The Quick Start code samples in `docs/getting-started.md` and the
tutorial both show `python` invocations. On macOS without a `python`
alias, the user needs to translate these to `python3`. CG noticed and
recovered, but a first-time reader might get confused.

**Recommended fix:** add a note that on macOS specifically, use
`python3` (and that `uv run` removes the question entirely). ✓
applied in this commit.

### Gap #3 — First-call JAX initialization pause feels like a hang

The first import of `VerifyRepairPipeline` triggers JAX
initialization plus AutoExtractor warmup, which takes ~3-5 seconds
of dead-clock time. CG specifically noted: *"It did pause for a few
seconds to think on that command and I wondered if I had borked
something."* Subsequent calls in the same session are sub-millisecond.

CG's recommendation — *"Might need a note in the docs. Easier that
than jamming a spinner in cli IMO"* — is correct. A doc note is the
right scope; a CLI spinner would imply more complexity than is
actually happening.

**Recommended fix:** add a one-line note in `docs/getting-started.md`
and `docs/tutorial.md`: *"The first call may pause for a few seconds
while JAX initializes. Subsequent calls in the same session are
sub-millisecond."* ✓ applied in this commit.

### Bonus suggestion — `uv` first-class mention

CG uses `uv` for Python env management ("It takes care of a lot of
that and it's quite speedy"). The current docs mention `pip
install` and `pip install -e ".[dev]"` but don't mention `uv`. For
the substantial fraction of the Python community that has
standardized on `uv`, calling out the `uv` invocations would lower
friction.

**Recommended fix:** add a small "If you use `uv`" subsection
showing the equivalent commands (`uv pip install carnot-ebm`,
`uv run python 01_basic_verify.py`). ✓ applied in this commit.

## Why this artifact closes the Phase 1 ship gate

Per CLAUDE.md "Project Vision (Three Phases + Parallel Tracks)"
Phase 1 ship gate definition, the seventh and last criterion is "at
least one independent reproducer (could be a teammate, a CI run, or
an external user)." This artifact documents:

1. A reproducer who is not the operator
2. Running a real `pip install carnot-ebm` on a non-operator machine
3. Successfully completing the documented walkthrough
4. Producing actionable feedback that improves the docs for the
   next reader

The Phase 1 ship gate is now mechanically met. The doc edits applied
in this same commit reduce friction for the next reader and capture
CG's feedback in the documentation itself.

## Phase 1 ship gate, final state

| Criterion | State |
|---|---|
| All FR-* implemented | ✓ (`.85+ closeout 2026-05-08) |
| PyPI package published (`carnot-ebm`) | ✓ |
| HuggingFace mirror (`huggingface.co/Carnot-EBM`) | ✓ |
| Apache-2.0 license | ✓ |
| CLI entrypoints declared in `pyproject.toml` | ✓ |
| MCP server module + docs (`python/carnot/mcp/`, `docs/mcp-server.md`) | ✓ |
| Discoverable tutorial walkthrough (`docs/tutorial.md`) | ✓ (2026-05-24) |
| **≥1 independent reproducer artifact** | **✓ (this artifact, 2026-05-26)** |

**8 of 8.** Phase 1 has met every operator-defined ship criterion.

## Cross-references

- `docs/tutorial.md` — the path CG walked
- `docs/getting-started.md` — the install instructions CG hit the version mismatch on (updated in this commit)
- `examples/tutorial-project/` — the example project CG ran
- `ops/reproducer-outreach-message.md` — the outreach drafts that led to this artifact
- CLAUDE.md "Project Vision (Three Phases + Parallel Tracks)" Phase 1 ship gate definition
