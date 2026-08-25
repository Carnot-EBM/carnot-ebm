#!/usr/bin/env python3
"""Periodic adversarial AI audit of the QA/reconciliation layer itself -- the
mechanical guards that DECIDE whether everything else in this project counts as
clean, preserved, or fabricated. Targets are listed in WHOLE_FILE_TARGETS /
GUARD_TARGETS / CHUNKED_FILE_TARGETS below and include adversarial_verify.py,
exclusion_manifest_lint.py, in_process_doc_reconcile.py, and (since 2026-07-29)
the record-preservation guards: determination_preservation_lint.py,
test_suite_mutation_check.py, operator_curated_docs_lint.py, and the
artifact/claim-integrity lints.

This is the sibling of scripts/verifier_authenticity_audit.py -- that audit
polices whether python/carnot/verify/*.py verifiers do what they claim.
Nobody polices the auditor. Every check in this project's fabrication/gate/
reconciliation machinery is itself hand-written pattern-matching code, and
pattern-matching code has exactly the bug class this audit hunts for.

Origin: 2026-07-03 operator question, after a single outer-loop session found
FOUR real bugs in this exact layer in one sitting:
  - in_process_doc_reconcile.py's map_status_label was missing "success" from
    its win-token list, misclassifying ~19% of the ENTIRE historical artifact
    corpus (352+13 of ~4160 artifacts) for an unknown number of months.
  - adversarial_verify.py's _inference_substrate_text() did str(d.get(...))
    on a field that is commonly a {"principle": ..., "value": ...} dict per
    CLAUDE.md's own "Principle-Annotated Artifact Fields" discipline --
    silently defeating substrate recognition on 176 artifacts corpus-wide.
  - adversarial_verify.py's _flips_gate() did a plain substring check
    ("diffusiongemma_met" in verdict) that matched inside the unrelated word
    "meta" (as in meta_tensor, a real PyTorch term) -- no word-boundary
    awareness.
  - adversarial_verify.py had no substrate category for live embedding-only
    LLM calls, so a genuinely fast-but-real 35s embedding-extraction run got
    floored against the 60s full-generation threshold.
None of these were caught by the linter's own 254 pre-existing unit tests
(tests test what the author thought to test, not the shape of real corpus
diversity) or by any existing adversarial audit (out of scope for both
Layer-2 audits that currently exist). The operator's question: "shouldn't
the adversarial agent be catching these?" -- the honest answer was no
adversarial agent has this layer in scope. This script closes that gap.

SECOND ORIGIN: 2026-07-29 -- "also update the adversarial model to ensure it
enforces this."
--------------------------------------------------------------------------
The overnight session found that the test suite silently rewrites the research
record: one run modified 41 tracked files, including README.md; the 2026-07-27
incident blamed on a "conductor re-run" that stripped seven fabrication-gate
stamps was in fact a TEST RUN, and five of the seven reproduced on demand. The
lesson that belongs to THIS audit is not the rewrite itself -- it is that
`scripts/determination_preservation_lint.py` ALREADY EXISTED to refuse exactly
that commit, and DID NOT FIRE. The destroyed field was named
`inference_substrate_correction_note`: a corrigendum in substance that the
guard's pattern list (`flagged_adversarial` plus keys containing the literal
`corrigendum`) simply did not name. A guard that is trusted and silent is worse
than no guard, because the loop stops looking.

Layer 1 cannot detect its own blind spots -- a pattern list cannot notice the
concept it is narrower than. That is precisely what a hostile independent
reviewer is for, so the prompts below now hunt four further classes that the
original three targets' bug taxonomy did not cover:

  * SILENT NON-FIRING -- name a real input this check is SUPPOSED to catch and
    does NOT. (The tonight class; the highest-value question in this file.)
  * PATTERN LIST NARROWER THAN ITS CONCEPT -- matching `corrigendum` but not
    `*_correction_note`; enumerating a fixed field list where the artifact
    corpus has ~31,510 distinct top-level keys.
  * UNTESTED PATTERN -- a check whose deletion leaves the test suite green,
    because another rule double-covers it. Mutation testing on the WIDENED
    determination lint found this twice: deleting the `correction` marker (the
    pattern its own incident test is NAMED for) left the suite green, and so did
    blanking half the rule-1/rule-2 dedup. Coverage that exists on paper and
    does not bite.
  * HARDCODED ABSOLUTE WRITE TARGET -- `PROJECT_ROOT = "/home/ianblenke/..."`.
    139 scripts and 99 files under python/carnot/ do this, so no worktree or env
    var can isolate them, and a fresh clone writes into the operator's checkout
    (independently a G2 reproducibility defect). Note that canonical_url_lint.py
    correctly PERMITS that literal -- its rule is about the GitHub URL, not the
    filesystem -- so nothing in the project watched it as a WRITE TARGET. A
    genuine gap BETWEEN two guards, which is the kind only a reviewer holding
    both in view can see.
  * TEST MUTATES TRACKED STATE -- a test that imports an experiment module and
    calls main() (1,696 test files import one; 704 call it), or otherwise writes
    results/**, openspec/**, output/**, or an operator-curated doc.

Two audit granularities, chosen by target-file size:
  - Small files (exclusion_manifest_lint.py, in_process_doc_reconcile.py):
    whole-file audit, mirroring verifier_authenticity_audit.py's pattern.
  - adversarial_verify.py (5600+ lines): function-level chunking. Extracts
    every top-level function via `ast`, filters to functions whose body
    does field extraction / string pattern-matching (the exact bug class
    this audit hunts), and audits each chunk independently with its own
    docstring as context. Auditing the whole file in one LLM call would
    either exceed context or produce a shallow, unfocused review.

The audit produces a structured report at
ops/qa_layer_authenticity_audit_report.md. It does NOT edit any file --
operator decides what to act on. Reuses the same audit-integrity guard
(Layer 1.5) as verifier_authenticity_audit.py: an LLM hostile reviewer can
hallucinate its smoking gun, so any flagged verdict whose quoted
high-specificity evidence does not literally appear in the source is
auto-downgraded to CANNOT_DETERMINE rather than acted on.

Usage:
    python scripts/qa_layer_authenticity_audit.py [--model gemini|claude|codex]
                                                   [--file FILE]
                                                   [--limit N]
    python scripts/qa_layer_authenticity_audit.py --check-targets

`--check-targets` is the audit's own defence against the bug class it hunts.
The target list above is itself a pattern list, and a pattern list goes stale
silently: a guard added to .pre-commit-config.yaml is trusted from the moment it
is wired, and nothing would have told anyone it was never audited. So the mode
reads .pre-commit-config.yaml, extracts every hook whose entry runs a
scripts/*.py, and refuses (exit 1) if any of them is NEITHER audited NOR listed
in ACKNOWLEDGED_NON_QA_LAYER with a written reason. It never silently drops a
guard on the floor; it forces a decision and records it.

Designed to be called from the conductor's milestone-close path (see
scripts/research_conductor.py:_run_operational_retrospective's caller),
alongside the existing verifier and landing-page audits.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "ops" / "qa_layer_authenticity_audit_report.md"
PRECOMMIT_CONFIG = PROJECT_ROOT / ".pre-commit-config.yaml"

# Whole-file targets: small enough to audit in one shot.
WHOLE_FILE_TARGETS = (
    PROJECT_ROOT / "scripts" / "exclusion_manifest_lint.py",
    PROJECT_ROOT / "scripts" / "in_process_doc_reconcile.py",
)

# ---------------------------------------------------------------------------------------
# GUARD TARGETS (added 2026-07-29).
#
# These are the RECORD-PRESERVATION and CLAIM-INTEGRITY guards: code whose job is to
# refuse a commit, quarantine an artifact, or reject a claim. Their characteristic failure
# is not a crash and not a false alarm -- it is SILENCE. determination_preservation_lint.py
# printed "OK" on the very commit that destroyed a hand-written corrigendum, because the
# destroyed key was called `inference_substrate_correction_note` and the guard's pattern
# list only knew the literal `corrigendum`. Nothing downstream could tell the difference
# between "checked and clean" and "checked with the wrong pattern".
#
# They get their own prompt (PER_GUARD_PROMPT) rather than sharing PER_FILE_PROMPT,
# because the question that matters for them is inverted: not "is this logic wrong" but
# "name a real input this is SUPPOSED to catch and does NOT."
#
# Entries are (path, one-line reason it is in QA-layer scope). A file that is absent from
# disk is skipped rather than crashing the run -- a guard may be mid-migration in another
# workflow, and an audit that hard-fails on a concurrent rename is an audit nobody runs.
# ---------------------------------------------------------------------------------------
GUARD_TARGETS: tuple[tuple[Path, str], ...] = (
    (
        PROJECT_ROOT / "scripts" / "substrate_alias_evidence_lint.py",
        "decides whether a commit may WIDEN adversarial_verify's no-LLM allowlist, and "
        "every name added there exempts artifacts from the fabrication gate's duration "
        "check -- a silent non-firing here lets an experiment clear the gate by naming "
        "its own substrate in the gate (measured 2026-08-23: 19 of 38 aliases added in "
        "two days, each in the same commit as the artifact it exempted)",
    ),
    (
        PROJECT_ROOT / "scripts" / "determination_preservation_lint.py",
        "refuses commits that drop a fabrication-gate determination; the 2026-07-29 "
        "silent non-firing is this audit's second origin incident",
    ),
    (
        PROJECT_ROOT / "scripts" / "test_suite_mutation_check.py",
        "detects a test run rewriting tracked files; its pre-commit --gate is the only "
        "interlock between a green test run and a published rewrite of the record",
    ),
    (
        PROJECT_ROOT / "scripts" / "operator_curated_docs_lint.py",
        "enforces the Public Documentation Discipline at the COMMIT layer -- it only sees "
        "files that reach `git add`, which is why it never saw README.md being rewritten "
        "on every test run",
    ),
    (
        PROJECT_ROOT / "python" / "carnot" / "testing" / "operator_curated_doc_guard.py",
        "the RUNTIME half of the same discipline: fires at the moment a test writes an "
        "operator-curated doc, naming the writer. Not a pre-commit hook, so --check-targets "
        "would never have found it -- listed here explicitly",
    ),
    (
        PROJECT_ROOT / "python" / "carnot" / "testing" / "child_results_guard.py",
        "carries the results/ write redirect across the subprocess boundary, so it decides "
        "whether a test's child process may rewrite committed evidence. Its own origin is a "
        "SILENT_NON_FIRING (2026-08-24): the two in-process guards are per-interpreter and "
        "could not see a child, so every subprocess writer stripped flagged_adversarial and "
        "the corrigendum fields while the suite reported green",
    ),
    (
        PROJECT_ROOT / "scripts" / "artifact_freshness_lint.py",
        "decides whether an analyser-produced artifact is stale, i.e. whether its numbers "
        "may still be cited",
    ),
    (
        PROJECT_ROOT / "scripts" / "arc_artifact_lint.py",
        "decides whether an ARC artifact's inference-substrate declaration is admissible",
    ),
    (
        PROJECT_ROOT / "scripts" / "arc_count_integrity_lint.py",
        "decides whether a reproduced-level count is honest -- the ARC headline metric",
    ),
    (
        PROJECT_ROOT / "scripts" / "arc_llm_on_liveness_lint.py",
        "decides whether an ARC row claiming the LLM tier actually had a live generator",
    ),
    (
        PROJECT_ROOT / "scripts" / "verifier_authenticity_lint.py",
        "Layer 1 of the Verifier Authenticity Discipline; note its sibling Layer-2 audit "
        "reviews python/carnot/verify/*.py, NOT this linter, so the linter itself was "
        "unaudited by anything until now",
    ),
    (
        PROJECT_ROOT / "scripts" / "arc_orphan_solver_lint.py",
        "decides whether an ARC solver counts as live-path-reachable, i.e. whether the "
        "work counts at all",
    ),
    (
        PROJECT_ROOT / "python" / "carnot" / "testing" / "tracked_results_guard.py",
        "RUNTIME guard for the sibling half of the Test-Run Record Integrity Discipline: "
        "fires at the moment a test writes a tracked results/** file, the exact mechanism "
        "that silently rewrote the research record (2026-07-29 incident). Same shape as "
        "operator_curated_doc_guard.py above -- not a pre-commit hook, so --check-targets "
        "alone would never have found it -- listed here explicitly, added 2026-08-05 after "
        "landing 7 hours unclassified.",
    ),
    (
        PROJECT_ROOT / "scripts" / "research_complete_ledger_lint.py",
        "decides whether the milestone ledger -- the planner's failure record -- may "
        "change; the regression lock for truthful archival (REQ-CONDUCTOR-ARCHIVE-2, "
        "2026-08-21: 57 phantom OK rows and 1,841 duplicate entries landed silently)",
    ),
    (
        PROJECT_ROOT / "scripts" / "mutation_marker_lint.py",
        "decides whether a mutation proof's marker may reach a commit. Its silent "
        "non-firing publishes a mutated line into the record on a path nothing else "
        "watches: a marked no-op statement is valid Python that clears every other "
        "hook here, and on 2026-08-25 one sat in arc_executable_world_model.py on the "
        "LIVE ARC scored path. It is also the ONLY mechanical half of "
        "REQ-OPS-MUTATION-PROOF -- the session wrapper beside it is opt-in -- so a hole "
        "here is the whole defense. (The marker token is deliberately not spelled out "
        "in this string: the lint scans this file, and quoting it would brick every "
        "commit.)",
    ),
    (
        PROJECT_ROOT / "scripts" / "audit_findings_ledger.py",
        "decides whether a flagged audit verdict is ever answered, so its silent non-firing "
        "makes every OTHER audit decorative. Its own origin is that failure (2026-08-25): it "
        "read one report while its stated concept was 'flagged audit verdicts someone must "
        "answer', so 7 SILENT_NON_FIRING findings from two milestone closes reached no ledger "
        "and were overwritten by the next regeneration. It is not pre-commit wired, so "
        "--check-targets would never have found it -- listed here explicitly",
    ),
    (
        PROJECT_ROOT / "scripts" / "run_stop_authority.py",
        "decides whether a live measurement RUN or an orphaned llama-server dies "
        "(REQ-CONDUCTOR-AUTHORITY-1/2). Both failure directions touch the record: "
        "silent non-firing lets a dead-tier run write hours of invalid rows that read "
        "as a wrong conclusion about the mechanism under test (the 2026-08-22 ar25 "
        "levels=0 near-miss), and a false fire destroys valid evidence mid-write. Its "
        "conjunctive predicate is exactly the pattern-list-vs-concept shape this audit "
        "hunts, and nothing else reviews it on a cadence",
    ),
)

# Pre-commit-wired scripts deliberately OUT of QA-layer scope, each with the reason. This
# is not a convenience list -- it is the audit trail for `--check-targets`. Adding a name
# here is a decision someone has to write down, which is the whole point: the failure mode
# being defended against is a guard drifting into the trusted set with nobody noticing.
ACKNOWLEDGED_NON_QA_LAYER: dict[str, str] = {
    "canonical_url_lint.py": (
        "string discipline over one URL literal; its failure publishes a wrong link, it "
        "does not admit or quarantine a research claim"
    ),
    "paper_v6_narrowing_lint.py": (
        "prose-phrasing discipline for the paper draft; failure over-claims in text that "
        "is operator-reviewed before submission anyway"
    ),
    "pages_fever_dream_lint.py": (
        "presentation-quality lint for the landing page; failure costs professional "
        "polish, not a determination. NOTE its sibling Layer-2 (pages_adversarial_audit) "
        "reviews the PAGE, not this linter -- so this is an acknowledged, recorded gap. "
        "RE-DECIDED 2026-07-29 ON EVIDENCE, not on this description: the file was read in "
        "full and audited once via `--file --guard-prompt --model codex`, which returned "
        "SILENT_NON_FIRING. Every defect it named was independently reproduced and every "
        "one is UNDER-DETECTION OF PRESENTATION BLOAT -- `Exp. 1001`, `exp1001`, "
        "`experiment 1001` all evade EXP_ID_RE, which only knows `Exp 1001`; "
        "`inference_mode=calibrated` and `n_seeds=5` evade FLAG_SYNTAX_RE, which only "
        "knows booleans; and a missing docs/index.html returns 0, so pre-commit sees "
        "clean. Consequence of every one is jargon or bloat reaching the landing page. "
        "None reads an artifact, admits a claim, or preserves a record, so it cannot let "
        "a false research claim through -- which is the criterion GUARD_TARGETS is drawn "
        "on. Kept OUT so audit attention stays on guards whose silence costs a "
        "determination. THE STANDING GAP IS UNCHANGED AND DELIBERATE: nothing reviews "
        "this linter on a cadence, so if it ever grows a rule that gates a CLAIM rather "
        "than polish, this entry must be revisited -- there is currently no mechanism "
        "that forces that revisit when the file changes"
    ),
    "overdue_priority_lint.py": (
        "planning-cadence forcing function; failure delays work rather than admitting a false claim"
    ),
    "arc_levelup_guarantee_lint.py": (
        "planning-cadence forcing function (ARC slot floor); same reasoning as "
        "overdue_priority_lint"
    ),
    "batching_precommit_check.py": "build/perf convention, no record-integrity role",
    "check_spec_coverage.py": "build-time spec/test traceability, no record-integrity role",
    "precommit_sync_technical_report.py": "mechanical md->html sync, no judgement to get wrong",
    "check_torch_cuda.py": "environment capability probe, no record-integrity role",
    "arc_precondition_nocov_lint.py": "pytest-invocation convention, no record-integrity role",
    "arc_nocov_precondition_lint.py": "pytest-invocation convention, no record-integrity role",
}

# Function-chunked target: too large for a whole-file audit.
CHUNKED_FILE_TARGETS = (PROJECT_ROOT / "scripts" / "adversarial_verify.py",)

# Above this size a whole-file review stops being a review and starts being a skim, so the
# file is function-chunked instead. Deliberately generous (adversarial_verify.py is 6,200
# lines): a SILENT-NON-FIRING judgement needs to see EVERY rule at once, because the finding
# is "no rule here covers input X" -- which cannot be reached from one chunk in isolation.
# Chunking is therefore a last resort for this audit, not a default.
CHUNK_THRESHOLD_LINES = 1200

# A function body is audit-worthy (does field extraction / pattern-matching --
# the exact bug class this audit hunts) if it contains any of these markers.
# Excludes pure scaffolding (argparse setup, report string-building, Flag
# dataclass definitions) that has no field-shape or substring-boundary risk.
RISKY_BODY_MARKERS = (
    ".get(",
    " in ",
    ".lower()",
    ".upper()",
    "re.search(",
    "re.match(",
    "re.compile(",
    ".startswith(",
    ".endswith(",
    "isinstance(",
)

# ---------------------------------------------------------------------------------------
# The 2026-07-29 bug classes. EVERY prompt in this file must cover all five --
# see BUG_CLASS_MARKERS below for how that is enforced.
#
# The original three questions ask "is this logic WRONG". These five ask the harder
# question: "is this logic SILENT where it should speak". A wrong check announces itself
# eventually -- someone hits the false positive and complains. A silent check never does,
# and every day it stays silent it earns more trust it has not got.
#
# The worked example is deliberately the real one, in full detail, because a hostile
# reviewer given an abstract instruction ("look for gaps") returns abstract findings. Given
# a concrete precedent with the exact shape of the miss, it returns concrete ones.
#
# THIS COMMENT USED TO LIE, and the lie is worth recording because it is the very class of
# bug this file hunts. It read "shared by every prompt in this file" while PER_GUARD_PROMPT
# -- the prompt used for the ONLY targets that are themselves guards -- spliced none of it.
# Classes D and E therefore never reached a guard review at all. That was caught by review,
# not by any test, and the demonstration was concrete: an audit fixture that wrote into
# `results/` on every run produced eight findings from the guard prompt and not one of them
# mentioned the write. A constant whose comment claims a scope wider than its actual call
# sites is the same shape as a lint whose pattern list is narrower than its concept -- the
# deliverable was committing the bug it was written to detect. BUG_CLASS_MARKERS + the test
# that asserts every prompt contains every marker is what makes the claim checkable rather
# than merely stated.
# ---------------------------------------------------------------------------------------
SHARED_BUG_CLASSES = """\
NOW THE HARDER CLASSES. Everything above asks whether this code does the wrong
thing. These five ask whether it does NOTHING where it was supposed to act --
which is worse, because a check that is silent is trusted, and a check that is
wrong eventually annoys somebody into looking at it.

The worked precedent (2026-07-29, this project, real):
`determination_preservation_lint.py` exists to refuse any commit that drops a
fabrication-gate determination or a corrigendum record from a results artifact.
A test run overwrote `results/experiment_3946_r11l_first_solve.json` and
destroyed a hand-written field named `inference_substrate_correction_note`.
The guard printed "determination-preservation-lint: OK" and the commit went
through. Why: its pattern list was the literal field name `flagged_adversarial`
plus "any key containing the substring `corrigendum`". The destroyed key was a
corrigendum IN SUBSTANCE -- a recorded human correction, exactly the thing the
guard's own docstring says it protects -- but it did not contain that literal
substring, and nothing anywhere compared the pattern list against the concept.
The guard was not wrong. It was NARROW, and silence looks identical to safety.

A. **SILENT NON-FIRING -- answer this first, and concretely.**
   Name a REAL, plausible input that this code is SUPPOSED to catch (per its own
   docstring, name, or the discipline it enforces) and DOES NOT. Not a
   hypothetical shape -- an actual string, field name, file path, or value a
   real agent in this project would plausibly produce. If you cannot name one,
   say so explicitly; do not invent a contrived one to look thorough.

B. **PATTERN LIST NARROWER THAN ITS CONCEPT.**
   For every hardcoded list, tuple, set, prefix, or regex alternation of names,
   tokens, or markers: state the CONCEPT the list is standing in for, in one
   sentence, then name a member of that concept the list omits. Watch for the
   `corrigendum` / `*_correction_note` shape specifically: a list that enumerates
   a handful of field names when the artifact corpus has roughly 31,510 distinct
   top-level keys is a sample of the concept, not a definition of it.

C. **UNTESTED PATTERN.**
   For each individual pattern/rule/branch: if it were DELETED, would anything
   fail? Or is it double-covered by a broader neighbouring rule, so that the
   named rule is decorative and its test is actually passing because of the
   other one? (Mutation testing on the widened version of the lint above found
   exactly this twice, including on the pattern its own incident regression test
   is NAMED for.) Name any rule you believe is deletable with the suite still
   green.

D. **HARDCODED ABSOLUTE WRITE TARGET.**
   Does this code write to, or compute, an ABSOLUTE path baked into the source
   (e.g. `PROJECT_ROOT = "/home/<someone>/..."`) rather than deriving it from
   `Path(__file__)`, an env var, or an argument? Any such path makes the code
   unrunnable in a worktree or a fresh clone WITHOUT FAILING -- it succeeds, and
   writes into the original checkout. Treat it as both a correctness bug and a
   reproducibility defect. Note that this project's canonical_url_lint.py
   deliberately PERMITS that same literal string, because its rule concerns the
   GitHub URL and not the filesystem -- so do not assume another guard covers it.

E. **TEST/SIDE-EFFECT MUTATION OF TRACKED STATE.**
   Does this code (or a test that would naturally be written for it) WRITE to
   `results/**`, `openspec/**`, `output/**`, `ops/**`, or an operator-curated doc
   (README.md, docs/index.html, docs/roadmap.md, docs/getting-started.md,
   docs/cli-usage.md, docs/mcp-server.md, docs/blog/*, docs/CNAME) as a side
   effect of merely running? Writing to a FIXED artifact path that a committed
   historical artifact already occupies is the specific hazard: it overwrites the
   research record on a green run, with no failure and no diff anyone reads.

F. **DEFAULT BRANCH DISABLES THE CHECK.**
   Does a recognizer chain -- a run of `if matches_X: return floor_X` clauses --
   end in a default that means NO CHECK rather than "unrecognized, say so"? The
   tell is a terminal `return None` / `return` / `pass` whose caller reads it as
   permission to skip. That inverts the safe direction: an input the chain has
   never been taught reads as approved rather than as unverified.
   Worked precedent, 2026-08-21: `adversarial_verify.duration_floor_for_artifact`
   walked ~15 substrate recognizers and returned None when none matched, and
   `check_duration_vs_claim` then returned before applying any floor. Measured
   over the corpus, 720 artifacts declared a substrate and received no duration
   floor at all; 237 of those ran in under a second. The same shape had ALREADY
   been fixed once for a single substrate on 2026-07-30 -- a one-case patch of a
   general defect, which is why this class exists. Ask: what does this code do
   with an input NONE of its branches recognize, and would a reader of the output
   be able to tell that case apart from a genuine pass?

G. **METRIC COMPUTED BEFORE THE WORK IT MEASURES.**
   Is a duration, counter, or other measurement evaluated BEFORE -- or
   independently of -- the operation it purports to describe? The commonest form
   is a value computed inline as a CALL ARGUMENT, since arguments are evaluated
   before the callee runs. Such a field is a constant wearing a measurement's
   name, and any check downstream that trusts it is inert.
   Worked precedent, 2026-08-21: four experiment scripts wrote
   `build_artifact(..., duration_s=max(time.monotonic() - start, 0.0001), ...)`
   where `start` was set on the preceding line, so the stored duration was always
   exactly the 0.0001 floor no matter how long the run took -- while that field's
   own documented principle is that wall time catches a comparison which skipped
   the expensive path. Ask: does the value actually span the work, and would it
   differ between a real run and a run that did nothing?
"""

# The canonical name of each bug class above. Every prompt in this file MUST contain every
# one of these strings, and `test_every_prompt_covers_every_bug_class` fails the build if
# one does not. The marker is the class NAME rather than a splice of SHARED_BUG_CLASSES on
# purpose: PER_GUARD_PROMPT already asks classes A-D in its own framing, aimed specifically
# at guards, and splicing the shared text in as well would state the same worked precedent
# twice in one prompt. Duplication dilutes a prompt; what actually needed enforcing is
# COVERAGE, not identical wording. So each prompt phrases the class however suits its
# target and simply has to use the canonical name while doing it.
# Suffixes a reviewer could legitimately cite as evidence-by-path. Kept deliberately short:
# every entry widens the audit-integrity guard's un-void path, and un-voiding is the only
# direction this exemption moves. Source, config, artifacts, docs -- the things that
# actually exist in this repo and that a finding would name.
CITABLE_PATH_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".toml",
    ".cfg",
    ".sh",
    ".tex",
    ".html",
)

BUG_CLASS_MARKERS: tuple[str, ...] = (
    "SILENT NON-FIRING",
    "PATTERN LIST NARROWER THAN ITS CONCEPT",
    "UNTESTED PATTERN",
    "HARDCODED ABSOLUTE WRITE TARGET",
    "TEST/SIDE-EFFECT MUTATION OF TRACKED STATE",
    "DEFAULT BRANCH DISABLES THE CHECK",
    "METRIC COMPUTED BEFORE THE WORK IT MEASURES",
)

SHARED_OUTPUT_TAIL = """\
## MISSED INPUT
<the concrete real-world input from class A that this code should catch and does
 NOT -- a literal field name, verdict string, path, or value. Or "none found".>

## RECOMMENDATION
<one of: KEEP | ADD_WORD_BOUNDARY | ADD_FIELD_UNWRAP | ADD_TOKEN | WIDEN_PATTERN_TO_CONCEPT |
 ADD_TEST_CASE | DERIVE_PATH_FROM_FILE | ISOLATE_WRITE_TARGET | NEEDS_REDESIGN>

## RATIONALE
<2-3 sentences>
```

Be hostile. If you find no problems, say CLEAN and move on -- do not invent
issues to seem thorough. Inventing a finding is not a harmless excess of
caution here: a fabricated smoking gun wastes an operator's investigation and
teaches the project to distrust this audit.

IMPORTANT about quoting: inside `## FINDINGS`, `## CLAIM` and `## RATIONALE`,
only put a backtick-quoted code span around text that LITERALLY APPEARS in the
code you were given -- those sections are machine-checked against the source and
a verdict citing text that is not there is automatically discarded. Constructed
inputs that do NOT appear in the source (your counterexample, your missed input)
belong in `## COUNTEREXAMPLE` and `## MISSED INPUT`, which are exempt from that
check by design.
"""

PER_CHUNK_PROMPT = """\
You are a hostile software reviewer auditing a piece of QA/fabrication-detection
code from the Carnot project's adversarial_verify.py -- the mechanical linter
that runs on every research-experiment artifact to catch fabricated or
implausible results (fake AUROC numbers, impossibly-short compute durations,
gamed statistical tests, etc).

This code is CRITICAL infrastructure: a bug here either lets fabrication
through (a false negative -- dangerous, the linter's whole purpose fails
silently) or falsely quarantines honest work (a false positive -- costly,
wastes investigation time and can mislead the project into thinking real
progress is suspect). A single outer-loop session on 2026-07-03 found FOUR
real bugs of this kind in one sitting:

1. A field-shape assumption bug: code did `str(d.get("some_field"))` assuming
   the field is always a bare string/number, but the project's own
   "Principle-Annotated Artifact Fields" convention (documented in CLAUDE.md)
   allows ANY field to be written as `{"principle": "...", "value": ...}`.
   `str()` on that dict produces a Python repr matching nothing -- silent
   failure across 176 real artifacts.

2. A substring-boundary bug: `"diffusiongemma_met" in verdict.lower()` matched
   inside the unrelated word "meta" (as in "meta_tensor", a real PyTorch
   term) purely by coincidence -- no word-boundary check.

3. A substrate-taxonomy gap: a real, honest, fast-but-genuine compute pattern
   (embedding-only LLM calls, much cheaper than full generation) had no
   matching category, so it fell through to a floor calibrated for a
   completely different (much more expensive) workload.

Your job: find more bugs like these in the function below. Answer THIS
structured set of questions. Do not soften the answers. Do not rationalize.

1. **Field extraction assumptions.** For every `d.get("field_name")` or
   similar dict-field read: does the code assume a specific TYPE (bare
   string, bare number, bare bool) without handling the case where the
   field might be a `{"principle": ..., "value": ...}` wrapped dict, a list,
   or None? Quote the specific line.

2. **String/substring matching without boundaries.** For every `in`,
   `.startswith(`, `.endswith(`, or `re.search`/`re.match` against free-text
   fields (honest_verdict, inference_substrate, docstrings, etc.): could the
   pattern match INSIDE an unrelated, longer word or phrase that happens to
   contain it as a substring? Try to construct a concrete counterexample
   string that would falsely match (or falsely fail to match).

3. **Negation / context blindness.** Does any check that scans free text for
   a forbidden/flagged phrase fail to distinguish "the artifact DOES X" from
   "the artifact explicitly did NOT do X" / "the artifact correctly avoided
   X" / "this check exists to detect X"? (E.g. would a verdict saying
   "blocked_X_not_attempted" incorrectly trigger a check meant to catch
   artifacts that DID X?)

4. **Boundary / off-by-one errors.** For any numeric threshold, comparison,
   or floor: is the comparison operator correct (`>` vs `>=`, `<` vs `<=`)?
   Does it handle the exact-equal-to-threshold case the way the docstring
   implies it should?

5. **Does the implementation match what the function's own docstring or name
   claims it detects?** Quote the claim, then say whether the actual logic
   is narrower, broader, or different from that claim.

6. **Construct one concrete, realistic artifact fragment** (a plausible
   honest_verdict string, or a plausible field dict) that would be
   MIS-classified by this function -- either a false positive (flags honest
   work) or a false negative (misses something it should catch). If you
   cannot construct one, say so explicitly.

{SHARED}
Output format -- exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: CLEAN | MINOR_RISK | REAL_BUG | SILENT_NON_FIRING | CANNOT_DETERMINE>

## CLAIM
<what the function's docstring/name claims to do, 1 sentence>

## FINDINGS
<numbered list of concrete issues found per the questions above, or "none found">

## COUNTEREXAMPLE
<the concrete mis-classifying input you constructed, or "none constructed">

{TAIL}""".replace("{SHARED}", SHARED_BUG_CLASSES).replace("{TAIL}", SHARED_OUTPUT_TAIL)

PER_FILE_PROMPT = """\
You are a hostile software reviewer auditing a QA/reconciliation script from
the Carnot project -- code that classifies, gates, or reconciles research
artifacts and roadmap tasks. A single outer-loop session on 2026-07-03 found
a real bug of exactly this shape: in_process_doc_reconcile.py's
map_status_label was missing "success" from its list of terminal-success
tokens, silently misclassifying ~19% of the entire historical artifact
corpus (352+13 of ~4160 artifacts, going back many months) as something
other than Complete.

Your job: review the WHOLE FILE below for the same class of bug. Answer:

1. **Classification/token lists.** For every list of tokens, prefixes, or
   markers used to classify a status/verdict/pattern: is it plausibly
   INCOMPLETE (missing an obvious synonym or common real-world phrasing)?
   Cross-check against the project's documented "Verdict Terminal-Prefix
   Discipline" (CLAUDE.md): terminal verdicts start with complete:/complete_/
   success:/success_/passed:/passed_/shipped:/shipped_. Does every one of
   those 8 prefix forms have matching recognition logic?

2. **Substring matching without boundaries.** Same question as for
   adversarial_verify.py -- could a pattern match inside an unrelated
   longer word?

3. **Field-shape assumptions.** Does this file assume fields are bare
   strings/numbers when the project's "Principle-Annotated Artifact Fields"
   convention allows `{"principle": ..., "value": ...}` wrapping?

4. **Negation / context blindness.** Does any forbidden-pattern scan fail
   to distinguish "did X" from "explicitly did not do X" / "correctly
   avoided X"?

5. **Construct one concrete, realistic counterexample** (a plausible
   verdict string or roadmap task fragment) that this file would
   mis-classify.

{SHARED}
Output format -- exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: CLEAN | MINOR_RISK | REAL_BUG | SILENT_NON_FIRING | CANNOT_DETERMINE>

## FINDINGS
<numbered list of concrete issues, or "none found">

## COUNTEREXAMPLE
<the concrete mis-classifying input you constructed, or "none constructed">

{TAIL}""".replace("{SHARED}", SHARED_BUG_CLASSES).replace("{TAIL}", SHARED_OUTPUT_TAIL)


PER_GUARD_PROMPT = """\
You are a hostile software reviewer auditing a GUARD from the Carnot project --
a script whose entire job is to REFUSE something: refuse a commit, quarantine an
artifact, reject a claim, or report that a run rewrote files it should not have.
Most of these are wired into pre-commit, so they run unattended, on every commit,
in front of an autonomous research loop that runs ~30 milestones a day.

READ THIS BEFORE YOU START, because it defines what "a bug" means here.

For ordinary code, the dangerous failure is doing the wrong thing. For a guard,
the dangerous failure is doing NOTHING. A guard that raises a false alarm gets
noticed within a day -- somebody's honest commit is refused and they come
looking. A guard that stays silent on a real violation is never noticed at all;
worse, every silent day increases the trust placed in it, and the project stops
performing the manual check the guard was supposed to replace. The record then
degrades with a green light on.

This has already happened here, on 2026-07-29, to a guard in this same family.
`determination_preservation_lint.py` was written to refuse any commit that drops
a fabrication-gate determination or a corrigendum record from a results
artifact. A test run overwrote a results artifact and destroyed a hand-written
field named `inference_substrate_correction_note`. The guard printed
"determination-preservation-lint: OK". The commit went through. The reason was
not subtle in hindsight: the guard matched the literal field name
`flagged_adversarial`, plus any key containing the substring `corrigendum`. The
destroyed field was a corrigendum in substance -- a recorded human correction,
precisely what the guard's docstring says it protects -- but it did not contain
that substring. The guard was not incorrect. It was NARROWER THAN ITS OWN
CONCEPT, and from the outside that is indistinguishable from working.

Your job is to find the NEXT one of those. Answer these questions about the
guard below.

1. **What is this guard's CONCEPT?** In one sentence, what class of bad thing is
   it supposed to prevent -- as stated by its docstring, its name, and the
   discipline it enforces? State the concept, not the implementation.

2. **SILENT NON-FIRING -- the central question.** Name a REAL, plausible input
   that falls inside that concept and that this guard DOES NOT catch. Be
   concrete: an actual field name, verdict string, file path, config value, or
   diff shape that an agent in this project would plausibly produce. Trace the
   code and say exactly which line lets it through. If you genuinely cannot find
   one, say so explicitly -- do not manufacture one.

3. **PATTERN LIST NARROWER THAN ITS CONCEPT.** For every hardcoded
   list/tuple/set/regex of names, tokens, prefixes or markers: is it a
   DEFINITION of the concept, or a SAMPLE of it? Name omitted members. The
   `corrigendum` /
   `*_correction_note` shape above is the archetype; look also for a fixed field
   list where the domain is open-ended (this project's artifact corpus has about
   31,510 distinct top-level keys).

4. **Scope holes.** What does the guard NOT look at -- which directories, file
   types, diff kinds (added / deleted / renamed files), or execution paths are
   outside its reach entirely? A guard that inspects `results/*.json` says
   nothing about `docs/`; a guard that reads the staged diff says nothing about
   a run that never staged anything. State the hole and whether anything else in
   the project covers it. Do NOT assume a sibling guard covers a hole unless you
   can see it in this file.

5. **Bypass.** How would an agent, or an ordinary automation, defeat this guard
   WITHOUT intending to? (Not an attacker -- the realistic threat here is a
   careless `git add -A`, a test that calls a script's `main()`, a re-run that
   overwrites a fixed output path, a `--no-verify`, a rewritten file that is
   restored to a *superficially* similar state.)

6. **UNTESTED PATTERN.** For each individual rule or branch: if it were deleted,
   would any test fail, or is it double-covered by a broader neighbouring rule
   so the named rule is decorative? Name any rule you believe is deletable with
   the suite still green.

7. **HARDCODED ABSOLUTE WRITE TARGET.** Does the guard bake an absolute path
   (`/home/<someone>/...`) into the source rather than deriving it from
   `Path(__file__)`, an env var, or an argument? Two distinct consequences, and
   the second is the one people miss: it acts on the wrong tree, AND if it
   WRITES, a run from a worktree or a fresh clone silently writes into the
   original operator checkout instead of failing. Treat that as a
   reproducibility defect as well as a correctness bug. Note that this project's
   `canonical_url_lint.py` deliberately PERMITS that same literal string,
   because its rule concerns the GitHub URL and not the filesystem -- so do not
   assume another guard covers it. Nothing was watching it as a write target.

8. **Failure mode on error.** If the guard's own machinery breaks (a file it
   reads is missing, a subprocess returns non-zero, JSON fails to parse), does
   it fail CLOSED (refuse, alarm) or fail OPEN (return "OK")? A guard that
   returns clean when it could not actually perform its check is the silent
   failure in its purest form. Quote the specific except/fallback.

8b. **DEFAULT BRANCH DISABLES THE CHECK.** Distinct from 8, and easier to miss,
   because nothing here errors. Does the guard dispatch through a chain of
   recognizers -- `if looks_like_X: return threshold_X` -- ending in a default
   that means NO CHECK rather than "unrecognized, say so"? The tell is a
   terminal `return None` / bare `return` whose caller treats it as permission
   to skip. An input the chain was never taught then reads as approved rather
   than unverified. Measured precedent, 2026-08-21:
   `adversarial_verify.duration_floor_for_artifact` returned None when no
   substrate recognizer matched, so 720 artifacts declaring a substrate got no
   duration floor and 237 of those ran under a second. That shape had already
   been patched for ONE substrate on 2026-07-30; the general case survived.
   Name the default branch and the input class it silently exempts.

8c. **METRIC COMPUTED BEFORE THE WORK IT MEASURES.** Does the guard read, or
   write, a duration/count/size that was evaluated BEFORE the operation it
   describes? The commonest form is a value computed inline as a CALL ARGUMENT,
   since arguments are evaluated before the callee runs -- producing a constant
   with a measurement's name, and rendering inert every downstream check that
   trusts it. Measured precedent, 2026-08-21: four experiment scripts passed
   `duration_s=max(time.monotonic() - start, 0.0001)` INTO the builder that did
   the work, so the value was always the 0.0001 floor. If the guard consumes a
   field like this, ask whether the field can actually differ between a real run
   and a run that did nothing.

9. **TEST/SIDE-EFFECT MUTATION OF TRACKED STATE.** Does this guard -- or a test
   that would naturally be written for it, or a fixture it needs -- WRITE to
   `results/**`, `openspec/**`, `output/**`, `ops/**`, or an operator-curated
   doc (README.md, docs/index.html, docs/roadmap.md, docs/getting-started.md,
   docs/cli-usage.md, docs/mcp-server.md, docs/blog/*, docs/CNAME) merely as a
   side effect of running? Writing to a FIXED path that a committed historical
   artifact already occupies is the specific hazard: it overwrites the research
   record on a green run, with no failure and no diff anyone reads. A guard is
   not exempt from this because it is a guard -- state-writing test fixtures are
   how the 2026-07-29 incident happened, and a guard that corrupts the record it
   protects while verifying it is the worst version of the problem.

Output format -- exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: CLEAN | MINOR_RISK | REAL_BUG | SILENT_NON_FIRING | CANNOT_DETERMINE>

## CONCEPT
<the class of bad thing this guard is supposed to prevent, 1 sentence>

## FINDINGS
<numbered list of concrete issues per the 8 questions above, or "none found">

## COUNTEREXAMPLE
<a concrete input that gets through, or "none constructed">

{TAIL}""".replace("{TAIL}", SHARED_OUTPUT_TAIL)


@dataclass
class Chunk:
    label: str  # e.g. "adversarial_verify.py::_flips_gate"
    body: str
    source_file: Path


def extract_risky_functions(path: Path) -> list[Chunk]:
    """Extract top-level function defs whose body does field extraction or
    string pattern-matching -- the exact bug class this audit hunts. Skips
    pure scaffolding (argparse, report formatting, dataclasses) with no such
    risk."""
    try:
        source = path.read_text()
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if getattr(node, "col_offset", 0) != 0:
            continue  # only top-level functions, not nested/methods
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        body = "\n".join(lines[start:end])
        if not any(marker in body for marker in RISKY_BODY_MARKERS):
            continue
        if len(body) < 40:
            continue
        chunks.append(Chunk(label=f"{path.name}::{node.name}", body=body, source_file=path))
    return chunks


def call_gemini(prompt: str, body: str, model: str = "gemini-3.1-pro-preview") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            ["gemini", "--model", model, "--yolo", "-p", full],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"gemini exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_claude(prompt: str, body: str, model: str = "claude-opus-4-8") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            ["claude", "--model", model, "--effort", "max", "--print", full],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"claude exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_codex(prompt: str, body: str, model: str = "gpt-5.5") -> tuple[bool, str]:
    """Codex (gpt-5.5) hostile reviewer -- quota-conserve path, mirrors verifier_authenticity_audit.py."""
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
                "--model",
                model,
                "--cd",
                str(PROJECT_ROOT),
                "--ephemeral",
                "-",
            ],
            input=full,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"codex exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_model(model_kind: str, model_name: str | None, prompt: str, body: str) -> tuple[bool, str]:
    if model_kind == "gemini":
        return call_gemini(prompt, body, model=model_name or "gemini-3.1-pro-preview")
    if model_kind == "codex":
        return call_codex(prompt, body, model=model_name or "gpt-5.5")
    return call_claude(prompt, body, model=model_name or "sonnet")


def parse_verdict(report: str) -> str:
    m = re.search(r"##\s*VERDICT\s*\n\s*(\S+)", report)
    return m.group(1).strip() if m else "UNKNOWN"


# Sections whose content is, BY PROMPT DESIGN, an input that does NOT appear in the
# audited source. The reviewer was explicitly asked to construct these. Fact-checking them
# against the source would void precisely the findings the 2026-07-29 extension exists to
# surface -- the whole point of a SILENT_NON_FIRING finding is to name a real input the
# code does not mention anywhere. Without this exemption the extension would have been
# decorative: its own integrity guard would have discarded every finding it added.
CONSTRUCTED_INPUT_SECTIONS = ("COUNTEREXAMPLE", "MISSED INPUT")


def strip_constructed_sections(report: str) -> str:
    """Return only the sections that make CLAIMS ABOUT THE AUDITED SOURCE.

    The integrity guard's question is "did the reviewer invent a line of code and attribute
    it to this file". That question is meaningless for a section that was commissioned to
    contain something the file does NOT have.
    """
    parts = re.split(r"(?m)^\s*##\s*(.+?)\s*$", report)
    if len(parts) == 1:
        return report  # unstructured output: nothing to scope, check the whole thing
    kept = [parts[0]]
    for i in range(1, len(parts) - 1, 2):
        heading, content = parts[i].strip().upper(), parts[i + 1]
        if any(heading.startswith(s) for s in CONSTRUCTED_INPUT_SECTIONS):
            continue
        kept.append(content)
    return "\n".join(kept)


def verify_quoted_evidence(
    report: str, body: str, repo_root: Path | None = None
) -> tuple[list[str], list[str]]:
    """Audit-integrity guard (Layer 1.5) -- descended from
    verifier_authenticity_audit.py's guard of the same name. An LLM hostile
    reviewer can hallucinate its smoking gun (a quoted line, a fabricated
    counterexample string, a path that doesn't exist). We can't fact-check
    prose, but we CAN fact-check high-specificity quoted spans (backtick
    code spans that look like a file path, a line of code, or contain a
    distinctive identifier) against the actual source. Missing evidence ->
    the flagged verdict is auto-downgraded rather than acted on.

    TWO CORRECTIONS ADDED 2026-07-29, both of which the extension needs to not eat itself:

    1. Constructed-input sections are exempt (see CONSTRUCTED_INPUT_SECTIONS). The new
       SILENT_NON_FIRING class asks the reviewer to name an input the code does NOT
       contain. Checking that answer against the code and voiding it when absent would
       reject every correct answer -- a guard that fires only on true findings, which is
       the exact inversion of what it is for.

    2. A quoted path to a real file in the repo is EVIDENCE, not hallucination. The
       original check asked only "does this string appear in the audited chunk", so a
       reviewer citing `results/experiment_3946_r11l_first_solve.json` -- a genuine
       artifact, and the right thing to cite when explaining what a guard failed to
       protect -- was treated as fabricated because that filename naturally does not
       appear inside a linter's source. Existence in the repo now counts. This only ever
       UN-voids a span, so it cannot let a hallucination through: a path that exists is,
       by construction, not made up.
    """
    norm_body = re.sub(r"\s+", "", body)
    scoped = strip_constructed_sections(report)

    def exists_in_repo(core: str) -> bool:
        """A quoted repo-relative path to a real FILE counts as evidence.

        Narrowed 2026-07-29 from `.exists()` to `.is_file()` plus a known suffix. The
        loose version un-voided any quoted span containing a slash that resolved to
        anything on disk, so the bare string `scripts/` -- eight characters, contains a
        slash, obviously a directory -- was accepted as high-specificity evidence. That
        is not a citation; it is a gesture at a folder. Requiring a real file with a
        source/artifact extension keeps the property this exemption exists for (a
        reviewer citing a genuine artifact by path is not hallucinating) while removing
        the breadth that was never intended.

        This relaxation only ever UN-voids a span, so every widening of it weakens the
        integrity guard. That asymmetry is why it is kept as tight as it can be while
        still admitting real citations.
        """
        if repo_root is None or "/" not in core:
            return False
        candidate = core.strip().strip("`'\"()[],.")
        if not candidate or candidate.startswith("/") or ".." in candidate:
            return False
        if not candidate.endswith(CITABLE_PATH_SUFFIXES):
            return False
        try:
            return (repo_root / candidate).is_file()
        except OSError:
            return False

    def is_present(core: str) -> bool:
        if re.sub(r"\s+", "", core) in norm_body:
            return True
        if exists_in_repo(core):
            return True
        toks = re.findall(r"[A-Za-z0-9_./-]{6,}", core)
        return any(re.sub(r"\s+", "", t) in norm_body for t in toks)

    high: list[str] = []
    missing: list[str] = []
    for span in re.findall(r"`([^`]+)`", scoped):
        core = span.strip()
        if len(core) < 6:
            continue
        is_high_specificity = bool(
            re.search(r"\.(json|py|ya?ml|txt|md)\b", core)
            or "/" in core
            or re.search(
                r"\b(d\.get|re\.search|re\.match|\.lower\(\)|\.startswith|\.endswith|"
                r"np\.random|numpy\.random|time\.sleep|torch\.load)\b",
                core,
            )
        )
        if not is_high_specificity:
            continue
        high.append(core)
        if not is_present(core):
            missing.append(core)
    return high, missing


"""Verdicts that put a unit on the operator's action list.

`SILENT_NON_FIRING` is the 2026-07-29 addition and is deliberately its own verdict rather
than folded into REAL_BUG. The two call for different operator responses: REAL_BUG means
"this logic is wrong, fix the logic", while SILENT_NON_FIRING means "this logic is right as
far as it goes and does not go far enough" -- a widening, usually with a regression test
named for the input that got through. Collapsing them would lose that distinction in the
one report where it matters most.
"""
FLAGGED_VERDICTS = frozenset({"REAL_BUG", "NEEDS_REDESIGN", "SILENT_NON_FIRING"})

VERDICT_ORDER = (
    "CLEAN",
    "MINOR_RISK",
    "REAL_BUG",
    "SILENT_NON_FIRING",
    "CANNOT_DETERMINE",
    "NEEDS_REDESIGN",
    "UNKNOWN",
)


def parse_section(report: str, heading: str) -> str:
    """Pull one `## HEADING` section's body out of a structured audit reply.

    Used for `## MISSED INPUT`, which is the single most actionable line the audit can
    produce -- it is the literal input a guard lets through -- and therefore gets hoisted
    into the report summary instead of being left buried in a per-unit section that nobody
    scrolls to.
    """
    m = re.search(
        rf"(?ms)^\s*##\s*{re.escape(heading)}\s*$\n(.*?)(?=^\s*##\s|\Z)",
        report,
    )
    if not m:
        return ""
    text = m.group(1).strip().strip("`").strip()
    if text.lower().startswith(("none found", "none constructed", "n/a", "none.")):
        return ""
    return text


def _run_one(
    label: str,
    body: str,
    prompt: str,
    args: argparse.Namespace,
    out: list[str],
    counts: dict[str, int],
    flagged: list[tuple[str, str]],
    integrity_voids: list[tuple[str, str, list[str]]],
    missed_inputs: list[tuple[str, str]] | None = None,
) -> None:
    flagged_verdicts = FLAGGED_VERDICTS
    ok, report = call_model(args.model, args.model_name, prompt, body)
    if not ok:
        out.append(f"## {label}\n\n(audit call failed: {report[:200]})\n")
        counts["UNKNOWN"] = counts.get("UNKNOWN", 0) + 1
        return
    verdict = parse_verdict(report)
    guard_note = ""
    if verdict in flagged_verdicts:
        _high, missing = verify_quoted_evidence(report, body, repo_root=PROJECT_ROOT)
        if missing:
            integrity_voids.append((label, verdict, missing[:6]))
            guard_note = (
                "\n> **AUDIT-INTEGRITY GUARD (Layer 1.5) — VERDICT AUTO-DOWNGRADED.** "
                f"The `{verdict}` verdict cited high-specificity evidence (code spans / "
                "file paths / distinctive identifiers) that does NOT appear in the source "
                "chunk (checked literally + by distinctive sub-token). This is the auditor "
                "hallucinating its smoking gun. Verdict downgraded to `CANNOT_DETERMINE` "
                "and removed from the action list; DO NOT act on this basis. Absent "
                "evidence: " + "; ".join(f"`{m}`" for m in missing[:6]) + "\n"
            )
            print(
                f"    [integrity-guard] VOIDED {label}: {verdict} cited "
                f"{len(missing)} absent evidence string(s)",
                file=sys.stderr,
            )
            verdict = "CANNOT_DETERMINE"
    counts[verdict] = counts.get(verdict, 0) + 1
    if verdict in flagged_verdicts:
        flagged.append((label, verdict))
        # Hoist the named missed input, but ONLY off a flagged verdict that survived the
        # integrity guard. A voided verdict's constructed inputs are exactly as unreliable
        # as the evidence that voided it, and this list is the part an operator acts on.
        if missed_inputs is not None:
            missed = parse_section(report, "MISSED INPUT")
            if missed:
                missed_inputs.append((label, missed))
    out.append(f"## {label}\n")
    out.append(f"**Verdict:** `{verdict}`\n")
    out.append(report.strip())
    if guard_note:
        out.append(guard_note)
    out.append("\n")


def build_units(path: Path) -> list[tuple[str, str, str]]:
    """Turn one target file into (label, body, prompt) audit units.

    Routing is by ROLE first, then size:
      * a GUARD_TARGETS file gets PER_GUARD_PROMPT -- the silent-non-firing interrogation;
      * anything over CHUNK_THRESHOLD_LINES is split into risky functions, because a
        whole-file review of six thousand lines is a skim;
      * everything else is a whole-file review.

    A missing file yields no units instead of raising. Guards get renamed and migrated by
    other workflows, and an audit that dies on a concurrent rename is an audit that stops
    being run at exactly the moment the tree is changing fastest.
    """
    resolved = path.resolve()
    guard_paths = {p.resolve() for p, _ in GUARD_TARGETS}
    try:
        source = resolved.read_text()
    except OSError:
        return []
    if resolved in guard_paths:
        return [(resolved.name, source, PER_GUARD_PROMPT)]
    if resolved in {p.resolve() for p in CHUNKED_FILE_TARGETS} or (
        source.count("\n") + 1 > CHUNK_THRESHOLD_LINES
    ):
        return [(c.label, c.body, PER_CHUNK_PROMPT) for c in extract_risky_functions(resolved)]
    return [(resolved.name, source, PER_FILE_PROMPT)]


def all_target_paths() -> list[Path]:
    """Every file this audit considers in scope, in a stable order.

    Guards come FIRST on purpose: they are the newly-covered surface and the ones with a
    live incident behind them, so a bounded run should reach them before anything else.

    Ordering alone does NOT deliver that, which is worth stating because this docstring
    used to claim it did. The rotation offset is persisted across runs, so putting the
    guards at the head of the list while the stored offset pointed at unit 45 of 168 left
    them unvisited for roughly seven bounded milestone-closes. `units_signature` is the
    other half: when the unit list changes, the offset is discarded and rotation restarts
    here, at the guards. Ordering expresses the priority; the signature enforces it.
    """
    return [p for p, _ in GUARD_TARGETS] + list(WHOLE_FILE_TARGETS) + list(CHUNKED_FILE_TARGETS)


def _now() -> float:
    """Monotonic clock behind the wall-clock budget, wrapped so tests can drive it."""
    return time.monotonic()


def units_signature(units: list[tuple[str, str, str]]) -> str:
    """Stable fingerprint of the unit LIST -- its length and the labels, not the bodies.

    Deliberately insensitive to edits inside a unit. Rewriting a function body should not
    throw away rotation progress; that would mean an actively-developed file perpetually
    re-audits its own head slice and never reaches the tail. What must invalidate the
    offset is a change to what the INDICES MEAN: a target added or removed, a function
    renamed, a file crossing the chunking threshold.
    """
    labels = "\n".join(label for label, _, _ in units)
    return f"{len(units)}:{hashlib.sha256(labels.encode()).hexdigest()[:16]}"


def resolve_rotation_offset(prior: object, signature: str) -> int:
    """Where this run should start, given the persisted state and the current signature.

    Extracted from main() so it can actually be tested. The rotation slice math already in
    the suite is a REIMPLEMENTATION of what main() does inline -- it passes whether or not
    main() is correct, which is the decorative-coverage shape this whole audit hunts. This
    decision is the part that matters (it is what left the guards dormant at offset 45), so
    it is a real function with real callers on both sides.

    A missing, malformed, or mismatched signature all resolve to 0. That is deliberately
    fail-safe in the direction of MORE auditing: the cost of restarting rotation is
    re-reviewing some units, and the cost of trusting a stale offset is a whole newly-added
    guard surface never being looked at.
    """
    if not isinstance(prior, dict):
        return 0
    if str(prior.get("units_signature", "")) != signature:
        return 0
    try:
        return max(0, int(prior.get("offset", 0)))
    except (TypeError, ValueError):
        return 0


def _precommit_script_hooks(config_text: str) -> dict[str, str]:
    """Map `scripts/<name>.py` -> the hook id that runs it, from .pre-commit-config.yaml.

    Parsed with a regex rather than a YAML load on purpose: this runs inside the audit and
    inside a unit test, and taking a hard dependency on PyYAML for a scope self-check would
    trade a real capability for a cosmetic one. The shape being matched is fixed by
    pre-commit's own schema (`entry:` is a command line), so a regex is sufficient and
    cannot silently mis-parse into a WRONG answer -- it either finds the script or does not.
    """
    hooks: dict[str, str] = {}
    current_id = ""
    for line in config_text.splitlines():
        m_id = re.match(r"\s*-?\s*id:\s*(\S+)", line)
        if m_id:
            current_id = m_id.group(1)
            continue
        if re.match(r"\s*entry:", line):
            for script in re.findall(r"scripts/([A-Za-z0-9_]+\.py)", line):
                hooks.setdefault(script, current_id)
    return hooks


# Guards do not all live in pre-commit. The runtime doc guard landed 2026-07-29 as a pytest
# audit hook under python/carnot/testing/, wired through conftest.py -- a location a
# pre-commit-only scan would never reach. Rather than treat that as a one-off, this directory
# is swept too: it is the project's designated home for test-time guards, so anything
# guard-shaped appearing in it should have to be classified like everything else.
RUNTIME_GUARD_DIR = PROJECT_ROOT / "python" / "carnot" / "testing"
RUNTIME_GUARD_NAME_MARKERS = ("guard", "lint", "audit", "check")


def _runtime_guard_modules(guard_dir: Path | None = None) -> dict[str, str]:
    """Guard-shaped modules under the runtime-guard directory -> a short origin label."""
    d = guard_dir if guard_dir is not None else RUNTIME_GUARD_DIR
    found: dict[str, str] = {}
    try:
        entries = sorted(d.glob("*.py"))
    except OSError:
        return found
    for p in entries:
        if p.name.startswith("_"):
            continue
        if any(marker in p.stem.lower() for marker in RUNTIME_GUARD_NAME_MARKERS):
            found[p.name] = f"runtime guard module ({d.name}/)"
    return found


def discover_unaudited_guards(
    config_text: str | None = None, guard_dir: Path | None = None
) -> list[tuple[str, str]]:
    """Wired guards that are NEITHER audited NOR acknowledged as out of scope.

    This is the audit applying its own class-B finding to itself. The target list above is a
    pattern list, and the whole lesson of 2026-07-29 is that a pattern list drifts narrower
    than its concept without anybody being told. A guard wired into pre-commit (or into
    conftest) is trusted from that moment; nothing else in the project would ever have said
    "and it has never been reviewed".

    Two sources, because one source is how the concept gets narrowed: pre-commit hooks that
    run a `scripts/*.py`, and guard-shaped modules in the runtime-guard directory.

    Returns (name, origin) pairs. Empty means the scope list is complete.
    """
    if config_text is None:
        try:
            config_text = PRECOMMIT_CONFIG.read_text()
        except OSError:
            config_text = ""
    covered = {p.name for p in all_target_paths()}
    covered.add(Path(__file__).name)  # this audit is reviewed by its operator, not by itself
    candidates: dict[str, str] = {}
    for script, hook_id in _precommit_script_hooks(config_text).items():
        candidates[script] = f"pre-commit hook: {hook_id}"
    for name, origin in _runtime_guard_modules(guard_dir).items():
        candidates.setdefault(name, origin)
    out: list[tuple[str, str]] = []
    for name, origin in sorted(candidates.items()):
        if name in covered or name in ACKNOWLEDGED_NON_QA_LAYER:
            continue
        out.append((name, origin))
    return out


def check_targets() -> int:
    """`--check-targets`: refuse (exit 1) when a wired guard is unclassified."""
    unaudited = discover_unaudited_guards()
    if not unaudited:
        print("qa-layer-audit --check-targets: OK — every wired guard is either in the")
        print("  audit's target list or explicitly acknowledged as out of scope.")
        return 0
    print("qa-layer-audit --check-targets: UNCLASSIFIED GUARD(S).")
    print(
        "  These run on every commit (or every test run), so the project trusts them, but\n"
        "  nothing reviews them and nobody has written down that they are out of scope.\n"
        "  That is the exact shape of the 2026-07-29 incident: trusted and unexamined.\n"
    )
    for name, origin in unaudited:
        print(f"  - {name}  ({origin})")
    print(
        "\n  Fix by EITHER adding the file to GUARD_TARGETS in this script (with a one-line\n"
        "  reason it is in scope), OR adding it to ACKNOWLEDGED_NON_QA_LAYER with the reason\n"
        "  its failure does not admit or destroy a research determination. Do not leave it\n"
        "  unlisted -- unlisted is how the last one got missed."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude",  # matches verifier_authenticity_audit.py's default; gemini is never
        # the default per the 2026-06-10 global-stall directive.
        choices=["gemini", "claude", "codex"],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--file",
        default=None,
        help="Audit a single file. Routing (guard prompt / chunked / whole-file) is decided "
        "by build_units() exactly as it is for a full run, so a --file run reviews the "
        "target the same way the scheduled run would.",
    )
    parser.add_argument(
        "--guard-prompt",
        action="store_true",
        help="Force the guard (silent-non-firing) prompt for --file. Use when auditing a "
        "guard that is not in GUARD_TARGETS -- e.g. an older revision recovered from git "
        "history, or a guard another workflow has not landed yet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N chunks/files total (for time-bounded sampling)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=0,
        help="Wall-clock budget (REQ-CONDUCTOR-RECEIPT-1). When it runs out the audit "
        "stops reviewing further units, writes a PARTIAL report with whatever "
        "completed, and advances rotation by that count only. A deadline the "
        "program knows about produces a partial report; a caller-imposed kill "
        "timeout produces nothing — which is how this audit went silent for 23 "
        "days after the 2026-07-29 scope extension outgrew the caller's 900s.",
    )
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Do not audit; report wired guards that are neither audited nor acknowledged "
        "out of scope, and exit 1 if any exist.",
    )
    args = parser.parse_args()

    if args.check_targets:
        return check_targets()

    units: list[tuple[str, str, str]] = []  # (label, body, prompt)

    if args.file:
        # Resolve BEFORE routing. The previous code compared a possibly-relative
        # `Path(args.file)` against the absolute CHUNKED_FILE_TARGETS, so it never matched:
        # `--file scripts/adversarial_verify.py` silently fell through to a whole-file
        # review of 6,200 lines. Silent mis-routing in the auditor is the same failure the
        # auditor exists to find.
        p = Path(args.file).resolve()
        if args.guard_prompt:
            try:
                units.append((p.name, p.read_text(), PER_GUARD_PROMPT))
            except OSError:
                pass
        else:
            units.extend(build_units(p))
        rotated = False
    else:
        for p in all_target_paths():
            units.extend(build_units(p))
        rotated = True

    # Rotation state: adversarial_verify.py alone has 150+ risky-function chunks --
    # far more than a single milestone-close pass (--limit ~20-25) can cover. Without
    # rotation, a fixed head-slice would always re-audit the same first N units and
    # NEVER reach the rest (the exact limitation verifier_authenticity_audit.py has,
    # unaddressed, for its own --limit 20). Persist an offset so successive full-corpus
    # runs advance through the whole list over time; --file runs (single-target,
    # rotated=False) don't touch rotation state.
    pending_rotation: tuple[Path, int, str, int] | None = None
    if args.limit > 0 and rotated and units:
        state_path = PROJECT_ROOT / "ops" / ".qa_layer_audit_rotation.json"
        signature = units_signature(units)
        # A persisted offset is only meaningful against the unit list it was measured
        # on. When the list changes -- a target added, a function renamed, a file grown
        # past the chunking threshold -- index N no longer denotes what it did, and
        # worse, newly-added units sit at the HEAD of the list while the offset points
        # deep into the tail. Adding the guards at offset 45 of 168 with --limit 20
        # would have left the entire new guard surface dormant for ~7 milestone-closes,
        # and that surface was added because of a live incident. Detect the change and
        # start over from the top. See resolve_rotation_offset.
        try:
            prior_state = json.loads(state_path.read_text())
        except Exception:
            prior_state = None
        offset = resolve_rotation_offset(prior_state, signature)
        offset = offset % len(units)
        total_units = len(units)
        rotated_units = units[offset:] + units[:offset]
        units = rotated_units[: args.limit]
        # Receipt-before-state (REQ-CONDUCTOR-RECEIPT-1, SCENARIO-CONDUCTOR-
        # RECEIPT-4): the offset used to be written HERE, before any unit ran.
        # From 2026-07-29 the caller's 900s timeout killed every run before the
        # report, so the coverage ledger advanced 20 units per close while zero
        # coverage happened -- the phantom-OK defect class, inside the tool
        # built to catch it. The state write now happens AFTER the report
        # lands, advanced by the number of units actually reviewed.
        pending_rotation = (state_path, offset, signature, total_units)
    elif args.limit > 0:
        units = units[: args.limit]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    # `out` holds ONLY the per-unit bodies. The previous version appended header and bodies
    # to one list and then spliced the summary in at a hardcoded `out[:5]`, which silently
    # became the wrong offset the moment a header line was added -- the summary landed in
    # the middle of the header. Keeping the three parts separate removes the offset entirely.
    out: list[str] = []

    counts: dict[str, int] = {v: 0 for v in VERDICT_ORDER}
    flagged: list[tuple[str, str]] = []
    integrity_voids: list[tuple[str, str, list[str]]] = []
    missed_inputs: list[tuple[str, str]] = []

    deadline = _now() + args.budget_seconds if args.budget_seconds > 0 else None
    completed = 0
    truncated = False
    for i, (label, body, prompt) in enumerate(units, 1):
        if deadline is not None and _now() >= deadline:
            truncated = True
            print(
                f"[budget] wall-clock budget exhausted after {completed}/{len(units)} unit(s)",
                file=sys.stderr,
            )
            break
        print(f"[{i}/{len(units)}] {label}", file=sys.stderr)
        _run_one(label, body, prompt, args, out, counts, flagged, integrity_voids, missed_inputs)
        completed += 1

    # Header is built AFTER the loop so it can state what actually happened
    # (completed count, PARTIAL marker) rather than what was planned.
    header = [
        f"<!-- generated by scripts/qa_layer_authenticity_audit.py — {today} -->",
        "<!-- per CLAUDE.md 'QA-Layer Authenticity Discipline' — advisory only -->",
        "",
        f"# qa_layer_authenticity_audit_report — {today}",
        "",
        f"Scanned {completed} of {len(units)} selected unit(s) with {args.model} as the "
        f"hostile reviewer. "
        f"Guards ({len(GUARD_TARGETS)}): "
        f"{', '.join(p.name for p, _ in GUARD_TARGETS)}. "
        f"Whole-file: {', '.join(p.name for p in WHOLE_FILE_TARGETS)}. "
        f"Function-chunked: {', '.join(p.name for p in CHUNKED_FILE_TARGETS)}.",
        "",
    ]
    if truncated:
        header += [
            f"**PARTIAL RUN** — wall-clock budget {args.budget_seconds:.0f}s exhausted "
            f"after {completed} of {len(units)} unit(s); rotation advances by "
            f"{completed} only (SCENARIO-CONDUCTOR-RECEIPT-3).",
            "",
        ]

    summary = [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for v, n in counts.items():
        summary.append(f"| `{v}` | {n} |")
    if missed_inputs:
        # Hoisted above FLAGGED deliberately. This is the shortest path from an audit run to
        # a repair: each line is a literal input a guard lets through, which is both the fix
        # and the regression test, already written.
        summary.append("")
        summary.append("### MISSED INPUTS — a real input each guard does NOT catch")
        summary.append(
            "The 2026-07-29 class. Each line names an input that falls inside the guard's own "
            "stated concept and gets through anyway. Treat each as a widening plus a "
            "regression test NAMED for the input — a widening without the named test is how "
            "the last one came back."
        )
        for label, missed in missed_inputs:
            one_line = " ".join(missed.split())
            summary.append(f"- `{label}` — {one_line[:400]}")
    if flagged:
        summary.append("")
        summary.append("### FLAGGED — operator action recommended")
        for label, verdict in flagged:
            summary.append(f"- `{label}` — **{verdict}**")
    unaudited = discover_unaudited_guards()
    if unaudited:
        summary.append("")
        summary.append("### SCOPE GAP — wired guards this audit does not cover")
        summary.append(
            "These run on every commit or every test run, so they are trusted, but they are "
            "neither in this audit's target list nor recorded as out of scope. Classify each "
            "one (see `--check-targets`)."
        )
        for name, origin in unaudited:
            summary.append(f"- `{name}` — {origin}")
    if integrity_voids:
        summary.append("")
        summary.append(
            "### AUDIT-INTEGRITY GUARD — flags voided (auditor hallucinated its evidence)"
        )
        summary.append(
            "These verdicts were FLAGGED by the LLM reviewer but cited concrete code/path "
            "strings that do NOT exist in the source chunk. Auto-downgraded to "
            "`CANNOT_DETERMINE`; **do NOT act on them.** They indicate the audit RUN was "
            "partly unreliable, not that the code is buggy."
        )
        for label, verdict, missing in integrity_voids:
            ev = "; ".join(f"`{m}`" for m in missing) if missing else "(none captured)"
            summary.append(f"- `{label}` — was **{verdict}**; absent evidence: {ev}")
    summary.append("")
    summary.append("---")
    summary.append("")

    REPORT_PATH.write_text("\n".join(header + summary + out))
    # Consumption state moves ONLY with the receipt (REQ-CONDUCTOR-RECEIPT-1,
    # SCENARIO-CONDUCTOR-RECEIPT-4): the rotation offset is written after the
    # report exists, advanced by the units actually reviewed. If the report
    # write raises, the offset stays put and the next run re-covers the slice
    # — re-reviewing is cheap; a slice never reviewed while counted as covered
    # is the 2026-07-29..08-21 silent-death incident.
    if pending_rotation is not None:
        state_path, offset, signature, total_units = pending_rotation
        try:
            state_path.write_text(
                json.dumps(
                    {
                        "offset": (offset + completed) % total_units,
                        "units_signature": signature,
                    }
                )
            )
        except Exception:
            pass
    print(f"audit complete — report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  scanned: {len(units)} unit(s)")
    print(f"  flagged: {len(flagged)}")
    for label, verdict in flagged:
        print(f"    {label}: {verdict}")
    for label, missed in missed_inputs:
        print(f"    MISSED INPUT [{label}]: {' '.join(missed.split())[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
