# Confirm local serving and constrain tool transport

Status: implemented and locally verified, 2026-09-05; global repository gates retain existing failures.

The operator requested real CPU confirmation before adopting serving flags.
REQ-ARC-WMTE-7043 records the experiment and its limits. REQ-ARC-WMTE-7044 and
REQ-ARC-WMTE-7045 gate and consume JSON tool calls on the live induction path.

Acceptance: real-binary transcripts, tests written before implementation,
call-site mutation failures with byte-identical restoration, live-path checks,
whole-tree collection, spec coverage, lint, and installed commit hooks.

Grammar transport ranks first for this bounded change. KV restart works for an
idle completed slot. Full run recovery needs an agent/environment journal and
exclusive slot ownership. That larger feature is deferred, with no recovery claim.
No GPU run or 27B efficacy measurement is part of this story.

Implementation commits: d5d72141ca and 33c63162f8; CPU evidence b288f4553e.
109 focused tests pass. Forty distinct mutations have final assertion RED,
byte-identical cmp restoration and GREEN; 45 executions preserve one original
GREEN survivor and two exception-only failures before proof corrections.
The live 0.8B loop returned two empty-argument calls and no engine. The production
grammar defeated BANANA but a nested-value copy truncated. No efficacy claimed.
Whole collection: 61,816 tests, eight existing errors. Global spec traceability:
1,178 existing violations; changed-file coverage and installed hooks pass.
See docs/research-notes/local-serving-confirmation-2026-09-05.md and its receipts.
