# Confirm local serving and constrain tool transport

Status: specified, 2026-09-05.

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
