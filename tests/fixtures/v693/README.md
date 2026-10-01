# V693 private authority fixture

`active.yaml.gz` contains the exact active authority captured by Exp7992 before
validation. Its decompressed SHA-256 is
`ae3a0133d3251da33b197553765f08ecbdd4630305cf2c9f1475d154f09cf56c`.
The source is the matching immutable snapshot under
`results/raw/experiment_7992_v693_contract_methods/authority_snapshots/`.
The gzip timestamp is zero so the fixture bytes are reproducible.

REQ-REPORT-7992-V693 and SCENARIO-REPORT-7992-FROZEN-FIXTURE require private
validation to remain independent of mutable live roadmaps. The fixture's design
is generated from these saved tasks and is explicitly circular evidence.
Live authority is assessed from the caller's paths and never borrows these
fixture bytes. Consumed staging remains absent.
