# Jev Tetris SemIf Comparison Specification

**Capability:** jev-tetris-semif-comparison  
**Version:** 0.1.0  
**Status:** Implemented  
**Traces to:** FR-11, FR-12

## Requirements

### REQ-JEV-TETRIS-001: Local option-logit Tetris comparison

The experiment SHALL preserve the six named upstream Jev Tetris files byte for
byte. A local loopback client SHALL replace only the cloud client. The local
server SHALL score each choice question with one next-token logit readout. It
SHALL not use autoregressive generation or an API key.

The server SHALL use at most 16 single-token labels. For at most 16 placements,
it SHALL name every placement. Before truncating a set of more than 16
placements, it SHALL rank the placements using the exact fields returned by
the vendored engine's `evaluate(board, placement)` call. The ascending
tie-break order SHALL be `holesCreated`, `bumpinessDelta`, `maxHeight`, and
`aggregateHeight`, followed by `cleared` descending and the original engine
order ascending. The client SHALL send those evaluation fields with each
choice question. The server SHALL name the first 15 ranked placements and add
one delegate option. If the delegate wins, it SHALL select the first placement
in the remaining ranked tail. It SHALL assign zero probability to later ranked
tail placements.

The server SHALL skip score and noul questions. It SHALL process the five
composite choice questions sequentially. Each response SHALL name the loaded
model in provenance.

The experiment SHALL run the upstream keyword and El-Tetris controls before a
GPU run. It SHALL stop with a `blocked_*` verdict if a required precondition
fails. It SHALL record real and shuffled runs without treating cited cloud
numbers as locally reproduced results.

#### SCENARIO-JEV-TETRIS-001-A: Named placements fit the token budget

**Given** no more than 16 placement criteria and a tokenizer with usable labels  
**When** the server constructs one choice prompt  
**Then** every placement receives one distinct single-token label  
**And** the prompt contains the state, instruction, labels, and placement prose.

#### SCENARIO-JEV-TETRIS-001-B: Oversized placement sets rank before delegation

**Given** more than 16 placements in engine order with valid `evaluate()` fields  
**When** the server builds the readout options  
**Then** the best 15 placements under the documented evaluation tie-break are named  
**And** a best-ranked placement outside the first 16 engine-order entries remains named  
**And** the sixteenth label delegates to the first remaining ranked placement  
**And** later ranked-tail placements receive zero probability  
**And** probabilities map back to every original placement ID.

#### SCENARIO-JEV-TETRIS-001-C: The local response matches the Jev client shape

**Given** choice, score, and noul questions and a mocked logit scorer  
**When** one request is evaluated  
**Then** every choice answer contains `choice`, `probabilities`, and `confidence`  
**And** score and noul answers are absent  
**And** response provenance identifies the selected open-weight model.

#### SCENARIO-JEV-TETRIS-001-D: Runs remain local and auditable

**Given** an endpoint override  
**When** the Node client validates the endpoint  
**Then** only loopback hosts are accepted  
**And** no API key is read or sent  
**And** the result records model files, seeds, checksums, duration, and limits.

## Implementation status

REQ-JEV-TETRIS-001 is implemented by
`scripts/experiments/jev_tetris_semif/readout_server.py` and the local files
under `vendor/jev-tetris/`. The six focused tests in
`tests/python/test_jev_tetris_local_readout.py` cover prompt construction,
single-token labels, evaluate-ranked overflow delegation, probability mapping,
response shape, and malformed input. Experiment 10004 preserves the original
three-seed model arms and their shuffle controls. Experiment 10005 records the
overflow correction and the bounded single-seed check.
