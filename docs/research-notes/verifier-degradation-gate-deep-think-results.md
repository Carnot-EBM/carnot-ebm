# Deep Think Response — Verifier Ensemble Degradation Gate Methodology

**Status:** Response received 2026-05-01. Methodology-only response with
one self-flagged soft drift on `min_transient_confidence` (Deep Think
noted in-line that the threshold "must be derived empirically from
telemetry distribution, not hardcoded"). 5-field telemetry schema,
per-verifier classification rules with explicit transient-vs-systematic
discriminators, hybrid-structural default behavior with migration path.
Strongest cross-validation-aware response of the day.
**Date received:** 2026-05-01
**Source prompt:** `verifier-degradation-gate-deep-think-prompt.md`

---

## TL;DR — Hybrid Structural Default

| Failure class | Action | Rationale |
|---|---|---|
| **Systematic blind spot** | **fail-closed** | Verification principle: undecidable / over-context inputs are *fundamentally unverified*; pretending otherwise falsifies the core premise |
| **Transient infrastructure** | **fail-open with `Decentralization-Degraded k=4` tagged artifact** | Decentralization principle: hardware-constrained nodes must not be locked out by transient OOM / network partition |

The artifact returned in degraded mode is **fully actionable, not a
stub**, but cryptographically tagged if `k_effective < 5`. The user
gets working output and can verify the degradation tier post-hoc.

---

## Telemetry schema (verbatim — 5 fields)

| Field | Type | What it captures |
|---|---|---|
| `wall_clock_duration` | float | Total elapsed time invocation → exit / RPC return / timeout kill |
| `cpu_time_consumed` | float | Active CPU time via host OS process wrappers (distinguishes compute-saturation from idle-blocking) |
| `exit_interface_state` | variant | Unparsed return mechanism (OS exit status, HTTP status class, Python exception type, literal stdout tokens like Z3's `UNKNOWN`) |
| `stderr_diagnostic_tail` | string | Final fixed-byte tail of stderr / unhandled exception traceback (catches CUDA OOM, syntax-validation rejection strings) |
| `input_scale_proxy` | integer | Verifier-specific projection of input complexity calculated *prior* to invocation (AST node count, sequence token length, JSON nesting depth) |

These are **shippable as-is** — no further specification needed.

---

## Per-verifier classification rules (verbatim)

### Z3-AST

- **Transient:** OS-level kill signal (e.g., SIGKILL) from host memory pressure, or wall-clock timeout paired with **near-zero `cpu_time_consumed`** (host-level thread starvation)
- **Systematic:** `exit_interface_state == "UNKNOWN"` (undecidable theory) OR timeout with `cpu_time_consumed ≈ wall_clock_duration` (intractable pathological constraint search)
- **Discriminator:** Isolate explicit epistemic surrender (`"UNKNOWN"`). For timeouts/crashes, evaluate ratio `cpu_time / wall_clock` to distinguish compute-bound intractability from host-level starvation.

### gVisor

- **Transient:** `exit_interface_state` originates from host orchestrator (`runsc` fails to mount sandbox cgroups) BEFORE guest code execution
- **Systematic:** Guest process boots successfully, then unhandled guest trap (seccomp SIGSYS violation), OR timeout with maximum `cpu_time_consumed` (deterministic infinite loop in untrusted code)
- **Discriminator:** Architectural origin — host infrastructure (transient) vs. guest user-space (systematic)

### Semantic embedding

- **Transient:** Server/Network errors (HTTP 5xx, socket drops) OR `stderr_diagnostic_tail` shows local shared-GPU VRAM allocation failures
- **Systematic:** Client/Payload errors (HTTP 4xx Bad Request) OR deterministic Exceptions citing Sequence Length / Context Window exceedance
- **Discriminator:** Strict map of exception and protocol classes — infrastructure/network = transient, payload validation + context bounds = systematic

### ThinkPRM

- **Transient:** Inference server unreachable, GPU driver crashes, or fast failures where `stderr_diagnostic_tail` shows model weights failed to load from disk
- **Systematic:** Timeouts highly correlated with `input_scale_proxy` approaching known context maximums, OR deterministic JSON/regex parsing failures on probe's output (model hallucinated unparsable reasoning format)
- **Discriminator:** Execution phase — pre-inference initialization + connection (transient), context-window saturation + post-generation parsing (systematic)

### JSON schema

- **Transient:** OS-level filesystem/network exceptions (connection reset, permission denied) when resolving external schema file paths
- **Systematic:** Native JSON decoder exceptions (schema file logically/syntactically malformed) OR structural schema validation exceptions (candidate violates schema rules)
- **Discriminator:** Exception class hierarchy — system I/O faults vs. application-level schema/payload validation

---

## Operator-facing API (verbatim)

```python
def verify(
    llm_output: CandidatePayload,
    on_transient: Literal["fail_open_degraded", "fail_closed"] = "fail_open_degraded",
    on_systematic: Literal["fail_open_degraded", "fail_closed"] = "fail_closed",
    # [FLAG: Qualitative drift prevention] Specific numerical thresholds violate the
    # Carnot prediction-error pattern. This threshold must be derived empirically
    # from telemetry distribution, not hardcoded.
    min_transient_confidence: float = DEPLOYMENT_CONFIG.empirical_p_threshold,
) -> VerificationArtifact:
    """
    VerificationArtifact contains:
      - is_accepted (bool)
      - k_effective_score (int)
      - telemetry_trace (dict)
      - validated_output (CandidatePayload) — fully actionable, not a stub,
        but cryptographically tagged if k_effective < 5
    """
```

**Confidence score:** `p_transient ∈ [0.0, 1.0]`
- `1.0` = clean infrastructure fault (HTTP 503)
- `0.0` = clean epistemic limit (Z3 `UNKNOWN`)
- Intermediate values = ambiguous telemetry (timeout with moderate CPU usage)

**Per-call vs per-deployment:** kwargs override per-deployment baseline. Batch pipelines enforce strict global verification; interactive/edge clients can dynamically request graceful degradation.

---

## Default behavior recommendation (verbatim)

> **Hybrid Structural Default — Fail-closed on Systematic blind spots, and Fail-open (yielding a tagged `Decentralization-Degraded k=4` artifact) on Transient failures.**

### Why this works

- **Verification principle** ("energy is ground truth") governs systematic limits: if an output is mathematically undecidable or contextually over-bound, it is **fundamentally unverified**. Pretending otherwise by failing open *falsifies the system's core premise*.
- **Decentralization principle** ("survive infrastructure constraints") governs transient limits: a hardware-constrained node suffering a network partition or OOM spike must not be permanently locked out. Yielding a *truthfully labeled, degraded k=4 artifact* fulfills the hardware portability mandate **without lying to the operator about the verification state**.

### Migration path (verbatim)

> Deploy this hybrid default in exp1121 tonight, but implement a **shadow-monitoring sink for the `p_transient` classification distribution**. If empirical runtime telemetry reveals an adversarial attack pattern (e.g., SOTA outputs crafted to deterministically trigger a "transient" GPU OOM to bypass the semantic probe), operators must flip the deployment config for `on_transient` to `fail_closed` until the telemetry discriminators are patched to recognize the attack signature.

This is excellent: deploy the hybrid as default, instrument the distribution, retain operator override for adversarial conditions.

---

## Drift check (per `feedback_carnot_prediction_pattern.md`)

- ✅ **Telemetry schema:** types and semantic descriptions, no specific values
- ✅ **Classification rules:** signatures and discriminators, not threshold numbers
- ✅ **API structure:** per-call kwargs + per-deployment baseline (architecture, not parameters)
- ⚠️ **Self-flagged drift:** `min_transient_confidence: float = DEPLOYMENT_CONFIG.empirical_p_threshold` — Deep Think *itself* added an inline comment: *"This threshold must be derived empirically from telemetry distribution, not hardcoded."* The drift was caught and explicitly disclaimed in the response. **This is the cross-validation discipline working as intended.**

This is the cleanest drift-detection response of the day — Deep Think identified its own potential parameter prescription and disclaimed it in-line.

---

## Operational implications

### Action 1: Audit what exp1121 actually wired

The conductor finished exp1121 at 22:57Z (~5 min before this Deep Think response arrived). The agent picked SOME default verifier-failure handling. We need to determine which:

- **If exp1121 wired pure fail-closed:** refactor to hybrid (decentralization principle violated)
- **If exp1121 wired pure fail-open:** refactor to hybrid (verification principle violated)
- **If exp1121 wired hybrid (unlikely without prompt guidance):** validate
- **If exp1121 didn't address verifier-failure at all:** add the layer

### Action 2: Wire the 5-field telemetry schema

The 5 telemetry fields are shippable as-is. The Carnot core
(`python/carnot/pipeline/verify_repair.py`) needs a wrapper layer that
collects these fields on every verifier invocation. This is .87 or .88
infrastructure work.

### Action 3: Implement the per-verifier classifiers

Each of the 5 verifiers needs a small `classify_failure()` function
that takes the 5-field telemetry record and returns
`(p_transient: float, classification: "transient"|"systematic")`.
Per-verifier rules are spelled out exactly enough to implement.

### Action 4: Wire the operator-facing API

`verify(output, on_transient=..., on_systematic=..., min_transient_confidence=...)` per the sample signature, with `VerificationArtifact` containing `is_accepted`, `k_effective_score`, `telemetry_trace`, `validated_output`.

### Action 5: Shadow-monitoring sink

Build a simple distribution logger for `p_transient` values across
production calls. .88 or .89 work. Operators flip `on_transient` to
`fail_closed` if adversarial patterns appear in the distribution.

### Action 6: Cryptographic tagging of degraded outputs

When `k_effective < 5`, tag the artifact so downstream consumers can
audit the verification tier. Implementation TBD — could be a HMAC over
the artifact + telemetry trace, signed by the verifier's deployment
key.

---

## Cross-validation status — All four Q's answered

This is the **fourth Deep Think response of the day**. All four
stayed in the methodology lane:

| Question | Drift | Outcome |
|---|---|---|
| Q1 Energy inversion | None | Hypothesis A confirmed empirically (exp1120) |
| Q2 Phase-3 attacks | None | 7 attacks with abort thresholds |
| Q3 GRPO + SP-IWPER | None | Outcome C, hybrid Decoupled Dual-Stream mandatory |
| **Q4 Verifier degradation** | **Self-flagged** | Hybrid Structural Default + 5-field telemetry |

Q4 is special: Deep Think *self-detected* the only drift in its own
response and disclaimed it inline. This is the cross-validation
pattern fully working — no Round 2 follow-up needed.

---

## Recommended next steps

1. ✅ **Save this response** (done).
2. **Audit exp1121's actual default** — read `python/carnot/pipeline/verify_repair.py` to see what got wired.
3. **If exp1121 didn't wire hybrid:** ship the structural fix (small refactor, single function).
4. **In .87 retro (exp1126):** surface this as a follow-up task for .88 — the 5-field telemetry + classifier + API + shadow monitoring.
5. **For .88 planner:** include the **Telemetry + Failure Classification** task as a reserved infrastructure slot, paired with the Phase-3 prototype kickoff.
