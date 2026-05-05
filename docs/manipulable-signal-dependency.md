# Manipulable Signal Dependency Template

`manipulable_signal_dependency` is an opt-in `ConstraintTemplateLibrary`
template for reasoning chains where a load-bearing conclusion depends on one
external signal with no independent corroboration.

Use it when the failure mode is structural trust, not arithmetic truth:

- RAG answers that treat one retrieved document as ground truth.
- Web-search answers that accept one search result without cross-checking.
- Tool/API/sensor outputs that are themselves attackable or unauthenticated.
- LLM-generated intermediate text reused as if it were an independent source.

Do not use it as a truthfulness detector. It does not decide whether a source is
correct; it flags reasoning that makes a strong conclusion from a single
manipulable support node.

## Defaults

The built-in defaults live in `DEFAULT_MANIPULABILITY_PRIORS`:

| Source class | Default prior |
|---|---:|
| `first_party_telemetry` | 0.25 |
| `authenticated_api` | 0.35 |
| `single_sensor` | 0.75 |
| `third_party_api` | 0.80 |
| `unauthenticated_tool` | 0.80 |
| `user_supplied_document` | 0.85 |
| `web_search` | 0.90 |
| `rag_open_corpus` | 0.90 |
| `llm_generated` | 0.95 |

The template fires when exactly one detected source class has prior greater than
or equal to `manipulability_threshold` and the response contains at least
`load_bearing_threshold` conclusion cue, such as "therefore", "thus", or
"conclude", with no corroboration cue such as "independent", "cross-check", or
"attested".

## Tuning

Override priors when a deployment has stronger source guarantees:

```python
from carnot.pipeline import manipulable_signal_dependency_template

result = manipulable_signal_dependency_template(
    response,
    source_priors={"third_party_api": 0.2},
    manipulability_threshold=0.7,
)
```

Set `load_bearing_threshold` above `1` when short, conversational answers create
too many false positives.

Spec: `REQ-LEARN-018-4`, `REQ-LEARN-018-5`,
`SCENARIO-LEARN-018-4`, `SCENARIO-LEARN-018-5`.
