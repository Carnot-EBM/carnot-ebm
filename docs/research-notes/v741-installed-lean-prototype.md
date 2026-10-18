# Installed lean inference prototype

Exp8633 exposes the existing bounded resident worker as an installed module.
It uses the original 109-parameter spline head and64 exposed research requests.
Python and native arithmetic retain the same scaler, temperature and fallback policy.
The old `scripts/experiments/resident_worker_8608.py` command remains available.

After installing the retained private wheel with local dependencies, start a pipe session:

```sh
python -m carnot.inference.resident_protocol /absolute/head.json sha256:CONFIG_DIGEST python
```

Use `native` for the existing hash-bound extension. The configuration contains `head`
and `extension`. Its digest covers its complete bytes. The native extension requires
an explicit absolute path and SHA256. There is no network listener. Standard output
contains length-prefixed JSON frames. Configuration messages go to standard error.

The frame limit remains65536 bytes. A session permits1025 requests. Requests retain
their monotonic deadlines, unique IDs and configuration hash. Errors contain no
successful payload. The worker exits on EOF or an acknowledged shutdown request.
The package CLI also rejects nonfinite or nonpositive idle timeouts.

`carnot` resolves its original core, model, sampler and training exports lazily.
The original `__all__` order, from-import identities, version and optional JAX/Rust
flags remain covered. Accessing those APIs still loads their original modules.
The inference entry itself imports no training, reporting or JAX modules.

The runner builds an offline private wheel, installs it into private scratch and
launches it outside the checkout with PYTHONPATH unset. Existing local dependencies
remain available through a private environment path. The installed distribution,
RECORD, dependency versions, module origin and import trace are retained. Scratch
is deleted after the run. The wheel and caller-owned immutable head remain reusable.

The readiness score qualifies executable mechanics. It gives no speed, learning,
generalization or factual-confidence credit. Exp8634 owns full public surface
qualification. Exp8635 owns complete installed cost measurements. Historical nulls
and quarantines remain unchanged. Future V741 outputs are not prerequisites here.

Replay receives an external authority path and hash. Rehashing a changed candidate
cannot replace sealed original operands or validation receipts. Terminal publication
uses the unchanged adversarial, strict-row and primary-publication readers.

Run the producer with the command in the task. Its JSON links exact invocation logs,
real child coverage, private E2E-018/021 receipts and the terminal sidecar. The conductor
owns ops and traceability reconciliation for this task.
