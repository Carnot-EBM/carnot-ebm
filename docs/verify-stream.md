# Streaming Verification API

Spec: REQ-VERIFY-1411, REQ-VERIFY-1412, REQ-VERIFY-1413,
SCENARIO-VERIFY-1411

`verify_stream(...)` verifies a pool of candidate answers and yields
`VerdictRecord` objects as each candidate finishes. This is useful when a caller
wants the first decisive low-energy candidate without waiting for every response
in a beam or multi-agent pool.

```python
import asyncio

from carnot.pipeline import verify_stream


async def main() -> None:
    candidates = [
        {"id": "a", "question": "What is 2 + 2?", "answer": "2 + 2 = 4."},
        {"id": "b", "question": "What is 2 + 2?", "answer": "2 + 2 = 5."},
    ]

    async for record in verify_stream(
        candidates,
        domain="arithmetic",
        max_concurrency=2,
        top_k=1,
        early_stop_margin=1.0,
    ):
        print(record.extras["candidate_id"], record.verdict, record.energy)


asyncio.run(main())
```

Candidate dictionaries require `id`, `question`, and either `answer` or
`response`. Each candidate may also provide `domain`; otherwise the stream-level
`domain` is used. If no pipeline is supplied, the API creates
`VerifyRepairPipeline(model=None)`.

Each emitted record includes stream metadata in `record.extras`:

- `candidate_id`: source candidate identifier.
- `stream_index`: zero-based completion-order index.
- `stream_rank`: provisional energy rank among completed candidates.
- `stream_end`: present on the final early-stop record when top-k margin
  stopping triggers.

The async generator cancels pending workers when the consumer closes it:

```python
stream = verify_stream(candidates, max_concurrency=4)
first = await anext(stream)
await stream.aclose()
```

The MCP server exposes the same primitive as a JSON event payload. The stdio
handler returns one object, so `events` contains ordered verdict events and
`stream_end` contains the terminal summary:

```python
from carnot.mcp import server

payload = server.verify_stream(
    candidates=[
        {"id": "a", "question": "What is 2 + 2?", "answer": "2 + 2 = 4."},
        {"id": "b", "question": "What is 2 + 2?", "answer": "2 + 2 = 5."},
    ],
    domain="arithmetic",
    top_k=1,
    early_stop_margin=1.0,
)

for event in payload["events"]:
    record = event["record"]
    print(record["extras"]["candidate_id"], record["verdict"])
```

Streaming transports can forward each `{"event": "verdict", "record": ...}`
entry incrementally and then forward the final `stream_end` summary unchanged.
