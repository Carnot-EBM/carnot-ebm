"""Use the direct Rust decision service without changing the JSON-lines default."""

from pathlib import Path

from carnot.pipeline.native_calibrated_decision_service import NativeServiceClient


extension = Path("/path/to/carnot/_rust.so")
client = NativeServiceClient.from_extension(
    state_path=Path("calibrated-decision-state.json"),
    extension_path=extension,
)
decision = client.predict("request-1", 0.12)
if decision.available:
    acknowledgment = client.release_feedback("request-1", label=0)
    print(decision.action, acknowledgment.durable)
else:
    print("escalate", decision.error)
