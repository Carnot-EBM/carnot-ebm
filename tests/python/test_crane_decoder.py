import json
from pathlib import Path

from carnot.inference.crane_decoder import CRANEDecoder, run_experiment


def test_crane_decoder_req_inference_2089(tmp_path: Path):
    """
    Test that REQ-INFER-CRANE-2089 is met: CRANEDecoder applies standard logits
    until a trigger token, then strictly applies grammar constraint.
    """
    decoder = CRANEDecoder(trigger_token_id=5)

    # Before trigger token
    logits_before = decoder.decode(current_token_id=2)
    assert not decoder.is_constrained
    assert logits_before["mode"] == "unconstrained"

    # Hit trigger token
    logits_after = decoder.decode(current_token_id=5)
    assert decoder.is_constrained
    assert logits_after["mode"] == "constrained"

    # Run experiment
    output_path = tmp_path / "experiment_2089_crane_decoder.json"

    run_experiment(output_path)

    assert output_path.exists()
    with output_path.open("r") as f:
        data = json.load(f)
    assert data["crane_ready"] is True
    assert "unsloth/gemma-4-31B-it-GGUF" in data["models_used"]
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in data["models_used"]
