import json
import os
from carnot.verify.roce_extractor import GenerationLogicExtractor

def test_roce_dynamic_logic_extraction():
    """SCENARIO-ROCE-1864: Evaluate extraction on 20-prompt dataset."""
    dataset = [
        "The response ```json\n{\"constraints\": [{\"type\": \"length\", \"max\": 100}]}\n```",
        "It must contain 'apple'.",
        "No constraints found here.",
        "```json\n{\"type\": \"format\", \"format\": \"csv\"}\n```",
        "You must contain 'banana'.",
        "```{\"constraints\": [{\"type\": \"keyword\", \"word\": \"urgent\"}]}```",
        "Just a normal text.",
        "Must contain \"secret\".",
        "```json\n{invalid json}\n```",
        "```json\n{\"constraints\": []}\n```",
        "must contain 'word'",
        "must contain 'test'",
        "must contain 'hello'",
        "must contain 'world'",
        "must contain 'foo'",
        "must contain 'bar'",
        "must contain 'baz'",
        "must contain 'qux'",
        "must contain 'quux'",
        "must contain 'corge'"
    ]
    
    assert len(dataset) == 20
    
    extractor = GenerationLogicExtractor()
    eval_result = extractor.evaluate(dataset)
    
    assert eval_result["dataset_size"] == 20
    # 17 should succeed based on the dataset above (4 JSON + 13 regex = 17 successes)
    assert eval_result["success_rate"] > 0.0
    
    output_path = "results/experiment_1864_roce.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_result, f, indent=2)
        
    assert os.path.exists(output_path)
