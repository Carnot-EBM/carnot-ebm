import json
import sys
import random
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_delta():
    pipeline = VerifyRepairPipeline()
    violations = []
    with open('data/fover_corpus.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex['label'] == 'incorrect':  # fover_violations implies incorrect label?
                violations.append(ex)
            if len(violations) >= 20:
                break

    results = []
    for ex in violations:
        # What is prompt vs response?
        # Maybe 'question_id' is prompt and 'step_text' is response? 
        # Or maybe the whole verify method signature only requires 'response'?
        # Let's inspect VerifyRepairPipeline.verify signature.
        pass
