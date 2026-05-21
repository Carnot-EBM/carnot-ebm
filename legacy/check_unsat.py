import json
from carnot.extraction.nsvif_extractor import NsvifExtractor

extractor = NsvifExtractor()
unsat = []
with open("results/arm_ebm_logprob_telemetry_manifest_1556.jsonl", "r") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        response = entry.get("response_text", "")
        result = extractor.verify(response)
        if not result.get("satisfiable", True):
            unsat.extend(result.get("violations", []))
print(f"Found {len(unsat)} unsat patterns from 1556")
