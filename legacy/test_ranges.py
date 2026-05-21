import json
import numpy as np
from sklearn.model_selection import train_test_split
from carnot.verify.tier0e_eorm import EORMVerifier
from carnot.verify.tier0f_semantic_calibration import SemanticCalibratedVerifier
from carnot.verify.tier0r_curry_howard import Tier0rVerifier
from carnot.verify.tier0s_halluguard import Tier0sVerifier
from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

with open("data/fover_corpus.jsonl", "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines if line.strip()]
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
texts = [d["step_text"] for d in eval_data]

verifiers = {
    "tier0e": EORMVerifier(),
    "tier0f": SemanticCalibratedVerifier(),
    "tier0r": Tier0rVerifier(),
    "tier0s": Tier0sVerifier(),
    "tier0u": Tier0uVerifier()
}

for name, v in verifiers.items():
    if hasattr(v, "verify"):
        s = [v.verify(t) for t in texts[:100]]
    elif hasattr(v, "score"):
        s = [v.score(t) for t in texts[:100]]
    else:
        s = [v.halluguard_ntk_score(t) for t in texts[:100]]
    print(f"{name}: min={min(s)}, max={max(s)}")
