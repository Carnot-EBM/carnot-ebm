import json
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier
from carnot.verify.semantic_energy import binary_auroc

try:
    from carnot.verify.nla_verifier_v3 import IsingVerifier
except ImportError:
    from carnot.verify.semantic_energy import IsingVerifier

v1 = SemanticConsistencyVerifier()
print("v1 init success:", hasattr(v1, 'score'))

try:
    v2 = IsingVerifier()
    print("v2 init success:", hasattr(v2, 'energy'))
except NameError:
    print("IsingVerifier not found")
