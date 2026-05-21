import sys; sys.path.insert(0, 'python')
from carnot.verify.tier0g_semantic_energy import SemanticEnergyVerifier
sev = SemanticEnergyVerifier()
print(sev.verify("Q", "A"))
