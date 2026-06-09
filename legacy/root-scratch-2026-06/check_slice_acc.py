import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
from carnot.verify.verifier_ensemble_diversity import load_fover_corpus, make_adversarial_slice
import numpy as np

records = load_fover_corpus("data/fover_corpus.jsonl")
slice1 = make_adversarial_slice(records, slice_size=200, rng=np.random.default_rng(42))
acc1 = sum(1 for r in slice1 if r.get('label') == 'correct') / len(slice1)
slice2 = make_adversarial_slice(records, slice_size=200, rng=np.random.default_rng(43))
acc2 = sum(1 for r in slice2 if r.get('label') == 'correct') / len(slice2)
slice3 = make_adversarial_slice(records, slice_size=200, rng=np.random.default_rng(44))
acc3 = sum(1 for r in slice3 if r.get('label') == 'correct') / len(slice3)
print(f"Slice 1: {acc1:.3f}")
print(f"Slice 2: {acc2:.3f}")
print(f"Slice 3: {acc3:.3f}")
