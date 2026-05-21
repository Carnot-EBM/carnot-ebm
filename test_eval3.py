import json
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from carnot.verify.tier0v_set_consistency import SetConsistencyVerifier

random.seed(42)

corpus = []
with open('data/fover_corpus.jsonl') as f:
    for line in f:
        corpus.append(json.loads(line))

corrects = [d for d in corpus if d.get('label') == 'correct']
incorrects = [d for d in corpus if d.get('label') == 'incorrect']

incorrect_by_qid = {}
for d in incorrects:
    qid = d['question_id']
    if qid not in incorrect_by_qid:
        incorrect_by_qid[qid] = []
    incorrect_by_qid[qid].append(d['step_text'])

all_incorrects = [d['step_text'] for d in incorrects]
all_texts = [d['step_text'] for d in corpus]

sets_X = []
sets_y = []

for c in corrects:
    qid = c['question_id']
    text = c['step_text']
    lines = text.split('\n')
    if len(lines) < 3:
        continue
    
    q = lines[0]
    mid = max(1, len(lines)//2)
    p1 = '\n'.join(lines[1:mid])
    p2 = '\n'.join(lines[mid:])
    
    if not p2.strip(): p2 = p1
    if not p1.strip(): p1 = q
    
    # Consistent set
    sets_X.append([q, p1, p2])
    sets_y.append(0)
    
    # Inconsistent set
    if qid in incorrect_by_qid:
        inc = random.choice(incorrect_by_qid[qid])
    else:
        inc = random.choice(all_incorrects)
        
    sets_X.append([q, p1, inc])
    sets_y.append(1)

indices = list(range(len(sets_X)))
random.shuffle(indices)
test_idx = indices[:int(len(indices)*0.2)]

test_X = [sets_X[i] for i in test_idx]
test_y = [sets_y[i] for i in test_idx]

# Without fitting on corpus
verifier = SetConsistencyVerifier()
energies = [verifier.compute_energy(s) for s in test_X]
auroc = roc_auc_score(test_y, energies)
print("AUROC without fit:", auroc)

verifier.fit(all_texts)
energies = [verifier.compute_energy(s) for s in test_X]
auroc = roc_auc_score(test_y, energies)
print("AUROC with fit:", auroc)
