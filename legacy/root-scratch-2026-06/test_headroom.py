import json
def build_strong_sc(records):
    results = []
    for rec in records:
        gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
        answers = [
            s.get("extracted_answer_norm") or s.get("extracted_answer")
            for s in (rec.get("samples") or [])
        ]
        scores = {}
        for i, ans in enumerate(answers):
            scores[ans] = scores.get(ans, 0.0) + 1.0 / (i + 1)
        voted = max(scores.items(), key=lambda x: x[1])[0]
        results.append((voted, voted == gold))
    return results

def compute_headroom_stats(records):
    n = len(records)
    strong_sc = build_strong_sc(records)
    oracle_correct = 0
    sc_correct = 0
    for i, rec in enumerate(records):
        gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
        answers = [
            s.get("extracted_answer_norm") or s.get("extracted_answer")
            for s in (rec.get("samples") or [])
        ]
        if answers and any(a == gold for a in answers):
            oracle_correct += 1
        if strong_sc[i][1]:
            sc_correct += 1
    
    oracle_acc = oracle_correct / n
    sc_acc = sc_correct / n
    return {
        "oracle_accuracy": oracle_acc,
        "strong_sc_accuracy": sc_acc,
        "selectable_headroom": oracle_acc - sc_acc,
        "oracle_exceeds_sc": oracle_acc > sc_acc,
        "n": n,
    }

records = []
with open("data/p01_greedy_wrong_headroom_corpus.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(compute_headroom_stats(records))
