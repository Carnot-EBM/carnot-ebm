import json, sys
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
            correct = sum(1 for t in lines if t.get("is_correct", False))
            acc = correct / len(lines) if lines else 0
            print(f"{path}: acc={acc:.4f} n={len(lines)}")
    except Exception as e:
        print(f"{path}: error {e}")
