import os
import json

sum_attempts = 0
sum_successes = 0

for root, dirs, files in os.walk('results'):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            try:
                with open(path) as fp:
                    data = json.load(fp)
                    # it could be at the top level or nested.
                    # Let's search recursively
                    def search(d):
                        global sum_attempts, sum_successes
                        if isinstance(d, dict):
                            if 'n_repair_attempts' in d:
                                sum_attempts += d['n_repair_attempts']
                            if 'n_repair_successes' in d:
                                sum_successes += d['n_repair_successes']
                            for v in d.values():
                                search(v)
                        elif isinstance(d, list):
                            for v in d:
                                search(v)
                    search(data)
            except Exception:
                pass

print(f"sum_attempts={sum_attempts}")
print(f"sum_successes={sum_successes}")
