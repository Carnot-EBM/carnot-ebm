import json
import glob
import re

files = glob.glob('results/experiment_*.json')
for f in files:
    match = re.search(r'experiment_(\d+)', f)
    if match:
        num = int(match.group(1))
        if 1582 <= num <= 1845:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    # print keys or specific finding fields
                    for k, v in data.items():
                        if isinstance(k, str) and ('find' in k.lower() or 'result' in k.lower() or 'metric' in k.lower() or 'empirical' in k.lower() or 'summary' in k.lower()):
                            print(f"{f}: {k} = {v}")
            except Exception as e:
                pass
