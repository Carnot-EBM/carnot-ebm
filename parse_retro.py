import json
import glob

expected_tasks = [f"24{i:02d}" for i in range(59, 71)]
files = glob.glob("results/experiment_2459*.json") + glob.glob("results/experiment_246*.json") + glob.glob("results/experiment_247*.json")

for mf in sorted(files):
    if "scores" in mf: continue
    print(f"--- {mf} ---")
    try:
        with open(mf, "r") as f:
            data = json.load(f)
            print("verdict:", data.get("honest_verdict", "None"))
            for k, v in data.items():
                if k != "honest_verdict":
                    print(f"  {k}: {v}")
    except Exception as e:
        print("Error:", e)
