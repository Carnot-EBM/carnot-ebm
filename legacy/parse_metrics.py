import json
import glob
import os

tasks = list(range(2406, 2419))
outcomes = []
completed = 0
failed = 0
gate_blocks = 0

auroc_vals = []
kl_vals = []
fr11_satisfied = False
kv260_yosys_succeeded = False
phase1_ship_gate_met = False

for task in tasks:
    pattern = f"/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_{task}*.json"
    files = glob.glob(pattern)
    if not files:
        print(f"Task {task} missing")
        continue
    file = files[0]
    with open(file, 'r') as f:
        data = json.load(f)
        status = data.get('status', 'unknown')
        if status == 'unknown':
            status = data.get('execution_status', {}).get('overall', 'unknown')
        if status == 'unknown':
             status = data.get('run_status', 'unknown')
             
        # Fallbacks for status parsing depending on typical carnot schemas
        if status == 'unknown' and 'overall' in data and 'status' in data['overall']:
            status = data['overall']['status']

        # Task 2408-2410 AUROC
        if task in [2408, 2409, 2410]:
            auroc = data.get('metrics', {}).get('auroc') or data.get('auroc')
            if auroc is None:
                # search globally
                for k, v in data.items():
                    if isinstance(v, dict) and 'auroc' in v:
                        auroc = v['auroc']
                    if isinstance(v, float) and 'auroc' in k:
                        auroc = v
            if auroc is not None:
                auroc_vals.append(auroc)
                
        # Task 2411
        if task == 2411:
            fr11_satisfied = data.get('fr11_nsvif_online_passed', False)
        
        # Task 2413
        if task == 2413:
            kv260_yosys_succeeded = data.get('synthesis_succeeded', False)
            
        # Task 2417
        if task == 2417:
            phase1_ship_gate_met = data.get('phase1_ship_gate_met', False)
            
        # Task 2414-2416 KL
        if task in [2414, 2415, 2416]:
            kl = data.get('kl_delta') or data.get('metrics', {}).get('kl_delta')
            if kl is not None:
                kl_vals.append(kl)
                
        # To determine completed vs failed based on status:
        if status in ['OK', 'complete', 'success', 'passed']:
            completed += 1
            simple_status = 'OK'
        else:
            failed += 1
            simple_status = 'FAIL'

        outcomes.append({
            "task_id": f"exp{task}",
            "status": simple_status,
            "raw_status": status
        })

print(json.dumps({
    "completed": completed,
    "failed": failed,
    "auroc_vals": auroc_vals,
    "kl_vals": kl_vals,
    "fr11_satisfied": fr11_satisfied,
    "kv260_yosys_succeeded": kv260_yosys_succeeded,
    "phase1_ship_gate_met": phase1_ship_gate_met,
    "outcomes": outcomes
}, indent=2))
