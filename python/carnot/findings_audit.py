import json
import glob
import re
import os

def load_referenced_experiments(doc_paths=None):
    if doc_paths is None:
        doc_paths = ["research-program.md", "docs/technical-report.md", "paper-v6"]
    refs = set()
    for p in doc_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                content = f.read()
                matches = re.findall(r'1[5678][0-9][0-9]', content)
                refs.update(matches)
    return refs

def audit_underclaimed_findings(results_dir="results", start_id=1582, end_id=1845):
    refs = load_referenced_experiments()
    pattern = os.path.join(results_dir, "experiment_*.json")
    files = glob.glob(pattern)
    underclaimed = []
    read_count = 0
    
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'experiment_(\d+)', basename)
        if match:
            exp_id = int(match.group(1))
            if start_id <= exp_id <= end_id:
                read_count += 1
                str_id = str(exp_id)
                
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                    
                    nums = {k: v for k, v in data.items() if isinstance(v, (int, float)) and k not in ['experiment', 'experiment_id']}
                    verdict = str(data.get('honest_verdict', '')).strip()
                    
                    is_finding = False
                    summary = ""
                    
                    # We look for empirical signals or a definitive completion verdict
                    if len(nums) > 0:
                        is_finding = True
                        metric_str = ", ".join([f"{k}={v}" for k, v in nums.items()])
                        summary = f"Metrics: {metric_str}. Verdict: {verdict}"
                    elif 'complete:' in verdict.lower() or 'success' in verdict.lower():
                        is_finding = True
                        summary = f"Qualitative finding: {verdict}"
                    
                    # Some files are retro reports which are not findings
                    if "retro" in basename.lower() or "archive" in basename.lower() or "milestone" in verdict.lower():
                        is_finding = False

                    # Additional check for blocked
                    if "blocked" in verdict.lower():
                        is_finding = False
                        
                    is_referenced = str_id in refs
                    if is_finding and not is_referenced:
                        summary = summary.replace('\n', ' ').replace('\r', '')
                        if len(summary) > 200:
                            summary = summary[:197] + "..."
                        
                        underclaimed.append({
                            "experiment_id": str_id,
                            "summary": summary,
                            "artifact_path": f
                        })
                except Exception:
                    pass

    underclaimed.sort(key=lambda x: int(x["experiment_id"]))
    return underclaimed, read_count

def generate_audit_report(out_path="results/experiment_1852_findings_audit.json", results_dir="results"):
    underclaimed, read_count = audit_underclaimed_findings(results_dir)
    count = len(underclaimed)
    
    recent_count = sum(1 for f in underclaimed if int(f["experiment_id"]) >= 1650)
    passed = count >= 3 and recent_count >= 1
    
    verdict = f"complete: findings_audit_surfaced_{count}_underclaimed_results" if passed else f"complete: findings_audit_underclaimed_findings_below_threshold_only_{count}"
    
    milestones = [f"2026.05.{i}" for i in range(130, 144)]
    
    report = {
        "schema": "carnot.findings_audit.v1",
        "milestones_audited": milestones,
        "artifacts_read_count": read_count,
        "underclaimed_findings": underclaimed,
        "underclaimed_findings_count": count,
        "acceptance_gate_passed": passed,
        "honest_verdict": verdict
    }
    
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
    return report

if __name__ == "__main__":
    generate_audit_report()
