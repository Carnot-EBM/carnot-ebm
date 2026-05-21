import subprocess
import json

def run_test():
    res = subprocess.run(['.venv/bin/python', 'test_baseline.py'], capture_output=True, text=True)
    if 'delta=' in res.stdout:
        delta_str = res.stdout.split('delta=')[1].strip()
        return float(delta_str)
    return 0.0

def main():
    commits_out = subprocess.run(['git', 'log', '--oneline', '--', 'python/carnot/pipeline/'], capture_output=True, text=True)
    commits = [line.split()[0] for line in commits_out.stdout.strip().split('\n')[:30]]
    
    bisect_commits_checked = []
    regression_commit = None
    last_working_commit = None
    
    for i, commit in enumerate(commits):
        bisect_commits_checked.append(commit)
        subprocess.run(['git', 'checkout', commit, '--', 'python/carnot/pipeline/'])
        
        delta = run_test()
        print(f"Commit {commit} delta={delta}")
        
        if delta > 0.05:
            last_working_commit = commit
            if i > 0:
                regression_commit = commits[i-1]
            break
            
    subprocess.run(['git', 'checkout', 'HEAD', '--', 'python/carnot/pipeline/'])
    
    result = {
        'regression_commit': regression_commit,
        'last_working_commit': last_working_commit,
        'bisect_commits_checked': bisect_commits_checked
    }
    with open('bisect_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Regression commit: {regression_commit}")

if __name__ == "__main__":
    main()
