import re
import os
import subprocess

def update_files():
    files = ['docs/index.html', 'README.md', 'docs/technical-report.md']
    
    replacements = [
        ('3,234', '3,255'),
        ('25,061', '25,087'),
        ('209 Archived Milestone Records', '213 Archived Milestone Records'),
        ('209 archived records', '213 archived records'),
        ('209</div><div class="stat-label">archived records through .194', '213</div><div class="stat-label">archived records through .200'),
        ('211 milestones up to .197', '213 milestones up to .200'),
        ('211 artifact-backed completed milestone records archived through 2026.05.197', '213 artifact-backed completed milestone records archived through 2026.05.200'),
        ('through 2026.05.195', 'through 2026.05.200'),
        ('24,316 Python test items collected', '25,087 Python test items collected'),
        ('Milestone 2026.05.194 Operational Retrospective', 'Milestone 2026.05.198 Operational Retrospective'),
        ('Milestone 2026.05.194 operational retrospective complete. Analyzed 19.8 min wall time / 12 experiments. All 12 tasks were synthesis-only, so GPUs correctly idled at 0% utilization throughout. Synthesis tasks (Exp 1909, 1910, 1911) remain the primary bottleneck for optimization.', 'Milestone 2026.05.198 operational retrospective complete. Analyzed 18.6 min wall time / 10 experiments. Slowest path: Exp 1985 (8 min, synthesis-only). GPU utilization on the 2 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis tasks and retrospectives remain the primary bottleneck for optimization.'),
        ('19.8 min wall time', '18.6 min wall time'),
        ('completed 12 experiments in 19.8 minutes', 'completed 10 experiments in 18.6 minutes'),
        ('2,528 task records', '2,548 task records'),
    ]

    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for old, new in replacements:
            content = content.replace(old, new)
            
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
            
    # Also update the card in docs/index.html and the section in README.md
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('**207** milestone records (latest 2026.05.192)', '**213** milestone records (latest 2026.05.200)')
    content = content.replace('**209** milestone records (latest 2026.05.195)', '**213** milestone records (latest 2026.05.200)')
    content = content.replace('| .194 |', '| .198 |').replace('12 experiments, 19.8 min wall time', '10 experiments, 18.6 min wall time')
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Add new section to docs/technical-report.md
    with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
        tr = f.read()

    new_finding = """
### 4.14 Recent Additions (Milestones .198 to .200)

**Carnot CoT2-Meta Routing Prototype**
Experiment 1983 implemented the CoT2-Meta routing framework over the Fast-Slow variant, meeting iteration reduction goals with stable pass rates and extending ODAR with an explicit fallback path.

**CLaRa-V Continuous Latent Representation Schema**
Experiment 1994 integrated the CLaRa-V continuous latent variables schema with EBM abstractions, and Experiment 1995 successfully projected these variables using PiNet Douglas-Rachford operator splitting.

**Continuous Test-Time Reasoning on Gemma-4-31B**
Experiment 1998 successfully evaluated continuous latent reasoning on 5 Sudoku problems using the gemma-4-31B-it-GGUF model.

**E2E Pipeline Verification of Phase 4 Continuous Sampling**
Experiment 2000 verified the end-to-end pipeline (E2E-008 passed), mapping CLaRa-V continuous latent variables to ContinuousEBM, evaluating energy correctly, and projecting via PiNet.
"""
    if '### 4.14 Recent Additions' not in tr:
        if '## 5. Operations and' in tr:
            tr = tr.replace('## 5. Operations and', new_finding + '\n## 5. Operations and')
        else:
            tr += '\n' + new_finding
            
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr)
        
    # Re-render HTML
    print("Running update_html.py")
    subprocess.run(['python', 'update_html.py'])

if __name__ == '__main__':
    update_files()
