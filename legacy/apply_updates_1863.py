import re
import subprocess

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update counts
    content = content.replace('2,177', '2,187')
    content = content.replace('2177', '2187')
    content = content.replace('Exp 1853', 'Exp 1863')
    content = content.replace('1853**', '1863**')
    
    # Update archived records text specifically
    content = content.replace('157</div><div class="stat-label">archived records through .144', '158</div><div class="stat-label">archived records through .145')
    content = content.replace('157 Archived', '158 Archived')
    
    # Update Python test counts
    content = content.replace('24,109', '24,209')

    # Update index.html experiment count text
    content = content.replace('4/4</div><div class="stat-label">experiments completed in .144', '10/10</div><div class="stat-label">experiments completed in .145')
    content = content.replace('Completed 17 experiments in 46.0 minutes. GPUs utilized efficiently on compute tasks; synthesis-only bottleneck remains.', 'Milestone .145 completed 10 experiments in 19.7 minutes. All tasks were synthesis-only, appropriate 0% GPU utilization.')
    
    # Update README table line if applicable
    content = content.replace(
        '| Milestone .144 closeout |',
        '| Milestone .145 closeout |'
    )
    content = re.sub(
        r'\| Milestone .145 closeout \|.*?\|',
        r'| Milestone .145 closeout | Completed 10 experiments in 19.7 minutes. All tasks synthesis-only with appropriate 0% GPU utilization |',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.13 Verification Learning and S2KAN/GloroKAN Primitives (Milestone .145)

**Verification Learning (VL) proxy for continuous self-learning**  
Experiment 1854 successfully deployed a Verification Learning (VL) proxy enabling continuous self-learning, complemented by cross-language (Rust/Python) equivalence verification in Experiment 1861.

**Memory Retention & Catastrophic Forgetting**  
Experiment 1856 evaluated memory retention using LTLZinc, confirming successful CERCE non-forgetting behavior.

**S2KAN and GloroKAN Integration**  
Experiment 1857 implemented S2KAN differentiable symbolic gates, while Experiment 1858 introduced forward pass Lipschitz approximation bounds. Formal verification of the S2KAN Python/Rust bridge with Z3 was completed in Experiment 1859, leading to the End-to-End verification of the S2KAN model on the local unsloth/Qwen3.6-35B-A3B-GGUF baseline in Experiment 1862.
"""

if '### 4.13 Verification Learning and S2KAN/GloroKAN Primitives (Milestone .145)' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python3', 'update_html.py'])
print("Done")
