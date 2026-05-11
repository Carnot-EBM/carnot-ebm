import re

files_to_update = ['README.md', 'docs/technical-report.md', 'docs/index.html']

for filepath in files_to_update:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update experiment counts
        content = content.replace('2,049', '2,072')
        content = content.replace('2049 Experiments', '2072 Experiments')
        content = content.replace('2049 Experiments', '2072 Experiments')
        
        # Update experiment records tracking
        content = content.replace('Exp 1721', 'Exp 1745')
        content = content.replace('1721**', '1745**')
        content = content.replace('through 1721', 'through 1745')

        # Update milestone counts
        content = content.replace('146', '148')
        
        # Update milestone IDs
        content = content.replace('.132', '.134')
        content = content.replace('2026.05.132', '2026.05.134')
        
        # Update Python test count
        content = content.replace('23,849', '24,113')
        content = content.replace('23849', '24113')

        # Update results table in README if needed
        # We need to make sure the latest closeout reflects .134
        if 'Milestone .132 closeout' in content:
            content = re.sub(r'\|\s*Milestone \.132 closeout.*?\|\s*Exp 1721\s*\|', 
                             r'| Milestone .134 closeout | Analyzed wall time / 10 experiments. Both RTX 3090s completely idle at 0% utilization. | Exp 1745 |', 
                             content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        print(f"Failed to update {filepath}: {e}")

print("Update complete")
