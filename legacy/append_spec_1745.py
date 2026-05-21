import re

def append_to_table(filepath, header_text, new_rows):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the end of the markdown table that follows the header_text
    # We look for the header, then skip to the end of the table
    header_idx = content.find(header_text)
    if header_idx == -1:
        return False
    
    # Find the table start after the header
    table_start_idx = content.find('|', header_idx)
    if table_start_idx == -1:
        return False
        
    # Find the end of the table (first empty line after the table)
    match = re.search(r'(\n\|.*)+\n\n', content[table_start_idx-1:])
    if match:
        table_end_idx = table_start_idx - 1 + match.end() - 1
        
        # Insert new rows
        new_content = content[:table_end_idx] + new_rows + content[table_end_idx:]
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

# New rows based on changelog:
new_rows = """
| Constraint-Informed KAN (CIKAN) | Baseline KAN | CIKAN Boundary | FourierCSP constraint compiled into fixed CIKAN boundary and preserved | Exp 1723/1725 |
| EqM Sampler GPU Integration | CPU Gibbs | GPU EqM | Equilibrium Matching (EqM) Gradient Sampler converged faster on GPU | Exp 1727/1740 |
| Live Telemetry Streamer | Batch-only telemetry | Streamer load tested | Live Telemetry Streamer for Continual Learning load test successful | Exp 1738 |
| Milestone .134 closeout | .133 | Phase 4 synthesis | Analyzed wall time / experiments for .134 with phase_4_synthesis_complete | Exp 1745 |"""

append_to_table('docs/technical-report.md', '## Headline Results', new_rows)
append_to_table('README.md', '## Headline Results', new_rows) # Wait, README.md might not have '## Headline Results'

# Let's check README.md table header
with open('README.md', 'r') as f:
    readme = f.read()
    if '### Latest follow-ons' in readme:
        readme = readme.replace('### Latest follow-ons', '### Latest follow-ons\n' + '\n'.join(['- **' + row.split('|')[1].strip() + ':** ' + row.split('|')[4].strip() + ' - ' + row.split('|')[3].strip() for row in new_rows.strip().split('\n')]))
        with open('README.md', 'w') as out:
            out.write(readme)

