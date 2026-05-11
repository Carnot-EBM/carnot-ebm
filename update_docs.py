import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Update experiment counts
    content = re.sub(r'(\d+,?\d*)\s+experiment records tracked through Exp\s+\d+', r'2,072 experiment records tracked through Exp 1745', content)
    content = re.sub(r'Exp \d+ data', r'Exp 1745 data', content)
    
    # Update milestone counts
    content = re.sub(r'(\d+)\s+archived completed milestone records', r'148 archived completed milestone records', content)
    content = re.sub(r'(\d+)\s+archived records through \.\d+', r'148 archived records through .134', content)
    
    # Update python test items collected
    content = re.sub(r'(\d+,?\d*)\s+Python test items collected', r'24,113 Python test items collected', content)
    content = re.sub(r'(\d+,?\d*)\s+Python items collected', r'24,113 Python items collected', content)

    # Update milestones in stats
    content = re.sub(r'through Exp \d+', r'through Exp 1745', content)
    
    with open(filepath, 'w') as f:
        f.write(content)

update_file('docs/index.html')
update_file('README.md')
update_file('docs/technical-report.md')

