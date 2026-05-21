import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacements
    content = content.replace("1,738", "1,778")
    content = content.replace("Exp 1587", "Exp 1627")
    content = content.replace("134 archived", "136 archived")
    content = content.replace("134-record", "136-record")
    content = content.replace("in 134", "in 136")
    content = content.replace("134 artifact-backed", "136 artifact-backed")
    content = content.replace("archived through .120", "archived through .121")
    content = content.replace("extending through .121", "extending through .122")
    content = content.replace("stops at .120", "stops at .121")
    
    with open(filepath, 'w') as f:
        f.write(content)

update_file("README.md")
update_file("docs/technical-report.md")
