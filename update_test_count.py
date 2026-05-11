import re

def update_test_count(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Update test count from 23,849 to 24,113
    content = content.replace("23,849 Python", "24,113 Python")
    content = content.replace("23,849\nPython", "24,113\nPython")
    content = content.replace("**23,849** Python", "**24,113** Python")
    content = content.replace("23849", "24113")
    
    # Also fix the 23,714 inside the report if present
    content = content.replace("**23,714** Python test items are currently collected", "**24,113** Python test items are currently collected")
    
    with open(filepath, 'w') as f:
        f.write(content)

update_test_count("README.md")
update_test_count("docs/technical-report.md")
update_test_count("docs/index.html")
print("Test counts updated.")
