import re

with open('README.md', 'r') as f:
    text = f.read()

text = text.replace('Archived Milestones:** 225', 'Archived Milestones:** 226')
text = text.replace('Tests:** 25,287', 'Tests:** 25,305')

# Add new result to table
if 'PREM Architecture' not in text:
    text = text.replace('| Adversarial GSM8K | Apple Math | Credibility validation | Verified resistance to superficial changes |', 
                        '| Adversarial GSM8K | Apple Math | Credibility validation | Verified resistance to superficial changes |\n| Process-Reward | PREM Architecture | Dynamic Test-Time Compute (TTC) | Scaled by energy variance |')

with open('README.md', 'w') as f:
    f.write(text)
