import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime

queries = [
    'all:"Energy-Based Model" AND all:reasoning',
    'all:"Energy-Based" AND all:verification',
    'all:"Constraint satisfaction" AND all:"neural network"',
    'all:"Ising model" AND all:"machine learning"',
    'all:"hallucination" AND all:"detection" AND all:"LLM"',
    'all:"Kolmogorov-Arnold Network"',
    'all:"Energy-guided decoding"',
    'all:"constrained generation"',
]

results_text = []

for q in queries:
    # URL encode
    encoded_q = urllib.parse.quote(q)
    url = f'http://export.arxiv.org/api/query?search_query={encoded_q}&sortBy=submittedDate&sortOrder=desc&max_results=3'
    try:
        req = urllib.request.urlopen(url)
        xml_data = req.read()
        root = ET.fromstring(xml_data)
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')[:200] + '...'
            results_text.append(f"Date: {published}\nTitle: {title}\nSummary: {summary}\n")
    except Exception as e:
        results_text.append(f"Error querying {q}: {e}")

with open('arxiv_results.txt', 'w') as f:
    f.write('\n'.join(results_text))
