import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime

queries = [
    'all:"energy-based model" AND (all:verification OR all:reasoning)',
    'all:"Kolmogorov-Arnold Network" OR all:"KAN"',
    'all:"constraint satisfaction" AND all:"neural network"',
    'all:"energy-guided decoding" OR all:"constrained generation"'
]

print("Arxiv Search Results:")
for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=desc&max_results=3'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')[:200]
            print(f"- {published[:10]} | {title}\n  {summary}...")
    except Exception as e:
        print(f"Error querying {q}: {e}")
