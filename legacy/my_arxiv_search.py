import urllib.request
import xml.etree.ElementTree as ET

queries = [
    'all:energy+AND+all:reasoning',
    'all:kolmogorov-arnold',
    'all:energy-guided+AND+all:decoding',
]

for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=3&sortBy=submittedDate&sortOrder=desc'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        print(f"--- Results for {q} ---")
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Summary: {summary[:200]}...")
            print()
    except Exception as e:
        print(f"Error querying {q}: {e}")
