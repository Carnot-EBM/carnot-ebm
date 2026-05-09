import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timedelta

def search(query, max_results=5):
    url = f'http://export.arxiv.org/api/query?search_query=all:"{urllib.parse.quote(query)}"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=desc'
    try:
        data = urllib.request.urlopen(url).read()
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', namespace).text.strip()
            summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
            authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            print(f"Title: {title}")
            print(f"Published: {published}")
            print(f"Authors: {', '.join(authors)}")
            print(f"Summary: {summary[:500]}...")
            print("-" * 40)
    except Exception as e:
        print(f"Error searching {query}: {e}")

print("SEARCHING: Energy-Based Models for verification")
search("Energy-Based Models verification reasoning", 3)

print("SEARCHING: Kolmogorov-Arnold Networks")
search("Kolmogorov-Arnold Networks KAN", 3)

print("SEARCHING: constraint satisfaction neural networks")
search("constraint satisfaction neural networks LLM", 3)

print("SEARCHING: Energy-guided decoding")
search("guided decoding energy", 3)

print("SEARCHING: continual online learning constraint")
search("continual learning online constraint", 3)
