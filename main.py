import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

url = input("Enter the URL: ")
html_content = requests.get(url).text
soup = BeautifulSoup(html_content, "html.parser")
all_text = soup.get_text()
text = [
    chunk.strip().lower()
    for chunk in re.split(
        r'[^a-zA-Z]+', all_text
    )
    if chunk.strip()
]
term_counts = Counter(text)
ranked_terms = term_counts.most_common()

print("\nRanked terms by frequency (most common first):")
for rank, (term, count) in enumerate(ranked_terms, 1):
    print(f"{rank}. {term}: {count}")