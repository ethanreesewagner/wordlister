import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

url = input("Enter the URL: ") # Replace with your URL
html_content = requests.get(url).text
soup = BeautifulSoup(html_content, "html.parser")

# Extract all text
all_text = soup.get_text()
# Step 1: First split by numbers
text = re.split(r'\d+', all_text)
# Step 2: Remove all numbers from the text
text = [re.sub(r'\d+', '', chunk) for chunk in text]
# Step 3: Split by punctuation
text = [re.split(r'[^\w\s]', chunk) for chunk in text]
text = [item for sublist in text for item in sublist]  # Flatten list
# Step 4: Split by new lines
text = [re.split(r'\n+', chunk) for chunk in text]
text = [item for sublist in text for item in sublist]  # Flatten list
# Step 5: Split every time there's a new capital letter
text = [re.split(r'(?=[A-Z])', chunk) for chunk in text]
text = [item for sublist in text for item in sublist]  # Flatten list
# Step 6: Make everything lowercase
text = [chunk.lower() for chunk in text]
# Step 7: Split by spaces
text = [re.split(r'\s+', chunk) for chunk in text]
text = [item for sublist in text for item in sublist]  # Flatten list
# Split by underscores and ampersands
text = [re.split(r'[_&]', chunk) for chunk in text]
text = [item for sublist in text for item in sublist]  # Flatten list
# Clean up: remove empty strings and strip whitespace
text = [chunk.strip() for chunk in text if chunk.strip()]
# Count frequency of each term
term_counts = Counter(text)
# Rank terms by frequency (most common first)
ranked_terms = term_counts.most_common()

print("\nRanked terms by frequency (most common first):")
for rank, (term, count) in enumerate(ranked_terms, 1):
    print(f"{rank}. {term}: {count}")