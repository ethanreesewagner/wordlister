import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import streamlit as st
import pandas as pd

def scraper(url):
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
    return term_counts.most_common()
    
st.write("Here's our first attempt at using data to create a table:")
st.text_input("Website", key="site")
if st.button("Get rankings!"):
    if st.session_state.site.startswith("http://") or st.session_state.site.startswith("https://"):
        url = st.session_state.site
    else:
        url = "http://" + st.session_state.site
    st.write(pd.DataFrame(scraper(url)))