import streamlit as st
import pandas as pd
import requests
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from rapidfuzz import fuzz
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import textwrap
import os

# ===== Initialize APIs from Replit secrets =====
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    base_url="https://models.github.ai/inference"
)

# ===== Helper functions =====
def search_urls(topic, max_results=50):
    query = f"best {topic} of all time ranked list"
    results = tavily.search(query=query, max_results=max_results)
    urls = []
    for r in results["results"]:
        url = r.get("url")
        if url and url.startswith("http"):
            urls.append(url)
    return urls

async def fetch(session, url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, timeout=12, headers=headers) as resp:
            return await resp.text()
    except:
        return None

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, u) for u in urls]
        pages = await asyncio.gather(*tasks)
    results = []
    for url, html in zip(urls, pages):
        if html and len(html) > 1500:
            results.append({"url": url, "html": html})
    return results

def extract_from_ordered_lists(soup):
    for ol in soup.find_all("ol"):
        items = ol.find_all("li")
        if len(items) >= 5:
            ranked = []
            for i, li in enumerate(items, 1):
                text = li.get_text(" ", strip=True)
                ranked.append((text, i))
            return ranked
    return None

def extract_from_tables(soup):
    for table in soup.find_all("table"):
        try:
            df = pd.read_html(str(table))[0]
            if len(df) >= 5:
                ranked = []
                for i, row in df.iterrows():
                    ranked.append((str(row[0]), i + 1))
                return ranked
        except:
            pass
    return None

def extract_from_numbered_text(text):
    pattern = r"\n\s*(\d{1,3})[\.\)]\s+([^\n]+)"
    matches = re.findall(pattern, text)
    if len(matches) >= 5:
        ranked = []
        for rank, item in matches:
            ranked.append((item.strip(), int(rank)))
        return ranked
    return None

def extract_ranked_list(html):
    soup = BeautifulSoup(html, "html.parser")
    result = extract_from_ordered_lists(soup)
    if result:
        return result
    result = extract_from_tables(soup)
    if result:
        return result
    text = soup.get_text("\n")
    result = extract_from_numbered_text(text)
    if result:
        return result
    return None

def extract_with_llm(topic, html):
    text = BeautifulSoup(html, "html.parser").get_text()
    prompt = textwrap.dedent(f"""
        Extract a ranked list about '{topic}'.
        Return JSON like [{{"item":"Example","rank":1}}].
        Only extract the ranked items. Text:
        {text[:3500]}
    """)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        items = re.findall(r'"item"\s*:\s*"([^"]+)"', content)
        ranks = re.findall(r'"rank"\s*:\s*(\d+)', content)
        ranked = []
        for item, rank in zip(items, ranks):
            ranked.append((item, int(rank)))
        if len(ranked) >= 3:
            return ranked
    except:
        pass
    return None

def normalize_item(x):
    x = x.lower()
    x = re.sub(r"\(.*?\)", "", x)
    x = re.sub(r"\d{4}", "", x)
    x = re.sub(r"[^a-z0-9 ]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def aggregate_lists(lists):
    scores = defaultdict(float)
    for ranked in lists:
        for item, rank in ranked:
            scores[normalize_item(item)] += 1 / rank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame([item for item, score in ranked[:50]], columns=["item"])
    return df

def run_pipeline(topic):
    urls = search_urls(topic)
    pages = asyncio.run(fetch_all(urls))
    extracted_lists = []
    for page in pages:
        ranked = extract_ranked_list(page["html"])
        if not ranked:
            ranked = extract_with_llm(topic, page["html"])
        if ranked:
            extracted_lists.append(ranked)
    aggregated = aggregate_lists(extracted_lists)
    return extracted_lists, aggregated

def scraper(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    counts = Counter(words)
    return counts.most_common()

# ===== Streamlit UI =====
st.title("Consensus Ranking Engine")

topic = st.text_input("Topic", placeholder="movies, books, albums, philosophers...")

if st.button("Build Ranking"):
    if not topic.strip():
        st.warning("Enter a topic")
    else:
        with st.spinner("Analyzing rankings..."):
            lists, aggregated = run_pipeline(topic)
        st.subheader("Consensus Ranking")
        st.dataframe(aggregated)
        st.subheader("Extracted Lists")
        for lst in lists:
            df = pd.DataFrame(lst, columns=["item", "rank"])
            st.dataframe(df)

st.write("---")
st.subheader("Zipf's Law")

site = st.text_input("Website")

if st.button("Analyze Text"):
    if not site.startswith("http"):
        site = "http://" + site
    df = pd.DataFrame(scraper(site), columns=["term", "count"])
    df = df.sort_values(by="count", ascending=False)
    st.bar_chart(df.head(40), x="term", y="count", sort=False, stack=False, horizontal=True)

