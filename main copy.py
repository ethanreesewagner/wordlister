import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "5000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
import sys
import logging
import streamlit as st
import pandas as pd
import requests
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import textwrap

# ===== Streamlit environment configuration =====
os.environ["STREAMLIT_FIRST_RUN_DISABLED"] = "true"
os.environ["BROWSER"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_LOG_LEVEL"] = "info"

# ===== Logging configuration =====
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===== Asyncio patch for Streamlit =====
import nest_asyncio
nest_asyncio.apply()

# ===== Lazy initialization for API clients =====
tavily_client = None
llm_client = None

def get_clients():
    global tavily_client, llm_client
    if tavily_client is None:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            logger.info("TavilyClient initialized")
        except Exception:
            logger.exception("TavilyClient init failed")
            tavily_client = None
    if llm_client is None:
        try:
            from langchain_openai import ChatOpenAI
            llm_client = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o",
                base_url="https://models.github.ai/inference"
            )
            logger.info("LLM client initialized")
        except Exception:
            logger.exception("LLM client init failed")
            llm_client = None
    return tavily_client, llm_client

# ===== Helper functions =====
def search_urls(topic, max_results=50):
    tavily, _ = get_clients()
    if not tavily:
        logger.warning("TavilyClient unavailable, returning empty URL list")
        return []

    query = f"best {topic} of all time ranked list"

    try:
        results = tavily.search(query=query, max_results=max_results)
        urls = [
            r.get("url")
            for r in results.get("results", [])
            if r.get("url", "").startswith("http")
        ]
        logger.info(f"Found {len(urls)} URLs for topic '{topic}'")
        return urls
    except Exception:
        logger.exception(f"Tavily search failed for topic '{topic}'")
        return []

async def fetch(session, url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, timeout=12, headers=headers) as resp:
            html = await resp.text()
            logger.info(f"Fetched {url} ({len(html)} chars)")
            return html
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, u) for u in urls]
        pages = await asyncio.gather(*tasks)

    results = [
        {"url": url, "html": html}
        for url, html in zip(urls, pages)
        if html and len(html) > 1500
    ]

    logger.info(f"Fetched {len(results)} valid pages out of {len(urls)} URLs")
    return results

def extract_from_ordered_lists(soup):
    for ol in soup.find_all("ol"):
        items = ol.find_all("li")

        if len(items) >= 5:
            return [
                (li.get_text(" ", strip=True), i + 1)
                for i, li in enumerate(items)
            ]

    return None

def extract_from_tables(soup):
    for table in soup.find_all("table"):
        try:
            df = pd.read_html(str(table))[0]

            if len(df) >= 5:
                return [
                    (str(row[0]), i + 1)
                    for i, row in df.iterrows()
                ]

        except Exception as e:
            logger.debug(f"Skipping table due to error: {e}")

    return None

def extract_from_numbered_text(text):
    matches = re.findall(r"\n\s*(\d{1,3})[\.\)]\s+([^\n]+)", text)

    if len(matches) >= 5:
        return [(item.strip(), int(rank)) for rank, item in matches]

    return None

def extract_ranked_list(html):
    soup = BeautifulSoup(html, "html.parser")

    result = extract_from_ordered_lists(soup) or extract_from_tables(soup)

    if result:
        return result

    text = soup.get_text("\n")

    return extract_from_numbered_text(text) or None

def extract_with_llm(topic, html):
    _, llm = get_clients()

    if not llm:
        logger.warning("LLM client unavailable, skipping LLM extraction")
        return None

    text = BeautifulSoup(html, "html.parser").get_text()

    prompt = textwrap.dedent(f"""
    Extract a ranked list about '{topic}'.
    Return JSON like [{{"item":"Example","rank":1}}].
    Only extract the ranked items.

    Text:
    {text[:3500]}
    """)

    try:
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        items = re.findall(r'"item"\s*:\s*"([^"]+)"', content)
        ranks = re.findall(r'"rank"\s*:\s*(\d+)', content)

        ranked = [(item, int(rank)) for item, rank in zip(items, ranks)]

        logger.info(f"LLM extracted {len(ranked)} items for topic '{topic}'")

        if len(ranked) >= 3:
            return ranked

    except Exception:
        logger.exception("LLM extraction failed")

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

    logger.info(f"Aggregated {len(ranked)} items across lists")

    return pd.DataFrame(
        [item for item, _ in ranked[:50]],
        columns=["item"]
    )

def run_pipeline(topic):
    logger.info(f"Running pipeline for topic '{topic}'")

    urls = search_urls(topic)

    if not urls:
        logger.warning(f"No URLs found for topic '{topic}'")
        return [], pd.DataFrame(columns=["item"])

    pages = asyncio.run(fetch_all(urls))

    extracted_lists = []

    for page in pages:
        ranked = extract_ranked_list(page["html"]) or extract_with_llm(
            topic,
            page["html"]
        )

        if ranked:
            extracted_lists.append(ranked)

    aggregated = aggregate_lists(extracted_lists)

    return extracted_lists, aggregated

def scraper(url):
    try:
        html = requests.get(url).text

        words = [
            w.lower()
            for w in re.findall(r"[a-zA-Z']+", html)
        ]

        counts = Counter(words)

        return counts.most_common()

    except Exception:
        logger.exception(f"Scraper failed for URL: {url}")
        return []

# ===== Streamlit UI =====
st.title("Consensus Ranking Engine")

topic = st.text_input(
    "Topic",
    placeholder="movies, books, albums, philosophers..."
)

if st.button("Build Ranking"):

    if not topic.strip():
        st.warning("Enter a topic")

    else:
        with st.spinner("Analyzing rankings..."):
            try:
                lists, aggregated = run_pipeline(topic)

                st.subheader("Consensus Ranking")
                st.dataframe(aggregated)

                st.subheader("Extracted Lists")

                for lst in lists:
                    st.dataframe(
                        pd.DataFrame(
                            lst,
                            columns=["item", "rank"]
                        )
                    )

            except Exception as e:
                logger.exception("Pipeline failed")
                st.error(f"Error running pipeline: {e}")

st.write("---")

st.subheader("Zipf's Law")

site = st.text_input("Website")

if st.button("Analyze Text"):

    if not site.startswith("http"):
        site = "http://" + site

    try:
        df = pd.DataFrame(
            scraper(site),
            columns=["term", "count"]
        )

        df = df.sort_values(by="count", ascending=False)

        st.bar_chart(
            df.head(40),
            x="term",
            y="count",
            sort=False,
            stack=False,
            horizontal=True
        )

    except Exception as e:
        logger.exception("Zipf analysis failed")
        st.error(f"Error analyzing text: {e}")
