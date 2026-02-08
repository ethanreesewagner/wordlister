import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ddgs import DDGS

import requests
from bs4 import BeautifulSoup

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use OpenAI key

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o"
)

def get_duckduckgo_links(query, n_results=10):
    """Use DDGS to get a set of result links for a topic."""
    links = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=n_results):
            url = result.get('href') or result.get('url')
            if url and url.startswith("http"):
                links.append({'title': result.get('title', ''), 'url': url, 'snippet': result.get('body', '')})
            if len(links) >= n_results:
                break
    return links

def scrape_title_desc(url, timeout=10):
    """Try to get the <title> and first meta/description from the page to help the agent."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc = ""
        descmeta = soup.find("meta", attrs={"name": "description"})
        if descmeta and descmeta.get("content"):
            desc = descmeta["content"].strip()
        if not desc:
            ogdesc = soup.find("meta", attrs={"property": "og:description"})
            if ogdesc and ogdesc.get("content"):
                desc = ogdesc["content"].strip()
        return title, desc
    except Exception:
        return "", ""

def extract_and_parse_lists(topic: str, n_lists: int = 5) -> list:
    """
    Use DuckDuckGo to find links, then let GPT-4o with the extracted URLs
    produce/normalize n_lists top lists for a topic.
    Returns: list of pd.DataFrame objects, all with columns: item, rank, source
    """
    search_links = get_duckduckgo_links(f"greatest {topic} of all time ranked list", n_results=n_lists * 2)

    # Optionally, pass title/desc to help the agent, and filter out likely irrelevant links
    enriched = []
    for link in search_links:
        title, desc = scrape_title_desc(link["url"])
        if title or desc:
            enriched.append({
                "url": link["url"],
                "title": title or link["title"],
                "snippet": desc or link["snippet"]
            })
        else:
            enriched.append(link)
        if len(enriched) >= n_lists:
            break

    if not enriched:
        return []

    # Construct prompt including the found URLs
    sources_prompt = "\n".join([
        f"- URL: {x['url']}\n  Title: {x.get('title','')}\n  Desc/Snippet: {x.get('snippet','')}" for x in enriched[:n_lists]
    ])

    prompt = f"""
You are a top-list aggregation assistant.

INSTRUCTIONS:
- Given the following search results containing likely sources of ranked or scored lists about '{topic}', pick the {n_lists} best, reputable, independent sources.
- For each, visit the source (from the URL) and extract its actual ranked or scored list (e.g., top X {topic}); output as a markdown table with columns: 'item' and 'rank' (with 1 always meaning 'best').
- The table must contain at least 5 items if present.
- If the list uses a float/integer score or rating instead of rank, use that in the 'rank' column.
- Clarify if a higher or lower number is better.
- Do NOT make up or guess data: You must only reflect what's actually present in the list/table found in the page at the given URL. Output nothing if you cannot find a real list for a source.
- Only provide a table and the source for each, nothing else.

SOURCES TO USE:
{sources_prompt}

OUTPUT FORMAT:
For each list:
Source: <URL>
| item | rank |
|-------------------|-------|
| ... | ... |
"""

    # LLM call, no tool use (it gets the real URLs/snippets in the prompt, so agent must do extraction/format)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", None)
    if content is None and hasattr(response, "message"):
        content = response.message.content
    if content is None:
        raise RuntimeError("No content returned from GPT-4o call.")

    # Parse markdown tables from the response, as before
    import re
    import io

    blocks = re.split(r'(?=Source\s*:)', content)
    extracted_lists = []
    for block in blocks:
        url_match = re.search(r"Source\s*:\s*(\S+)", block)
        md_table_match = re.search(
            r"(\|+\s*item\s*\|+\s*rank\s*\|.*?)(?:\n\s*\n|\Z)", block, re.DOTALL | re.IGNORECASE
        )
        if url_match and md_table_match:
            url = url_match.group(1).strip()
            md_table = md_table_match.group(1).strip()
            md_lines = [line.strip() for line in md_table.splitlines() if "|" in line]
            if not md_lines:
                continue
            # Make sure the first header line and second separator line start with '|'
            if not md_lines[0].startswith("|"):
                md_lines[0] = "|" + md_lines[0]
            if len(md_lines) > 1 and not md_lines[1].startswith("|"):
                md_lines[1] = "|" + md_lines[1]
            # Only keep rows with at least two columns (to filter out empty/invalid tables)
            filtered_lines = []
            for line in md_lines:
                if line.count("|") >= 2:
                    filtered_lines.append(line)
            md_table_cleaned = "\n".join(filtered_lines)
            try:
                df = pd.read_csv(io.StringIO(md_table_cleaned), sep="|", engine='python')
                # Remove all-whitespace and unnamed columns
                df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).lower().startswith('unnamed')]]
                df.columns = [c.strip().lower() for c in df.columns]
                if 'item' in df.columns and 'rank' in df.columns:
                    pre_row_count = len(df)
                    df = df.dropna(subset=['item', 'rank'])
                    # Remove rows where item is empty string or rank is empty
                    df = df[df['item'].astype(str).str.strip() != ""]
                    df = df[df['rank'].astype(str).str.strip() != ""]
                    # Remove duplicate header/separator rows that some LLMs include in body
                    df = df[~df['item'].str.lower().isin(['item', '---', '———————', ''])]
                    df = df[~df['rank'].str.lower().isin(['rank', '---', '———', ''])]
                    # --- BEGIN PATCH: Remove first row if it's just dashes ---
                    if len(df) > 0:
                        first_row_val = df.iloc[0].astype(str).apply(lambda s: s.strip())
                        # Check if all columns in first row are dashes
                        if all(re.fullmatch(r"-+", v) for v in first_row_val):
                            df = df.iloc[1:]
                    # --- END PATCH ---
                    post_row_count = len(df)
                    if post_row_count > 0:
                        df['source'] = url
                        extracted_lists.append(df[['item', 'rank', 'source']])
            except Exception as e:
                continue
    return extracted_lists

user_topic = st.text_input("Enter a topic for top lists:")

if user_topic and st.button("Find and Aggregate Top Lists"):
    with st.spinner("Searching via DuckDuckGo and extracting lists with GPT-4o..."):
        try:
            dfs = extract_and_parse_lists(user_topic)
        except Exception as e:
            st.write(f"Failed: {e}")
            dfs = []

    if dfs:
        # Only aggregate if we actually have non-empty frames
        non_empty = [df[['item', 'rank']] for df in dfs if df is not None and not df.empty]
        if non_empty:
            combined = pd.concat(non_empty)
            combined['rank'] = pd.to_numeric(combined['rank'], errors='coerce')
            agg = combined.groupby('item', as_index=False)['rank'].mean().dropna().sort_values('rank')
            agg.columns = ['item', 'average_rank']

            st.write("---")
            st.write("### Aggregated List by Average Rank")
            st.dataframe(agg)
        else:
            st.info("No lists with extracted data found. (Check if the topic yields usable ranked lists!)")

        st.write("---")
        st.write("### Extracted Lists:")
        for i, df in enumerate(dfs, 1):
            if df is not None and not df.empty:
                st.write(f"**Extracted {i}:** (Source: {df['source'].iloc[0]})")
                st.dataframe(df[['item', 'rank']])
    else:
        st.info("No lists extracted. (Check if the topic yields relevant sources!)")
else:
    st.info("Enter a topic to get top lists.")