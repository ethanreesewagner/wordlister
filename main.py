import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langsmith import traceable
from typing import Optional, Tuple, List, Dict, Any
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

import re
import io

load_dotenv()

# Load LangSmith environment variables from .env lines 6-10 (see @file_context_0)
langsmith_tracing = os.getenv("LANGSMITH_TRACING")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT")

# Set LangSmith tracing and config
if langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = langsmith_tracing
if langsmith_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
if langsmith_project:
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

openai_api_key = os.getenv("OPENAI_API_KEY")  # Use OpenAI key
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o"
)

# --- Tool definitions ---

@tool("duckduckgo_search", return_direct=False)
def ddg_tool(query: str, max_results: int = 25) -> List[Dict[str, str]]:
    """Search DuckDuckGo for top results. Returns a list of dicts with title, url, snippet."""
    links = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            url = result.get('href') or result.get('url')
            if url and url.startswith("http"):
                links.append({'title': result.get('title', ''), 'url': url, 'snippet': result.get('body', '')})
            if len(links) >= max_results:
                break
    return links

def _get_preceding_context(elem, soup, max_prev=3) -> str:
    """Get headings and text from elements preceding this one (siblings or ancestors)."""
    context = []
    prev = elem.find_previous_siblings(limit=max_prev)
    for s in reversed(prev):
        t = s.get_text(strip=True)
        if s.name in ("h1", "h2", "h3", "h4", "h5", "h6") and t:
            context.append(f"  [{s.name}]: {t}")
        elif s.name == "caption" and t:
            context.append(f"  [caption]: {t}")
    # Check parent section
    parent = elem.parent
    for _ in range(2):
        if parent and parent.name:
            if parent.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                t = parent.get_text(strip=True)
                if t:
                    context.append(f"  [parent {parent.name}]: {t}")
            elif parent.get("class"):
                context.append(f"  [container class]: {' '.join(parent.get('class', []))}")
            parent = parent.parent
        else:
            break
    return "\n".join(context) if context else ""

def _get_html_snippet(elem, max_len: int = 2000) -> str:
    """Get raw HTML for element and its children, truncated."""
    html = str(elem)
    if len(html) > max_len:
        html = html[:max_len] + "\n... [truncated]"
    return html

@tool("extract_page_data", return_direct=False)
def extract_page_data_tool(url: str, timeout: int = 10, include_html: bool = True) -> str:
    """Extract data from a webpage: tables and lists with surrounding context (headings, captions, container classes) and optionally the raw HTML markup for each block."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = []

        # Extract HTML tables with context
        table_elems = soup.find_all("table")
        for i, table in enumerate(table_elems[:5]):
            try:
                df = pd.read_html(str(table))[0]
            except Exception:
                continue
            if df is None or df.empty or len(df) < 2:
                continue
            ctx = _get_preceding_context(table, soup)
            block = f"TABLE {i + 1}:\n"
            if ctx:
                block += "Context (headings/captions):\n" + ctx + "\n"
            block += "Data:\n" + df.to_string(index=False) + "\n"
            if include_html:
                block += "HTML snippet:\n```html\n" + _get_html_snippet(table) + "\n```\n"
            parts.append(block)

        # Ordered lists with context
        for i, ol in enumerate(soup.find_all("ol", limit=5)):
            items = [li.get_text(strip=True) for li in ol.find_all("li", recursive=False) if li.get_text(strip=True)]
            if len(items) < 3:
                continue
            ctx = _get_preceding_context(ol, soup)
            block = f"ORDERED LIST {i + 1}:\n"
            if ctx:
                block += "Context:\n" + ctx + "\n"
            block += "Items:\n" + "\n".join(f"  {j+1}. {t}" for j, t in enumerate(items)) + "\n"
            if include_html:
                block += "HTML snippet:\n```html\n" + _get_html_snippet(ol) + "\n```\n"
            parts.append(block)

        # Unordered lists
        for i, ul in enumerate(soup.find_all("ul", limit=3)):
            items = [li.get_text(strip=True) for li in ul.find_all("li", recursive=False) if li.get_text(strip=True)]
            if len(items) < 5:
                continue
            ctx = _get_preceding_context(ul, soup)
            block = f"LIST {i + 1}:\n"
            if ctx:
                block += "Context:\n" + ctx + "\n"
            block += "Items:\n" + "\n".join(f"  {j+1}. {t}" for j, t in enumerate(items)) + "\n"
            if include_html:
                block += "HTML snippet:\n```html\n" + _get_html_snippet(ul) + "\n```\n"
            parts.append(block)

        if parts:
            return "Extracted data from page:\n\n" + "\n".join(parts)
        # Fallback: extract main text (many sites use JS or custom markup for lists)
        for selector in ["article", "main", ".content", ".article-body"]:
            try:
                els = soup.select(selector)
                for el in els[:2]:
                    t = el.get_text(separator="\n", strip=True)
                    if len(t) > 200:
                        return "No tables/lists found. Main page text:\n\n" + t[:6000]
            except Exception:
                pass
        body = soup.find("body")
        if body:
            t = body.get_text(separator="\n", strip=True)
            if len(t) > 300:
                return "No tables/lists found. Page text:\n\n" + t[:6000]
        return "No tables or lists with sufficient items found on this page."
    except Exception as e:
        return f"Error extracting page data: {e}"

# --- Deterministic extraction: search all URLs, extract all, LLM parses each ---

def _format_lists_for_aggregation(dfs: list) -> str:
    """Format extracted DataFrames for the aggregation prompt."""
    parts = []
    for i, df in enumerate(dfs):
        if df is None or df.empty:
            continue
        source = df["source"].iloc[0] if "source" in df.columns else "unknown"
        parts.append(f"List {i + 1} (Source: {source}):\n" + df[["item", "rank"]].to_string(index=False) + "\n")
    return "\n".join(parts)

def agent_aggregate_lists(dfs: list, topic: str, llm):
    """
    Use the LLM to aggregate multiple ranked lists into a single consensus ranking.
    Returns a string (markdown table) and optionally a DataFrame.
    """
    if not dfs or all(df is None or df.empty for df in dfs):
        return None, "No lists to aggregate."
    
    formatted = _format_lists_for_aggregation(dfs)
    prompt = f"""You are an expert at synthesizing rankings from multiple sources.

You have {len([d for d in dfs if d is not None and not d.empty])} ranked lists about "{topic}". 
Aggregate them into a single consensus ranking. Consider:
- Items that appear in more lists should generally rank higher
- Average rank across lists
- Consistency of placement

Output a single markdown table with columns: rank, item, num_sources, avg_rank
Order by consensus score (best first). Include at least the top 15-20 items.

Input lists:
{formatted}

Output format (strict markdown table):
| rank | item | num_sources | avg_rank |
|------|------|-------------|----------|
| 1 | ... | ... | ... |
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    return content, content

@traceable(name="wordlister_pipeline")
def extract_and_parse_lists(topic: str, llm, status_placeholder=None) -> tuple:
    """
    Deterministically: 1) Search DuckDuckGo for all URLs. 2) Extract page data from
    every URL. 3) Use LLM to parse each page. 4) Use LLM to aggregate lists.
    Returns: (list of DataFrames, aggregated_markdown_string)
    """
    search_query = f"top {topic} ranked list best of all time"
    
    # 1. Get ALL search results (no agent - we control this)
    urls_to_process = []
    with DDGS() as ddgs:
        for result in ddgs.text(search_query, max_results=25):
            url = result.get("href") or result.get("url")
            if url and url.startswith("http") and url not in [u["url"] for u in urls_to_process]:
                urls_to_process.append({"url": url, "title": result.get("title", "")})
    
    if not urls_to_process:
        raise RuntimeError("No URLs found from DuckDuckGo search.")
    
    if status_placeholder is None:
        status_placeholder = st.empty()
    
    extracted_lists = []
    prompt_template = """Extract the primary ranked/list data from this webpage about "{topic}".
The data may be in tables, ordered lists, or numbered text. Output a markdown table with columns "item" and "rank" (rank 1 = best).
If you cannot find any ranked list with at least 3 items, respond with exactly: NONE

Output format (use exactly this):
Source: {url}
| item | rank |
|------|------|
| Item1 | 1 |
| Item2 | 2 |
| ... | ... |
"""
    
    # 2. Extract data from EACH URL, then LLM-parse each
    for idx, entry in enumerate(urls_to_process):
        url = entry["url"]
        status_placeholder.caption(f"Processing {idx + 1}/{len(urls_to_process)}...")
        try:
            raw_data = extract_page_data_tool.invoke({"url": url})
        except Exception:
            continue
        
        if raw_data.strip().endswith("on this page.") and "No tables or lists" in raw_data:
            continue
        
        # 3. LLM parses this page's data into item|rank format
        user_msg = prompt_template.format(topic=topic, url=url) + "\n\nExtracted data:\n" + raw_data[:8000]  # Truncate if huge
        try:
            response = llm.invoke([HumanMessage(content=user_msg)])
            content = response.content if hasattr(response, "content") else str(response)
        except Exception:
            continue
        
        first_line = content.strip().split("\n")[0].strip().upper()
        if first_line == "NONE" or content.strip().upper().startswith("NONE"):
            continue
        
        # 4. Parse the LLM output
        blocks = re.split(r"(?=Source\s*:)", content)
        for block in blocks:
            url_match = re.search(r"Source\s*:\s*(\S+)", block)
            md_table_match = re.search(
                r"(\|+[^\n]+\|+[^\n]+\|[^\n]*\n(?:\|[^\n]+\|[^\n]+\|[^\n]*\n)+)",
                block
            ) or re.search(
                r"(\|+\s*(?:item|name|title)\s*\|+\s*(?:rank|#|position)\s*\|.*?)(?:\n\s*\n|\Z)",
                block, re.DOTALL | re.IGNORECASE
            )
            if not url_match or not md_table_match:
                continue
            parsed_url = url_match.group(1).strip()
            md_table = md_table_match.group(1).strip()
            md_lines = [line.strip() for line in md_table.splitlines() if "|" in line and line.count("|") >= 2]
            if len(md_lines) < 2:
                continue
            if not md_lines[0].startswith("|"):
                md_lines[0] = "|" + md_lines[0]
            md_table_cleaned = "\n".join(md_lines)
            try:
                df = pd.read_csv(io.StringIO(md_table_cleaned), sep="|", engine='python')
                df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).lower().startswith('unnamed')]]
                df.columns = [c.strip().lower() for c in df.columns]
                # Map common column names to item/rank; fallback to first two columns
                if len(df.columns) < 2:
                    continue
                col0, col1 = df.columns[0], df.columns[1]
                item_col = next((c for c in df.columns if str(c).lower() in ("item","name","title","movie","song","game","entry")), col0)
                rank_col = next((c for c in df.columns if str(c).lower() in ("rank","#","position","no","no.","number") or "rank" in str(c).lower() or "#" in str(c)), col1)
                df = df.rename(columns={item_col: "item", rank_col: "rank"})
                if 'item' in df.columns and 'rank' in df.columns:
                    df = df.dropna(subset=['item', 'rank'])
                    df = df[df['item'].astype(str).str.strip() != ""]
                    df = df[df['rank'].astype(str).str.strip() != ""]
                    df = df[~df['item'].str.lower().isin(['item', '---', '———————', ''])]
                    df = df[~df['rank'].str.lower().isin(['rank', '---', '———', ''])]
                    if len(df) > 0:
                        first_row_val = df.iloc[0].astype(str).apply(lambda s: s.strip())
                        if all(re.fullmatch(r"-+", v) for v in first_row_val):
                            df = df.iloc[1:]
                    if len(df) > 0:
                        df = df.copy()
                        df['source'] = parsed_url
                        extracted_lists.append(df[['item', 'rank', 'source']])
            except Exception:
                continue
    status_placeholder.empty()
    
    # Agent aggregation (LLM aggregates all lists into one consensus ranking)
    status_placeholder.caption("Aggregating lists...")
    aggregated_str, _ = agent_aggregate_lists(extracted_lists, topic, llm)
    status_placeholder.empty()
    
    return extracted_lists, aggregated_str

user_topic = st.text_input("Enter a topic for top lists:", placeholder="e.g. movies, albums, books")
find_clicked = st.button("Find and Aggregate Top Lists")

if find_clicked:
    if not user_topic or not user_topic.strip():
        st.warning("Please enter a topic first.")
    else:
        status_ph = st.empty()
        with st.spinner("Searching, extracting, parsing, and aggregating..."):
            try:
                dfs, aggregated_str = extract_and_parse_lists(user_topic.strip(), llm, status_ph)
            except Exception as e:
                st.write(f"Failed: {e}")
                dfs = []
                aggregated_str = None

        if dfs:
            if aggregated_str:
                st.write("---")
                st.write("### Aggregated List (Agent Consensus)")
                st.markdown(aggregated_str)
            st.write("---")
            st.write("### Extracted Lists:")
            for i, df in enumerate(dfs, 1):
                if df is not None and not df.empty:
                    st.write(f"**Extracted {i}:** (Source: {df['source'].iloc[0]})")
                    st.dataframe(df[['item', 'rank']])
        else:
            st.info("No lists extracted. Try a different topic or check your connection.")
else:
    st.info("Enter a topic and click **Find and Aggregate Top Lists** to start.")