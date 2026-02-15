import streamlit as st
import json
import os
from dotenv import load_dotenv
import pandas as pd
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langsmith import traceable
import requests
from bs4 import BeautifulSoup
import re
import io

load_dotenv()

langsmith_tracing = os.getenv("LANGSMITH_TRACING")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT")

if langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = langsmith_tracing
if langsmith_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
if langsmith_project:
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o"
)

@tool("duckduckgo_search", return_direct=False)
def ddg_tool(query: str, max_results: int = 25):
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
    context = []
    prev = elem.find_previous_siblings(limit=max_prev)
    for s in reversed(prev):
        t = s.get_text(strip=True)
        if s.name in ("h1", "h2", "h3", "h4", "h5", "h6") and t:
            context.append(f"  [{s.name}]: {t}")
        elif s.name == "caption" and t:
            context.append(f"  [caption]: {t}")
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
    html = str(elem)
    if len(html) > max_len:
        html = html[:max_len] + "\n... [truncated]"
    return html

@tool("extract_page_data", return_direct=False)
def extract_page_data_tool(url: str, timeout: int = 10, include_html: bool = True) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = []
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

def _parse_llm_list_output(content: str, parsed_url: str):
    md_table_match = re.search(
        r"(\|+(?:.+\|)+\n(?:\|.+\|.*\n)+)", content
    )
    if not md_table_match:
        return None
    md_table = md_table_match.group(1).strip()
    md_lines = [line.strip() for line in md_table.splitlines() if "|" in line and line.count("|") >= 2]
    if len(md_lines) < 2:
        return None
    if not md_lines[0].startswith("|"):
        md_lines[0] = "|" + md_lines[0]
    md_table_cleaned = "\n".join(md_lines)
    try:
        df = pd.read_csv(io.StringIO(md_table_cleaned), sep="|", engine='python')
        df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).lower().startswith('unnamed')]]
        df.columns = [c.strip().lower() for c in df.columns]
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
                return df[['item', 'rank', 'source']]
    except Exception:
        return None
    return None

def _format_lists_for_aggregation(dfs: list) -> str:
    parts = []
    for df in dfs:
        if df is None or df.empty:
            continue
        src = df['source'].iloc[0] if 'source' in df.columns else "unknown"
        srcblock = f"Source: {src}\n| item | rank |\n|------|------|\n"
        for _, row in df.iterrows():
            srcblock += f"| {row['item']} | {row['rank']} |\n"
        parts.append(srcblock.strip())
    return "\n\n".join(parts)

def _parse_markdown_table_to_df(md_content: str):
    """
    Parses markdown table from a string and returns a DataFrame with columns ['rank', 'item'] if possible.
    """
    md_table_match = re.search(
        r"(\|+\s*rank\s*\|\s*item\s*\|.*(?:\|.*\|.*\n?)+)", md_content, re.IGNORECASE
    )
    if not md_table_match:
        return None

    md_table = md_table_match.group(1).strip()
    md_lines = [line.strip() for line in md_table.splitlines() if "|" in line and line.count("|") >= 2]
    if len(md_lines) < 2:
        return None

    if not md_lines[0].startswith("|"):
        md_lines[0] = "|" + md_lines[0]
    md_table_cleaned = "\n".join(md_lines)
    try:
        df = pd.read_csv(io.StringIO(md_table_cleaned), sep="|", engine='python')
        df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).lower().startswith('unnamed')]]
        df.columns = [c.strip().lower() for c in df.columns]
        if "rank" in df.columns and "item" in df.columns:
            return df[["rank", "item"]]
        else:
            return None
    except Exception:
        return None

def agent_aggregate_lists(list_dfs: list, topic: str, llm, status_placeholder):
    """
    Perform aggregation of lists using an LLM agent.
    Passes all lists (markdown) in one prompt and asks LLM to output a single
    consensus markdown table with only columns: rank and item, ordered as the agent decides.
    Returns the result as a DataFrame.
    """
    aggregation_prompt = f"""You are an expert at synthesizing rankings from multiple sources.

You are given multiple lists about "{topic}". Each list comes from a different source.
Aggregate these lists into a single consensus ranking as you see fit, using your own best method.

Output a single markdown table with columns: rank, item.

Order by your consensus score (best first). Include at least the top 15-20 items.

Input lists (each starts with a Source line):

{_format_lists_for_aggregation(list_dfs)}

Respond ONLY with the table in markdown (NO intro sentence, NO comments):

| rank | item |
|------|------|
| 1 | ... |
"""
    status_placeholder.caption("Aggregating lists (via agent LLM)...")
    try:
        response = llm.invoke([HumanMessage(content=aggregation_prompt)])
        table_markdown = response.content if hasattr(response, "content") else str(response)

        # Parse agent output into DataFrame (required by instructions)
        df = _parse_markdown_table_to_df(table_markdown)
        if df is not None:
            return df
        else:
            # If parsing fails, return an empty DataFrame with columns
            return pd.DataFrame(columns=["rank", "item"])
    except Exception as e:
        # On agent error, return an empty DataFrame for consistent output
        return pd.DataFrame(columns=["rank", "item"])

@traceable(name="wordlister_pipeline")
def extract_and_parse_lists(topic: str) -> list:
    """
    every URL. 3) Use LLM to parse each page's data into item|rank format.
    Returns: list of pd.DataFrame objects with columns: item, rank, source
    Deterministically: 1) Search DuckDuckGo for all URLs. 2) Extract page data from
    every URL. 3) Use LLM to parse each page. 4) Use LLM to aggregate lists.
    Returns: (list of DataFrames, aggregated_output) where aggregated_output is a DataFrame.
    """
    status_placeholder = st.empty()
    search_query = f"top {topic} ranked list best of all time"
    urls_to_process = []
    with DDGS() as ddgs:
        for result in ddgs.text(search_query, max_results=25):
            url = result.get("href") or result.get("url")
            if (
                url
                and url.startswith("http")
                and url not in [u["url"] for u in urls_to_process]
            ):
                urls_to_process.append(
                    {"url": url, "title": result.get("title", "")}
                )
    extracted_lists = []
    extract_prompt_template = """Extract the primary ranked/list data from this webpage about "{topic}".
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
    for idx, entry in enumerate(urls_to_process):
        url = entry["url"]
        status_placeholder.caption(f"Processing {idx + 1}/{len(urls_to_process)}...")
        try:
            raw_data = extract_page_data_tool.invoke({"url": url})
        except Exception as e:
            continue
        if raw_data.strip().endswith("on this page.") and "No tables or lists" in raw_data:
            continue
        parse_prompt = extract_prompt_template.format(topic=topic, url=url) + "\n\nExtracted data:\n" + raw_data[:8000]
        try:
            response = llm.invoke([HumanMessage(content=parse_prompt)])
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            continue
        first_line = content.strip().split("\n")[0].strip().upper()
        if first_line == "NONE" or content.strip().upper().startswith("NONE"):
            continue
        # parse markdown table for aggregation and for user
        blocks = re.split(r"(?=Source\s*:)", content)
        for block in blocks:
            url_match = re.search(r"Source\s*:\s*(\S+)", block)
            if not url_match:
                continue
            parsed_url = url_match.group(1).strip()
            df = _parse_llm_list_output(block, parsed_url)
            if df is not None and not df.empty:
                extracted_lists.append(df)
    status_placeholder.empty()
    if extracted_lists:
        status_placeholder.caption("Aggregating lists...")
        aggregated_df = agent_aggregate_lists(extracted_lists, topic, llm, status_placeholder)
        status_placeholder.empty()
        return extracted_lists, aggregated_df
    status_placeholder.empty()
    # Return aggregated_output as empty DataFrame for API contract
    return extracted_lists, pd.DataFrame(columns=["rank", "item"])

st.title("Find and Aggregate Top Lists")

col1, col2 = st.columns([3,1])
with col1:
    user_topic = st.text_input("Enter a topic for top lists:", placeholder="e.g. movies, albums, books")
with col2:
    do_find = st.button("Find and Aggregate Top Lists")
    
if 'do_find' not in locals():
    do_find = False

if do_find:
    if not user_topic or not user_topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Searching, extracting, and parsing all URLs..."):
            try:
                dfs, aggregated_result = extract_and_parse_lists(user_topic.strip())
            except Exception as e:
                st.write(f"Failed: {e}")
                dfs = []
                aggregated_result = pd.DataFrame(columns=["rank", "item"])
        if dfs:
            non_empty = [df[['item', 'rank']] for df in dfs if df is not None and not df.empty]
            if non_empty:
                st.write("---")
                st.write("### Aggregated List (Agent Consensus)")
                if aggregated_result is not None and isinstance(aggregated_result, pd.DataFrame):
                    if not aggregated_result.empty:
                        st.dataframe(aggregated_result)
                    else:
                        st.info("No aggregate list found.")
                else:
                    st.info("No aggregate list found.")
            st.write("---")
            st.write("### Extracted Lists:")
            for i, df in enumerate(dfs, 1):
                if df is not None and not df.empty:
                    st.write(f"**Extracted {i}:** (Source: {df['source'].iloc[0]})")
                    st.dataframe(df[['item', 'rank']])
        else:
            st.info("No lists extracted. (Check if the topic yields relevant sources!)")
else:
    st.info("Enter a topic and click **Find and Aggregate Top Lists** to start.")