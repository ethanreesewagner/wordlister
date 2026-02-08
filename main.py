import requests
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI  # Changed import to openai
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from ddgs import DDGS

from langchain.agents import initialize_agent, AgentType  # Use classic LangChain zero-shot agent

# Store extracted lists for aggregation
extracted_lists = []

@tool
def extract_list_from_url(url: str) -> str:
    """
    Fetches the HTML content of a URL and asks the LLM to extract and format any top/ranked lists found on the page.
    This tool is used to get top lists from web pages, even if the page does not contain HTML tables.

    Args:
        url: The URL to extract the list from

    Returns:
        A string representation of the extracted list data, or a message if no ranked list was found.
    """
    global extracted_lists
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return f"Failed to retrieve {url}: HTTP {response.status_code}"

        html = response.text

        # Ask the LLM to extract the list from this HTML
        system_prompt = (
            "You are a web list extraction assistant. Given the following webpage HTML and its URL, "
            "identify and extract any ordered or ranked lists on the topic (such as top 10 lists, rankings, etc.). "
            "If such lists exist, output them as markdown tables with columns for item and ranking/position. "
            "If none are found, just state that clearly. Do NOT hallucinate any list."
        )
        user_prompt = f"""URL: {url}
---
HTML Content:
{html[:9000]}
---
Please extract any top/ranked lists you find on this page. If possible, output a markdown table or ranked list (item and rank/score if available)."""

        # Use a simple chat prompt to the LLM
        llm_response = llm([
            HumanMessage(
                content=f"{system_prompt}\n\n{user_prompt}"
            )
        ]).content

        # Try to parse the LLM result as a pandas DataFrame if it's a markdown table
        extracted_df = None
        if "No ranked list" not in llm_response and "|" in llm_response:
            try:
                # Find the markdown table in the LLM response (heuristic)
                import io
                lines = llm_response.strip().splitlines()
                table_lines = [line for line in lines if '|' in line and '-' not in line]
                if len(table_lines) >= 2:
                    # Insert the header line and possible delimiter
                    for i, line in enumerate(lines):
                        if "|" in line and ("---" in line or "---" in lines[i+1] if i+1 < len(lines) else False):
                            header_idx = i
                            break
                    else:
                        header_idx = 0
                    md_table = "\n".join(lines[header_idx:])
                    extracted_df = pd.read_csv(io.StringIO(md_table), sep="|").dropna(axis=1, how="all")
                    # Clean up: drop index column if present from markdown output
                    if extracted_df.columns[0].strip() == "":
                        extracted_df = extracted_df.iloc[:, 1:]
            except Exception:
                pass

        if extracted_df is not None and not extracted_df.empty:
            extracted_lists.append(extracted_df)
            return f"Successfully extracted ranked list from {url}:\n{extracted_df.to_string(index=False)}"
        else:
            # Save the LLM output text anyway, could combine later
            extracted_lists.append(llm_response)
            return f"LLM extracted from {url}:\n{llm_response}"

    except Exception as e:
        return f"Error extracting list from {url}: {str(e)}"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use OpenAI key

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o"
)

@tool
def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo's direct API (duckduckgo-search library) for information about a topic.
    Returns search results as a formatted string with URLs and snippets.

    Args:
        query: The search query to execute

    Returns:
        A string containing search results with URLs and descriptions
    """
    try:
        # Use DuckDuckGo search directly via the ddgs library
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10))

        if results and len(results) > 0:
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                url = result.get('href', result.get('url', 'No URL'))
                body = result.get('body', result.get('snippet', 'No description'))
                formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {body}\n")
            return "\n".join(formatted_results)
        else:
            return f"Search completed but no results found for: '{query}'. Try a more specific search query or different keywords."

    except Exception as e:
        error_msg = str(e)
        return f"Error searching DuckDuckGo for '{query}': {error_msg}. Please try again with a different query."

# Set up tools
tools = [search_web, extract_list_from_url]

# Use classic LangChain agent since LangGraph's create_react_agent raises an import error
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Check if the tool input is a valid string or a URL."
)

user_topic = st.text_input("Enter a topic for top lists:")

if user_topic and st.button("Find and Aggregate Top Lists"):
    # Reset extracted lists
    extracted_lists.clear()

    # Use agent to find and extract top lists
    user_prompt = (
        f"Find websites that contain high-quality top lists on the topic '{user_topic}'. "
        f"Search for reputable sources like Wikipedia, news articles, or ranking sites. "
        f"For at least 3-5 relevant URLs, fetch their content and extract any ranked/top lists directly from the content, even if they are not tables. "
        f"Attempt to return any ranked list as structured markdown tables (with item and ranking or score columns, if possible)."
    )

    with st.spinner("Agent is searching and extracting data..."):
        # Use agent to respond to user_prompt
        try:
            result = agent({"input": user_prompt})
        except Exception as e:
            result = {"output": f"Agent failed due to: {str(e)}"}

    st.write("Agent Response:")
    # Write the agent's output (zero-shot agent)
    if isinstance(result, dict) and "output" in result:
        st.write(result["output"])
    else:
        st.write(str(result))

    # Display extracted lists
    if extracted_lists:
        st.write("---")
        st.write("### Extracted Lists:")
        for i, table in enumerate(extracted_lists, 1):
            st.write(f"**Extracted {i}:**")
            if isinstance(table, pd.DataFrame):
                st.dataframe(table)
            else:
                st.write(table)

        # Attempt to aggregate only the DataFrames
        dfs = [table for table in extracted_lists if isinstance(table, pd.DataFrame)]
        if len(dfs) > 1:
            st.write("---")
            st.write("### Aggregated List:")
            import numpy as np

            def find_ranking_col(df):
                number_cols = df.select_dtypes(include=np.number).columns.tolist()
                if number_cols:
                    return number_cols[0]
                for col in df.columns:
                    if any(kw in str(col).lower() for kw in ['rank', 'score', 'points', 'position']):
                        return col
                return None

            merged = None
            for i, df in enumerate(dfs):
                name_col = None
                # Try to find column likely to be the entity/item
                object_cols = [c for c in df.columns if df[c].dtype == object]
                if object_cols:
                    name_col = object_cols[0]
                else:
                    name_col = df.columns[0]
                rank_col = find_ranking_col(df)
                use_df = df[[name_col, rank_col]].copy() if rank_col and rank_col in df.columns else df[[name_col]].copy()
                use_df = use_df.rename(columns={name_col: "item", rank_col: f"rank_{i}" if rank_col else f"score_{i}"})
                if merged is None:
                    merged = use_df
                else:
                    merged = pd.merge(merged, use_df, on="item", how="outer")

            if merged is not None and len(merged) > 0:
                rank_cols = [c for c in merged.columns if 'rank' in c or 'score' in c]
                if rank_cols:
                    merged['average'] = merged[rank_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
                    merged_sorted = merged.sort_values('average')
                    st.dataframe(merged_sorted[['item', 'average'] + rank_cols])
                else:
                    st.dataframe(merged)
    else:
        st.info("No lists were extracted. The agent may need to search for more specific URLs.")
else:
    st.info("Enter a topic to get top lists.")