# --- add these imports near the top of your file ---
import os
import re
import json
import requests
from huggingface_hub import InferenceClient
# we import the internal get_session used by huggingface_hub
import huggingface_hub.inference._client as _hf_inference_client
from dotenv import load_dotenv
import builtins

load_dotenv('env')

user_filters = []


def _safe_chat_completion(client: InferenceClient, model: str, messages, retries: int = 1):
    """
    Call client.chat.completions.create(...) but if a ContentDecodingError occurs,
    update the huggingface-hub session to request identity (no gzip) and retry once.
    """
    try:
        return client.chat.completions.create(model=model, messages=messages)
    except requests.exceptions.ContentDecodingError as e:
        if retries <= 0:
            raise
        # Get the session object used internally by huggingface_hub and disable gzip decoding by requesting identity
        try:
            session = _hf_inference_client.get_session()
            # force future responses to be uncompressed
            session.headers.update({"Accept-Encoding": "identity"})
        except Exception:
            # If we cannot access internal session, re-raise original error
            raise
        # retry once
        return _safe_chat_completion(client, model, messages, retries=retries - 1)


def get_filter_key_names_llm(user_filters):
    """
    Use LLM to extract meaningful key names for each user filter.
    Returns a list of dicts: [{"key": ..., "value": ...}, ...]
    """
    client = InferenceClient(
        provider="together",
        api_key=os.getenv('HF_TOKEN'),
    )
    prompt = (
        "For each of the following user-entered filters, identify the most meaningful key name (e.g., Funding, Employee size, Category, Revenue, Location, Funded By, etc.) and its value. "
        "Return a JSON object with two fields: 'filters' (a JSON array of objects with 'key' and 'value'), and 'query' (a single string that would be an efficient search query for a web search engine to find companies matching all the filters). "
        "Filters: " + ", ".join(user_filters) + "\n"
        "Example: {\"filters\": [{\"key\": \"Funding\", \"value\": \"1M\"}, {\"key\": \"Employee size\", \"value\": \"less than 50\"}], \"query\": \"B2B SaaS companies funded 1M with less than 50 employees\"} "
        "Return only the JSON object."
    )

    messages = [{"role": "user", "content": prompt}]

    completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
    response_text = completion.choices[0].message.content
    try:
        result = json.loads(response_text)
        filters = result.get('filters', [])
        query = result.get('query', "")
        return filters, query
    except Exception:
        # fallback to previous behavior if not a valid object
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), ""
        return [], ""




def extract_company_info_with_llm(tavily_result):
    client = InferenceClient(
        provider="together",
        api_key=os.getenv('HF_TOKEN'),
    )
    # concatenate sources
    sources_text = ""
    relevant_sources = [src for src in tavily_result.get('sources', [])]
    if relevant_sources:
        for src in relevant_sources:
            # be cautious about extremely long content — truncate for safety
            content = src.get('content', '')
            sources_text += (content + "\n\n") 

    # dynamic keys
    filter_keys = getattr(builtins, 'filter_keys', None)
    # Always include 'Company Name' in the output keys
    if not filter_keys:
        filter_keys = ['Company Name', 'Category', 'Funding Type', 'Funding', 'Employee size']
    elif 'Company Name' not in filter_keys:
        filter_keys = ['Company Name'] + filter_keys
    keys_str = ", ".join(filter_keys)

    prompt = (
        f"Given the following information about SaaS companies, "
        f"extract a JSON array where each object has these keys: {keys_str}. "
        "For each company, always extract the company name and only fill keys for which data is available; set missing keys to empty string. "
        "Do not invent or guess missing data. Extract from tables and text. Only include companies that match the filters. "
        "Here is the data:\n"
        f"Source: {sources_text}\n"
        "Return only the JSON array."
    )
    messages = [{"role": "user", "content": prompt}]

    completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
    response_text = completion.choices[0].message.content
    print(response_text)
    try:
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            try:
                result_json = json.loads(match.group(0))
            except Exception:
                # Try to fix incomplete JSON (e.g., missing closing bracket)
                fixed = match.group(0)
                if not fixed.endswith(']'):
                    fixed += ']'
                try:
                    result_json = json.loads(fixed)
                except Exception:
                    result_json = [{k: "" for k in filter_keys}]
        else:
            result_json = [{k: "" for k in filter_keys}]
    except Exception:
        result_json = [{k: "" for k in filter_keys}]
    return result_json


import os
from tavily import TavilyClient


def search_query_with_tavily(query: str):
    """
    Perform a web search using Tavily and return the answer and sources.

    Parameters:
        query (str): The search query string.

    Returns:
        dict: A dictionary with 'answer' and 'sources' keys.
    """
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY environment variable not set.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, include_answer=True,
                             search_depth="advanced",  # or "deep"
                             include_raw_content=True,  # optional: fetch full-page content
                             llm_options={"max_tokens": 800})  # try increasing answer size)

    answer = response.get("answer", "No answer found.")
    sources = [
        {"title": res["title"], "url": res["url"], "content": res["content"]}
        for res in response.get("results", [])
    ]

    return {"answer": answer, "sources": sources}

def get_user_filters():
        print("Enter your filters one by one below. Type 'done' when finished.")
        while True:
            filter_input = input()
            if filter_input.lower() == 'done':
                break
            user_filters.append(filter_input)
        return user_filters

if __name__ == "__main__":


    get_user_filters()

    # Get meaningful key names for each filter using LLM
    filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    print("\nDetected filter keys and values:")
    for item in filter_key_values:
        print(f"{item['key']}: {item['value']}")

    print(f"\nSuggested Tavily query: {suggested_query}")

    # Store dynamic keys globally for use in LLM extraction
    import builtins
    builtins.filter_keys = [item['key'] for item in filter_key_values if item['key']]

    # Use the LLM-suggested query for Tavily search
    result = search_query_with_tavily(suggested_query)

    # Store dynamic keys globally for use in LLM extraction
    import builtins
    builtins.filter_keys = [item['key'] for item in filter_key_values if item['key']]

    # Build Tavily query from user filters (or from key-value pairs if needed)
    query = "list of companies that " + ", ".join(f for f in user_filters)
    result = search_query_with_tavily(query)

    # print("\nAnswer:", result["answer"])
    print("\nSources:")
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['title']} - {src['url']}")
        print(f"   {src['content']}\n")

    # Usage after Tavily result
    company_info = extract_company_info_with_llm(result)
    print(company_info)
    # Save to JSON
    OUTPUT_FILE = "companies.json"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(company_info, f, ensure_ascii=False, indent=4)
    print(f"[+] Saved {len(company_info)} companies to {OUTPUT_FILE}")

"""
    {
        Company Name:
        Category:
        Funding Type:
        Funding:
        Employee size: -1 (default)
    }

"""