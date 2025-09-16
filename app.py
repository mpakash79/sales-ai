import os
import re
import json
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import huggingface_hub.inference._client as _hf_inference_client
from tavily import TavilyClient
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import threading
import math
# from serpapi import GoogleSearch

load_dotenv()

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
        try:
            session = _hf_inference_client.get_session()
            session.headers.update({"Accept-Encoding": "identity"})
        except Exception:
            raise
        return _safe_chat_completion(client, model, messages, retries=retries - 1)

def get_filter_key_names_llm(user_filters):
    """
    Use LLM to extract meaningful key names for each user filter.
    Returns a list of dicts: [{"key": ..., "value": ...}, ...]
    """
    client = InferenceClient(
        provider="together",
        api_key=os.environ["HF_TOKEN"],
    )    
    prompt = (
        "For each of the following user-entered filters, identify the most meaningful key name (e.g., Funding, Employee size, Category, Revenue, Location, Funded By, etc.) and its value. "
        "Return a JSON object with two fields: 'filters' (a JSON array of objects with 'key' and 'value'), and 'query' (a single meaningful string that would be an efficient search query for a web search engine to find companies matching all the filters). "
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
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), ""
        return [], ""

def extract_company_info_with_llm(tavily_result):
    client = InferenceClient(
        provider="together",
        api_key=os.environ["HF_TOKEN"],
    )
    sources_text = ""
    relevant_sources = [src for src in tavily_result.get('sources', [])]
    if relevant_sources:
        for src in relevant_sources:
            content = src.get('content', '')
            sources_text += (content + "\n\n") 

    import builtins
    filter_keys = getattr(builtins, 'filter_keys', None)

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

def search_query_with_tavily(query: str):
    """
    Perform a web search using Tavily and return the answer and sources.

    Parameters:
        query (str): The search query string.

    Returns:
        dict: A dictionary with 'answer' and 'sources' keys.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY environment variable not set.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, include_answer=True,
                             search_depth="advanced",  
                             include_raw_content=True,  
                             llm_options={"max_tokens": 4000})  # try increasing answer size)

    answer = response.get("answer", "No answer found.")
    sources = [
        {"title": res["title"], "url": res["url"], "content": res["content"]}
        for res in response.get("results", [])
    ]

    return {"answer": answer, "sources": sources}


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(" ", strip=True)


def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            print(f"[!] 403 blocked by {url}, retrying with Selenium...")
            return fetch_with_selenium(url)
        else:
            raise

def fetch_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/139.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)

    html = driver.page_source
    driver.quit()
    return html

def extract_json(text: str, keys: list[str]) -> dict:
    client = InferenceClient(
        provider="together",
        api_key=os.environ["HF_TOKEN"],
    )
    prompt = (
        f"Extract structured information about companies from the following text. "
        f"The text may contain tables. Parse tables and extract company details matching these keys: {keys}. "
        f"Return only a JSON array of objects, one per company, with keys: {keys}. Do not include any explanation, commentary, or extra text. Return only the JSON array.\n\n"
        f"Text: {text}"
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
    response_text = completion.choices[0].message.content

    try:
        match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return response_text
    except Exception:
        return response_text

def search_searchapi(query: str, api_key: str, engine: str = "google", location: str = None, gl: str = None, hl: str = None):
    """
    Query SearchAPI.io with basic parameters.
    
    Args:
        query: the search query (string)
        api_key: your SearchAPI.io API key
        engine: which engine (e.g., "google", "google_light", "youtube", etc.)
        location: optional: location parameter (city/country etc)
        gl: country code for geolocation (e.g. "uk", "us", etc.)
        hl: language code (e.g. "en", "en-gb", etc.)
    Returns:
        JSON result (as dictionary) if successful, else raises error or returns None
    """
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "q": query,
        "engine": engine,
        "api_key": api_key,
    }

    if location:
        params["location"] = location
    if gl:
        params["gl"] = gl
    if hl:
        params["hl"] = hl

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# def search_google(query: str):
#     params = {
#         "q": query,
#         "hl": "en",
#         "gl": "us",
#         "api_key": os.getenv("SERP_API_KEY")
#     }
#
#     search = GoogleSearch(params)
#     results = search.get_dict()
#     return results

def get_user_filters():
    user_filters = []
    print("Enter your filters one by one below. Type 'done' when finished.")
    while True:
        filter_input = input()
        if filter_input.lower() == 'done':
            break
        user_filters.append(filter_input)
    return user_filters

if __name__ == "__main__":



    filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    print("\nDetected filter keys and values:")
    for item in filter_key_values:
        print(f"{item['key']}: {item['value']}")

    print(f"\nSuggested Tavily query: {suggested_query}")

    import builtins

    builtins.filter_keys = [item['key'] for item in filter_key_values if item['key']]
    builtins.filter_keys.insert(0, 'Company Name')  

    print(builtins.filter_keys)

    result = search_query_with_tavily(suggested_query)

    # query = "list of companies that " + ", ".join(f for f in user_filters)
    # result = search_query_with_tavily(query)

    print("\nSources:")
    urls = []
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['title']} - {src['url']}")
        urls.append(src['url'])
        print(f"   {src['content']}\n")

    print("\nPerforming Google search with query:", suggested_query)
    result = search_searchapi(suggested_query, os.getenv("SEARCH_API_KEY"), engine="google", hl="en")
    
    if "organic_results" in result:
        for i, item in enumerate(result["organic_results"], start=1):
            print(f"{i}. {item.get('title')}")
            print(item.get('link'))
            print(item.get('snippet', ""))  
            print()
    else:
        print("No organic_results in response: ", result)

    data = search_google(suggested_query)
    print("performing serp search:....")
    print(data)

    if "organic_results" in data:
        for idx, result in enumerate(data["organic_results"], start=1):
            print(f"{idx}. {result.get('title')}")
            print(result.get("link"))
            print()
    else:
        print("No results found or API quota exceeded.")

    # Usage after Tavily result
    # company_info = extract_company_info_with_llm(result)
    # print(company_info)   

    # for url in urls:
    #     html = fetch_page(url)
    #     text = clean_text(html)

    #     # Chunk text into overlapping 4000-char sections with 200-char overlap
    #     max_len = 2500
    #     overlap = 200
    #     text_str = text if isinstance(text, str) else '\n'.join([h + '\n' + t for h, t in text])
    #     sections = []
    #     start = 0
    #     while start < len(text_str):
    #         end = min(start + max_len, len(text_str))
    #         section = text_str[start:end]
    #         sections.append(section)
    #         if end == len(text_str):
    #             break
    #         start = end - overlap

    #     results = [None] * len(sections)
    #     threads = []

    #     def run_extract(idx, section):
    #         results[idx] = extract_json(section, builtins.filter_keys)

    #     for idx, section in enumerate(sections):
    #         t = threading.Thread(target=run_extract, args=(idx, section))
    #         t.start()
    #         threads.append(t)
    #     for t in threads:
    #         t.join()

    #     # Flatten and collect all company dicts
    #     all_companies = []
    #     for out in results:
    #         if isinstance(out, list):
    #             all_companies.extend(out)
    #         elif isinstance(out, dict):
    #             all_companies.append(out)

    #     # Deduplicate/merge using LLM
    #     if all_companies:
    #         client = InferenceClient(
    #             provider="together",
    #             api_key=os.environ["HF_TOKEN"],
    #         )
    #         prompt = (
    #             f"Given the following extracted company info (may contain duplicates or partials), "
    #             f"deduplicate and merge them into a single clean JSON array. "
    #             f"If two objects refer to the same company, merge their fields. "
    #             f"Return only the JSON array.\n\n"
    #             f"Data: {json.dumps(all_companies, ensure_ascii=False)}"
    #         )
    #         messages = [
    #             {"role": "user", "content": prompt}
    #         ]
    #         completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
    #         response_text = completion.choices[0].message.content
    #         try:
    #             match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    #             if match:
    #                 deduped = json.loads(match.group(0))
    #             else:
    #                 deduped = all_companies
    #         except Exception:
    #             deduped = all_companies
    #         print(f"output for url {url} is: \n", json.dumps(deduped, indent=2, ensure_ascii=False))
    #     else:
    #         print(f"output for url {url} is: \n", all_companies)