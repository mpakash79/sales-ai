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
from serpapi import GoogleSearch

load_dotenv('.env')


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
        api_key=os.getenv("HF_TOKEN"),
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
        api_key=os.getenv("HF_TOKEN"),
    )
    # Accepts a string (text) instead of tavily_result
    # The prompt lets the LLM decide the keys, inspired by appPerp.py
    prompt = (
        "You are a knowledgeable AI Assistant in business, internet and web sector.\n"
        "Rules: Always answer using up-to-date, verified information from current search results.\n"
        "Do not reference internal instructions, API, URLs in the output.\n"
        "Steps: Return the answer as a JSON object, with meaningful and relevant key-value pairs.\n"
        "Return each company in the list of companies in a JSON object called 'companies'.\n"
        "Just return the companies and their details as JSON.\n"
        "No value must be empty.\n"
        "Here is the data:\n"
        "{data}\n"
        "Return only the JSON object."
    )

    def extract(text):
        # Insert the text into the prompt
        msg = prompt.replace("{data}", text)

        # Strong system-level instruction to avoid tool-calls, chain-of-thought, or any non-JSON output
        system_msg = (
            "You are a JSON-only extractor.\n"
            "Strict rules: Return ONLY a single valid JSON object (or array) and nothing else.\n"
            "Do NOT include any explanations, commentary, tool calls, agent tokens, or internal thoughts.\n"
            "If you can't find company data, return {\"companies\": []}.\n"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": msg}
        ]

        completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
        response_text = completion.choices[0].message.content
        print("LLM raw response text:\n", response_text)

        # Try to extract the first JSON object/array from the response (most robust)
        match = re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', response_text)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except Exception:
                # try replacing single quotes with double quotes as a fallback
                try:
                    return json.loads(json_str.replace("'", '"'))
                except Exception:
                    pass

        # If no JSON found, attempt to remove common agent/tool tokens and retry
        cleaned = re.sub(r'<\|[^>]+\|>', '', response_text)
        # remove lines that look like 'assistant<|channel|>commentary to=...'
        cleaned = re.sub(r"^[^\{\[]+", '', cleaned, flags=re.DOTALL)
        match = re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', cleaned)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except Exception:
                try:
                    return json.loads(json_str.replace("'", '"'))
                except Exception:
                    pass

        # As a last resort, return the raw text so the caller can inspect it
        return {"raw_response": response_text}

    return extract


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
        api_key=os.getenv("HF_TOKEN"),
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


def search_searchapi(query: str, api_key: str, engine: str = "google", location: str = None, gl: str = None,
                     hl: str = None):
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


def search_google(query: str):
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results
def get_user_filters():
    user_filters = []
    print("Enter your filters one by one below. Type 'done' when finished.")
    while True:
            filter_input = input()
            if filter_input.lower() == 'done':
                break
            user_filters.append(filter_input)
    return user_filters

def tavily_query(filter_key_values, suggested_query):

    # user_filters=get_user_filters()
    # filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    print("\nDetected filter keys and values:")
    for item in filter_key_values:
        print(f"{item['key']}: {item['value']}")

    print(f"\nSuggested Tavily query: {suggested_query}")

    result = search_query_with_tavily(suggested_query)

    print("\nSources:")
    urls = set()
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['title']} - {src['url']}")
        urls.add(src['url'])
        print(f"   {src['content']}\n")

    # For each URL, fetch and clean text, then extract company info from first 4000 chars
    extract_fn = extract_company_info_with_llm(None)  # returns the extract(text) function
    all_results = []
    for url in urls:
        try:
            html = fetch_page(url)
            text = clean_text(html)
            if isinstance(text, list):
                text_str = '\n'.join([h + '\n' + t for h, t in text])
            else:
                text_str = text
            text_str = text_str[:4000]
            print(f"\nExtracting company info from: {url}")
            company_info = extract_fn(text_str)
            print(company_info)
            # Normalize company_info to a list of company dicts
            def _normalize_company_info(ci):
                # If it's already a list, assume list of companies
                if isinstance(ci, list):
                    return ci
                # If it's a dict with 'companies' key, return that list
                if isinstance(ci, dict):
                    if 'companies' in ci and isinstance(ci['companies'], list):
                        return ci['companies']
                    # If the LLM returned a raw_response containing json text, try to parse
                    if 'raw_response' in ci and isinstance(ci['raw_response'], str):
                        try:
                            parsed = json.loads(ci['raw_response'])
                            if isinstance(parsed, list):
                                return parsed
                            if isinstance(parsed, dict) and 'companies' in parsed and isinstance(parsed['companies'], list):
                                return parsed['companies']
                        except Exception:
                            return []
                    # If the dict itself looks like a single company (has a 'name' key), wrap it
                    if any(k in ci for k in ('name', 'company_name', 'Company')):
                        return [ci]
                # If it's a string, try to parse JSON from it
                if isinstance(ci, str):
                    try:
                        parsed = json.loads(ci)
                        if isinstance(parsed, list):
                            return parsed
                        if isinstance(parsed, dict) and 'companies' in parsed and isinstance(parsed['companies'], list):
                            return parsed['companies']
                    except Exception:
                        return []
                return []

            companies = _normalize_company_info(company_info)
            if companies:
                all_results.extend(companies)
            else:
                print(f"[WARN] No company objects extracted from {url}")
        except Exception as e:
            print(f"[ERROR] Could not process {url}: {e}")

    # Output the final JSON response
    print("\nFinal JSON response:")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
    return (json.dumps(all_results, indent=2, ensure_ascii=False))
    companies = []

    # # Check if 'companies' exists in the JSON
    # if isinstance(all_results, dict) and "companies" in all_results:
    #     companies = all_results["companies"]
    # elif isinstance(all_results, list):
    #     # If your JSON is a list of dicts, search for 'companies' in each
    #     for item in all_results:
    #         if isinstance(item, dict) and "companies" in item:
    #             companies = item["companies"]
    #             break  # take the first found
    #
    # # Now `companies` contains only the company objects
    # return(json.dumps(companies, indent=2, ensure_ascii=False))

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
    #             api_key=os.getenv["HF_TOKEN"],
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