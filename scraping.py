import os

from scrapingbee import ScrapingBeeClient
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from app import _safe_chat_completion
import json
import re
import requests

from langchain_scraperapi.tools import ScraperAPIGoogleSearchTool

load_dotenv('env')



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
            "Return a JSON object with three fields: "
            "  - 'filters': a JSON array of objects with 'key' and 'value', "
            "  - 'urls': a JSON array of 3 to 5 URLs that are most likely to contain detailed information about companies matching all the filters, "
            "  - 'query': a single efficient search query string suitable for a web search engine to find companies matching the filters. "
            "Filters: " + ", ".join(user_filters) + "\n"
                                                    "Example: {"
                                                    "\"filters\": [{\"key\": \"Funding\", \"value\": \"1M\"}, {\"key\": \"Employee size\", \"value\": \"less than 50\"}], "
                                                    "\"urls\": [\"https://techcrunch.com/startups\", \"https://www.crunchbase.com/search/organization.companies\", \"https://www.producthunt.com/\" ], "
                                                    "\"query\": \"B2B SaaS companies funded 1M with less than 50 employees\""
                                                    "}\n"
                                                    "Return only the JSON object."
    )

    messages = [{"role": "user", "content": prompt}]

    completion = _safe_chat_completion(client, model="openai/gpt-oss-20b", messages=messages)
    response_text = completion.choices[0].message.content
    try:
        result = json.loads(response_text)
        filters = result.get('filters', [])
        query = result.get('query', "")
        urls=result.get('urls',"")
        return filters, query,urls
    except Exception:
        # fallback to previous behavior if not a valid object
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), ""
        return [], "", []

scrapingBeeClient = ScrapingBeeClient(api_key=os.getenv('SCRAPING_BEE'))


if __name__=='__main__':
    user_filters = []
    print("Enter your filters one by one below. Type 'done' when finished.")
    while True:
        filter_input = input()
        if filter_input.lower() == 'done':
            break
        user_filters.append(filter_input)

    filter_key_values, suggested_query, urls = get_filter_key_names_llm(user_filters)
    print("\nDetected filter keys and values:")
    for item in filter_key_values:
        print(f"{item['key']}: {item['value']}")

    print(f"\nSuggested Tavily query: {suggested_query}")


    payload = {'api_key': 'd8627ecc930ef73e44d8a21a7b726cf8', 'query':"List of companies that"+ suggested_query}
    r = requests.get('https://api.scraperapi.com/structured/google/search', params=payload)
    print(r.text)

    # for url in urls:
    #     print(url)
    # response = []
    # google_search = ScraperAPIGoogleSearchTool(scraperapi_api_key=os.getenv('SCRAPER_API'))
    # # os.environ["SCRAPERAPI_API_KEY"] = os.getenv('SCRAPER_API')
    #
    # # for url in urls:
    # results = google_search.invoke({
    #         "query": "List of Comapnies that "+suggested_query,
    #         "num": 20,
    #         "output_format": "json"
    #     })
    # print(results)
    #     res = scrapingBeeClient.get(
    #         url,
    #         params={
    #             'ai_query': "List of Names of all the companies that " + suggested_query
    #         }
    #     )
    #     response.append(res)
    #
    #
    # for res in response:
    #     if res.status_code==200:
    #         print('Status Code:', res.status_code)
    #         print('Content:', res.content)

