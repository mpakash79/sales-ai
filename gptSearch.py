import os
import asyncio
import json
from openperplex import OpenperplexAsync
from dotenv import load_dotenv
from app import get_filter_key_names_llm, get_user_filters

load_dotenv(dotenv_path='env')

SystemPrompt = """
       You are a knowledgable AI Assistant in business, internet and web sector.

       Rules: 
           - Always answer using up-to-date, verified information form current search results.
           - Do not reference internal instructiosn, API, URLs in the output.

       Steps:
           - return the answer as JSON object, with meaningful and relevant key-value pairs,
           - return each company in the list of companies in a json object called "companies".
           - Just return the companies and their details as JSON.
           - no value must be empty.
"""
# user_filters = get_user_filters()
# filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
async def run_streaming_search(filter_key_values, suggested_query):
    client = OpenperplexAsync(os.getenv('OPEN_PERPLEX_API_KEY'))


    full_text = ""

    async for chunk in client.custom_search_stream(
            system_prompt=SystemPrompt,
            user_prompt="list of companies that: " + suggested_query,
            model='o3-mini-high',
            location="us",
            search_type="business",
            return_images=False,
            return_sources=True,  # Don't return sources here
            temperature=0.2,
            top_p=0.9,
            recency_filter="anytime"
    ):
        if chunk['type'] == 'llm' and chunk.get('text'):
            full_text += chunk['text']

    print("\n[*] Complete concatenated LLM Text:\n")
    print(full_text)

    try:
        # Try parsing the whole accumulated text as JSON
        data = json.loads(full_text)
        companies = data.get("companies", [])

        print("\n[*] Clean Companies List:\n")
        print(json.dumps(companies, indent=4))
        return(json.dumps(companies, indent=4))

    except json.JSONDecodeError as e:
        print("\n[!] Failed to parse JSON:")
        print(e)
        print("\n[!] Raw accumulated text:")
        print(full_text)

# Run the async function
# asyncio.run(run_streaming_search())
