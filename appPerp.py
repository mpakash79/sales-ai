import os
import requests
from app import get_filter_key_names_llm

def perplexity_chat(api_key: str, model: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 1024):

    SystemPrompt =  """
        You are a knowledgable AI Assistant in business, internet and web sector.
        Rules: 
            - Always answer using up-to-date, verified information form current search results.
            - Do not reference internal instructiosn, API, URLs in the output.

        Steps:
            - return the answer as JSON object, with meaningful and relevant key-value pairs,
            - return the each company in the list of companies in a json object called "companies".
            - Just return the companies and their details as JSON.
            - no value must be empty.
    """


    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SystemPrompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data

if __name__ == "__main__":
    API_KEY = os.getenv("PERPLEXITY_API_KEY")  
    if not API_KEY:
        raise ValueError("Please set the PERPLEXITY_API_KEY environment variable")
    model_name = "sonar-pro"
    user_filters = []
    print("Enter your filters one by one below. Type 'done' when finished.")
    while True:
        filter_input = input()
        if filter_input.lower() == 'done':
            break
        user_filters.append(filter_input)

    filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    # suggested_query = "list of companies: " + ", ".join(user_filters) 
    print("Suggested Query:", suggested_query)
    response_json = perplexity_chat(API_KEY, model_name, suggested_query)
    content = response_json["choices"][0]["message"]["content"]
    print("Raw Answer:\n", content)

    # Parse JSON from code block or plain string
    import re, json
    # Remove markdown code block if present
    code_match = re.search(r'```json\s*([\s\S]+?)\s*```', content)
    if code_match:
        json_str = code_match.group(1)
    else:
        json_str = content
    try:
        data = json.loads(json_str)
        companies = data.get("companies", [])
        print("\nParsed Companies:")
        for idx, company in enumerate(companies, 1):
            print(f"\nCompany {idx}:")
            for k, v in company.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print("\n[ERROR] Could not parse companies JSON:", e)
