import asyncio
import json
from app import get_filter_key_names_llm, get_user_filters
from gptSearch import run_streaming_search
from app import tavily_query

user_filters=get_user_filters()
filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
openperplex_json_str=asyncio.run(run_streaming_search(filter_key_values, suggested_query))
tavily_json_str=tavily_query(filter_key_values, suggested_query)

tavily_list = json.loads(tavily_json_str)
openperplex_list = json.loads(openperplex_json_str)
merged_list = tavily_list + openperplex_list  # simple merge, duplicates not removed


# --- ENRICHMENT SECTION ---

import re
from app import search_google, extract_company_info_with_llm


def enrich_companies_with_person_info(merged_list, roles=["CEO", "CTO"], extra_roles=None):
	"""
	For each company in merged_list, search for CEO/CTO/other roles using Google Search API and LLM (same model/provider as app.py).
	LLM is prompted to return a structured JSON for all requested roles at once, using the top search snippets as context.
	Adds info and source to each company dict.
	"""
	if extra_roles:
		roles = list(set(roles + extra_roles))
	from huggingface_hub import InferenceClient
	import os
	import json
	client = InferenceClient(provider="together", api_key=os.getenv("HF_TOKEN"))
	for company in merged_list:
		# Try to get company name from common keys
		name = None
		for k in ["name", "company_name", "Company Name", "Company"]:
			if k in company:
				name = company[k]
				break
		if not name:
			continue
		query = f"{' and '.join(roles)} of {name}"
		results = search_google(query)
		# Gather top 3 snippets
		snippets = " ".join([res.get("snippet", "") for res in results.get("organic_results", [])[:3]])
		# Prompt LLM for all roles at once, ask for JSON output
		prompt = (
			f"You are a business information assistant.\n"
			f"Given the following web search snippets, extract the names of the following roles for the company '{name}': {', '.join(roles)}.\n"
			f"Return a JSON object with each role as a key, and value as an object with 'name' and 'source' fields.\n"
			f"If a role is not found, set its value to an empty string.\n"
			f"Use only the information in the snippets.\n"
			f"Snippets: {snippets}\n"
			f"Example output: {{\n  'CEO': {{'name': 'John Doe', 'source': 'snippet or url'}}, 'CTO': {{'name': 'Jane Smith', 'source': 'snippet or url'}} }}\n"
			f"Return only the JSON object."
		)
		messages = [{"role": "user", "content": prompt}]
		try:
			completion = client.chat.completions.create(model="openai/gpt-oss-20b", messages=messages)
			response_text = completion.choices[0].message.content
			# Try to extract JSON from code block or plain string
			import re
			code_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
			if code_match:
				json_str = code_match.group(1)
			else:
				json_str = response_text
			try:
				role_info = json.loads(json_str.replace("'", '"'))
			except Exception:
				role_info = {}
			for role in roles:
				val = role_info.get(role)
				if isinstance(val, dict) and val.get("name"):
					company[role] = val
				elif isinstance(val, str) and val:
					company[role] = {"name": val, "source": "LLM+Google Search"}
		except Exception as e:
			pass
	return merged_list



# Example usage: ask user for extra roles if needed
roles = ["CEO", "CTO"]
extra_roles = []
print("\nIf you want to search for other roles (e.g. CFO, Founder), enter them one by one. Type 'done' when finished.")
while True:
	role_input = input()
	if role_input.lower() == 'done':
		break
	if role_input.strip():
		extra_roles.append(role_input.strip())


# Move enrichment after merged_list is defined
enriched_list = enrich_companies_with_person_info(merged_list, roles=roles, extra_roles=extra_roles)

print('\nFinal enriched company list:')
print(json.dumps(enriched_list, indent=2, ensure_ascii=False))


