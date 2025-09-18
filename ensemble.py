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


print('Final tavily and gpt search queries:')
print(json.dumps(merged_list, indent=2, ensure_ascii=False))


