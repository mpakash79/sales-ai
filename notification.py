import os
import json
import time
import builtins
import smtplib
import re
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from appPerp import(
perplexity_chat
)


from app import (
    get_filter_key_names_llm,
    get_user_filters
)

load_dotenv(dotenv_path='env')


SEEN_FILE = 'companies.json'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'your_email@gmail.com')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'YOUR_APP_PASSWORD')
RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'receiver_email@gmail.com')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
MODEL_NAME = "sonar-pro"


def send_email_notification(company):
    subject = f"🚀 New Company Found: {company.get('name', 'Unknown')}"

    # Build body dynamically from all key-value pairs
    body_lines = ["📢 A new company matching your criteria was found!\n"]
    for key, value in company.items():
        # Convert key to readable format: replace underscores with spaces and capitalize words
        pretty_key = key.replace('_', ' ').title()
        body_lines.append(f"🏢 {pretty_key}: {value}")

    body = "\n".join(body_lines)

    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = RECEIVER_EMAIL
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[+] Email sent for {company.get('name')}")
    except Exception as e:
        print(f"[!] Failed to send email: {e}")


def load_seen_companies():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_seen_companies(companies):
    with open(SEEN_FILE, 'w') as f:
        json.dump(companies, f, indent=4)


def poll_and_notify():
    seen_companies = load_seen_companies()
    seen_company_names = {c.get('name', '').strip().lower() for c in seen_companies if isinstance(c, dict) and c.get('name')}

    user_filters = get_user_filters()
    filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    builtins.filter_keys = [item['key'] for item in filter_key_values if item['key']]

    print("[*] Suggested Query:", suggested_query)

    response_json = perplexity_chat(PERPLEXITY_API_KEY, MODEL_NAME, suggested_query)
    content = response_json["choices"][0]["message"]["content"]
    print("[*] Full API Response Content:")
    print(content)

    code_match = re.search(r'```json\s*([\s\S]+?)\s*```', content)
    json_str = code_match.group(1) if code_match else content

    print("[*] Extracted JSON String:")
    print(json_str)

    try:
        data = json.loads(json_str)
        new_companies = data.get("companies", [])

        print(f"[*] Found {len(new_companies)} companies from Perplexity API")

        updated_companies = seen_companies.copy()

        for company in new_companies:
            if not isinstance(company, dict):
                print(f"[!] Skipping invalid company entry (not a dict): {company}")
                continue

            company_name = company.get('name', '')
            if not company_name:
                print(f"[!] Skipping company with missing 'name': {company}")
                continue

            if company_name not in seen_company_names:
                send_email_notification(company)
                seen_company_names.add(company_name)
                updated_companies.append(company)

        save_seen_companies(updated_companies)
        print(f"[*] Total companies stored: {len(updated_companies)}")

    except json.JSONDecodeError:
        print("[!] Invalid JSON received from Perplexity API. Skipping entire batch.")
        print("[*] Raw API content:")
        print(content)
    except Exception as e:
        print(f"[!] Unexpected error: {e}")


if __name__ == "__main__":
    if not PERPLEXITY_API_KEY:
        raise ValueError("Please set PERPLEXITY_API_KEY environment variable")

    while True:
        print("[*] Polling for new companies...")
        try:
            poll_and_notify()
        except Exception as e:
            print(f"[!] Error during polling: {e}")

        print("[*] Sleeping for 24 hours...")
        time.sleep(24 * 60 * 60)
