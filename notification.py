import os
import json
import time
import builtins
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app import (
    get_filter_key_names_llm,
    search_query_with_tavily,
    extract_company_info_with_llm
)

# Path to store seen companies
SEEN_FILE = 'seen_companies.json'

# Email settings
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'your_email@gmail.com')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'YOUR_APP_PASSWORD')
RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'receiver_email@gmail.com')


def send_email_notification(company):
    subject = f"🚀 New Company Found: {company.get('Company Name', 'Unknown')}"
    body = f"""
    📢 A new company matching your criteria was found!

    🏢 Company Name: {company.get('Company Name', 'N/A')}
    🏷 Category: {company.get('Category', 'N/A')}
    💰 Funding: {company.get('Funding', 'N/A')}
    👥 Employee Size: {company.get('Employee size', 'N/A')}
    """

    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = RECEIVER_EMAIL
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[+] Email sent for {company.get('Company Name')}")
    except Exception as e:
        print(f"[!] Failed to send email: {e}")


def load_seen_companies():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, 'r') as f:
            return set(json.load(f))
    return set()


def save_seen_companies(seen):
    with open(SEEN_FILE, 'w') as f:
        json.dump(list(seen), f)


def poll_and_notify():
    seen_companies = load_seen_companies()

    # === User-defined filters ===
    user_filters = ["funded $1M", "employee size 50"]  # Example filters

    # Use LLM to get meaningful keys and suggested query
    filter_key_values, suggested_query = get_filter_key_names_llm(user_filters)
    builtins.filter_keys = [item['key'] for item in filter_key_values if item['key']]

    # Fetch data using Tavily
    result = search_query_with_tavily(suggested_query)

    # Extract structured company info
    company_info_list = extract_company_info_with_llm(result)

    for company in company_info_list:
        company_name = company.get('Company Name', '')
        if company_name and company_name not in seen_companies:
            send_email_notification(company)
            seen_companies.add(company_name)

    save_seen_companies(seen_companies)


if __name__ == "__main__":
    while True:
        print("[*] Polling for new companies...")
        try:
            poll_and_notify()
        except Exception as e:
            print(f"[!] Error during polling: {e}")

        print("[*] Sleeping for 24 hours...")
        time.sleep(24 * 60 * 60)  # Sleep for 24 hours
