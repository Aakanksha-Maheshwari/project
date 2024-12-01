import streamlit as st
import requests
import json
from multiprocessing import Process, Manager, Queue
import openai
from bespokelabs import BespokeLabs, DefaultHttpxClient
import httpx
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import os


# Initialize Clients and Keys
client = chromadb.PersistentClient()
openai.api_key = st.secrets["openai"]["api_key"]
bespoke_key = st.secrets["bespoke_labs"]["api_key"]
bl = BespokeLabs(auth_token=bespoke_key)

# API URLs
ALPHA_VANTAGE_KEY = st.secrets['alpha_vantage']['api_key']
news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_KEY}&limit=50"
tickers_url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_KEY}"

# App Title
st.title("Multi-Agent Financial Newsletter Application")


### Helper Functions ###
def fetch_data(api_url):
    """Fetch data from an API."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        st.write(f"Fetched Data: {api_url}")  # Debugging log
        return data
    except Exception as e:
        st.error(f"Error fetching data from {api_url}: {e}")
        return None


def store_data_in_chromadb(data, collection_name):
    """Store data in ChromaDB."""
    if not data:
        st.warning(f"No data available for {collection_name}.")
        return

    collection = client.get_or_create_collection(collection_name)
    records = data.get("feed", []) + data.get("top_gainers", []) + data.get("top_losers", []) + data.get("most_actively_traded", [])
    
    for i, record in enumerate(records, start=1):
        try:
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(record)],
                metadatas=[{"source": record.get("source", "N/A")}],
            )
        except Exception as e:
            st.error(f"Error storing data in {collection_name}: {e}")
    st.success(f"Stored {len(records)} records in {collection_name}.")


def retrieve_data(collection_name, query_text, top_k=5):
    """Retrieve data from ChromaDB."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        return [json.loads(doc) for doc in results["documents"]]
    except Exception as e:
        st.error(f"Error retrieving data from {collection_name}: {e}")
        return []


def call_openai(prompt):
    """Call OpenAI for text generation."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI: {e}")
        return "Error generating response."


def assess_accuracy(newsletter, data):
    """Assess newsletter accuracy using Bespoke Labs."""
    try:
        response = bl.minicheck.factcheck.create(claim=newsletter, context=json.dumps(data))
        return round(response.support_prob * 100, 2) if hasattr(response, "support_prob") else 0
    except Exception as e:
        st.error(f"Error assessing accuracy: {e}")
        return 0


### Agents ###
class CompanyAnalystAgent:
    """Analyze company performance."""
    def process(self):
        data = fetch_data(news_url)
        if data:
            store_data_in_chromadb(data, "company_data")
            return retrieve_data("company_data", "Company performance insights")
        return []


class MarketTrendsAnalystAgent:
    """Analyze market trends."""
    def process(self):
        data = fetch_data(tickers_url)
        if data:
            store_data_in_chromadb(data, "market_data")
            return retrieve_data("market_data", "Market trends insights")
        return []


class RiskManagementAgent:
    """Analyze risks based on company and market data."""
    def process(self, company_data, market_data):
        if not company_data and not market_data:
            return "No risk data available."
        prompt = f"Analyze the following data for risks:\nCompany Data: {json.dumps(company_data)}\nMarket Data: {json.dumps(market_data)}"
        return call_openai(prompt)


class NewsletterAgent:
    """Generate the financial newsletter."""
    def process(self, company_summary, market_summary, risk_summary):
        prompt = f"""
        Create a financial newsletter:
        - Company Insights: {company_summary}
        - Market Trends: {market_summary}
        - Risk Analysis: {risk_summary}
        """
        return call_openai(prompt)


### Main Logic ###
if st.button("Generate Financial Newsletter"):
    try:
        st.subheader("Company Analyst Agent")
        company_agent = CompanyAnalystAgent()
        company_data = company_agent.process()
        st.write(f"Retrieved {len(company_data)} company records.")

        st.subheader("Market Trends Analyst Agent")
        market_agent = MarketTrendsAnalystAgent()
        market_data = market_agent.process()
        st.write(f"Retrieved {len(market_data)} market records.")

        st.subheader("Risk Management Agent")
        risk_agent = RiskManagementAgent()
        risk_summary = risk_agent.process(company_data, market_data)
        st.write(f"Risk Summary: {risk_summary}")

        st.subheader("Newsletter Agent")
        newsletter_agent = NewsletterAgent()
        newsletter = newsletter_agent.process(company_data, market_data, risk_summary)
        st.markdown(newsletter)

        st.subheader("Accuracy Assessment")
        accuracy = assess_accuracy(newsletter, company_data + market_data)
        st.success(f"Newsletter Accuracy: {accuracy}%")
    except Exception as e:
        st.error(f"Error during processing: {e}")
