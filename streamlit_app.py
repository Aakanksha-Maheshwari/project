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

import streamlit as st
import requests
import json
from bespokelabs import BespokeLabs
import chromadb
import openai

# Initialize Clients and Keys
client = chromadb.PersistentClient()
openai.api_key = st.secrets["openai"]["api_key"]
bespoke_key = st.secrets["bespoke_labs"]["api_key"]
bl = BespokeLabs(auth_token=bespoke_key)

# API URLs
news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={st.secrets['alpha_vantage']['api_key']}&limit=50"
tickers_url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={st.secrets['alpha_vantage']['api_key']}"

# App Title
st.title("Multi-Agent Financial Newsletter Application")


### Helper Functions ###
def fetch_data(api_url):
    """Fetch data from an API."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def store_data_in_chromadb(data, collection_name):
    """Store data in ChromaDB."""
    if not data or "feed" not in data:
        st.warning(f"No valid data for {collection_name}")
        return

    collection = client.get_or_create_collection(collection_name)
    for i, item in enumerate(data["feed"], start=1):
        try:
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}],
            )
        except Exception as e:
            st.error(f"Error storing data in {collection_name}: {e}")


def retrieve_data(collection_name, query_text, top_k=5):
    """Retrieve data from ChromaDB."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        return [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"]]
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return []


def call_openai(prompt):
    """Call OpenAI for generating text."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI: {e}")
        return "Error generating response."


def assess_accuracy(newsletter, data):
    """Assess newsletter accuracy using Bespoke Labs."""
    try:
        response = bl.minicheck.factcheck.create(claim=newsletter, context=json.dumps(data))
        return round(response.support_prob * 100, 2) if hasattr(response, "support_prob") else 0
    except Exception as e:
        st.error(f"Error with Bespoke Labs: {e}")
        return 0


### Agents ###
class CompanyAnalystAgent:
    """Analyze company performance."""
    def process(self):
        data = fetch_data(news_url)
        store_data_in_chromadb(data, "company_data")
        return retrieve_data("company_data", "Company performance insights")


class MarketTrendsAnalystAgent:
    """Analyze market trends."""
    def process(self):
        data = fetch_data(tickers_url)
        store_data_in_chromadb(data, "market_data")
        return retrieve_data("market_data", "Market trends insights")


class RiskManagementAgent:
    """Analyze risks based on company and market data."""
    def process(self, company_data, market_data):
        if not company_data and not market_data:
            return "No risk data available."
        
        prompt = f"""
        Analyze the following data for risks:
        Company Data: {json.dumps(company_data)}
        Market Data: {json.dumps(market_data)}
        """
        return call_openai(prompt)


class NewsletterAgent:
    """Generate the newsletter."""
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
    # Company Analyst Agent
    st.subheader("Company Analyst Agent")
    company_agent = CompanyAnalystAgent()
    company_data = company_agent.process()
    st.write(f"Company Data: {len(company_data)} records retrieved.")

    # Market Trends Analyst Agent
    st.subheader("Market Trends Analyst Agent")
    market_agent = MarketTrendsAnalystAgent()
    market_data = market_agent.process()
    st.write(f"Market Data: {len(market_data)} records retrieved.")

    # Risk Management Agent
    st.subheader("Risk Management Agent")
    risk_agent = RiskManagementAgent()
    risk_summary = risk_agent.process(company_data, market_data)
    st.write(f"Risk Summary: {risk_summary}")

    # Newsletter Agent
    st.subheader("Newsletter Agent")
    newsletter_agent = NewsletterAgent()
    newsletter = newsletter_agent.process(company_data, market_data, risk_summary)
    st.markdown(newsletter)

    # Accuracy Assessment
    st.subheader("Assessing Newsletter Accuracy")
    combined_data = company_data + market_data
    accuracy = assess_accuracy(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy}%")
