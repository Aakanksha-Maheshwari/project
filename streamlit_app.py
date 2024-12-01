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

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]
bespoke_key = st.secrets["bespoke_labs"]["api_key"]

# Initialize Bespoke Labs
bespoke = BespokeLabs(auth_token=bespoke_key)

# API URLs
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Multi-Agent Financial Newsletter Application")

### Helper Functions ###
def fetch_data(api_url):
    """Fetch data from API."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def store_data_in_chromadb(data, collection_name):
    """Store data in ChromaDB."""
    if not data:
        st.warning(f"No data to store for collection: {collection_name}")
        return

    collection = client.get_or_create_collection(collection_name)
    for i, item in enumerate(data.get("feed", []), start=1):
        try:
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
        except Exception as e:
            st.error(f"Error storing data: {e}")

def retrieve_data(collection_name, query_text, top_k=5):
    """Retrieve data from ChromaDB."""
    collection = client.get_or_create_collection(collection_name)
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return [json.loads(doc) for doc in results["documents"]]

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error with GPT-4: {e}")
        return None

def evaluate_with_bespoke(newsletter, data):
    """Evaluate accuracy with Bespoke Labs."""
    try:
        response = bespoke.minicheck.factcheck.create(claim=newsletter, context=json.dumps(data))
        return response.support_prob * 100 if response.support_prob else 0
    except Exception as e:
        st.error(f"Error evaluating with Bespoke: {e}")
        return 0

### Multi-Agent System ###
class CompanyAnalystAgent:
    """Agent for analyzing company performance."""
    def process(self):
        collection_name = "company_data"
        data = fetch_data(news_url)
        store_data_in_chromadb(data, collection_name)
        return retrieve_data(collection_name, "Company performance insights")

class MarketTrendsAnalystAgent:
    """Agent for analyzing market trends."""
    def process(self):
        collection_name = "market_data"
        data = fetch_data(tickers_url)
        store_data_in_chromadb(data, collection_name)
        return retrieve_data(collection_name, "Market trends insights")

class RiskManagementAgent:
    """Agent for assessing risks."""
    def analyze(self, company_data, market_data):
        prompt = f"""
        Analyze the following data for potential risks:
        Company Data: {json.dumps(company_data)}
        Market Data: {json.dumps(market_data)}
        Provide insights on risks for investors.
        """
        return call_openai_gpt4(prompt)

class NewsletterAgent:
    """Agent for generating a financial newsletter."""
    def generate(self, company_summary, market_summary, risk_summary):
        prompt = f"""
        Create a financial newsletter combining:
        - Company Insights: {company_summary}
        - Market Trends: {market_summary}
        - Risk Analysis: {risk_summary}
        """
        return call_openai_gpt4(prompt)

### Main Logic ###
if st.button("Generate Financial Newsletter"):
    st.subheader("Company Analyst Agent")
    company_agent = CompanyAnalystAgent()
    company_data = company_agent.process()
    st.write(f"Company Data: {len(company_data)} records retrieved.")

    st.subheader("Market Trends Analyst Agent")
    market_agent = MarketTrendsAnalystAgent()
    market_data = market_agent.process()
    st.write(f"Market Data: {len(market_data)} records retrieved.")

    st.subheader("Risk Management Agent")
    risk_agent = RiskManagementAgent()
    risk_summary = risk_agent.analyze(company_data, market_data)
    st.write(f"Risk Summary: {risk_summary}")

    st.subheader("Newsletter Agent")
    newsletter_agent = NewsletterAgent()
    newsletter = newsletter_agent.generate(
        json.dumps(company_data),
        json.dumps(market_data),
        risk_summary
    )
    st.markdown(newsletter)

    st.subheader("Assessing Newsletter Accuracy")
    combined_data = company_data + market_data
    accuracy = evaluate_with_bespoke(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy}%")
