import streamlit as st
import requests
import json
import openai
from bespokelabs import BespokeLabs
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

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
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None

def store_data_in_chromadb(data, collection_name, record_keys):
    """Store structured data in ChromaDB."""
    if not data:
        st.warning(f"No valid data for {collection_name}.")
        return

    records = []
    for key in record_keys:
        if key in data:
            records.extend(data[key])

    if not records:
        st.warning(f"No records found for keys: {record_keys}")
        return

    collection = client.get_or_create_collection(collection_name)
    for i, record in enumerate(records, start=1):
        try:
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(record)],
                metadatas=[{"source": record.get("source", "N/A")}]
            )
        except Exception as e:
            st.error(f"Error storing record in {collection_name}: {e}")

    st.success(f"Stored {len(records)} records in {collection_name}.")

def retrieve_data(collection_name, query_text, top_k=5):
    """Retrieve data from ChromaDB."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        documents = results.get("documents", [])
        return [json.loads(doc) if isinstance(doc, str) else doc for doc in documents]
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
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI: {e}")
        return "Error generating response."

def summarize_highlights(data, category):
    """Summarize highlights for company or market data."""
    if not data:
        return f"No {category} data available."

    highlights = []
    for record in data:
        if isinstance(record, dict):
            name = record.get("ticker", record.get("title", "Unknown"))
            sentiment = record.get("overall_sentiment_label", "Neutral")
            details = record.get("summary", "Details not available.")
            highlights.append(f"{name}: {sentiment} - {details}")
        else:
            st.warning(f"Invalid record format: {record}")

    if not highlights:
        return f"No valid {category} data available for summarization."

    return "\n".join(highlights[:5])  # Limit to top 5 highlights

def summarize_risks(company_data, market_data):
    """Analyze and summarize risks."""
    if not company_data and not market_data:
        return "No risk data available."

    prompt = f"""
    Analyze the following data for risks:
    - Company Data: {json.dumps(company_data[:5], indent=2)} 
    - Market Data: {json.dumps(market_data[:5], indent=2)}
    """
    return call_openai(prompt)

def assess_accuracy_with_bespoke(newsletter, company_data, market_data):
    """Assess accuracy using Bespoke Labs."""
    try:
        context = f"""
        Company Highlights:
        {summarize_highlights(company_data, 'Company')}
        
        Market Highlights:
        {summarize_highlights(market_data, 'Market')}
        """
        response = bl.minicheck.factcheck.create(claim=newsletter, context=context)
        support_prob = getattr(response, "support_prob", None)
        if support_prob is None:
            st.error("No support probability found in Bespoke Labs response.")
            return 0
        return round(support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0


### Agents ###

class CompanyAnalystAgent:
    """Analyze company performance."""
    def process(self):
        data = fetch_data(news_url)
        if data:
            store_data_in_chromadb(data, "company_data", ["feed"])
            return retrieve_data("company_data", "Company performance insights")
        return []

class MarketTrendsAnalystAgent:
    """Analyze market trends."""
    def process(self):
        data = fetch_data(tickers_url)
        if data:
            store_data_in_chromadb(data, "market_data", ["top_gainers", "top_losers", "most_actively_traded"])
            return retrieve_data("market_data", "Market trends insights")
        return []


### Main Logic ###

if st.button("Generate Financial Newsletter"):
    try:
        # Company Analyst Agent
        st.subheader("Company Analyst Agent")
        company_agent = CompanyAnalystAgent()
        company_data = company_agent.process()
        company_summary = summarize_highlights(company_data, "Company")
        st.markdown(f"### Company Highlights\n{company_summary}")

        # Market Trends Analyst Agent
        st.subheader("Market Trends Analyst Agent")
        market_agent = MarketTrendsAnalystAgent()
        market_data = market_agent.process()
        market_summary = summarize_highlights(market_data, "Market")
        st.markdown(f"### Market Highlights\n{market_summary}")

        # Risk Analysis
        st.subheader("Risk Management Summary")
        risk_summary = summarize_risks(company_data, market_data)
        st.write(risk_summary)

        # Newsletter Generation
        st.subheader("Generated Financial Newsletter")
        newsletter = call_openai(f"""
        Generate a financial newsletter with:
        - Company Highlights: {company_summary}
        - Market Highlights: {market_summary}
        - Risk Analysis: {risk_summary}
        """)
        st.markdown(newsletter)

        # Accuracy Assessment
        st.subheader("Accuracy Assessment")
        accuracy = assess_accuracy_with_bespoke(newsletter, company_data, market_data)
        st.success(f"Newsletter Accuracy: {accuracy}%")

    except Exception as e:
        st.error(f"Error during processing: {e}")
