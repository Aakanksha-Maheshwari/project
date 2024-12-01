import streamlit as st
import requests
import json
import openai
from bespokelabs import BespokeLabs, DefaultHttpxClient
import httpx
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]
bespoke_key = st.secrets["bespoke_labs"]["api_key"]

# Initialize Bespoke Labs with DefaultHttpxClient
bl = BespokeLabs(
    auth_token=bespoke_key
)

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, OpenAI GPT-4, and Bespoke Labs")

### Helper Functions ###

def retrieve_from_chromadb(collection_name, query, top_k=5):
    """Retrieve relevant documents from ChromaDB."""
    collection = client.get_or_create_collection(collection_name)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents']
    except Exception as e:
        st.error(f"Error retrieving data from ChromaDB: {e}")
        return []

def validate_rag_data(rag_data, source):
    """Validate RAG data for completeness and relevance."""
    if not rag_data:
        st.warning(f"No data retrieved from {source}.")
    else:
        st.write(f"Retrieved {len(rag_data)} items from {source}.")
    return rag_data

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "Error generating response."

def assess_accuracy_with_bespoke(newsletter, rag_data):
    """Assess the accuracy of the newsletter using Bespoke Labs."""
    try:
        response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(rag_data)
        )
        support_prob = getattr(response, "support_prob", None)
        if support_prob is None:
            st.error("Bespoke Labs response does not contain 'support_prob'.")
            return 0

        st.write("Bespoke Labs Response:", response)
        return round(support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0

def fetch_and_update_news_data():
    """Fetch news data from the API and update ChromaDB."""
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        st.write("News API Response:", data)
        if 'feed' in data:
            collection = client.get_or_create_collection("news_sentiment_data")
            for i, item in enumerate(data['feed'], start=1):
                collection.add(
                    ids=[str(i)],
                    documents=[json.dumps(item)],
                    metadatas=[{"source": item.get("source", "N/A")}]
                )
            st.success("News data updated in ChromaDB.")
        else:
            st.error("No news data found in API response.")
    except Exception as e:
        st.error(f"Error updating news data: {e}")

def fetch_and_update_ticker_trends_data():
    """Fetch ticker trends data from the API and update ChromaDB."""
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()
        st.write("Ticker Trends API Response:", data)
        if "top_gainers" in data:
            collection = client.get_or_create_collection("ticker_trends_data")
            collection.add(
                ids=[str(i) for i in range(len(data["top_gainers"]))],
                documents=[json.dumps(item) for item in data["top_gainers"]],
                metadatas=[{"type": "gainer"} for _ in data["top_gainers"]]
            )
            st.success("Ticker trends data updated in ChromaDB.")
        else:
            st.error("Invalid data format received from API.")
    except Exception as e:
        st.error(f"Error updating ticker trends data: {e}")

### Newsletter Generation ###

def generate_newsletter_with_accuracy():
    """Generate the newsletter using RAG and measure its accuracy."""
    company_insights = retrieve_from_chromadb("news_sentiment_data", "Company performance insights", top_k=5)
    company_insights = validate_rag_data(company_insights, "News Sentiment Data")

    market_trends = retrieve_from_chromadb("ticker_trends_data", "Market trends insights", top_k=5)
    market_trends = validate_rag_data(market_trends, "Ticker Trends Data")

    if not company_insights or not market_trends:
        st.error("Insufficient RAG data for newsletter generation.")
        return

    summarized_insights = call_openai_gpt4(f"""
    Summarize the following company insights for a professional newsletter:
    {company_insights}
    """)

    summarized_trends = call_openai_gpt4(f"""
    Summarize the following market trends, including top gainers and losers, for a professional newsletter:
    {market_trends}
    """)

    prompt = f"""
    Generate a professional daily newsletter with:
    - Key Company Insights: {summarized_insights}
    - Major Market Trends: {summarized_trends}
    """
    newsletter = call_openai_gpt4(prompt)
    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    rag_context = company_insights + market_trends
    accuracy_score = assess_accuracy_with_bespoke(newsletter, rag_context)
    st.success(f"Newsletter Accuracy: {accuracy_score}%")

### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter and Measure Accuracy"):
    generate_newsletter_with_accuracy()
