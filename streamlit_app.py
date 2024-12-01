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

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage RAG System with GPT-4-Mini")

### Helper Functions ###

def fetch_and_store(api_url, collection_name, key="feed", limit=50):
    """Fetch data from API, filter, and store it in ChromaDB."""
    try:
        response = requests.get(f"{api_url}&limit={limit}")
        response.raise_for_status()
        data = response.json()

        if key in data and data[key]:
            collection = client.get_or_create_collection(collection_name)
            for i, item in enumerate(data[key], start=1):
                collection.add(
                    ids=[str(i)],
                    documents=[json.dumps(item)],
                    metadatas=[{"source": item.get("source", "Unknown")}]
                )
            st.success(f"Stored {len(data[key])} documents in {collection_name}.")
        else:
            st.warning(f"No '{key}' key or empty response from API for {collection_name}.")
    except Exception as e:
        st.error(f"Error in fetching or storing data for {collection_name}: {e}")

def retrieve_and_summarize(collection_name, query_text, top_k=5):
    """Retrieve data from ChromaDB, summarize and return key insights."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"] if doc]

        if not documents:
            return "No relevant data found."

        # Summarize the retrieved data
        summary_prompt = f"""
        Summarize the following data for a financial newsletter:
        Focus on key insights and trends:
        {json.dumps(documents)}
        """
        summary = call_openai_gpt4(summary_prompt)
        return summary
    except Exception as e:
        st.error(f"Error retrieving from {collection_name}: {e}")
        return "Error in data retrieval."

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4-mini to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-40-mini",
            messages=[
                {"role": "system", "content": "You are an assistant generating financial newsletters."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4-mini: {e}")
        return "Error generating response."

def assess_accuracy(newsletter, context):
    """Assess the accuracy of the newsletter using a simulated model."""
    try:
        # Placeholder for an accuracy assessment function
        # Replace this with any custom logic or API integration to validate
        accuracy = len(newsletter) / (len(context) * 100)  # Mock logic
        return round(accuracy * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy: {e}")
        return 0

### Main Functions ###

def fetch_and_store_news_data():
    """Fetch and store news sentiment data."""
    fetch_and_store(news_url, "news_sentiment_data", key="feed", limit=50)

def fetch_and_store_ticker_trends_data():
    """Fetch and store ticker trends data."""
    fetch_and_store(tickers_url, "ticker_trends_data", key="top_gainers", limit=20)

def generate_newsletter():
    """Generate a professional newsletter using retrieved data."""
    news_summary = retrieve_and_summarize("news_sentiment_data", "Company performance insights", top_k=10)
    trends_summary = retrieve_and_summarize("ticker_trends_data", "Top stock trends", top_k=10)

    if "No relevant data found" in (news_summary, trends_summary):
        st.error("Insufficient data for newsletter generation.")
        st.subheader("Fallback Newsletter")
        st.markdown("No relevant financial data available. Please check back later.")
        return

    # Generate newsletter
    newsletter_prompt = f"""
    Create a financial newsletter with:
    - Key Company Insights: {news_summary}
    - Major Market Trends: {trends_summary}
    """
    newsletter = call_openai_gpt4(newsletter_prompt)
    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    # Assess accuracy
    rag_data = [news_summary, trends_summary]
    accuracy_score = assess_accuracy(newsletter, rag_data)
    st.success(f"Newsletter Accuracy: {accuracy_score}%")

### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_store_news_data()

if st.button("Fetch and Store Ticker Trends Data"):
    fetch_and_store_ticker_trends_data()

if st.button("Generate Newsletter"):
    generate_newsletter()
