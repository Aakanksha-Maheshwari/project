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

# Initialize Bespoke Labs
bl = BespokeLabs(auth_token=bespoke_key)

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Improved RAG System with Alpha Vantage, OpenAI, and Bespoke Labs")

### Helper Functions ###

def fetch_and_store_data(api_url, collection_name, key="feed"):
    """Fetch data from API and store it in ChromaDB."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        st.write(f"Raw API Response for {collection_name}:", data)  # Debugging

        if key in data and data[key]:
            collection = client.get_or_create_collection(collection_name)
            for i, item in enumerate(data[key], start=1):
                collection.add(
                    ids=[str(i)],
                    documents=[json.dumps(item)],
                    metadatas=[{"source": item.get("source", "Unknown")}]
                )
            st.success(f"Data updated in {collection_name}. Total documents: {collection.count()}")
        else:
            st.warning(f"No '{key}' key or empty response for {collection_name}.")
    except Exception as e:
        st.error(f"Error fetching or updating {collection_name}: {e}")

def retrieve_data(collection_name, query_text, top_k=10):
    """Retrieve data from ChromaDB based on a query."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        st.write(f"Raw results from {collection_name}:", results)  # Debugging
        return [json.loads(doc) if isinstance(doc, str) else doc for doc in results['documents'] if doc]
    except Exception as e:
        st.error(f"Error retrieving data from {collection_name}: {e}")
        return []

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional assistant generating financial newsletters."},
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

        st.write("Bespoke Labs Response:", response)  # Debugging
        return round(support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0

def generate_newsletter():
    """Generate a professional newsletter using retrieved data and LLM."""
    news_data = retrieve_data("news_sentiment_data", "Company performance insights", top_k=10)
    trends_data = retrieve_data("ticker_trends_data", "Market trends insights", top_k=10)

    if not news_data and not trends_data:
        st.error("No relevant data available for newsletter generation.")
        st.subheader("Fallback Newsletter")
        st.markdown("No relevant financial data available. Please check back later for updates.")
        return

    # Prepare context for LLM
    summarized_news = call_openai_gpt4(f"Summarize the following news data:\n{json.dumps(news_data)}")
    summarized_trends = call_openai_gpt4(f"Summarize the following market trends:\n{json.dumps(trends_data)}")

    # Generate newsletter
    newsletter_prompt = f"""
    Generate a professional financial newsletter with:
    1. Key Company Insights: {summarized_news}
    2. Major Market Trends: {summarized_trends}
    """
    newsletter = call_openai_gpt4(newsletter_prompt)
    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    # Assess accuracy
    combined_data = news_data + trends_data
    accuracy_score = assess_accuracy_with_bespoke(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy_score}%")

### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_store_data(news_url, "news_sentiment_data", key="feed")

if st.button("Fetch and Store Ticker Trends Data"):
    fetch_and_store_data(tickers_url, "ticker_trends_data", key="top_gainers")

if st.button("Generate Newsletter and Measure Accuracy"):
    generate_newsletter()
