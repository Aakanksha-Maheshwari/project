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

def retrieve_from_multiple_rags(query, collections, top_k=5):
    """Search multiple collections for relevant RAG data."""
    results = []
    for collection_name in collections:
        collection_results = retrieve_from_chromadb(collection_name, query, top_k)
        results.extend([doc for doc in collection_results if doc])  # Filter empty results
    return results

def update_chromadb(collection_name, data):
    """Update ChromaDB with new data."""
    collection = client.get_or_create_collection(collection_name)
    for i, item in enumerate(data, start=1):
        collection.add(
            ids=[str(i)],
            metadatas=[{"source": item.get("source", "N/A"), "time_published": item.get("time_published", "N/A")}],
            documents=[json.dumps(item)]
        )

def fetch_and_update_news_data():
    """Fetch news data from the API and update ChromaDB."""
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        st.write("News API Response:", data)  # Debugging
        if 'feed' in data:
            update_chromadb("news_sentiment_data", data['feed'])
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
        st.write("Ticker Trends API Response:", data)  # Debugging
        if "top_gainers" in data:
            combined_data = [
                {"type": "top_gainers", "data": data["top_gainers"]},
                {"type": "top_losers", "data": data["top_losers"]},
                {"type": "most_actively_traded", "data": data["most_actively_traded"]}
            ]
            update_chromadb("ticker_trends_data", combined_data)
            st.success("Ticker trends data updated in ChromaDB.")
        else:
            st.error("Invalid data format received from API.")
    except Exception as e:
        st.error(f"Error updating ticker trends data: {e}")

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

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."

def validate_rag_data(rag_data, source):
    """Validate RAG data for completeness and relevance."""
    if not rag_data:
        st.warning(f"No data retrieved from {source}.")
    else:
        st.write(f"Retrieved {len(rag_data)} items from {source}.")
    return rag_data

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

### Newsletter Generation ###

def generate_newsletter_with_accuracy():
    """Generate the newsletter using RAG and measure its accuracy."""
    company_insights = retrieve_from_chromadb("news_sentiment_data", "Detailed insights about company performance", top_k=5)
    company_insights = validate_rag_data(company_insights, "News Sentiment Data")

    market_trends = retrieve_from_chromadb("ticker_trends_data", "Key market trends and top performers", top_k=5)
    market_trends = validate_rag_data(market_trends, "Ticker Trends Data")

    if not company_insights or not market_trends:
        st.error("Insufficient RAG data for newsletter generation.")
        return

    summarized_insights = call_openai_gpt4(f"""
    Summarize the following insights for a newsletter:
    {company_insights}
    """)
    summarized_trends = call_openai_gpt4(f"""
    Summarize the market trends from the following data, focusing on top gainers, losers, and trends:
    {market_trends}
    """)

    prompt = f"""
    Create a professional daily newsletter. Include:
    1. Key Company Insights: {summarized_insights}
    2. Major Market Trends: {summarized_trends}
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
