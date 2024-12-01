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
st.title("Alpha Vantage Financial Newsletter with RAG and GPT-4")

### Helper Functions ###

def retrieve_and_summarize(collection_name, query_text, top_k=10):
    """Retrieve data from ChromaDB and summarize."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"] if doc]

        if not documents:
            return "No relevant data found.", []

        # Summarize the retrieved data
        summary_prompt = f"""
        Summarize the following data for a financial newsletter:
        Focus on key insights and trends:
        {json.dumps(documents)}
        """
        summary = call_openai_gpt4(summary_prompt)
        return summary, documents
    except Exception as e:
        st.error(f"Error retrieving from {collection_name}: {e}")
        return "Error in data retrieval.", []

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial newsletter generator."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "Error generating response."

def assess_accuracy(newsletter, relevant_data):
    """
    Assess the accuracy of the newsletter by comparing included insights with relevant data.
    """
    try:
        # Ensure relevant_data contains dictionaries
        parsed_data = [json.loads(item) if isinstance(item, str) else item for item in relevant_data]
        
        # Count the number of insights included in the newsletter
        included_insights = sum(
            1 for insight in parsed_data if insight.get('title') and insight['title'] in newsletter
        )
        total_insights = len(parsed_data)

        if total_insights == 0:
            return 0  # Avoid division by zero

        # Calculate accuracy as a percentage
        accuracy = (included_insights / total_insights) * 100
        return round(accuracy, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy: {e}")
        return 0

def fetch_and_store_data(api_url, collection_name):
    """Fetch data from API and store in ChromaDB."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data:
            st.warning("Empty response from API.")
            return

        collection = client.get_or_create_collection(collection_name)
        for i, item in enumerate(data.get("feed", []), start=1):
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
        st.success(f"Data from {collection_name} updated successfully.")
    except Exception as e:
        st.error(f"Error fetching data for {collection_name}: {e}")

### Newsletter Generation ###

def generate_newsletter():
    """Generate the financial newsletter."""
    news_summary, news_data = retrieve_and_summarize("news_sentiment_data", "Company performance insights", top_k=50)
    trends_summary, trends_data = retrieve_and_summarize("ticker_trends_data", "Market trends insights", top_k=20)

    if not news_data and not trends_data:
        st.error("No relevant data available for newsletter generation.")
        st.subheader("Fallback Newsletter")
        st.markdown("No relevant financial data available. Please check back later for updates.")
        return

    newsletter_prompt = f"""
    Generate a professional financial newsletter:
    Key Company Insights: {news_summary}
    Major Market Trends: {trends_summary}
    """
    newsletter = call_openai_gpt4(newsletter_prompt)

    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    combined_data = news_data + trends_data
    accuracy = assess_accuracy(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy}%")

### Main Buttons ###

if st.button("Fetch and Store News Data"):
    fetch_and_store_data(news_url, "news_sentiment_data")

if st.button("Fetch and Store Trends Data"):
    fetch_and_store_data(tickers_url, "ticker_trends_data")

if st.button("Generate Newsletter"):
    generate_newsletter()
