import streamlit as st
import requests
import json
import openai
import os
from bespokelabs import curator
from pydantic import BaseModel, Field
from typing import List
from datasets import Dataset

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit App Title
st.title("RAG + GPT-4 + Curator Pipeline for Newsletter Generation and Accuracy Check")

### Helper Classes ###
class ClaimContext(BaseModel):
    claim: str = Field(description="Generated newsletter content")
    context: str = Field(description="RAG-based extracted data")

### Helper Functions ###
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
        news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
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
        tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'
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

### Curator Integration ###
def generate_newsletter_with_curator():
    """Generate newsletter using Curator and measure accuracy."""
    # Step 1: Fetch and process RAG-based data
    rag_data = retrieve_from_chromadb("news_sentiment_data", "Extract news insights", top_k=5)
    if not rag_data:
        st.error("No RAG data available.")
        return
    
    st.write("RAG Data Retrieved:", rag_data)

    # Step 2: Create synthetic dataset for newsletter generation
    dataset = Dataset.from_dict({"context": [json.dumps(rag_data)]})

    # Step 3: Define Curator pipeline
    class Newsletter(BaseModel):
        text: str = Field(description="Generated newsletter text")

    newsletter_generator = curator.Prompter(
        prompt_func=lambda row: f"Write a newsletter based on the following context:\n{row['context']}",
        model_name="gpt-4o-mini",
        response_format=Newsletter,
        parse_func=lambda row, result: [{"context": row["context"], "newsletter": result.text}],
    )

    # Step 4: Generate Newsletter
    synthetic_data = newsletter_generator(dataset)
    st.write("Generated Newsletter Dataset:", synthetic_data)

    if not synthetic_data:
        st.error("Newsletter generation failed.")
        return

    generated_newsletter = synthetic_data[0]["newsletter"]

    # Step 5: Measure Accuracy using Curator and Pydantic
    context = json.dumps(rag_data)
    claim = generated_newsletter

    accuracy_pipeline = curator.Prompter(
        prompt_func=lambda row: f"Measure the accuracy of the following claim:\nClaim: {row['claim']}\nContext: {row['context']}",
        model_name="gpt-4o-mini",
        response_format=ClaimContext,
        parse_func=lambda row, result: [{"claim": row["claim"], "accuracy": result.claim}],
    )

    accuracy_data = Dataset.from_dict({"claim": [claim], "context": [context]})
    accuracy_result = accuracy_pipeline(accuracy_data)

    st.write("Accuracy Measurement:", accuracy_result)

    st.subheader("Generated Newsletter")
    st.markdown(generated_newsletter)

    if accuracy_result:
        st.success(f"Measured Accuracy: {accuracy_result[0]['accuracy']}")

### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter and Measure Accuracy"):
    generate_newsletter_with_curator()
