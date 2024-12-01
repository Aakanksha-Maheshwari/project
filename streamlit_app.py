import streamlit as st
import requests
import json
from multiprocessing import Process, Manager, Queue
from openai import OpenAI
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
from multiprocessing import Process, Manager, Queue
from openai import OpenAI
import chromadb
import os

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai_api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client_openai = OpenAI(api_key=openai_api_key)

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Multi-Agent Financial Newsletter Generator with ChromaDB")

### Helper Functions ###

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for the given text using OpenAI."""
    try:
        text = text.replace("\n", " ")
        response = client_openai.embeddings.create(input=[text], model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def fetch_data_from_api(api_url):
    """Fetch data from Alpha Vantage API and return the JSON response."""
    try:
        st.info(f"Fetching data from {api_url}...")
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        st.write(f"API response: {data}")  # Debugging
        return data
    except Exception as e:
        st.error(f"Error fetching data from API: {e}")
        return None

def store_data_in_chromadb(api_url, collection_name):
    """Fetch data from API, generate embeddings, and store in ChromaDB."""
    data = fetch_data_from_api(api_url)
    if not data or "feed" not in data:
        st.warning(f"No valid data returned for {collection_name}.")
        return

    collection = client.get_or_create_collection(collection_name)
    for i, item in enumerate(data["feed"], start=1):
        summary = item.get('summary', '')
        if not summary:
            continue
        embedding = get_embedding(summary, model="text-embedding-ada-002")
        if embedding:
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
    st.success(f"Data successfully stored in ChromaDB for {collection_name}.")

def retrieve_data_from_chromadb(collection_name, query_text, top_k):
    """Retrieve data from ChromaDB using similarity search."""
    try:
        collection = client.get_or_create_collection(collection_name)
        query_embedding = get_embedding(query_text, model="text-embedding-ada-002")
        if not query_embedding:
            st.error("Failed to generate embedding for query text.")
            return []

        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"]]
        return documents
    except Exception as e:
        st.error(f"Error retrieving data from ChromaDB for {collection_name}: {e}")
        return []

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        st.info("Calling OpenAI GPT-4 for response...")
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial newsletter generator."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling GPT-4: {e}")
        return f"Error: {str(e)}"

def assess_accuracy_with_bespoke(newsletter, rag_summary):
    """Compare newsletter with RAG summary."""
    try:
        st.info("Assessing newsletter accuracy...")
        newsletter_embedding = get_embedding(newsletter, model="text-embedding-ada-002")
        if not newsletter_embedding:
            st.error("Failed to generate embedding for newsletter.")
            return 0, "Error generating newsletter embedding."

        rag_embeddings = [get_embedding(item["summary"], model="text-embedding-ada-002") for item in rag_summary]
        if not all(rag_embeddings):
            st.error("Failed to generate embeddings for RAG summaries.")
            return 0, "Error generating RAG embeddings."

        similarities = [cosine_similarity(newsletter_embedding, e) for e in rag_embeddings if e]
        accuracy_percentage = round((sum(similarities) / len(similarities)) * 100, 2) if similarities else 0

        return accuracy_percentage, None
    except Exception as e:
        st.error(f"Error assessing accuracy: {e}")
        return 0, f"Error: {str(e)}"

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude = (sum(a ** 2 for a in vec1) ** 0.5) * (sum(b ** 2 for b in vec2) ** 0.5)
    return dot_product / magnitude if magnitude else 0

### Main Logic ###

if st.button("Generate Financial Newsletter"):
    with Manager() as manager:
        output_queue = manager.Queue()

        # Populate ChromaDB with data from Alpha Vantage
        store_data_in_chromadb(news_url, "news_sentiment_data")
        store_data_in_chromadb(tickers_url, "ticker_trends_data")

        st.write("Data fetching completed. Proceed to summarization.")
