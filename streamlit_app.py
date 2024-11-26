import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import openai
import requests
from crewai import Agent, Crew, Task, Process
from bespokelabs import BespokeLabs

# OpenAI API Key Setup
openai.api_key = st.secrets["openai"]["api_key"]

# Initialize ChromaDB Client
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient()

# Initialize Bespoke Labs
bl = BespokeLabs(auth_token=st.secrets["bespoke_labs"]["api_key"])

# Custom RAG Functionality
class RAGHelper:
    def __init__(self, client):
        self.client = client

    def query(self, collection_name, query, n_results=5):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=n_results)
            documents = [doc for sublist in results["documents"] for doc in sublist]
            return documents
        except Exception as e:
            st.error(f"Error querying RAG: {e}")
            return []

    def add(self, collection_name, documents, metadata):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            collection.add(
                documents=documents,
                metadatas=metadata,
                ids=[str(i) for i in range(len(documents))]
            )
            st.success(f"Data successfully added to the '{collection_name}' collection.")
        except Exception as e:
            st.error(f"Error adding to RAG: {e}")

# Fetch Market News
def fetch_market_news():
    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": st.secrets["alpha_vantage"]["api_key"],
            "limit": 50,
            "sort": "RELEVANCE",
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()

        # Log and return fetched data
        news_feed = response.json().get("feed", [])
        st.write("Fetched News Feed:", news_feed)  # Debugging log
        return news_feed
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return []

# Fetch Gainers and Losers
def fetch_gainers_losers():
    try:
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": st.secrets["alpha_vantage"]["api_key"],
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()

        # Log and return fetched data
        gainers_losers = response.json()
        st.write("Fetched Gainers and Losers Data:", gainers_losers)  # Debugging log
        return gainers_losers
    except Exception as e:
        st.error(f"Error fetching gainers and losers: {e}")
        return {}

# Initialize RAG Helper
rag_helper = RAGHelper(client=st.session_state.chroma_client)

# Fetch and Add Data to RAG
if st.button("Fetch and Add Data to RAG"):
    news_data = fetch_market_news()
    if news_data:
        try:
            # Validate and log fetched news data
            documents = [article.get("summary", "No summary") for article in news_data]
            metadata = [{"title": article.get("title", ""), "source": article.get("source", "")} for article in news_data]
            st.write("News Data to Add to RAG:", documents, metadata)  # Debugging log

            # Add data to RAG
            rag_helper.add("news_collection", documents, metadata)
        except Exception as e:
            st.error(f"Error processing and adding news data: {e}")

    gainers_losers_data = fetch_gainers_losers()
    if gainers_losers_data:
        try:
            # Validate and log fetched gainers/losers data
            gainers = gainers_losers_data.get("top_gainers", [])
            documents = [f"{g['ticker']} - ${g['price']} ({g['change_percentage']}%)" for g in gainers]
            metadata = [{"ticker": g["ticker"], "price": g["price"], "change": g["change_percentage"]} for g in gainers]
            st.write("Gainers and Losers Data to Add to RAG:", documents, metadata)  # Debugging log

            # Add data to RAG
            rag_helper.add("trends_collection", documents, metadata)
        except Exception as e:
            st.error(f"Error processing and adding gainers/losers data: {e}")

# Generate Newsletter
if st.button("Generate Newsletter"):
    try:
        company_insights = rag_helper.query("news_collection", "latest company news")
        st.write("Queried Company Insights:", company_insights)  # Debugging log

        market_trends = rag_helper.query("trends_collection", "latest market trends")
        st.write("Queried Market Trends:", market_trends)  # Debugging log

        summarized_company = "No company insights available." if not company_insights else company_insights
        summarized_trends = "No market trends available." if not market_trends else market_trends

        newsletter = f"""
        **Company Insights:**
        {summarized_company}

        **Market Trends:**
        {summarized_trends}
        """
        st.markdown(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")
