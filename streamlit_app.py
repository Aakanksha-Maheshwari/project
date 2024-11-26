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

    def summarize(self, data, context="general insights"):
        try:
            input_text = "\n".join(data) if isinstance(data, list) else str(data)
            prompt = f"""
            Summarize the following {context} with detailed takeaways, actionable insights, and relevant examples:
            
            {input_text}
            
            Ensure the summary aligns strictly with the provided data and does not include assumptions.
            """
            messages = [
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ]
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error summarizing data: {e}")
            return "Summary unavailable due to an error."

    def generate_newsletter(self, company_insights, market_trends, risks):
        try:
            prompt = f"""
            Create a detailed daily market newsletter based on the following:

            **Company Insights:**
            {company_insights}

            **Market Trends:**
            {market_trends}

            **Risk Analysis:**
            {risks}

            Ensure the newsletter is factual, actionable, and references metadata such as sources, authors, and publication times.
            """
            messages = [
                {"role": "system", "content": "You are a professional newsletter creator."},
                {"role": "user", "content": prompt}
            ]
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating newsletter: {e}")
            return "Newsletter generation failed due to an error."

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
        return response.json().get("feed", [])
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
        return response.json()
    except Exception as e:
        st.error(f"Error fetching gainers and losers: {e}")
        return {}

# Initialize RAG Helper
rag_helper = RAGHelper(client=st.session_state.chroma_client)

# Fetch and Add Data to RAG
if st.button("Fetch and Add Data to RAG"):
    news_data = fetch_market_news()
    st.write("Fetched News Data:", news_data)  # Debugging fetched data
    if news_data:
        documents = [article.get("summary", "No summary") for article in news_data]
        metadata = [{"title": article.get("title", ""), "source": article.get("source", "")} for article in news_data]
        rag_helper.add("news_collection", documents, metadata)

    gainers_losers_data = fetch_gainers_losers()
    st.write("Fetched Gainers and Losers Data:", gainers_losers_data)  # Debugging fetched data
    if gainers_losers_data:
        gainers = gainers_losers_data.get("top_gainers", [])
        documents = [f"{g['ticker']} - ${g['price']} ({g['change_percentage']}%)" for g in gainers]
        metadata = [{"ticker": g["ticker"], "price": g["price"], "change": g["change_percentage"]} for g in gainers]
        rag_helper.add("trends_collection", documents, metadata)

# Generate Newsletter
if st.button("Generate Newsletter"):
    try:
        company_insights = rag_helper.query("news_collection", "latest company news")
        st.write("Queried Company Insights:", company_insights)  # Debugging queried data

        market_trends = rag_helper.query("trends_collection", "latest market trends")
        st.write("Queried Market Trends:", market_trends)  # Debugging queried data

        summarized_company = rag_helper.summarize(company_insights, context="company insights") if company_insights else "No company insights available."
        summarized_trends = rag_helper.summarize(market_trends, context="market trends") if market_trends else "No market trends available."
        risks = rag_helper.summarize([summarized_company, summarized_trends], context="risk assessment")

        newsletter = rag_helper.generate_newsletter(summarized_company, summarized_trends, risks)
        st.markdown(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")
