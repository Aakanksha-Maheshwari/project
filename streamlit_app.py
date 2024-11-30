import streamlit as st
import requests
import json
import openai
# Import pysqlite3 for chromadb compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG and OpenAI GPT-4")

### Helper Functions ###

def update_chromadb(collection_name, data):
    """Update ChromaDB with new data."""
    collection = client.get_or_create_collection(collection_name)
    collection.reset()  # Clear existing data
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
        st.write("Fetched News Data:", data)  # Print fetched data
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
        st.write("Fetched Ticker Data:", data)  # Print fetched data
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
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return f"Error generating response: {str(e)}"

### RAG-Agent Definition ###

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description):
        """Execute the task using RAG and GPT-4 summarization."""
        # Retrieve relevant data from ChromaDB
        collection_name = "news_sentiment_data" if "news" in self.goal.lower() else "ticker_trends_data"
        collection = client.get_or_create_collection(collection_name)
        documents = collection.query(query_texts=[task_description], n_results=5)["documents"]
        augmented_prompt = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\nRelevant Data:\n{json.dumps(documents)}"
        return call_openai_gpt4(augmented_prompt)

### Newsletter Generation ###

def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents."""
    agents = [
        RAGAgent(role="Researcher", goal="Process news data"),
        RAGAgent(role="Market Analyst", goal="Analyze trends"),
        RAGAgent(role="Writer", goal="Generate newsletter")
    ]
    tasks = [
        "Extract insights from news data",
        "Analyze market trends",
        "Write a financial newsletter"
    ]
    newsletter = []
    for agent, task in zip(agents, tasks):
        st.write(f"Executing task: {task}")
        result = agent.execute_task(task)
        newsletter.append(f"### {agent.role}\n{result}")
    st.markdown("\n".join(newsletter))

### Main Interface ###
if st.button("Update News Data"):
    fetch_and_update_news_data()

if st.button("Update Ticker Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter"):
    generate_newsletter_with_rag()
