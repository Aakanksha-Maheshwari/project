import streamlit as st
import requests
import json
import openai
import chromadb
from chromadb.config import Settings

# Import pysqlite3 for chromadb compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Update News Data", "Update Ticker Trends Data", "Generate Newsletter"]
)

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
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Debugging: Log the response structure
        print("API Response:", response)

        # Access the content using object attributes
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return f"Error generating response: {str(e)}"

### RAG-Agent Definition ###

class RAGAgent:
    def _init_(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description):
        """Execute the task using RAG and GPT-4 summarization."""
        # Retrieve relevant data from ChromaDB
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("ticker_trends_data", task_description)
        else:
            retrieved_data = []

        # Combine retrieved data with task description
        augmented_prompt = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\nRelevant Data:\n{json.dumps(retrieved_data)}"

        # Call GPT-4 for summarization
        summary = call_openai_gpt4(augmented_prompt)
        return summary

### Agents and Tasks ###

researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

tasks = [
    {"description": "Extract insights from news data", "agent": researcher},
    {"description": "Analyze market trends", "agent": market_analyst},
    {"description": "Analyze risk data", "agent": risk_analyst},
    {"description": "Write the newsletter", "agent": writer},
]

### Newsletter Generation ###

def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents."""
    newsletter_content = []
    for task in tasks:
        st.write(f"Executing: {task['description']} with {task['agent'].role}")
        result = task['agent'].execute_task(task['description'])
        if "Error" in result:
            st.error(f"Failed to complete task: {task['description']}")
        else:
            newsletter_content.append(f"## {task['agent'].role}\n{result}\n")
    st.subheader("Generated Newsletter")
    st.markdown("\n".join(newsletter_content))

### Main Logic ###
if option == "Update News Data":
    fetch_and_update_news_data()
elif option == "Update Ticker Trends Data":
    fetch_and_update_ticker_trends_data()
elif option == "Generate Newsletter":
    generate_newsletter_with_rag()