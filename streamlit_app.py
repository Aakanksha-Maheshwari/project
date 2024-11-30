import streamlit as st
import requests
import json
import openai
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

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'
bespoke_api_url = "https://api.bespokelabs.ai/minicheck"

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, GPT-4, and Bespoke Labs")

### **Helper Functions** ###

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
        response = openai.chat.completions.create(
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

def measure_newsletter_accuracy(context, claim):
    """Use Bespoke Labs API to measure the accuracy of the newsletter."""
    try:
        payload = {"context": context, "claim": claim}
        headers = {"Authorization": f"Bearer {bespoke_key}"}
        response = requests.post(bespoke_api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        st.write("Bespoke API Response:", result)  # Debugging
        return result.get("accuracy_score", "No accuracy score provided")
    except Exception as e:
        st.error(f"Error calling Bespoke Labs API: {e}")
        return "Error measuring accuracy"

### **RAG-Agent Definition** ###

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute the task using RAG and GPT-4 summarization."""
        # Retrieve relevant data from ChromaDB
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("ticker_trends_data", task_description)
        else:
            retrieved_data = []

        # Combine additional data if provided
        combined_data = retrieved_data
        if additional_data:
            combined_data.extend(additional_data)

        # Combine retrieved data with task description
        augmented_prompt = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\nRelevant Data:\n{json.dumps(combined_data)}"

        # Call GPT-4 for summarization
        summary = call_openai_gpt4(augmented_prompt)
        return summary

### Instantiate Agents ###
researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

### **Newsletter Generation with Accuracy Measurement** ###

def generate_newsletter_with_accuracy():
    """Generate the newsletter using RAG and agents and measure its accuracy."""
    newsletter_content = []

    # Step 1: Execute tasks
    st.write("Executing: Extract insights from news data (Researcher)")
    news_results = researcher.execute_task("Extract insights from news data")

    st.write("Executing: Analyze market trends (Market Analyst)")
    trends_results = market_analyst.execute_task("Analyze market trends")

    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_results = risk_analyst.execute_task("Analyze risk data")

    # Combine results
    combined_data = [
        {"role": "Researcher", "content": news_results},
        {"role": "Market Analyst", "content": trends_results},
        {"role": "Risk Analyst", "content": risk_results}
    ]

    # Generate newsletter
    st.write("Executing: Write the newsletter (Writer)")
    writer_task_description = "Write a cohesive newsletter based on insights from news, market trends, and risk analysis."
    newsletter = writer.execute_task(writer_task_description, additional_data=combined_data)

    # Measure accuracy
    if newsletter and combined_data:
        st.write("Measuring newsletter accuracy...")
        context = json.dumps(combined_data)
        claim = newsletter
        accuracy = measure_newsletter_accuracy(context, claim)
        st.success(f"Newsletter Accuracy: {accuracy}")
    else:
        st.error("Failed to generate or measure the newsletter.")

### **Main Page Buttons** ###

if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter and Measure Accuracy"):
    generate_newsletter_with_accuracy()
