import streamlit as st
import requests
import json
import openai
from multiprocessing import Process, Manager, Queue
from bespokelabs import BespokeLabs, DefaultHttpxClient
import httpx
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import threading

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


### Agent Functions ###

def company_analyst_agent(output_queue):
    """Fetch and analyze company performance data."""
    try:
        collection_name = "news_sentiment_data"
        fetch_and_store_data(news_url, collection_name)
        summary, data = retrieve_and_summarize(collection_name, "Company performance insights", top_k=50)
        output_queue.put(("company_summary", summary))
        output_queue.put(("company_data", data))
    except Exception as e:
        output_queue.put(("error", f"Error in Company Analyst Agent: {e}"))

def market_trends_analyst_agent(output_queue):
    """Fetch and analyze market trends data."""
    try:
        collection_name = "ticker_trends_data"
        fetch_and_store_data(tickers_url, collection_name)
        summary, data = retrieve_and_summarize(collection_name, "Market trends insights", top_k=20)
        output_queue.put(("market_summary", summary))
        output_queue.put(("market_data", data))
    except Exception as e:
        output_queue.put(("error", f"Error in Market Trends Analyst Agent: {e}"))

def risk_management_agent(input_queue, output_queue):
    """Evaluate risks based on company and market data."""
    try:
        company_data = input_queue.get("company_data", [])
        market_data = input_queue.get("market_data", [])
        if not company_data and not market_data:
            output_queue.put(("risk_summary", "No risk data available."))
            return

        risk_analysis_prompt = f"""
        Analyze the following data for potential risks:
        Company Data: {json.dumps(company_data)}
        Market Data: {json.dumps(market_data)}
        Provide insights on major risks for investors.
        """
        risk_summary = call_openai_gpt4(risk_analysis_prompt)
        output_queue.put(("risk_summary", risk_summary))
    except Exception as e:
        output_queue.put(("error", f"Error in Risk Management Agent: {e}"))

def newsletter_generator_agent(input_queue, output_queue):
    """Generate the newsletter using agent outputs."""
    try:
        company_summary = input_queue.get("company_summary", "No company summary available.")
        market_summary = input_queue.get("market_summary", "No market summary available.")
        risk_summary = input_queue.get("risk_summary", "No risk summary available.")

        newsletter_prompt = f"""
        Generate a financial newsletter combining the following:
        - Key Company Insights: {company_summary}
        - Major Market Trends: {market_summary}
        - Risk Management Insights: {risk_summary}
        Provide a professional and cohesive newsletter.
        """
        newsletter = call_openai_gpt4(newsletter_prompt)
        output_queue.put(("newsletter", newsletter))
    except Exception as e:
        output_queue.put(("error", f"Error in Newsletter Generator Agent: {e}"))


### Helper Functions ###

def fetch_and_store_data(api_url, collection_name):
    """Fetch data from API and store in ChromaDB."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data:
            return

        collection = client.get_or_create_collection(collection_name)
        for i, item in enumerate(data.get("feed", []), start=1):
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
    except Exception as e:
        print(f"Error fetching data for {collection_name}: {e}")

def retrieve_and_summarize(collection_name, query_text, top_k=10):
    """Retrieve data from ChromaDB and summarize."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"] if doc]

        if not documents:
            return "No relevant data found.", []

        summary_prompt = f"""
        Summarize the following data for a financial newsletter:
        Focus on key insights and trends:
        {json.dumps(documents)}
        """
        summary = call_openai_gpt4(summary_prompt)
        return summary, documents
    except Exception as e:
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
        return "Error generating response."

def assess_accuracy_with_bespoke(newsletter, relevant_data):
    """Use Bespoke Labs to assess the accuracy of the newsletter."""
    try:
        response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(relevant_data)
        )
        support_prob = getattr(response, "support_prob", None)
        return round(support_prob * 100, 2) if support_prob else 0
    except Exception:
        return 0


### Main Logic ###

if st.button("Generate Financial Newsletter"):
    manager = Manager()
    output_queue = manager.Queue()
    input_data = manager.dict()

    # Create agents as separate processes
    company_agent = Process(target=company_analyst_agent, args=(output_queue,))
    market_agent = Process(target=market_trends_analyst_agent, args=(output_queue,))
    risk_agent = Process(target=risk_management_agent, args=(input_data, output_queue))
    newsletter_agent = Process(target=newsletter_generator_agent, args=(input_data, output_queue))

    # Start agents
    company_agent.start()
    market_agent.start()

    # Collect data from company and market agents
    company_agent.join()
    market_agent.join()

    # Pass collected data to risk and newsletter agents
    while not output_queue.empty():
        key, value = output_queue.get()
        input_data[key] = value

    risk_agent.start()
    risk_agent.join()

    newsletter_agent.start()
    newsletter_agent.join()

    # Display the final newsletter
    newsletter = input_data.get("newsletter", "Error: Newsletter not generated.")
    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    # Assess accuracy
    combined_data = input_data.get("company_data", []) + input_data.get("market_data", [])
    accuracy = assess_accuracy_with_bespoke(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy}%")
