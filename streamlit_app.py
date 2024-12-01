import streamlit as st
import requests
import json
import openai
from multiprocessing import Process, Manager, Queue
from bespokelabs import BespokeLabs
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
st.title("Multi-Agent Financial Newsletter Generator")

### Helper Functions ###

def fetch_and_store_data(api_url, collection_name, output_queue):
    """Agent: Fetch and store data in ChromaDB."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data:
            output_queue.put((collection_name, None, "No data returned from API."))
            return

        collection = client.get_or_create_collection(collection_name)
        for i, item in enumerate(data.get("feed", []), start=1):
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
        output_queue.put((collection_name, "Data fetched and stored successfully.", None))
    except Exception as e:
        output_queue.put((collection_name, None, f"Error: {str(e)}"))

def retrieve_and_summarize(collection_name, query_text, top_k, output_queue):
    """Agent: Retrieve and summarize data from ChromaDB."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in results["documents"] if doc]

        if not documents:
            output_queue.put((collection_name, "No relevant data found.", None))
            return

        summary_prompt = f"""
        Summarize the following data for a financial newsletter:
        Focus on key insights and trends:
        {json.dumps(documents)}
        """
        summary = call_openai_gpt4(summary_prompt)
        output_queue.put((collection_name, summary, documents))
    except Exception as e:
        output_queue.put((collection_name, None, f"Error: {str(e)}"))

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
        return f"Error: {str(e)}"

def assess_accuracy_with_bespoke(newsletter, relevant_data):
    """Agent: Use Bespoke Labs to assess newsletter accuracy."""
    try:
        response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(relevant_data)
        )
        support_prob = getattr(response, "support_prob", None)
        if support_prob is None:
            return 0, "Bespoke Labs response does not contain 'support_prob'."

        return round(support_prob * 100, 2), None
    except Exception as e:
        return 0, f"Error: {str(e)}"

### Multi-Agent Execution ###

def company_analyst_agent(output_queue):
    """Company Analyst Agent."""
    fetch_and_store_data(news_url, "news_sentiment_data", output_queue)
    retrieve_and_summarize("news_sentiment_data", "Company performance insights", 50, output_queue)

def market_trends_agent(output_queue):
    """Market Trends Analyst Agent."""
    fetch_and_store_data(tickers_url, "ticker_trends_data", output_queue)
    retrieve_and_summarize("ticker_trends_data", "Market trends insights", 20, output_queue)

def risk_management_agent(company_data, market_data, output_queue):
    """Risk Management Agent."""
    try:
        if not company_data or not market_data:
            output_queue.put(("risk_management", "No risk data available.", None))
            return

        risk_prompt = f"""
        Analyze the following data for potential risks:
        Company Data: {json.dumps(company_data)}
        Market Data: {json.dumps(market_data)}
        Provide insights on major risks for investors.
        """
        risk_summary = call_openai_gpt4(risk_prompt)
        output_queue.put(("risk_management", risk_summary, None))
    except Exception as e:
        output_queue.put(("risk_management", None, f"Error: {str(e)}"))

def newsletter_agent(company_summary, market_summary, risk_summary, output_queue):
    """Newsletter Generator Agent."""
    try:
        prompt = f"""
        Generate a financial newsletter combining:
        - Company Insights: {company_summary}
        - Market Trends: {market_summary}
        - Risk Management Insights: {risk_summary}
        Provide a professional and cohesive newsletter.
        """
        newsletter = call_openai_gpt4(prompt)
        output_queue.put(("newsletter", newsletter, None))
    except Exception as e:
        output_queue.put(("newsletter", None, f"Error: {str(e)}"))

### Main Logic ###

if st.button("Generate Financial Newsletter"):
    with Manager() as manager:
        output_queue = manager.Queue()

        # Launch agents as processes
        company_process = Process(target=company_analyst_agent, args=(output_queue,))
        market_process = Process(target=market_trends_agent, args=(output_queue,))
        company_process.start()
        market_process.start()
        company_process.join()
        market_process.join()

        # Collect outputs
        company_summary, company_data, market_summary, market_data = None, None, None, None
        while not output_queue.empty():
            name, result, error = output_queue.get()
            if error:
                st.error(f"{name} Error: {error}")
            elif name == "news_sentiment_data":
                company_summary, company_data = result
            elif name == "ticker_trends_data":
                market_summary, market_data = result

        # Risk management agent
        risk_process = Process(
            target=risk_management_agent, args=(company_data, market_data, output_queue)
        )
        risk_process.start()
        risk_process.join()

        # Collect risk summary
        risk_summary = None
        while not output_queue.empty():
            name, result, error = output_queue.get()
            if error:
                st.error(f"{name} Error: {error}")
            elif name == "risk_management":
                risk_summary = result

        # Newsletter generation
        newsletter_process = Process(
            target=newsletter_agent,
            args=(company_summary, market_summary, risk_summary, output_queue),
        )
        newsletter_process.start()
        newsletter_process.join()

        # Collect newsletter
        newsletter = None
        while not output_queue.empty():
            name, result, error = output_queue.get()
            if error:
                st.error(f"{name} Error: {error}")
            elif name == "newsletter":
                newsletter = result

        # Display newsletter
        if newsletter:
            st.subheader("Generated Newsletter")
            st.markdown(newsletter)

        # Assess accuracy
        combined_data = (company_data or []) + (market_data or [])
        accuracy, error = assess_accuracy_with_bespoke(newsletter, combined_data)
        if error:
            st.error(f"Accuracy Error: {error}")
        else:
            st.success(f"Newsletter Accuracy: {accuracy}%")
