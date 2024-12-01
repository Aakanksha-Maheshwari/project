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
st.title("Multi-Agent Financial Newsletter Generator")

### Agent Functions ###

def company_analyst_agent():
    """Fetch and analyze company performance data."""
    try:
        collection_name = "news_sentiment_data"
        fetch_and_store_data(news_url, collection_name)
        summary, data = retrieve_and_summarize(collection_name, "Company performance insights", top_k=50)
        return summary, data
    except Exception as e:
        st.error(f"Error in Company Analyst Agent: {e}")
        return "No data from Company Analyst Agent.", []

def market_trends_analyst_agent():
    """Fetch and analyze market trends data."""
    try:
        collection_name = "ticker_trends_data"
        fetch_and_store_data(tickers_url, collection_name)
        summary, data = retrieve_and_summarize(collection_name, "Market trends insights", top_k=20)
        return summary, data
    except Exception as e:
        st.error(f"Error in Market Trends Analyst Agent: {e}")
        return "No data from Market Trends Analyst Agent.", []

def risk_management_agent(company_data, market_data):
    """Evaluate risks based on company and market data."""
    try:
        if not company_data and not market_data:
            return "No risk data available."

        risk_analysis_prompt = f"""
        Analyze the following data for potential risks:
        Company Data: {json.dumps(company_data)}
        Market Data: {json.dumps(market_data)}
        Provide insights on major risks for investors.
        """
        return call_openai_gpt4(risk_analysis_prompt)
    except Exception as e:
        st.error(f"Error in Risk Management Agent: {e}")
        return "Error analyzing risks."

def newsletter_generator_agent(company_summary, market_summary, risk_summary):
    """Generate the newsletter using agent outputs."""
    try:
        newsletter_prompt = f"""
        Generate a financial newsletter combining the following:
        - Key Company Insights: {company_summary}
        - Major Market Trends: {market_summary}
        - Risk Management Insights: {risk_summary}
        Provide a professional and cohesive newsletter.
        """
        return call_openai_gpt4(newsletter_prompt)
    except Exception as e:
        st.error(f"Error in Newsletter Generator Agent: {e}")
        return "Error generating newsletter."

### Helper Functions ###

def fetch_and_store_data(api_url, collection_name):
    """Fetch data from API and store in ChromaDB."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data:
            st.warning(f"No data returned from API for {collection_name}.")
            return

        collection = client.get_or_create_collection(collection_name)
        for i, item in enumerate(data.get("feed", []), start=1):
            collection.add(
                ids=[str(i)],
                documents=[json.dumps(item)],
                metadatas=[{"source": item.get("source", "N/A")}]
            )
        st.success(f"Data fetched and stored in {collection_name}.")
    except Exception as e:
        st.error(f"Error fetching data for {collection_name}: {e}")

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
        st.error(f"Error retrieving data from {collection_name}: {e}")
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

def assess_accuracy_with_bespoke(newsletter, relevant_data):
    """
    Use Bespoke Labs to assess the accuracy of the newsletter.
    """
    try:
        response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(relevant_data)
        )
        support_prob = getattr(response, "support_prob", None)
        if support_prob is None:
            st.error("Bespoke Labs response does not contain 'support_prob'.")
            return 0

        st.write("Bespoke Labs Response:", response)
        return round(support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0


### Main Logic ###

if st.button("Generate Financial Newsletter"):
    # Execute agents
    company_summary, company_data = company_analyst_agent()
    market_summary, market_data = market_trends_analyst_agent()
    risk_summary = risk_management_agent(company_data, market_data)

    # Generate newsletter
    newsletter = newsletter_generator_agent(company_summary, market_summary, risk_summary)

    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    # Assess accuracy using Bespoke
    combined_data = company_data + market_data
    accuracy = assess_accuracy_with_bespoke(newsletter, combined_data)
    st.success(f"Newsletter Accuracy: {accuracy}%")
