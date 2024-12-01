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

def store_data_in_chromadb(api_url, collection_name):
    """Fetch data from API, generate embeddings, and store in ChromaDB."""
    try:
        st.info(f"Fetching data for {collection_name} from {api_url}...")
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data or "feed" not in data:
            st.warning(f"No data returned for {collection_name}.")
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
    except Exception as e:
        st.error(f"Error storing data in ChromaDB for {collection_name}: {e}")

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

### Multi-Agent Execution ###

def company_analyst_agent(output_queue):
    """Company Analyst Agent."""
    data = retrieve_data_from_chromadb("news_sentiment_data", "Company performance insights", 50)
    summary_prompt = f"Summarize the following company data: {json.dumps(data)}"
    summary = call_openai_gpt4(summary_prompt)
    output_queue.put(("company", {"data": data, "summary": summary}, None))

def market_trends_agent(output_queue):
    """Market Trends Analyst Agent."""
    data = retrieve_data_from_chromadb("ticker_trends_data", "Market trends insights", 20)
    summary_prompt = f"Summarize the following market trends data: {json.dumps(data)}"
    summary = call_openai_gpt4(summary_prompt)
    output_queue.put(("market_trends", {"data": data, "summary": summary}, None))

def risk_management_agent(company_data, market_data, output_queue):
    """Risk Management Agent."""
    try:
        st.info("Performing risk management analysis...")
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
        st.error(f"Error in Risk Management Agent: {e}")
        output_queue.put(("risk_management", None, f"Error: {str(e)}"))

def newsletter_agent(company_summary, market_summary, risk_summary, output_queue):
    """Newsletter Generator Agent."""
    try:
        st.info("Generating newsletter...")
        prompt = f"""
        Generate a financial newsletter combining:
        - Company Insights: {company_summary}
        - Market Trends: {market_summary}
        - Risk Management Insights: {risk_summary}
        Provide a professional and cohesive newsletter.
        """
        newsletter = call_openai_gpt4(prompt)
        output_queue.put(("newsletter", newsletter, None))
        st.success("Newsletter generated successfully.")
    except Exception as e:
        st.error(f"Error in Newsletter Generator Agent: {e}")
        output_queue.put(("newsletter", None, f"Error: {str(e)}"))

### Main Logic ###

if st.button("Generate Financial Newsletter"):
    with Manager() as manager:
        output_queue = manager.Queue()

        # Populate ChromaDB with data from Alpha Vantage
        store_data_in_chromadb(news_url, "news_sentiment_data")
        store_data_in_chromadb(tickers_url, "ticker_trends_data")

        # Launch agents as processes
        company_process = Process(target=company_analyst_agent, args=(output_queue,))
        market_process = Process(target=market_trends_agent, args=(output_queue,))
        company_process.start()
        market_process.start()
        company_process.join()
        market_process.join()

        # Collect outputs
        company_summary, market_summary, rag_summary = None, None, []
        company_data, market_data = [], []
        while not output_queue.empty():
            name, result, error = output_queue.get()
            st.info(f"Queue data received: {name}, Result: {result}, Error: {error}")
            if error:
                st.error(f"{name} Error: {error}")
            elif name == "company":
                company_summary = result["summary"]
                company_data = result["data"]
                rag_summary.append(result)
            elif name == "market_trends":
                market_summary = result["summary"]
                market_data = result["data"]
                rag_summary.append(result)

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

        # Assess Accuracy
        accuracy, error = assess_accuracy_with_bespoke(newsletter, rag_summary)
        if error:
            st.error(f"Accuracy Error: {error}")
        else:
            st.success(f"Newsletter Accuracy: {accuracy}%")
