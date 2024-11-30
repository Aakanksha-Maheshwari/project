import streamlit as st
import requests
import json
import openai
from crewai import Agent,Task, Crew
import streamlit as st
import requests
import json
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["api_keys"]["alpha_vantage"]
openai.api_key = st.secrets["api_keys"]["openai"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Generate Newsletter"]
)

### Define Agents ###

# Agent to load and store news data
class NewsDataAgent(Agent):
    def handle(self):
        news_collection = client.get_or_create_collection("news_sentiment_data")

        try:
            response = requests.get(news_url)
            response.raise_for_status()
            data = response.json()

            if "feed" in data:
                for i, item in enumerate(data["feed"], start=1):
                    document = {
                        "id": str(i),
                        "title": item["title"],
                        "url": item["url"],
                        "time_published": item["time_published"],
                        "source": item["source"],
                        "summary": item.get("summary", "N/A"),
                        "topics": [topic["topic"] for topic in item.get("topics", [])],
                        "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
                        "overall_sentiment_score": item.get("overall_sentiment_score", "N/A"),
                        "ticker_sentiments": [
                            {
                                "ticker": ticker["ticker"],
                                "relevance_score": ticker["relevance_score"],
                                "ticker_sentiment_label": ticker["ticker_sentiment_label"],
                                "ticker_sentiment_score": ticker["ticker_sentiment_score"],
                            }
                            for ticker in item.get("ticker_sentiment", [])
                        ],
                    }
                    topics_str = ", ".join(document["topics"])
                    ticker_sentiments_str = json.dumps(document["ticker_sentiments"])

                    news_collection.add(
                        ids=[document["id"]],
                        metadatas=[{
                            "source": document["source"],
                            "time_published": document["time_published"],
                            "topics": topics_str,
                            "overall_sentiment": document["overall_sentiment_label"],
                            "ticker_sentiments": ticker_sentiments_str,
                        }],
                        documents=[json.dumps(document)]
                    )
                return {"status": "News data loaded successfully"}
            return {"status": "No news data found"}
        except requests.exceptions.RequestException as e:
            return {"error": f"API call failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}


# Agent to load and store ticker trends data
class TickerTrendsAgent(Agent):
    def handle(self):
        ticker_collection = client.get_or_create_collection("ticker_trends_data")

        try:
            response = requests.get(tickers_url)
            response.raise_for_status()
            data = response.json()

            if "top_gainers" in data:
                ticker_collection.add(
                    ids=["top_gainers"],
                    metadatas=[{"type": "top_gainers"}],
                    documents=[json.dumps(data["top_gainers"])],
                )
                ticker_collection.add(
                    ids=["top_losers"],
                    metadatas=[{"type": "top_losers"}],
                    documents=[json.dumps(data["top_losers"])],
                )
                ticker_collection.add(
                    ids=["most_actively_traded"],
                    metadatas=[{"type": "most_actively_traded"}],
                    documents=[json.dumps(data["most_actively_traded"])],
                )
                return {"status": "Ticker trends data loaded successfully"}
            return {"status": "No ticker trends data found"}
        except requests.exceptions.RequestException as e:
            return {"error": f"API call failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}


# Agent to retrieve data from ChromaDB and summarize
class DataSummarizationAgent(Agent):
    def handle(self, inputs):
        news_collection = client.get_or_create_collection("news_sentiment_data")
        ticker_collection = client.get_or_create_collection("ticker_trends_data")
        try:
            # Retrieve data from ChromaDB
            news_results = news_collection.get()
            news_data = [json.loads(doc) for doc in news_results["documents"]]
            ticker_data = {}
            for data_type in ["top_gainers", "top_losers", "most_actively_traded"]:
                results = ticker_collection.get(ids=[data_type])
                ticker_data[data_type] = json.loads(results["documents"][0])

            combined_data = {
                "news": news_data[:10],
                "tickers": ticker_data
            }
            return {"combined_data": combined_data}
        except Exception as e:
            return {"error": f"Error retrieving data: {e}"}


# Agent to generate a newsletter
class NewsletterAgent(Agent):
    def handle(self, inputs):
        combined_data = inputs.get("combined_data", {})
        news_data = combined_data.get("news", [])
        tickers = combined_data.get("tickers", {})
        
        input_text = f"""
        News Data: {json.dumps(news_data, indent=2)}
        Ticker Trends:
        Top Gainers: {json.dumps(tickers.get('top_gainers', []), indent=2)}
        Top Losers: {json.dumps(tickers.get('top_losers', []), indent=2)}
        Most Actively Traded: {json.dumps(tickers.get('most_actively_traded', []), indent=2)}
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant tasked with summarizing data into a concise newsletter."},
                    {"role": "user", "content": f"Summarize the following data into a newsletter:\n{input_text}"}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return {"newsletter": response.choices[0].message.content.strip()}
        except Exception as e:
            return {"error": f"Failed to generate newsletter: {e}"}


### Define Tasks ###
news_task = Task(
    description="Load news data from Alpha Vantage and store in ChromaDB.",
    expected_output="News data loaded successfully",
    agent=NewsDataAgent(),
)

ticker_task = Task(
    description="Load ticker trends from Alpha Vantage and store in ChromaDB.",
    expected_output="Ticker trends data loaded successfully",
    agent=TickerTrendsAgent(),
)

summarization_task = Task(
    description="Retrieve and summarize data from ChromaDB.",
    expected_output="Combined data ready for summarization",
    agent=DataSummarizationAgent(),
    context=[news_task, ticker_task],
)

newsletter_task = Task(
    description="Generate a newsletter summarizing all data.",
    expected_output="A concise financial newsletter",
    agent=NewsletterAgent(),
    context=[summarization_task],
)

### Assemble the Crew ###
my_crew = Crew(
    agents=[news_task.agent, ticker_task.agent, summarization_task.agent, newsletter_task.agent],
    tasks=[news_task, ticker_task, summarization_task, newsletter_task],
)

### Streamlit Function ###
def generate_newsletter():
    st.write("### Generating Newsletter")
    try:
        results = my_crew.kickoff(inputs={})
        newsletter = results["newsletter_task"]
        if "error" in newsletter:
            st.error(newsletter["error"])
        else:
            st.subheader("Generated Newsletter")
            st.text(newsletter["newsletter"])
    except Exception as e:
        st.error(f"Failed to generate newsletter: {e}")


### Main Logic ###
if option == "Generate Newsletter":
    generate_newsletter()
