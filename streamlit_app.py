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
bl = BespokeLabs(
    auth_token=st.secrets["bespoke_labs"]["api_key"]
)

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

# Bespoke Labs Accuracy Assessment
def assess_accuracy_with_bespoke(newsletter_content, rag_context):
    """
    Assess the accuracy of the newsletter using Bespoke Labs with debugging.
    """
    try:
        st.markdown("### Debugging Bespoke Accuracy")
        st.markdown("**Generated Newsletter Content:**")
        st.text(newsletter_content)

        st.markdown("**RAG Context Data (Detailed):**")
        st.json(rag_context)

        # Call Bespoke Labs API
        response = bl.minicheck.factcheck.create(
            claim=newsletter_content,
            context=rag_context,
        )

        # Log Bespoke response
        st.markdown("**Raw Bespoke Labs Response:**")
        st.json({
            "support_prob": response.support_prob,
            "other_info": str(response)
        })

        # Return accuracy score
        return round(response.support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0

# Alpha Vantage Data Fetching
def fetch_market_news():
    """
    Fetch detailed market news using Alpha Vantage API.
    """
    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": st.secrets["alpha_vantage"]["api_key"],
            "limit": 50,
            "sort": "RELEVANCE",
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        news_feed = response.json().get("feed", [])

        # Extract detailed news data
        detailed_news = []
        for article in news_feed:
            content = {
                "title": article.get("title", "No title"),
                "summary": article.get("summary", "No summary"),
                "url": article.get("url", ""),
                "time_published": article.get("time_published", ""),
                "authors": article.get("authors", []),
                "source": article.get("source", ""),
                "overall_sentiment_score": article.get("overall_sentiment_score", 0),
                "overall_sentiment_label": article.get("overall_sentiment_label", ""),
                "topics": article.get("topics", []),
                "ticker_sentiment": article.get("ticker_sentiment", []),
            }
            detailed_news.append(content)

        return detailed_news
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return []

def fetch_gainers_losers():
    """
    Fetch top gainers and losers from Alpha Vantage API.
    """
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

# Market Newsletter Crew
class MarketNewsletterCrew:
    def __init__(self):
        self.rag_helper = RAGHelper(client=st.session_state.chroma_client)

    def company_analyst(self, task_input):
        return self.rag_helper.summarize(task_input, context="company insights")

    def market_trends_analyst(self, task_input):
        return self.rag_helper.summarize(task_input, context="market trends")

    def risk_manager(self, company_insights, market_trends):
        prompt_risks = f"""
        Assess risks based on the following:

        **Company Insights:**
        {company_insights}

        **Market Trends:**
        {market_trends}

        Include macroeconomic, sector-specific, and stock-specific risks.
        """
        return self.rag_helper.summarize([prompt_risks], context="risk assessment")

    def newsletter_generator(self, company_insights, market_trends, risks):
        return self.rag_helper.generate_newsletter(company_insights, market_trends, risks)

# Streamlit Interface
st.title("Market Data Newsletter with CrewAI, OpenAI, and RAG")

crew_instance = MarketNewsletterCrew()

# Fetch and Add Data to RAG
if st.button("Fetch and Add Data to RAG"):
    news_data = fetch_market_news()
    if news_data:
        documents = [f"{article['title']} - {article['summary']}" for article in news_data]
        metadata = [
            {
                "title": article["title"],
                "url": article["url"],
                "authors": article["authors"],
                "source": article["source"],
                "time_published": article["time_published"],
                "overall_sentiment_score": article["overall_sentiment_score"],
                "overall_sentiment_label": article["overall_sentiment_label"],
                "topics": article["topics"],
                "ticker_sentiment": article["ticker_sentiment"],
            }
            for article in news_data
        ]
        crew_instance.rag_helper.add("news_collection", documents, metadata)

    trends_data = fetch_gainers_losers()
    if trends_data:
        gainers = trends_data.get("top_gainers", [])
        losers = trends_data.get("top_losers", [])
        trends_docs = [
            f"{item['ticker']} - ${item['price']} ({item['change_percentage']}%)"
            for item in gainers + losers
        ]
        trends_metadata = [{"ticker": item["ticker"]} for item in gainers + losers]
        crew_instance.rag_helper.add("trends_collection", trends_docs, trends_metadata)

# Generate Newsletter
if st.button("Generate Newsletter"):
    try:
        company_insights = crew_instance.rag_helper.query("news_collection", "latest company news")
        summarized_company = crew_instance.company_analyst(company_insights) if company_insights else "No company insights available."

        market_trends = crew_instance.rag_helper.query("trends_collection", "latest market trends")
        summarized_trends = crew_instance.market_trends_analyst(market_trends) if market_trends else "No market trends available."

        risks = crew_instance.risk_manager(summarized_company, summarized_trends)

        newsletter = crew_instance.newsletter_generator(summarized_company, summarized_trends, risks)
        st.markdown(newsletter)

        rag_context = "\n".join(company_insights + market_trends)
        accuracy_score = assess_accuracy_with_bespoke(newsletter, rag_context)
        st.markdown(f"**Accuracy Score:** {accuracy_score}%")
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")
