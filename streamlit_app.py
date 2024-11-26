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
            if documents and metadata:  # Ensure non-empty lists
                collection.add(
                    documents=documents,
                    metadatas=metadata,
                    ids=[str(i) for i in range(len(documents))]
                )
                st.success(f"Data successfully added to the '{collection_name}' collection.")
            else:
                st.warning(f"No valid data to add to '{collection_name}'.")
        except Exception as e:
            st.error(f"Error adding to RAG: {e}")

    def summarize(self, data, context="general insights"):
        try:
            input_text = "\n".join(data) if isinstance(data, list) else str(data)
            prompt = f"Summarize the following {context}:\n\n{input_text}\n\nProvide a concise summary."
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
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
            """
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating newsletter: {e}")
            return "Newsletter generation failed due to an error."

# Bespoke Labs Accuracy Assessment
def assess_accuracy_with_bespoke(newsletter_content, rag_context):
    try:
        context_data = "\n".join(rag_context) if isinstance(rag_context, list) else str(rag_context)
        response = bl.minicheck.factcheck.create(
            claim=newsletter_content,
            context=context_data
        )
        return round(response.support_prob * 100, 2)
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0

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
        
        # Log raw response for debugging
        raw_data = response.json()
        st.write("Raw Gainers and Losers Data:", raw_data)  # Debugging log

        # Validate the response structure
        gainers = raw_data.get("top_gainers", [])
        losers = raw_data.get("top_losers", [])
        if not gainers and not losers:
            st.warning("No gainers or losers data found in the response.")
        return raw_data
    except Exception as e:
        st.error(f"Error fetching gainers and losers: {e}")
        return {}

# Market Newsletter Crew
class MarketNewsletterCrew:
    def __init__(self):
        self.rag_helper = RAGHelper(client=st.session_state.chroma_client)

    def company_analyst(self, task_input):
        """Analyze company news."""
        return self.rag_helper.summarize(task_input, context="company insights")

    def market_trends_analyst(self, task_input):
        """Analyze market trends."""
        return self.rag_helper.summarize(task_input, context="market trends")

    def risk_manager(self, inputs):
        """Evaluate risks."""
        company_insights, market_trends = inputs
        risks_prompt = f"""
        Assess risks based on the following:

        **Company Insights:**
        {company_insights}

        **Market Trends:**
        {market_trends}
        """
        return self.rag_helper.summarize([risks_prompt], context="risk analysis")

    def newsletter_generator(self, inputs):
        """Generate the newsletter."""
        company_insights, market_trends, risks = inputs
        return self.rag_helper.generate_newsletter(company_insights, market_trends, risks)

    def crew(self):
        """Define the Crew process."""
        return Crew(
            agents=[
                Agent(name="Company Analyst", task=self.company_analyst),
                Agent(name="Market Trends Analyst", task=self.market_trends_analyst),
                Agent(name="Risk Manager", task=self.risk_manager),
                Agent(name="Newsletter Generator", task=self.newsletter_generator),
            ],
            process=Process.sequential
        )

# Streamlit Interface
st.title("Market Data Newsletter with CrewAI and RAG")

# Initialize Helpers and Crew
rag_helper = RAGHelper(client=st.session_state.chroma_client)
crew_instance = MarketNewsletterCrew()

# Fetch and Add Data
if st.button("Fetch and Add Data to RAG"):
    news_data = fetch_market_news()
    if news_data:
        documents = [article.get("summary", "No summary") for article in news_data]
        metadata = [{"title": article.get("title", ""), "source": article.get("source", "")} for article in news_data]
        rag_helper.add("news_collection", documents, metadata)

    # Fetch and Add Gainers/Losers Data
        gainers_losers_data = fetch_gainers_losers()
        if gainers_losers_data:
            gainers = gainers_losers_data.get("top_gainers", [])
            losers = gainers_losers_data.get("top_losers", [])

        # Check if data is available
        if gainers or losers:
            # Combine gainers and losers
            documents = [
                f"{item['ticker']} - ${item['price']} ({item['change_percentage']}%)"
                for item in gainers + losers
            ]
            metadata = [
                {
                    "ticker": item["ticker"],
                    "price": item["price"],
                    "change": item["change_percentage"]
                }
                for item in gainers + losers
            ]

        # Log processed data
        st.write("Processed Gainers and Losers Data:", documents, metadata)  # Debugging log

        # Add to RAG
        rag_helper.add("trends_collection", documents, metadata)
    else:
        st.warning("No valid gainers or losers data to add to RAG.")
else:
    st.error("Failed to fetch gainers and losers data.")


# Generate Newsletter
if st.button("Generate Newsletter"):
    try:
        company_insights = rag_helper.query("news_collection", "latest company news")
        market_trends = rag_helper.query("trends_collection", "latest market trends")

        crew = crew_instance.crew()
        insights = crew.run(inputs=(company_insights, market_trends))
        summarized_company, summarized_trends, risks, newsletter = insights

        # Assess Bespoke accuracy
        rag_context = company_insights + market_trends
        accuracy_score = assess_accuracy_with_bespoke(newsletter, rag_context)

        st.markdown(newsletter)
        st.markdown(f"**Accuracy Score:** {accuracy_score}%")
    except Exception as e:
        st.error(f"Error generating newsletter or assessing accuracy: {e}")
