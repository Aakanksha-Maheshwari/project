import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import openai
import requests
from crewai import Agent, Crew, Task, Process
from bespokelabs import BespokeLabs

# Explicitly set OpenAI API key
openai.api_key = st.secrets.get("openai", {}).get("api_key")
if not openai.api_key:
    st.error("OpenAI API Key is missing. Please set it in the secrets file.")

# Initialize Bespoke Labs
bl = BespokeLabs(auth_token=st.secrets["bespoke_labs"]["api_key"])
if not bl:
    st.error("Bespoke Labs API Key is missing. Please set it in the secrets file.")

# Initialize ChromaDB Client
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient()

# Custom RAG Functionality
class RAGHelper:
    def __init__(self, client):
        self.client = client

    def query(self, collection_name, query, n_results=5):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=n_results)
            documents = [doc for sublist in results["documents"] for doc in sublist]  # Flatten nested lists
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
        prompt = f"Summarize the following {context}:\n\n{input_text}\n\nProvide a concise summary."
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=openai.api_key  # Explicitly pass the API key
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error summarizing data: {e}")
        return "Summary unavailable due to an error."

    def generate_newsletter(self, company_insights, market_trends, risks):
        try:
            messages = [
                {"role": "system", "content": "You are a professional assistant tasked with creating market newsletters."},
                {"role": "user", "content": f"""
                Generate a professional daily market newsletter based on the following data:

                Company Insights:
                {company_insights}

                Market Trends:
                {market_trends}

                Risk Analysis:
                {risks}

                Format it concisely and professionally, focusing on insights.
                """}
            ]
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating newsletter: {e}")
            return "Newsletter generation failed due to an error."

# Accuracy Assessment with Bespoke Labs
def assess_accuracy_with_bespoke(newsletter_content, rag_context):
    try:
        response = bl.minicheck.factcheck.create(claim=newsletter_content, context=rag_context)
        return round(response.support_prob * 100, 2)  # Convert to percentage
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0

# Alpha Vantage Data Fetching
def fetch_market_news():
    try:
        params = {"function": "NEWS_SENTIMENT", "apikey": st.secrets["alpha_vantage"]["api_key"], "limit": 50}
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json().get("feed", [])
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return []

def fetch_gainers_losers():
    try:
        params = {"function": "TOP_GAINERS_LOSERS", "apikey": st.secrets["alpha_vantage"]["api_key"]}
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching gainers and losers: {e}")
        return {}

# Market Newsletter Crew with Multi-Agent System
class MarketNewsletterCrew:
    def __init__(self):
        self.rag_helper = RAGHelper(client=st.session_state.chroma_client)

    def company_analyst(self, task_input):
        """Analyze company news."""
        return self.rag_helper.summarize(task_input, context="company insights")

    def market_trends_analyst(self, task_input):
        """Analyze market trends."""
        return self.rag_helper.summarize(task_input, context="market trends")

    def risk_manager(self, task_input):
        """Evaluate risks based on inputs."""
        return self.rag_helper.summarize(task_input, context="risk analysis")

    def newsletter_generator(self, inputs):
        """Generate the newsletter."""
        company_insights, market_trends, risks = inputs
        return self.rag_helper.generate_newsletter(company_insights, market_trends, risks)

    def crew(self):
        """Define the multi-agent Crew process."""
        return Crew(
            agents=[
                Agent(name="Company Analyst", task=self.company_analyst),
                Agent(name="Market Trends Analyst", task=self.market_trends_analyst),
                Agent(name="Risk Manager", task=self.risk_manager),
                Agent(name="Newsletter Generator", task=self.newsletter_generator),
            ],
            process=Process.parallel
        )

# Streamlit Interface
st.title("Market Data Newsletter with Multi-Agent System")

# Initialize Crew
crew_instance = MarketNewsletterCrew()

if st.button("Fetch and Add Data to RAG"):
    # Fetch and add news to RAG
    news_data = fetch_market_news()
    if news_data:
        documents = [article.get("summary", "No summary") for article in news_data]
        metadata = [{"title": article.get("title", "")} for article in news_data]
        crew_instance.rag_helper.add("news_collection", documents, metadata)

    # Fetch and add gainers/losers to RAG
    gainers_losers_data = fetch_gainers_losers()
    if gainers_losers_data:
        gainers = gainers_losers_data.get("top_gainers", [])
        documents = [f"{g['ticker']} - ${g['price']} ({g['change_percentage']}%)" for g in gainers]
        metadata = [{"ticker": g["ticker"], "price": g["price"], "change": g["change_percentage"]} for g in gainers]
        crew_instance.rag_helper.add("trends_collection", documents, metadata)

if st.button("Generate Newsletter"):
    try:
        # Query and summarize data
        company_insights = crew_instance.rag_helper.query("news_collection", "latest company news")
        market_trends = crew_instance.rag_helper.query("trends_collection", "latest market trends")

        # Use the Crew to process tasks
        crew = crew_instance.crew()
        insights = crew.run(inputs=(company_insights, market_trends))
        summarized_company, summarized_trends, risks, newsletter = insights

        # Assess accuracy
        rag_context = "\n".join(company_insights + market_trends)
        accuracy_score = assess_accuracy_with_bespoke(newsletter, rag_context)

        # Display results
        st.markdown(newsletter)
        st.markdown(f"**Accuracy Score:** {accuracy_score}%")
    except Exception as e:
        st.error(f"Error generating newsletter or assessing accuracy: {e}")
