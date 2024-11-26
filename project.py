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
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes data."},
                    {"role": "user", "content": prompt}
                ]
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
 
# Bespoke Labs Accuracy Assessment
def assess_accuracy_with_bespoke(newsletter_content, rag_context):
    """
    Assess the accuracy of the newsletter content using Bespoke Labs.
    """
    try:
        st.markdown("### Debugging Information for Bespoke Labs")
        st.markdown("**Claim (Newsletter Content):**")
        st.text(newsletter_content)
 
        st.markdown("**Context (RAG Data):**")
        st.text(rag_context)
 
        # Call Bespoke Labs API
        response = bl.minicheck.factcheck.create(
            claim=newsletter_content,
            context=rag_context,
        )
 
        # Display the raw response
        st.markdown("**Raw Bespoke Labs Response:**")
        st.json({
            "support_prob": response.support_prob,
            "other_info": str(response)  # Log additional info
        })
 
        # Extract and return the support probability
        return round(response.support_prob * 100, 2)  # Convert to percentage
    except AttributeError as e:
        st.error(f"Error: Missing or incorrect response attribute. Details: {e}")
        return 0
    except Exception as e:
        st.error(f"Error assessing accuracy with Bespoke Labs: {e}")
        return 0
 
# Alpha Vantage Data Fetching
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
        """Analyze company news."""
        return self.rag_helper.summarize(task_input, context="company insights")
 
    def market_trends_analyst(self, task_input):
        """Analyze market trends."""
        return self.rag_helper.summarize(task_input, context="market trends")
 
    def risk_manager(self, company_insights, market_trends):
        """Evaluate risks based on insights."""
        prompt_risks = f"Assess risks based on:\n\nCompany Insights:\n{company_insights}\n\nMarket Trends:\n{market_trends}"
        return self.rag_helper.summarize([prompt_risks], context="risk assessment")
 
    def newsletter_generator(self, company_insights, market_trends, risks):
        """Generate the newsletter."""
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
st.title("Market Data Newsletter with CrewAI, OpenAI, and RAG")
 
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
        # Query and summarize company insights
        company_insights = crew_instance.rag_helper.query("news_collection", "latest company news")
        summarized_company = crew_instance.company_analyst(company_insights) if company_insights else "No company insights available."
 
        # Query and summarize market trends
        market_trends = crew_instance.rag_helper.query("trends_collection", "latest market trends")
        summarized_trends = crew_instance.market_trends_analyst(market_trends) if market_trends else "No market trends available."
 
        # Assess risks
        risks = crew_instance.risk_manager(summarized_company, summarized_trends) if summarized_company and summarized_trends else "No risk analysis available."
 
        # Generate newsletter
        newsletter = crew_instance.newsletter_generator(summarized_company, summarized_trends, risks)
        st.markdown(newsletter)
 
        # Assess accuracy with Bespoke Labs
        rag_context = "\n".join(company_insights + market_trends)
        accuracy_score = assess_accuracy_with_bespoke(newsletter, rag_context)
        st.markdown(f"**Accuracy Score:** {accuracy_score}%")
    except Exception as e:
        st.error(f"Error generating newsletter or assessing accuracy: {e}")