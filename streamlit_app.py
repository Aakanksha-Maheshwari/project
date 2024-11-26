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
                messages=[{"role": "user", "content": prompt}]
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

# Helper Functions for Refinement and Accuracy
def filter_context(rag_data, keywords):
    """Filter RAG data for relevance based on specific keywords."""
    return [item for item in rag_data if any(keyword.lower() in item.lower() for keyword in keywords)]

def refine_newsletter(content):
    """Refine newsletter content using GPT for better accuracy and clarity."""
    prompt = f"Refine the following newsletter content to improve accuracy and verifiability:\n\n{content}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error refining newsletter: {e}")
        return content

def summarize_context_for_claim(rag_context, claim):
    """Summarize RAG context for better alignment with the claim."""
    prompt = f"Summarize the following context to match the claim:\n\nClaim:\n{claim}\n\nContext:\n{rag_context}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error summarizing context for claim: {e}")
        return rag_context

def assess_accuracy_with_bespoke(newsletter_content, rag_context):
    """Assess the accuracy of the newsletter content using Bespoke Labs."""
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

# Streamlit Interface
st.title("Market Data Newsletter with Accuracy Improvement")

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
        summarized_company = crew_instance.company_analyst(company_insights)

        # Query and summarize market trends
        market_trends = crew_instance.rag_helper.query("trends_collection", "latest market trends")
        summarized_trends = crew_instance.market_trends_analyst(market_trends)

        # Assess risks
        risks = crew_instance.risk_manager(summarized_company, summarized_trends)

        # Generate and refine newsletter
        newsletter = crew_instance.newsletter_generator(summarized_company, summarized_trends, risks)
        refined_newsletter = refine_newsletter(newsletter)

        # Filter and summarize context
        keywords = ["Townsquare", "Intel", "Xerox", "market trends"]
        filtered_context = filter_context(company_insights + market_trends, keywords)
        rag_context = summarize_context_for_claim("\n".join(filtered_context[:5]), refined_newsletter)

        # Assess accuracy
        accuracy_score = assess_accuracy_with_bespoke(refined_newsletter, rag_context)

        # Display results
        st.markdown(refined_newsletter)
        st.markdown(f"**Accuracy Score:** {accuracy_score}%")
    except Exception as e:
        st.error(f"Error generating newsletter or assessing accuracy: {e}")
