import streamlit as st
import requests
import json
import openai
from crewai import Agent
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
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]


# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

### Define Agents ###

# Company Analyst Agent
class CompanyAnalystAgent:
    def handle(self):
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()

        if "feed" in data:
            company_news = [
                item for item in data["feed"]
                if "company" in [topic["topic"] for topic in item.get("topics", [])]
            ]
            return {"company_news": company_news[:5]}  # Limit to top 5 news for efficiency
        return {"company_news": []}

# Market Trends Analyst Agent
class MarketTrendsAnalystAgent:
    def handle(self):
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()

        if "top_gainers" in data:
            return {
                "market_trends": {
                    "top_gainers": data["top_gainers"][:5],  # Limit for efficiency
                    "top_losers": data["top_losers"][:5],
                    "most_actively_traded": data["most_actively_traded"][:5]
                }
            }
        return {"market_trends": {}}

# Risk Management Analyst Agent
class RiskManagementAnalystAgent:
    def handle(self):
        risks = [
            "High volatility in technology stocks.",
            "Potential downturn in global markets due to geopolitical tensions.",
        ]
        strategies = ["Consider hedging with options.", "Diversify portfolio."]
        return {"risks": risks, "strategies": strategies}

# Newsletter Generator Agent
class NewsletterGeneratorAgent:
    def __init__(self, company_news, market_trends, risks, strategies):
        self.company_news = company_news
        self.market_trends = market_trends
        self.risks = risks
        self.strategies = strategies

    def handle(self):
        input_text = f"""
        Company News: {json.dumps(self.company_news, indent=2)}
        Market Trends:
          Top Gainers: {json.dumps(self.market_trends.get('top_gainers', []), indent=2)}
          Top Losers: {json.dumps(self.market_trends.get('top_losers', []), indent=2)}
          Most Actively Traded: {json.dumps(self.market_trends.get('most_actively_traded', []), indent=2)}
        Risks: {json.dumps(self.risks, indent=2)}
        Strategies: {json.dumps(self.strategies, indent=2)}
        """
        
        try:
            # Use OpenAI's Chat API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Adjust model as per your access level
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing financial data into a concise newsletter."},
                    {"role": "user", "content": f"Summarize the following data into a newsletter:\n{input_text}"}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return {"newsletter": response.choices[0].message["content"].strip()}
        except Exception as e:
            return {"error": f"Failed to generate newsletter: {str(e)}"}

### Streamlit Functions ###

def generate_newsletter_with_agents():
    st.write("### Generating Newsletter with Multi-Agent System")

    # Instantiate and run agents manually
    company_analyst = CompanyAnalystAgent()
    market_trends_analyst = MarketTrendsAnalystAgent()
    risk_management_analyst = RiskManagementAnalystAgent()

    # Collect results from agents
    company_news_result = company_analyst.handle()
    market_trends_result = market_trends_analyst.handle()
    risk_management_result = risk_management_analyst.handle()

    # Use the results in the Newsletter Generator Agent
    newsletter_agent = NewsletterGeneratorAgent(
        company_news=company_news_result["company_news"],
        market_trends=market_trends_result["market_trends"],
        risks=risk_management_result["risks"],
        strategies=risk_management_result["strategies"]
    )
    newsletter_result = newsletter_agent.handle()

    # Display the generated newsletter or handle errors
    if "error" in newsletter_result:
        st.error(newsletter_result["error"])
    else:
        st.subheader("Generated Newsletter")
        st.text(newsletter_result["newsletter"])

# Function to Load News Data
def load_news_data():
    # Implementation remains the same as your original function
    ...

# Function to Retrieve News Data
def retrieve_news_data():
    # Implementation remains the same as your original function
    ...

# Function to Load Ticker Trends Data
def load_ticker_trends_data():
    # Implementation remains the same as your original function
    ...

# Function to Retrieve Ticker Trends Data
def retrieve_ticker_trends_data():
    # Implementation remains the same as your original function
    ...

### Main Logic ###
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    retrieve_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    retrieve_ticker_trends_data()
elif option == "Generate Newsletter":
    generate_newsletter_with_agents()
