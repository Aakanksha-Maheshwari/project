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
alpha_vantage_key = st.secrets["api_keys"]["alpha_vantage"]
openai.api_key = st.secrets["api_keys"]["openai"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent RAG System with Crew AI")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

### Define Crew AI Agents ###

# Company Analyst Agent
class CompanyAnalystAgent(Agent):
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
class MarketTrendsAnalystAgent(Agent):
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
class RiskManagementAnalystAgent(Agent):
    def handle(self):
        # Simulated risk analysis
        risks = [
            "High volatility in technology stocks.",
            "Potential downturn in global markets due to geopolitical tensions.",
        ]
        strategies = ["Consider hedging with options.", "Diversify portfolio."]
        return {"risks": risks, "strategies": strategies}

# Newsletter Generator Agent
class NewsletterGeneratorAgent(Agent):
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
        
        # Use OpenAI API to summarize the data
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Adjust model based on your access
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing financial data into a concise newsletter."},
                {"role": "user", "content": f"Summarize the following data into a newsletter:\n{input_text}"}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        return {"newsletter": response["choices"][0]["message"]["content"].strip()}

### Crew AI Multi-Agent System ###
system = System()
system.add_agent("company_analyst", CompanyAnalystAgent())
system.add_agent("market_trends_analyst", MarketTrendsAnalystAgent())
system.add_agent("risk_management_analyst", RiskManagementAnalystAgent())

### Streamlit Functions ###

def generate_newsletter_with_agents():
    st.write("### Generating Newsletter with Multi-Agent System")

    # Run the first three agents concurrently
    results = system.run_all()

    # Extract data from agent results
    company_news = results["company_analyst"]["company_news"]
    market_trends = results["market_trends_analyst"]["market_trends"]
    risks = results["risk_management_analyst"]["risks"]
    strategies = results["risk_management_analyst"]["strategies"]

    # Create and run the Newsletter Generator Agent
    newsletter_agent = NewsletterGeneratorAgent(
        company_news=company_news,
        market_trends=market_trends,
        risks=risks,
        strategies=strategies
    )
    system.add_agent("newsletter_generator", newsletter_agent)
    newsletter_result = system.run("newsletter_generator")

    # Display the generated newsletter
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
