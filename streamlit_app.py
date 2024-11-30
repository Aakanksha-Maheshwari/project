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

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent RAG System with Crew AI")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Generate Newsletter"]
)

### Define Agents ###

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
    def __init__(self):
        self.market_trends = None

    def handle(self, inputs):
        market_trends = inputs.get("market_trends", {})
        risks = []
        strategies = []

        if market_trends.get("top_gainers"):
            risks.append("Potential overvaluation in top gainers.")
        if market_trends.get("top_losers"):
            risks.append("Significant risks in top losers.")

        strategies.append("Diversify investments to reduce risk.")
        strategies.append("Focus on long-term trends for stability.")

        return {"risks": risks, "strategies": strategies}


# Newsletter Generator Agent
class NewsletterGeneratorAgent(Agent):
    def __init__(self):
        self.company_news = None
        self.market_trends = None
        self.risks = None
        self.strategies = None

    def handle(self, inputs):
        self.company_news = inputs.get("company_news", [])
        self.market_trends = inputs.get("market_trends", {})
        self.risks = inputs.get("risks", [])
        self.strategies = inputs.get("strategies", [])

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
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing financial data into a concise newsletter."},
                    {"role": "user", "content": f"Summarize the following data into a newsletter:\n{input_text}"}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return {"newsletter": response.choices[0].message.content.strip()}
        except Exception as e:
            return {"error": f"Failed to generate newsletter: {str(e)}"}


### Define Tasks ###
company_task = Task(
    description="Analyze company-related news.",
    expected_output="Company news insights",
    agent=CompanyAnalystAgent(),
)

market_trends_task = Task(
    description="Analyze market trends.",
    expected_output="Market trends insights",
    agent=MarketTrendsAnalystAgent(),
)

risk_task = Task(
    description="Perform risk analysis based on market trends.",
    expected_output="Risk analysis and mitigation strategies",
    agent=RiskManagementAnalystAgent(),
    context=[market_trends_task],
)

newsletter_task = Task(
    description="Generate a newsletter summarizing all data.",
    expected_output="A concise financial newsletter",
    agent=NewsletterGeneratorAgent(),
    context=[company_task, market_trends_task, risk_task],
)

### Assemble the Crew ###
my_crew = Crew(
    agents=[
        company_task.agent,
        market_trends_task.agent,
        risk_task.agent,
        newsletter_task.agent,
    ],
    tasks=[company_task, market_trends_task, risk_task, newsletter_task],
)

### Streamlit Function ###
def generate_newsletter():
    st.write("### Generating Newsletter")

    try:
        # Run the Crew AI tasks
        results = my_crew.kickoff(inputs={})

        # Display the newsletter
        newsletter = results["newsletter_task"]
        if "error" in newsletter:
            st.error(newsletter["error"])
        else:
            st.subheader("Generated Newsletter")
            st.text(newsletter["newsletter"])
    except Exception as e:
        st.error(f"Failed to run crew tasks: {e}")


### Main Logic ###
if option == "Generate Newsletter":
    generate_newsletter()
