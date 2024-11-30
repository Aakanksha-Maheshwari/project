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
import yaml

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Load Agent Configurations
with open("agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai_api_key = st.secrets["openai"]["api_key"]

# Set OpenAI API Key
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent RAG System")

# Sidebar Options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Generate Newsletter"]
)

### Define Agents ###
news_loader = Agent(
    role=agents_config["news_loader"]["role"],
    goal=agents_config["news_loader"]["goal"],
    backstory=agents_config["news_loader"]["backstory"]
)

ticker_loader = Agent(
    role=agents_config["ticker_loader"]["role"],
    goal=agents_config["ticker_loader"]["goal"],
    backstory=agents_config["ticker_loader"]["backstory"]
)

summarizer = Agent(
    role=agents_config["summarizer"]["role"],
    goal=agents_config["summarizer"]["goal"],
    backstory=agents_config["summarizer"]["backstory"]
)

newsletter_writer = Agent(
    role=agents_config["newsletter_writer"]["role"],
    goal=agents_config["newsletter_writer"]["goal"],
    backstory=agents_config["newsletter_writer"]["backstory"]
)

### Define Tasks ###
news_task = Task(
    description="Load news data from Alpha Vantage into ChromaDB.",
    expected_output="News data loaded successfully.",
    agent=news_loader,
    callback=lambda output: st.info(f"News Task Completed: {output.raw}"),
)

ticker_task = Task(
    description="Load ticker trends data from Alpha Vantage into ChromaDB.",
    expected_output="Ticker trends data loaded successfully.",
    agent=ticker_loader,
    callback=lambda output: st.info(f"Ticker Task Completed: {output.raw}"),
)

summarization_task = Task(
    description="Retrieve and summarize data from ChromaDB.",
    expected_output="Data summarized successfully.",
    agent=summarizer,
    context=[news_task, ticker_task],
    callback=lambda output: st.info(f"Summarization Task Completed: {output.raw}"),
)

newsletter_task = Task(
    description="Generate a concise newsletter summarizing market data.",
    expected_output="Newsletter generated successfully.",
    agent=newsletter_writer,
    context=[summarization_task],
    callback=lambda output: st.success(f"Newsletter:\n{output.raw}"),
)

### Assemble the Crew ###
my_crew = Crew(
    agents=[news_loader, ticker_loader, summarizer, newsletter_writer],
    tasks=[news_task, ticker_task, summarization_task, newsletter_task],
    verbose=True
)

### Streamlit ###
def generate_newsletter():
    st.write("### Generating Newsletter...")
    try:
        # Remove `inputs` from kickoff
        results = my_crew.kickoff()
        
        # Access the newsletter task output directly
        newsletter_output = newsletter_task.output.raw
        st.success(f"Newsletter Generated:\n{newsletter_output}")
    except Exception as e:
        st.error(f"Failed to generate newsletter: {e}")


### Main Logic ###
if option == "Generate Newsletter":
    generate_newsletter()
