import os
import json
import difflib
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

# Load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect database
db = SQLDatabase.from_uri("sqlite:///database.db")

# Initialize LLM only if API key is available
llm = None
toolkit = None
agent_executor = None

if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )
    except Exception as e:
        print(f"[Warning] Could not initialize ChatOpenAI: {e}")
else:
    print("[Info] No OPENAI_API_KEY set. AI features disabled.")

# Load FAQ data
faq_path = os.path.join(os.path.dirname(__file__), "water_faq.json")
with open(faq_path, "r") as f:
    faq_data = json.load(f)

def check_faq(question: str) -> str:
    question_lower = question.lower().strip()
    faq_questions = list(faq_data.keys())
    matches = difflib.get_close_matches(question_lower, faq_questions, n=1, cutoff=0.6)
    if matches:
        return faq_data[matches[0]]
    return None

def ask_question(question: str) -> str:
    # First check FAQ
    faq_answer = check_faq(question)
    if faq_answer:
        return faq_answer

    # Then try SQL agent if available
    if agent_executor:
        try:
            response = agent_executor.run(question)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return "AI service is not available because no API key is set."
