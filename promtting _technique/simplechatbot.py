from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
