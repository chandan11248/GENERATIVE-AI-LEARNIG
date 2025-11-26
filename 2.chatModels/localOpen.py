from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()

# Check if key loaded correctly
print("OpenAI Key =", os.getenv("OPENAI_API_KEY"))

# Create model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Call the model
result = llm.invoke("What is the capital of Nepal?")
print("Model Output:", result.content)