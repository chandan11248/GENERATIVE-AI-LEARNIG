from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-4" ,temperature=0)
result=llm.invoke("what is the capital of nepal")
print(result.content) 