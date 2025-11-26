from langchain_groq import ChatGroq
from langchain.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

chat_history=[
    SystemMessage(content="you are a helpful  assistant with "
    "funny tone and respect and clear the doubt in a  simple and detail manner "
    "like  you are teaching a  10 grade student "),

]
while True:
    user_input=input("You:")
    if user_input=="exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    result=llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content )

print(chat_history)