from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

# Define template 
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer service agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load chat history
chat_history = []
with open("chat_history.txt") as f:
    lines = f.readlines()
    chat_history = lines  
    # print(chat_history)


while True:
    user_input=input("You:")
    if user_input=="exit":
        break
    prompt = template.invoke({
    "query": user_input,
    "chat_history": chat_history})
    chat_history.append(HumanMessage(content=user_input))
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content )

