from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chat_history = []
print("AI: hi there !!")

while True:
    user_input = input("human: ")
    if user_input=="exit":
        break
    context = "\n".join(chat_history + [f"Human: {user_input}"])

    result = llm.invoke(context)
    chat_history.append(f"Human: {user_input}")
    chat_history.append(f"AI: {result.content}")
    print("AI:", result.content)
 