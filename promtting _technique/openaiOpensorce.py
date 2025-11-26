from langchain.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", 
                        task="text-generation")
model=ChatHuggingFace(llm=llm)
chat_history=[
    SystemMessage(content="tell me in advanced mathematics "),
]
while True:
    user_input=input("You:")
    if user_input=="exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content )

print(chat_history)  