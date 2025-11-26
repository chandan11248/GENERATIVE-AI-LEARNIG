from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="moonshotai/Kimi-K2-Thinking", 
                        task="text-generation")
model=ChatHuggingFace(llm=llm)
result=model.invoke("teh data you have from which year ?? ")
print(result.content)

