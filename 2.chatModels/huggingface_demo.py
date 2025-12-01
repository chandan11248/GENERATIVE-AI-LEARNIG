from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", 
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

result=model.invoke("/Users/owner/Desktop/LangChain/2.chatModels/7F750683-FC2E-44C8-A6D3-BFC83AD27060_1_201_a.jpeg")
print(result.content)

