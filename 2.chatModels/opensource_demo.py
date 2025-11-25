from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", 
                        task="text-generation")
model=ChatHuggingFace(llm=llm)
result=model.invoke("what is the capital of nepal? ")
print(result.content)
