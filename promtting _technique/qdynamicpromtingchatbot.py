from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional {domain}. Explain in {style} style."),
    ("human", "Explain the paper 'Attention Is All You Need'.")
])

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = template.invoke({
    "domain": "researcher",
    "style": "detailed"
})

result = model.invoke(prompt)
print(result.content)