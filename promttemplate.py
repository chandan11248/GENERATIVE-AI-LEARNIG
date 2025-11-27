from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")   # FIXED

template = PromptTemplate(
    template="tell me the name of a random {types} guy.",
    input_variables=['types'],
)

prompt = template.format(types="asian")   # FIXED format usage

result = model.invoke(prompt)
print(result.content)


#another technique 
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(
    "Tell me the name of a random {types} guy."
)

prompt = template.invoke({"types": "asian"})   # <-- DICT WORKS HERE