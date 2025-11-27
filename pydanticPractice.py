from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel,Field
from typing import Optional, Literal

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=1.2)  
template = PromptTemplate(
    template="tell me the name of a random {types} guy.",
    input_variables=['types'],
)

class infos (BaseModel):
    name:str
    age:int=Field(gt=0 ,lt=35,description=("the age of the person"))


structured_model=model.with_structured_output(infos)


prompt = template.format(types="asian")   # FIXED format usage

result = structured_model.invoke(prompt)
print(result)
