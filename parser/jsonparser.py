from langchain_groq  import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


templete=PromptTemplate(content="tell me the name of the  5 top rich {city} ",
                        input_variables=["city"])
promt=templete.invoke({"city":"city"})
parsor=JsonOutputParser()