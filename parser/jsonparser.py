from langchain_groq  import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
model=ChatGroq(model="llama-3.3-70b-versatile")

parsor=JsonOutputParser()
templete=PromptTemplate(template="""give name ,age and address of a frictional character  \n
                        {format_instruction}""",
                        input_variables=[],
                        partial_variables={"format_instruction":parsor.get_format_instructions()})
promt=templete.invoke({})
chain = templete | model | parsor
result = chain.invoke({})
print(result)