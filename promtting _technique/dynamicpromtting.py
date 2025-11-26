# using streamlit and promt engineering technique to see how will we can use it to  do the  things 
from langchain_groq import ChatGroq
import streamlit as st 
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
st.header("research tool")
# user_input=st.text_input("enter your prompt")

paper_input = st.selectbox("select the paper ",[ "Select...", "Attention Is All You Need", 
                                                "BERT: Pre-training of Deep Bidirectional Transformers",
                                                  "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style", 
                           ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox( "Select Explanation Length", 
                            ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

promt=["generate the  deep and brief explation of {paper_input} in {style_input} style in {length_input}"]

if st.button("Summarize"):
    prompt = f"Generate a deep and brief explanation of {paper_input} in {style_input} style in {length_input}"
    result = llm.invoke(prompt)
    st.write(result.content)
 