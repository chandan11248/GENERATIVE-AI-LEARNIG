from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "what is the capital of America?"
document={
    "hello","my name is chandan shah"," wanna learn ever things ...",
    "want to see the capacity of human brain "
}
vector = embedding.embed_documents(document)
print(vector)