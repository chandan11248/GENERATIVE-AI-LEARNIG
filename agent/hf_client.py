from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from .config import HF_DEEPSEEK_REPO_ID


def make_deepseek_model(temperature: float = 0.2) -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id=HF_DEEPSEEK_REPO_ID,
        task="text-generation",
        # Reasoning models can produce long outputs; keep decode reasonable
        # You can tune generation args via HfEndpoint kwargs if needed.
    )
    model = ChatHuggingFace(llm=llm, temperature=temperature)
    return model
