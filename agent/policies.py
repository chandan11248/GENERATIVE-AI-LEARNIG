from textwrap import dedent
from .hf_client import make_deepseek_model
from langchain_core.prompts import ChatPromptTemplate


POLICY_SNIPPETS = dedent(
    """
    Store Policies (Summary):
    - Shipping: Standard 3-5 business days, express 1-2 business days.
    - Returns: 30-day return window for unused items in original packaging.
    - Refunds: Issued to original payment method within 5-7 business days after inspection.
    - International: Shipping fees vary by region; duties/taxes are buyer responsibility.
    - Warranty: 1-year limited warranty on manufacturing defects for footwear and apparel.
    """
)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful support agent. Use the provided policy snippets to answer the user.
If the answer is not in the policy, say you don't know and keep it brief.

Policy snippets:
{policies}

User question:
{question}
"""
)


def answer_policy_question(question: str) -> str:
    model = make_deepseek_model()
    chain = prompt | model
    result = chain.invoke({"policies": POLICY_SNIPPETS, "question": question})
    return result.content
