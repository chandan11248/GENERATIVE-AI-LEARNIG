from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

json_schema = {
    "title": "ChatOutput",
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "sentiment": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["reply", "sentiment", "keywords"]
}

model = ChatGroq(model="llama-3.3-70b-versatile")
structured = model.with_structured_output(json_schema)

chat_history = []

while True:
    user_input = input("human: ")
    if user_input == "exit":
        break

    chat_history.append(f"Human: {user_input}")

    context = "\n".join(chat_history)

    result = structured.invoke(context)

    # store AI reply inside history
    chat_history.append(f"AI: {result['reply']}")

    print(json.dumps(result, indent=2))