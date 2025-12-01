from dotenv import load_dotenv
load_dotenv()   # loads OPENAI_API_KEY from .env automatically

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message["content"])