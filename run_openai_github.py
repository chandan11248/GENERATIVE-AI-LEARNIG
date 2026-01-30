import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Read the content of text.txt and truncate if necessary
file_path = "text.txt"
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        text_content = f.read()
    # Truncate to approximately 6000-7000 tokens to fit within 8000 limit
    # 25000 characters is a safe approximation.
    if len(text_content) > 25000:
        text_content = text_content[:25000] + "..."
else:
    print(f"Error: {file_path} not found.")
    sys.exit(1)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes text briefly.",
        },
        {
            "role": "user",
            "content": f"Please briefly summarize the following content:\n\n{text_content}",
        }
    ],
    temperature=1,
    top_p=1,
    model=model,
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
