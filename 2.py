import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.tools import tool

# Load environment variables from .env
load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found. Please set it in the .env file.")

client = InferenceClient(
    provider="nscale",
    
)

@tool
def generate_image(prompt: str):
    """Generate an image using SDXL text-to-image."""
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    save_path = "output.png"
    image.save(save_path)
    return f"Image generated and saved to {save_path}"

# Example run
if __name__ == "__main__":
    print(generate_image.run("Astronaut riding a horse"))