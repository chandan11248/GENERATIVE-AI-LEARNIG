import torch
from diffusers import DiffusionPipeline
import imageio
import os

# Enable MPS (Apple GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Load model (optimized for 2B)
pipe = DiffusionPipeline.from_pretrained(
    "THUDM/CogVideoX-2B",
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.to(device)

prompt = "A young man walking on a street, cinematic, 4k"

print("Generating video frames...")

video_frames = pipe(
    prompt,
    num_frames=48,        # best balance for MacBook
    guidance_scale=6.5,   # sharpness
    num_inference_steps=20
).frames

# Save output
output_path = "output.mp4"
imageio.mimsave(output_path, video_frames, fps=8)

print(f"Video saved to {output_path}")