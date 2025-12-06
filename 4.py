from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2B",
    torch_dtype=torch.float16,
)

pipe.to("mps")  # mps for Mac, cuda for NVIDIA

result = pipe(
    prompt="A young man walking on the street, 4k, realistic",
    guidance_scale=6.0,
    num_inference_steps=30
)

video = result["videos"][0]
video.save("output.mp4")
print("Saved output.mp4")