from PIL import Image
import torch
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from utils import blip2, mistral

task = "Have good posture"
proof = Image.open("bad_posture.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
blip2_model = blip2(device)
mistral_model = mistral(device, os.path.join(os.getcwd(), '..', '..', 'mistral'))

output = blip2_model.run_inference(proof, f"Determine if the task, {task}, is met from this image as proof (just answer 'yes' or 'no' or if the answer cannot be verified, respond with 'unknown')")
passed = output == "yes"
print(output)
print("Task completed" if passed else "Task not completed")