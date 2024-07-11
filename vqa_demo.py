from utils import blip2
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
blip2 = blip2(device=device)
question = "Does the person have good posture or bad posture?"

# Good posture
image = Image.open("good_posture.png")
answer = blip2.run_inference(image, question)
print(f"{question} {answer}")

# Bad posture
image = Image.open("bad_posture.png")
answer = blip2.run_inference(image, question)
print(f"{question} {answer}")