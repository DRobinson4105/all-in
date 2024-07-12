import os
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from utils import mistral, pose_count

device = "cuda" if torch.cuda.is_available() else "cpu"
mistral = mistral(device)

task = "Do 10 push ups"

pose_types = ["pushup", "pullup", "abworkout", "squat"]
pose_classification_prompt = f"""
Classify the following task into the correct activity and respond with that category exactly. If
the task is not any of these activities then respond with 'no'. These are the categories:
{','.join(pose_types)}. For example, if the task is 'Do a pull up', then the output would be
pullup.
"""

pose_type = mistral.run_inference(pose_classification_prompt, task)

if pose_type == 'no': raise ValueError("Not a valid video")
print(f"Task is classified as {pose_type}")

result = str(pose_count(pose_type))
print(f"{result} repetitions done")
task_verification_prompt = f"""
Given the result of the count from the pose estimation model, determine if the task, {task}, was
completed. The task can be more than completed. For example, if someone did 7 pull ups and the task
was to do 5 pull ups, the task would be completed. Just output 'yes' if the task was completed or
'no' if the task was not completed
"""

output = mistral.run_inference(task_verification_prompt, result)
passed = output == "yes"
print("Completed task" if passed else "Did not complete task")