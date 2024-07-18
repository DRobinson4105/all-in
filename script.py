"""
1. Pass task through mistral to classify task (video-exercise, apple-health, image-question-answering)
2. video-exercise:
    - Pass task through mistral to classify exercise (pushup, pullup, squat)
    - Pass video through yolov8 to get count
    - Pass task and count to determine if task is completed
   apple-health:
    - tbd
   image-analysis:
    - Pass task and image through fine-tuned blip2 to determine if task is completed
   text-analysis-in-image
"""

from PIL import Image
import cv2
from utils import blip2, mistral, pose_count
import torch

# task = "Do 10 push ups"
# proof = cv2.VideoCapture("demos/pose-estimation/video.mp4")
task = "Have good posture"
proof = Image.open("demos/visual-question-answering/good_posture.png")


device = "cuda" if torch.cuda.is_available() else "cpu"
mistral_model = mistral(device)
blip2_model = blip2(device)

def verify_task(task, proof):
    task_types = ['video_exercise', 'apple_health', 'text_analysis_in_image', 'image_analysis', 'unknown']

    task_classification_prompt = f"""
    Classify the following task into the correct category and respond with that category exactly.
    These are the categories: {','.join(task_types)}. For example, if the task is 'Do a pull up',
    then the output would be video_exercise. If the task is not any of these activities then just
    respond with 'unknown' exactly. Do not explain anything and just respond with the category that
    fits the best.
    """

    task_type = mistral_model.run_inference(task_classification_prompt, task)
    if (task_type == 'video_exercise'):
        return verify_exercise(task, proof)
    elif (task_type == 'apple_health'):
        return verify_apple_health(task, proof)
    elif (task_type == 'image_analysis' or (task_type == 'unknown' and isinstance(proof, Image.Image))):
        return verify_image_analysis(task, proof)
    elif (task_type == 'text_analysis_in_image'):
        return verify_text_analysis_in_image(task, proof)
    else:
        return 'unknown'
    
def process_output(output):
    if (output == 'yes'):
        return 'yes'
    elif (output == 'no'):
        return 'no'
    else:
        return 'unknown'

def verify_exercise(task, proof):
    if not isinstance(proof, cv2.VideoCapture):
        return 'unknown'
    
    pose_types = ["pushup", "pullup", "squat"]
    pose_classification_prompt = f"""
    Classify the following task into the correct activity and respond with that category exactly.
    If the task is not any of these activities then respond with 'unknown'. These are the
    categories: {','.join(pose_types)}. For example, if the task is 'Do a pull up', then the output
    would be pullup.
    """

    pose_type = mistral_model.run_inference(pose_classification_prompt, task)

    if pose_type not in pose_types:
        return 'unknown'

    result = str(pose_count(pose_type, proof))
    task_verification_prompt = f"""
    Given the result of the count from the pose estimation model, determine if the task, {task},
    was completed. The task can be more than completed. For example, if someone did 7 pull ups and
    the task was to do 5 pull ups, the task would be completed. Just output 'yes' if the task was
    completed or 'no' if the task was not completed.
    """

    output = mistral_model.run_inference(task_verification_prompt, result)
    return process_output(output)

def verify_apple_health(task, proof):
    return 'unknown'

def verify_image_analysis(task, proof):
    if not isinstance(proof, Image.Image):
        return 'unknown'
    
    prompt = f"Determine if the task, {task}, is met from this image as proof (just answer 'yes' or 'no' or if the answer cannot be verified, respond with 'unknown')"

    output = blip2_model.run_inference(proof, prompt)
    return process_output(output)

def verify_text_analysis_in_image(task, proof):
    return 'unknown'

print(verify_task(task, proof))