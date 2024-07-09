import cv2
from ultralytics import YOLO
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

yolov8_pose_model = YOLO("yolov8n-pose.pt")

class mistral:
    def __init__(self, device="cpu"):
        self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
 
        self.mistral_7b_model = AutoModelForCausalLM.from_pretrained(
            "mistral", 
            quantization_config=self.bnb_config, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.mistral_7b_tokenizer = AutoTokenizer.from_pretrained("mistral")
        self.mistral_7b_tokenizer.pad_token_id = self.mistral_7b_tokenizer.eos_token_id

        self.device = device
    
    def run_inference(self, context, prompt):
        messages = [{"role": "user", "content": f"{context}: {prompt}"}]
        encoded = self.mistral_7b_tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encoded.to(self.device)

        generated_ids = self.mistral_7b_model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        decoded = self.mistral_7b_tokenizer.batch_decode(generated_ids)

        output = decoded[0]
        result = output[output.find("[/INST]")+7:output.find("</s>")].replace(" ", "").lower()
        return result

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def pose_count(pose_type):    
    # line_thickness=2 #display
    kpts_to_check=[6, 8, 10]
    poseup_angle=145.0
    posedown_angle=90.0
    angle = None
    count = None
    stage = None

    cap = cv2.VideoCapture("video.mp4")
    assert cap.isOpened(), "Error reading video file"

    frame_count = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success: break

        frame_count += 1
        results = yolov8_pose_model.track(im0, verbose=False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Window closed by user.")
            break
        
        try:
            if frame_count == 1:
                count = [0] * len(results[0])
                angle = [0] * len(results[0])
                stage = ["-" for _ in results[0]]

            keypoints = results[0].keypoints.data
            # annotator = Annotator(im0, line_width=line_thickness) #display

            for ind, k in enumerate(reversed(keypoints)):
                if pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                    angle[ind] = calculate_angle(
                        k[int(kpts_to_check[0])].cpu(),
                        k[int(kpts_to_check[1])].cpu(),
                        k[int(kpts_to_check[2])].cpu(),
                    )
                    # im0 = annotator.draw_specific_points(k, kpts_to_check, shape=(640, 640), radius=10) #display

                    # Check and update pose stages and counts based on angle
                    if pose_type in {"abworkout", "pullup"}:
                        if angle[ind] > poseup_angle:
                            stage[ind] = "down"
                        if angle[ind] < posedown_angle and stage[ind] == "down":
                            stage[ind] = "up"
                            count[ind] += 1

                    elif pose_type in {"pushup", "squat"}:
                        if angle[ind] > poseup_angle:
                            stage[ind] = "up"
                        if angle[ind] < posedown_angle and stage[ind] == "up":
                            stage[ind] = "down"
                            count[ind] += 1

                    # annotator.plot_angle_and_count_and_stage( #display
                    #     angle_text=angle[ind],
                    #     count_text=count[ind],
                    #     stage_text=stage[ind],
                    #     center_kpt=k[int(kpts_to_check[1])],
                    # )
                    
                # annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True) #display

        except:
            continue

    cv2.destroyAllWindows()
    return max(count)