import os
import cv2
import numpy as np
from glob import glob 
from tqdm import tqdm, trange

VIDEOS_FOLDER = "/home/kylehatch/Desktop/VideoGlue/videoglue.github.io/static/videos"





# Function to add white box with text to the bottom of each frame
def add_white_box_with_text_to_frame(frame, box_height, text):
    height, width, _ = frame.shape
    new_height = height + box_height
    white_box = 255 * np.ones((box_height, width, 3), dtype=np.uint8)

    # Add text to the white box
    font = cv2.FONT_HERSHEY_COMPLEX # cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (white_box.shape[1] - text_size[0]) // 2
    text_y = (white_box.shape[0] + text_size[1]) // 2
    cv2.putText(white_box, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    frame_with_box = np.vstack((frame, white_box))
    return frame_with_box

# Function to add additional text within the frame
def add_additional_text_to_frame(frame):
    height, width, _ = frame.shape

    # Add "goal image" text
    goal_text = "generated subgoal image"
    font = cv2.FONT_HERSHEY_COMPLEX
    # font_scale = 0.7
    # font_thickness = 2
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)
    goal_text_size = cv2.getTextSize(goal_text, font, font_scale, font_thickness)[0]
    goal_text_x = (width - goal_text_size[0]) // 2
    goal_text_y = 10 + goal_text_size[1]
    cv2.putText(frame, goal_text, (goal_text_x, goal_text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Add "observation image" text
    observation_text = "observation image"
    observation_text_size = cv2.getTextSize(observation_text, font, font_scale, font_thickness)[0]
    observation_text_x = (width - observation_text_size[0]) // 2
    observation_text_y = (height // 2) + 10 + observation_text_size[1]
    cv2.putText(frame, observation_text, (observation_text_x, observation_text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return frame



TASKS_DICT = {"2024-06-11_17-14-12":"put banana in drawer",
        "2024-06-11_17-26-30":"put sushi in bowl",
        "2024-06-11_17-40-11":"put banana in drawer",
        "2024-06-11_17-43-41":"put sushi in bowl",
        "2024-06-11_17-48-38":"put sushi in bowl",
        "2024-06-11_17-55-14":"put banana in drawer",
        "2024-06-11_17-57-49":"put banana in drawer"}

input_video_paths = glob(os.path.join(VIDEOS_FOLDER, "2024-*.mp4"))
input_video_paths = [path for path in input_video_paths if "annotated" not in path]

for input_video_path in tqdm(input_video_paths):
    # output_video_path = input_video_path.replace(".mp4", "_annotated.mp4")
    output_video_path = input_video_path.replace(".mp4", "_annotated.webm")

    # Load video
    # input_video_path = 'input_video.mp4'
    # output_video_path = 'output_video_with_white_box.mp4'
    box_height = 30
    video_name = input_video_path.split("/")[-1].split(".")[0]
    text = TASKS_DICT[video_name]

    cap = cv2.VideoCapture(input_video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'H264')#(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height + box_height))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height + box_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add "goal image" and "observation image" text
        frame = add_additional_text_to_frame(frame)

        # Add white box with "Hello world" text
        frame_with_box = add_white_box_with_text_to_frame(frame, box_height, text)
        out.write(frame_with_box)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Finished processing the videos.")