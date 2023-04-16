import cv2
import mediapipe as mp
import numpy as np

# Helper functions
def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)

def draw_line(image, p1, p2, color=(0, 255, 0), thickness=2):
    cv2.line(image, p1, p2, color, thickness)

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Detection function
def detect_body_parts(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Extract head, hands, and shoulders landmarks
    landmarks = results.pose_landmarks.landmark
    head = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return head, left_hand, right_hand, left_shoulder, right_shoulder

# Video processing
def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    previous_shoulder_line = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        head, left_hand, right_hand, left_shoulder, right_shoulder = detect_body_parts(frame)

        # Draw head bounding box
        draw_bounding_box(frame, (int(head.x * width) - 50, int(head.y * height) - 50, 100, 100))

        # Draw hand bounding boxes
        draw_bounding_box(frame, (int(left_hand.x * width) - 30, int(left_hand.y * height) - 30, 60, 60))
        draw_bounding_box(frame, (int(right_hand.x * width) - 30, int(right_hand.y * height) - 30, 60, 60))

        # Draw shoulder line
        left_shoulder_point = (int(left_shoulder.x * width), int(left_shoulder.y * height))
        right_shoulder_point = (int(right_shoulder.x * width), int(right_shoulder.y * height))
        draw_line(frame, left_shoulder_point, right_shoulder_point)

        # Detect shoulder shrugging
        if previous_shoulder_line:
            shoulder_line_change = np.abs(np.array(left_shoulder_point) - np.array(previous_shoulder_line[0]))

            if shoulder_line_change > 5:  # Threshold for shoulder shrug detection
                draw_text(frame, "Shrug", (10, 50), font_scale=1, color=(255, 255, 255), thickness=2)
                draw_line(frame, left_shoulder_point, right_shoulder_point, color=(0, 0, 255))

        previous_shoulder_line = left_shoulder_point

        out.write(frame)
        cv2.imshow('Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "input1.mp4"
    output_video = "output1.mp4"
    process_video(input_video, output_video)
