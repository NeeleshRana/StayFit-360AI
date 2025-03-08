import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    angle = np.abs(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])) * 180.0 / np.pi
    return angle if angle <= 180 else 360 - angle

def track_pushups():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    pushup_counter = 0
    stage = None
    hip_error = ""
    knee_error = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (800, 600))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                keypoints = {
                    name: [landmarks[getattr(mp_pose.PoseLandmark, name).value].x,
                           landmarks[getattr(mp_pose.PoseLandmark, name).value].y]
                    for name in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
                                 "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
                }
                
                left_elbow_angle = calculate_angle(keypoints["LEFT_SHOULDER"], keypoints["LEFT_ELBOW"], keypoints["LEFT_WRIST"])
                right_elbow_angle = calculate_angle(keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_ELBOW"], keypoints["RIGHT_WRIST"])
                left_hip_angle = calculate_angle(keypoints["LEFT_SHOULDER"], keypoints["LEFT_HIP"], keypoints["LEFT_KNEE"])
                right_hip_angle = calculate_angle(keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_HIP"], keypoints["RIGHT_KNEE"])
                left_knee_angle = calculate_angle(keypoints["LEFT_HIP"], keypoints["LEFT_KNEE"], keypoints["LEFT_ANKLE"])
                right_knee_angle = calculate_angle(keypoints["RIGHT_HIP"], keypoints["RIGHT_KNEE"], keypoints["RIGHT_ANKLE"])

                if left_hip_angle < 160 or right_hip_angle < 160:
                    hip_error = "Raise your hips!"
                elif left_hip_angle > 180 or right_hip_angle > 180:
                    hip_error = "Lower your hips!"
                else:
                    hip_error = ""

                knee_error = "Straighten your knees!" if left_knee_angle < 160 or right_knee_angle < 160 else ""

                if left_elbow_angle > 160 and right_elbow_angle > 160:
                    if stage == "down":
                        pushup_counter += 1
                    stage = "up"
                elif left_elbow_angle < 140 and right_elbow_angle < 140:
                    if stage == "up":
                        stage = "down"

            cv2.rectangle(image, (0, 0), (800, 150), (245, 117, 16), -1)
            cv2.putText(image, f'Push-Ups: {pushup_counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Stage: {stage}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            for idx, (text, value) in enumerate(zip(["Left Elbow", "Right Elbow", "Left Hip", "Right Hip", "Left Knee", "Right Knee"],
                                                    [left_elbow_angle, right_elbow_angle, left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle])):
                cv2.putText(image, f'{text}: {int(value)}°', (10 + (idx // 3) * 300, 100 + (idx % 3) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if hip_error:
                cv2.putText(image, hip_error, (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if knee_error:
                cv2.putText(image, knee_error, (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Live Push-Up Tracker', image)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total Push-Ups: {pushup_counter}")

# Call the function to start tracking
track_pushups()