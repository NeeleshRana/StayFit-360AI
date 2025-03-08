import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import platform
import csv  

# Setup Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    angle = np.abs(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])) * 180.0 / np.pi
    return angle if angle <= 180 else 360 - angle

def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        try:
            os.system('play -nq -t alsa synth 0.3 sine 1000')
        except:
            os.system('afplay /System/Library/Sounds/Ping.aiff')

def track_bicep_curls():
    cap = cv2.VideoCapture(0)
    correct_reps, incorrect_reps = 0, 0
    stage = None
    warning_start_time = None
    warning_active = False
    reps_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            warning_text = None
            errors = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                index_finger = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle([elbow[0], elbow[1] - 0.1], shoulder, elbow)
                wrist_angle = calculate_angle(index_finger, wrist, elbow)

                if elbow_angle > 160:
                    stage = "down"
                elif elbow_angle < 50 and stage == "down":
                    stage = "up"
                    if shoulder_angle > 5 or wrist_angle > 171 or wrist_angle < 140:
                        incorrect_reps += 1
                    else:
                        correct_reps += 1
                    reps_data.append([correct_reps, incorrect_reps])

                if shoulder_angle > 5:
                    warning_text = "Keep your shoulder stable!"
                    errors = True
                elif wrist_angle < 140 or wrist_angle > 171:
                    warning_text = "Keep your wrist straight!"
                    errors = True

                if errors:
                    if warning_start_time is None:
                        warning_start_time = time.time()
                    elif time.time() - warning_start_time >= 0.5:
                        if not warning_active:
                            warning_active = True
                            beep()
                else:
                    warning_start_time = None
                    warning_active = False

            cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)
            cv2.putText(image, f'Correct Reps: {correct_reps}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Incorrect Reps: {incorrect_reps}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if warning_text:
                cv2.putText(image, warning_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Single Arm Curl Tracker', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    total_reps = correct_reps + incorrect_reps
    accuracy = (correct_reps / total_reps) * 100 if total_reps > 0 else 0

    final_file = "Biceps-final.csv"
    file_exists = os.path.exists(final_file)
    is_empty = os.stat(final_file).st_size == 0 if file_exists else True

    with open(final_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_empty:
            writer.writerow(["Accuracy (%)"])
        writer.writerow([round(accuracy, 2)])

    print("Data saved successfully!")
    cap.release()
    cv2.destroyAllWindows()
track_bicep_curls()
