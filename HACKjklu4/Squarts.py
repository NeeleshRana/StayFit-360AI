import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import platform
import csv

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        try:
            os.system('play -nq -t alsa synth 0.3 sine 1000')
        except:
            os.system('afplay /System/Library/Sounds/Ping.aiff')

def track_squats():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    correct_reps, incorrect_reps = 0, 0
    squat_position = "Up"
    warning_start_time = None
    warning_active = False
    reps_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
            
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] + 10])
            
            errors = []
            if knee_angle < 80:
                errors.append("Squat too low!")
            if hip_angle < 60:
                errors.append("Leaning forward too much!")
            if ankle_angle < 163:
                errors.append("Ankle too bent!")
            
            if knee_angle < 100:
                if squat_position == "Up":
                    squat_position = "Down"
            elif knee_angle > 140:
                if squat_position == "Down":
                    squat_position = "Up"
                    if errors:
                        incorrect_reps += 1
                    else:
                        correct_reps += 1
                    reps_data.append([correct_reps, incorrect_reps])
            
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
            
            for i, error_text in enumerate(errors):
                cv2.putText(frame, error_text, (50, 110 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(frame, f"Correct Reps: {correct_reps}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Incorrect Reps: {incorrect_reps}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Squat Analysis", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    df_reps = pd.DataFrame(reps_data, columns=["Correct Reps", "Incorrect Reps"])
    df_reps.to_csv("squats-reps.csv", index=False)
    
    total_reps = correct_reps + incorrect_reps
    accuracy = (correct_reps / total_reps) * 100 if total_reps > 0 else 0
    
    final_file = "squarts-final.csv"
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
    pose.close()

# Run the function
track_squats()
