import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def track_tricep_pushdown():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    reps_file = "Tricepreps.csv"
    final_file = "Tricepfinal.csv"
    
    if not os.path.exists(reps_file):
        with open(reps_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Rep No", "Correct"])
    
    cap = cv2.VideoCapture(0)
    rep_count, correct_reps, incorrect_reps = 0, 0, 0
    rep_position = "Up"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)
            wrist_angle = calculate_angle(elbow, wrist, [wrist[0], wrist[1] + 10])
            
            errors, correct = [], 1
            
            if elbow_angle < 70:
                errors.append(("Not bending enough!", (int(elbow[0]), int(elbow[1]))))
                correct = 0
            if shoulder_angle > 10:
                errors.append(("Keep shoulders stable!", (int(shoulder[0]), int(shoulder[1]))))
                correct = 0
            if wrist_angle < 70:
                errors.append(("Straighten wrist!", (int(wrist[0]), int(wrist[1]))))
                correct = 0
            
            if elbow_angle < 90 and rep_position == "Up":
                rep_position = "Down"
            elif elbow_angle > 150 and rep_position == "Down":
                rep_position = "Up"
                rep_count += 1
                with open(reps_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([rep_count, correct])
                if correct:
                    correct_reps += 1
                else:
                    incorrect_reps += 1
            
            for error_text, position in errors:
                cv2.putText(frame, error_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(frame, f"Reps: {rep_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Tricep Pushdown Analysis", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    total_reps = correct_reps + incorrect_reps
    accuracy = (correct_reps / total_reps) * 100 if total_reps > 0 else 0
    
    file_exists = os.path.exists(final_file)
    is_empty = os.stat(final_file).st_size == 0 if file_exists else True
    
    with open(final_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_empty:
            writer.writerow(["Accuracy (%)"])
        writer.writerow([round(accuracy, 2)])
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Call the function to start tracking
track_tricep_pushdown()
