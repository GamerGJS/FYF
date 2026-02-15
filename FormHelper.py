import mediapipe as mp
import cv2
import math
from playsound3 import playsound

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- NEW: State Variable ---
sound_played = False 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))
        wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))

        invisible_dist = math.sqrt((wrist[0] - shoulder[0]) ** 2 + (wrist[1] - shoulder[1]) ** 2)

        if invisible_dist >= 220:
            line_color = (0, 255, 0)  # Green
            
            # --- MODIFIED: Only play if it hasn't played yet ---
            if not sound_played:
                playsound('assets/GREEN.mp3', block=False)
                sound_played = True 
        else:
            line_color = (0, 0, 255)  # Red
            # --- MODIFIED: Reset when arm is pulled back ---
            sound_played = False

        # Drawing
        cv2.line(frame, shoulder, elbow, (255, 0, 0), 5)
        cv2.line(frame, elbow, wrist, (255, 0, 0), 5)
        cv2.line(frame, shoulder, wrist, line_color, 5)
        
        for pt in [shoulder, elbow, wrist]:
            cv2.circle(frame, pt, 10, (0, 0, 255), -1)

    cv2.putText(frame, f'Distance: {int(invisible_dist)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('FYF', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
