import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('full.mp4')      # REPLACE FILE NAME WITH WHATEVER FILE IS IN LOCAL DIRECTORY

# Calculates angles by using the dot product method cos(angle) = a * b / (||a|| ||b||)
def angleCalc(shoulder, elbow, wrist):
    shoulder = np.array([shoulder.x, shoulder.y])
    elbow = np.array([elbow.x, elbow.y])
    wrist = np.array([wrist.x, wrist.y])

    a = shoulder - elbow
    b = wrist - elbow

    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    """
    if wrist[1] > elbow[1]:       # makes sure wrist going above elbow means the angle is getting smaller
        angle = 180 - angle
        print(wrist[1], elbow[1])
    """    
    return angle

# Extension time feedback is updated during the extension part of the movement, when the angle > 130 degrees
def getExtensionTimeFeedback(time):
    feedbackText = ""
    if time > 2.5:
        feedbackText = f"Good Extension Time: {float(duration)}"
    else:
        feedbackText = f"Poor Extension Time: {float(duration)}" 
    return feedbackText

# Extension range feedback is updated during the "in between" part of the movement, as the angle reduces below 130 degrees
def getExtensionRangeFeedback(angle):
    feedbackText = ""
    if angle > 150:
        feedbackText = f"Good Extension Range: {int(max_extension)}"
    else:    
        feedbackText = f"Poor Extension Range: {int(max_extension)}"
    return feedbackText

# Extension range feedback is updated during the "in between" part of the movement, as the angle increases above 90 degrees
def getContractionRangeFeedback(angle):
    feedbackText = ""
    if angle < 80:
        feedbackText = f"Good Contraction Range: {int(max_contraction)}"
    else:    
        feedbackText = f"Poor Contraction Range: {int(max_contraction)}"
    return feedbackText

prevTime = None # initalizing prev time to none, this will "start" the time once the last action was completed, in this case the arm was curled
curling = False # self-explanatory
extensionFormFeedback = "" 
contractionFormFeedback = "" 
timeFeedback = ""
max_extension = 0 # Sets to lowest angle so it gets overwritten by next highest
max_contraction = 360 # Sets highest angle so it gets overwritten by next lowest
prevAngle = 0 # initializing prev angle to none, this will be used to count a rep

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # left arm
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # calculating elbow angle
            angle = angleCalc(right_shoulder, right_elbow, right_wrist)
            prevAngle = angle            

            # checking for full extension and speed
            if angle >= 130:  # Extended. 
                if curling:
                    end_time = time.time()
                    duration = end_time - previous_time
                    timeFeedback = getExtensionTimeFeedback(duration)                    
                    curling = False  # reset state
                if angle > max_extension: # Tracks max angle reached
                    max_extension = angle
                          
            elif angle <= 90:    # curled
                if not curling and angle - prevAngle < 0.5: # Added angle check to start clock when the weight is lowered
                    previous_time = time.time()  # starting the timing
                    curling = True
                if angle < max_contraction: # Tracks lowest angle reached
                    max_contraction = angle

            elif 90 < angle < 130:   # in between. Use this pahse to check if max and least angles were reached                
                if max_extension != 0:
                    extensionFormFeedback = getExtensionRangeFeedback(max_extension)
                    max_extension = 0                
                if max_contraction != 360:
                    contractionFormFeedback = getContractionRangeFeedback(max_contraction)
                    max_contraction = 360                    

            # displaying both angle and feedback on screen
            cv2.putText(frame, f"Elbow Angle: {int(angle)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)            
            cv2.putText(frame, extensionFormFeedback, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, contractionFormFeedback, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, timeFeedback, (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Person Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
