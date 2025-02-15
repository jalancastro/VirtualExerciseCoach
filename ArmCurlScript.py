import cv2
import mediapipe as mp
import numpy as np
import time


prevAngle = None    # initializing prev angle to none, this will be used to count a rep

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture('notallthewaydown.mp4')      # REPLACE FILE NAME WITH WHATEVER FILE IS IN LOCAL DIRECTORY

def angleCalc(shoulder, elbow, wrist):      # method to calculate angle
    shoulder = np.array([shoulder.x, shoulder.y])
    elbow = np.array([elbow.x, elbow.y])
    wrist = np.array([wrist.x, wrist.y])

    a = shoulder - elbow
    b = wrist - elbow

    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

prevTime = None     # initalizing prev time to none, this will "start" the time once the last action was completed, in this case the arm was curled
curling = False     # self-explanatory
feedback = ""       # hoping this says pass or fail

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)


        if results.pose_landmarks:

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks for left arm
            shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            # calculating elbow angle
            angle = angleCalc(shoulder, elbow, wrist)
            max_reach = 0

            # checking for full extension and speed
            if angle > 160:  # fully extended
                if curling:
                    end_time = time.time()
                    duration = end_time - previous_time
                    if duration < 1.5:  # too fasr
                        feedback = "Slow down"
                    else:
                        feedback = "Good rep"
                    curling = False  # reset state
            
            elif angle < 50:    # curled
                if not curling:
                    previous_time = time.time()  # starting the timing
                    curling = True

            elif angle < 160 and angle > 100:   # in between
                if curling == True:
                    if angle > max_reach:   #if max reach is less than 160, then we'll throw out the didn't extend all the way message
                        max_reach = angle
                    if max_reach < 160:
                        feedback = "Did not extend all the way"
                        curling = False

            # displaying both angle and feedback on screen
            cv2.putText(frame, f"Elbow Angle: {int(angle)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)



        cv2.imshow('Person Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()