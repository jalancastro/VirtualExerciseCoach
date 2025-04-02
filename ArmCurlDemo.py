import cv2
import mediapipe as mp
import numpy as np
import time

# mediapipe pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# rep counting
right_reps = 0
left_reps = 0
right_stage = None  # down or up
left_stage = None   # down or up
feedback_timer = 0
feedback_text = ""
detector = 0


# adjustable elbow swing threshold
ELBOW_SWING_THRESHOLD = 70  # ADJUST HIGHER IF TOO SENSITIVE
SHOULDER_SWING_THRESHOLD = 50
HIP_RANGE_THRESHOLD = 10

# adjustable partial rep treshold
REP_ATTEMPT_TRESHOLD = 160
REP_COMPLETION_TRESHOLD = 170
# partial rep flag
right_partial_rep_flag = False
left_partial_rep_flag = False

# video cap and writing
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("exercise_recording.avi", cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width, frame_height))

# to make sure we don't incorrectly throw errors or start counting too soon, need this to capture if body is
# in frame
def is_body_in_frame(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_ELBOW
    ]
    return all(landmarks[l].visibility > 0.7 for l in required_landmarks)  # Only if visible for 0.7+
POSE_CONNECTIONS = frozenset([(12,14),(14,16),(11,13),(13,15)])

def set_feedback_text(feedback):
    global feedback_text
    if feedback_text != "" and feedback != "":
        feedback_text += "\n"
    feedback_text += feedback

# rep counting function. counts reps and sets stages
def rep_counter(right_angle, left_angle):
    global feedback_timer, feedback_text, right_reps, right_stage, left_reps, left_stage
    if right_angle > 150:
        right_stage = "down"
    elif right_angle < 50 and right_stage == "down":
        right_stage = "up"
        right_reps += 1
        #set_feedback_text("Good rep - Right arm")
        feedback_text = "Good rep - Right arm"
        feedback_timer = time.time()

    # left arm rep counting
    if left_angle > 150:
       left_stage = "down"
    elif left_angle < 50 and left_stage == "down":
        left_stage = "up"
        left_reps += 1
        #set_feedback_text("Good rep - Left arm")
        feedback_text = "Good rep - Left arm"
        feedback_timer = time.time()    

# sets the partial rep flags if a certain angle was reached during contraction
# if a rep was completed successfully by setting the stage to "up", it clears the flag
def set_partial_rep_flag(right_angle, left_angle, right_stage, left_stage):
    global right_partial_rep_flag, left_partial_rep_flag
    if right_angle < REP_ATTEMPT_TRESHOLD:
        right_partial_rep_flag = True
    if left_angle < REP_ATTEMPT_TRESHOLD:
        left_partial_rep_flag = True
    if right_stage == "up":
        right_partial_rep_flag = False
    if left_stage == "up":
        left_partial_rep_flag = False    
# uses flags to determine if an angle of 160 or less was reached, but 50 degrees wasn't reached(full rep). 
# If so, when the rep is completed and the arm extended beyond 170 degrees, it sets the feedback text.
def set_partial_rep_feedback(right_angle, left_angle):
    global right_partial_rep_flag, left_partial_rep_flag, feedback_text, feedback_timer
    if right_angle > REP_COMPLETION_TRESHOLD and right_partial_rep_flag:
        right_partial_rep_flag = False
        #set_feedback_text("Partial rep - Right arm")
        feedback_text = "Partial rep - Right arm"
        feedback_timer = time.time()
    if left_angle > REP_COMPLETION_TRESHOLD and left_partial_rep_flag:
        print("left_partial_rep_flag: ", left_partial_rep_flag)
        left_partial_rep_flag = False
        #set_feedback_text("Partial rep - Left arm")
        feedback_text = "Partial rep - Left arm"
        feedback_timer = time.time()

# function to detect elbow swings depending on wrist and elbow positions
def set_elbow_swing_feedback(right_arm_position, left_arm_position, body_in_frame):
    global ELBOW_SWING_THRESHOLD, feedback_text, feedback_timer, detector    
    # elbow swing detection with adjusted threshold
    right_elbow_swing = right_arm_position > ELBOW_SWING_THRESHOLD
    left_elbow_swing = left_arm_position > ELBOW_SWING_THRESHOLD
    
    if right_elbow_swing and body_in_frame:
        #set_feedback_text("Right elbow swinging too much!")
        feedback_text = "Right elbow swinging too much!"
        feedback_timer = time.time()
        detector = 1
    if left_elbow_swing and body_in_frame:
        #set_feedback_text("Left elbow swinging too much!")
        feedback_text = "Left elbow swinging too much!"
        feedback_timer = time.time()
        detector = 1
        
# function to detect shoulder swings depending on shoulder positions
def set_shoulder_swing_feedback(right_shoulder_position, left_shoulder_position, body_in_frame):
    global ELBOW_SWING_THRESHOLD, feedback_text, feedback_timer, detector    
    # shoulder swing detection with threshold
    right_shoulder_swing = right_shoulder_position > SHOULDER_SWING_THRESHOLD
    left_shoulder_swing = left_shoulder_position > SHOULDER_SWING_THRESHOLD
    
    if right_shoulder_swing and body_in_frame:
        #set_feedback_text("You are swinging your right shoulder, don't use momentum!")
        feedback_text = "You are swinging your right shoulder, don't use momentum!"
        feedback_timer = time.time()
        detector = 1
    if left_shoulder_swing and body_in_frame:
        #set_feedback_text("You are swinging your left shoulder, don't use momentum!")
        feedback_text = "You are swinging your left shoulder, don't use momentum!"
        feedback_timer = time.time()
        detector = 1

# function to detect hip position error depending on each hip
def set_hip_position_feedback(right_hip_position, left_hip_position, body_in_frame):
    global ELBOW_SWING_THRESHOLD, feedback_text, feedback_timer, detector    
    # shoulder swing detection with threshold
    right_hip_swing = right_hip_position > HIP_RANGE_THRESHOLD
    left_hip_swing = left_hip_position > HIP_RANGE_THRESHOLD
    
    if right_hip_swing and body_in_frame:
        #set_feedback_text("Right hip too far out")
        feedback_text = "Right hip too far out"
        feedback_timer = time.time()
        detector = 1
    if left_hip_swing and body_in_frame:
        #set_feedback_text("Left hip too far out")
        feedback_text = "Left hip too far out"
        feedback_timer = time.time()
        detector = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)


    h, w, _ = frame.shape
    detector = 0
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # in frame detection
        body_in_frame = is_body_in_frame(landmarks)

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        for landmark in results.pose_landmarks.landmark:
            if landmark != right_shoulder and landmark != right_elbow and landmark != right_wrist and landmark != left_shoulder and landmark != left_elbow and landmark != left_wrist:
                landmark.visibility = 0

        #Draw landmarkpoints
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)
        cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 1, (0, 255, 0), 20)
        cv2.circle(frame, (int(right_elbow.x * w), int(right_elbow.y * h)), 1, (0, 255, 0), 20)
        cv2.circle(frame, (int(right_wrist.x * w), int(right_wrist.y * h)), 1, (0, 255, 0), 20)
        cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 1, (0, 255, 0), 20)
        cv2.circle(frame, (int(left_elbow.x * w), int(left_elbow.y * h)), 1, (0, 255, 0), 20)
        cv2.circle(frame, (int(left_wrist.x * w), int(left_wrist.y * h)), 1, (0, 255, 0), 20)

        cv2.line(frame,(int(right_shoulder.x * w), int(right_shoulder.y * h)),(int(right_elbow.x * w), int(right_elbow.y * h)),(255, 255, 255),2)

        cv2.line(frame, (int(right_elbow.x * w), int(right_elbow.y * h)),
                    (int(right_wrist.x * w), int(right_wrist.y * h)), (255, 255, 255), 2)

        cv2.line(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                 (int(left_elbow.x * w), int(left_elbow.y * h)), (255, 255, 255), 2)

        cv2.line(frame, (int(left_elbow.x * w), int(left_elbow.y * h)),
                 (int(left_wrist.x * w), int(left_wrist.y * h)), (255, 255, 255), 2)


        if left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5:
            # landmarks to pixel coordinates
            def to_pixel_coords(landmark):
                return int(landmark.x * w), int(landmark.y * h)

            r_shoulder, r_elbow, r_wrist = map(to_pixel_coords, [right_shoulder, right_elbow, right_wrist])
            l_shoulder, l_elbow, l_wrist = map(to_pixel_coords, [left_shoulder, left_elbow, left_wrist])

            # angle calculation
            def calculate_angle(a, b, c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)

                ab = a - b
                bc = c - b

                cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                return np.degrees(angle)

            right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

             # hip out of range detection
            r_shoulder, r_hip, l_hip,l_shoulder = map(to_pixel_coords, [right_shoulder, right_hip, left_hip,left_shoulder])     
            
            # set partial rep flag on the way up
            set_partial_rep_flag(right_angle, left_angle, right_stage, left_stage)

            # right arm rep counting                
            rep_counter(right_angle, left_angle)            
                
            # if treshold angle wasn't reached, set partial rep feedback
            set_partial_rep_feedback(right_angle, left_angle)

            # IF arm swing AND body is in frame
            set_elbow_swing_feedback(abs(r_wrist[0] - r_elbow[0]), abs(l_wrist[0] - l_elbow[0]), body_in_frame)            

            # IF shoulder swing AND body is in frame
            set_shoulder_swing_feedback(r_shoulder[1] - l_shoulder[1], l_shoulder[1] - r_shoulder[1], body_in_frame)
            
            # IF hip swing AND body is in frame
            set_hip_position_feedback(r_hip[1] - l_hip[1], l_hip[1] - r_hip[1], body_in_frame)

            if feedback_text and (time.time() - feedback_timer < 2):
                cv2.putText(frame, feedback_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                feedback_text = ""
            if detector == 1:
                for i in range(2):
                    out.write(frame)
            else:
                out.write(frame)
        else:
            cv2.putText(frame, "Will start recording once full body is in frame", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

    # saving the video frame

    else:
        cv2.putText(frame, "Will start recording once full body is in frame", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # rep count display text
    cv2.putText(frame, f"Right Reps: {right_reps}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Left Reps: {left_reps}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # display "Will start recording once full body is in frame" during recording phase ONLY
    # if not body_in_frame:
    #     cv2.putText(frame, "Will start recording once full body is in frame", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # show feedback for at least 2 seconds

    cv2.imshow("Arm Curl Tracker", frame)

    # exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

##################
# PLAYBACK PHASE #
##################
# open recently saved video
cap = cv2.VideoCapture("exercise_recording.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    h, w, _ = frame.shape

    if 1==0 :#results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # body in frame
        body_in_frame = is_body_in_frame(landmarks)

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # landmarks to pixels
        def to_pixel_coords(landmark):
            return int(landmark.x * w), int(landmark.y * h)

        r_shoulder, r_elbow, r_wrist = map(to_pixel_coords, [right_shoulder, right_elbow, right_wrist])
        l_shoulder, l_elbow, l_wrist = map(to_pixel_coords, [left_shoulder, left_elbow, left_wrist])

        # angle calcs
        right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # ELBOW SWING ERROR
        right_elbow_swing = abs(r_wrist[0] - r_elbow[0]) > ELBOW_SWING_THRESHOLD
        left_elbow_swing = abs(l_wrist[0] - l_elbow[0]) > ELBOW_SWING_THRESHOLD

        # SHOULDER SWING ERROR
        right_shoulder_swing = (l_shoulder[1] - r_shoulder[1]) > SHOULDER_SWING_THRESHOLD
        left_shoulder_swing = (r_shoulder[1] - l_shoulder[1]) > SHOULDER_SWING_THRESHOLD

        # detecting elbow swing during playback
        if right_elbow_swing and body_in_frame:
            feedback_text = "Right elbow swinging too much!"
            feedback_timer = time.time()

        if left_elbow_swing and body_in_frame:
            feedback_text = "Left elbow swinging too much!"
            feedback_timer = time.time()

        # IF shoulder swing AND body is in frame
        if right_shoulder_swing and body_in_frame:
            feedback_text = "You are swinging your right shoulder, don't use momentum!"
            feedback_timer = time.time()
            detector = 1
        if left_shoulder_swing and body_in_frame:
            feedback_text = "You are swinging your left shoulder, don't use momentum!"
            feedback_timer = time.time()
            detector = 1

    # making sure the feedback doesnt disappear right away
    if feedback_text and (time.time() - feedback_timer < 2):
        cv2.putText(frame, feedback_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # showing frame
    cv2.imshow("Arm Curl Tracker (Playback)", frame)

    # exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
