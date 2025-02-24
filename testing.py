
import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
num = 0
elbnum = 0
vec1len = 0
vec2len = 0
def anglecalc(hip,shoulder,elbow):
    shoulder = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, shoulder)]
    sh = np.array([shoulder.x, shoulder.y,shoulder.z])
    hip = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, hip)]
    hi = np.array([hip.x, hip.y,hip.z])
    elbow = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, elbow)]
    el = np.array([elbow.x, elbow.y,elbow.z])



    hish = np.array([np.abs(hip.x - shoulder.x), np.abs(hip.y - shoulder.y),np.abs(hip.z - shoulder.z)])
    shel = np.array([np.abs(shoulder.x - elbow.x), np.abs(shoulder.y - elbow.y),np.abs(shoulder.z - elbow.z)])
    hishM = np.linalg.norm(hish)
    shelM = np.linalg.norm(shel)
    angle = np.arccos((np.dot(hish, shel)) / (hishM * shelM))
    angledeg = np.abs(angle * 180.0/np.pi)
    if angledeg > 180.0:
        angledeg = 360 - angledeg
    #print(angle)
    return angledeg

def refanc(b,a,d2,tol):
    aref = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, a)]
    bref = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, b)]

    a_x = aref.x
    a_y = aref.y
    b_x = bref.x
    b_y = bref.y
    if (d2 == 0 or d2 == 2):
        #print(abs(b_x - a_x))
        if (abs(b_x - a_x) > tol):

            #print("error")
            return "error"
    if (d2 == 1 or d2 == 2):
        #print(abs(b_y - a_y))
        if (abs(b_y - a_y) > tol):
            #print("error")
            return "error"

# def imagelengthcap(hip,shoulder):
#     time.sleep(3)
#     shoulder = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, shoulder)]
#     sh = np.array([shoulder.x, shoulder.y, shoulder.z])
#     hip = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, hip)]
#     hi = np.array([hip.x, hip.y, hip.z])
#
#     veclen = np.sqrt(np.square(hip.x - shoulder.x) + np.square(hip.y - shoulder.y))
#
#     return veclen




with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)


        if results.pose_landmarks:
            # if num == 0:
            #     vec1len = imagelengthcap("RIGHT_WRIST","RIGHT_ELBOW")
            #     print(vec1len)
            #     vec2len = imagelengthcap("RIGHT_ELBOW", "RIGHT_SHOULDER")
            #     print(vec2len)
            #     num = 1


            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)



            sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            h, w, _ = frame.shape
            cx, cy = int(sh.x * w), int(sh.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            EL = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            h, w, _ = frame.shape
            cx, cy = int(EL.x * w), int(EL.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            WR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            h, w, _ = frame.shape
            cx, cy = int(WR.x * w), int(WR.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            out = str(anglecalc( "RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"))
            out2 = str(refanc("RIGHT_HIP", "RIGHT_ELBOW", 0, 0.04))
            cv2.putText(frame, out,(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, out2, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        cv2.imshow('VirtCoach', frame)


        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

cap.release()
cv2.destroyAllWindows()