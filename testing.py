
import cv2
import mediapipe as mp
import numpy as np
import csv


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture("testshouldbad1.mp4")
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


framecount = 0
with open ("testshbad.csv", mode = "w") as csvfile:
    fielnames = [
        "frame",
        "R_HIPx",
        "R_HIPy",
        "R_HIPz",
        "R_SHOULDERx",
        "R_SHOULDERy",
        "R_SHOULDERz",
        "R_ELBOWx",
        "R_ELBOWy",
        "R_ELBOWz",
        "R_WRISTx",
        "R_WRISTy",
        "R_WRISTz",
        "L_HIPx",
        "L_HIPy",
        "L_HIPz",
        "L_SHOULDERx",
        "L_SHOULDERy",
        "L_SHOULDERz",
        "L_ELBOWx",
        "L_ELBOWy",
        "L_ELBOWz",
        "L_WRISTx",
        "L_WRISTy",
        "L_WRISTz"
    ]
    writer = csv.DictWriter(csvfile,fieldnames=fielnames)


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)







            rhi = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            rsh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            rEL = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            rWR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lsh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            lEL = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            lWR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            lhi = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            framecount= framecount + 1
            writer.writerow(
                    {"frame": framecount, "R_HIPx": rhi.x - rhi.x, "R_HIPy": rhi.y - rhi.y, "R_HIPz": rhi.z - rhi.z,
                     "R_SHOULDERx": rsh.x - rhi.x,
                     "R_SHOULDERy": rsh.y - rhi.y,
                     "R_SHOULDERz": rsh.z - rhi.z,
                     "R_ELBOWx": rEL.x - rhi.x,
                     "R_ELBOWy": rEL.y - rhi.y,
                     "R_ELBOWz": rEL.z - rhi.z,
                     "R_WRISTx": rWR.x - rhi.x,
                     "R_WRISTy": rWR.y - rhi.y,
                     "R_WRISTz": rWR.z - rhi.z,
                     "L_HIPx": lhi.x - rhi.x,
                     "L_HIPy": lhi.y - rhi.y,
                     "L_HIPz": lhi.z - rhi.z,
                     "L_SHOULDERx": lsh.x - rhi.x,
                     "L_SHOULDERy": lsh.y - rhi.y,
                     "L_SHOULDERz": lsh.z - rhi.z,
                     "L_ELBOWx": lEL.x - rhi.x,
                     "L_ELBOWy": lEL.y - rhi.y,
                     "L_ELBOWz": lEL.z - rhi.z,
                     "L_WRISTx": lWR.x - rhi.x,
                     "L_WRISTy": lWR.y - rhi.y,
                     "L_WRISTz": lWR.z - rhi.z})





            #cv2.imshow('VirtCoach', frame)


            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

cap.release()
cv2.destroyAllWindows()