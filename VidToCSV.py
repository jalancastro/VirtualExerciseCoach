import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import csv

filepath = "C:/Users/HP/Pictures/Camera Roll/Poses/"

for i in os.listdir(filepath):
    numcount = 0
    if "CSV" in i:
        pass
    elif os.path.isdir(filepath+i+"CSV"):
        pass
    else:
        os.mkdir(filepath+i+"CSV")
    if "CSV" in i:
        pass
    else:
        for ii in os.listdir(filepath+i):
            numcount = numcount + 1
            framecount = 0
            print(ii)
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(filepath+i+"/"+ii)
            num = 0
            elbnum = 0
            vec1len = 0
            vec2len = 0


            with open(filepath+i+"CSV/"+i+str(numcount), mode="w") as csvfile:
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
                writer = csv.DictWriter(csvfile, fieldnames=fielnames)

                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

                            rhi = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                            rsh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                            rEL = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                            h, w, _ = frame.shape
                            rWR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                            lsh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            lEL = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                            lWR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                            lhi = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                            framecount = framecount + 1

                            writer.writerow({"frame" : framecount,"R_HIPx": rhi.x-rhi.x, "R_HIPy": rhi.y-rhi.y, "R_HIPz": rhi.z-rhi.z, "R_SHOULDERx": rsh.x-rhi.x,
                                             "R_SHOULDERy": rsh.y- rhi.y,
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


#path=glob.glob(filepath + "/**/*.mp4")
#subdir = os.path.basename(os.path.dirname(path))
#print(subdir)
#print(my_files)