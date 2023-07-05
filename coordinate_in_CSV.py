# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:28:43 2021

@author: srcdo
"""

import csv
import os
import numpy as np

import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
num_coords = 21
print(num_coords)


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
print(landmarks)

with open('coords1.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    
class_name = "Akanksha"
cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                           )
        
        # # 2. Right hand
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        #                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        #                           )
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        #                          )
        # Export coordinates
        try:
            # Extract Pose landmarks
            left = results.left_hand_landmarks.landmark
            left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
            
            # # Extract Face landmarks
            # right = results.right_hand_landmarks.landmark
            # right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
            
            # Concate rows
            #row = left_row+right_row
            row = left_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords1.csv', mode='a', newline='') as f:
                #f = f.iloc[1:]
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

