# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:27:15 2021

@author: srcdo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:16:54 2021

@author: srcdo
"""
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
import xlsxwriter

# print("Opening camera...")
# message_image = cv2.imread('h1.jpeg')  # Replace 'message_image.jpg' with the path to your custom message image
# cv2.imshow('Message', message_image)
# cv2.waitKey(2000)  # Wait for 2 seconds before opening the camera
# cv2.destroyAllWindows()

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Create Tkinter window
root = tk.Tk()
root.title("Output")

# Create a label for the message


# Display the Tkinter window


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
        cv2.putText(frame, "Camera is open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        #cv2.imshow('Camera', frame)

        # Display the message in a separate window
        # cv2.namedWindow('Message', cv2.WINDOW_NORMAL)
        # cv2.putText(frame, "Camera is open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # # cv2.imshow('Message', frame)
        # cv2.moveWindow('Message', 0, 0)
    

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        # print("Capture done successfully!")
        # cv2.putText(frame, "Capture done successfully!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('Camera', frame)
        # cv2.waitKey(2000) 

# cap.release()
# cv2.destroyAllWindows()
# In this code, a new window named 'Message' is created using cv2.namedWindow('Message', cv2.WINDOW_NORMAL). The message "Camera is open" is displayed in this window using cv2.putText(). The cv2.moveWindow('Message', 0, 0) function moves the 'Message' window to the top-left corner of the screen.

# Both the camera feed and the 'Message' window will be displayed simultaneously. You can customize the appearance of the message by modifying the parameters of the cv2.putText() function.







        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        filename = xlsxwriter.Workbook('hello1.xlsx')
        dict = {
                'item1': 1
                }
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                          )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # # 4. Pose Detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        #                          )
        # Export coordinates
        try:
            # # Extract Pose landmarks
            left = results.left_hand_landmarks.landmark
            left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
            
            # # # Extract Face landmarks
            # right = results.right_hand_landmarks.landmark
            # right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
            
            # Concate rows
            row = left_row
            
#             # Append class name 
#             row.insert(0, class_name)
            
#             # Export to CSV
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row) 

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            # # # Grab ear coords
            # coords = tuple(np.multiply(
            #                 np.array(
            #                     (results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.LEFT_EAR].x, 
            #                       results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.LEFT_EAR].y))
            #             , [640,480]).astype(int))
            
            # cv2.rectangle(image, 
            #               (coords[0], coords[1]+5), 
            #               (coords[0]+len(body_language_class)*20, coords[1]-30), 
            #               (245, 117, 16), -1)
            # cv2.putText(image, body_language_class, coords, 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # # Get status box
            # cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            # Display Class
            
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0,0,139)), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            id= body_language_class.split(' ')[0]
         
            print(id)
            
            print("hello")
            worksheet = filename.add_worksheet()
            if(id=="Saurabh"):
            # #         id='Shital'
                
                           worksheet.write('A1', '1')
                           worksheet.write('B1', 'Saurabh')
                           worksheet.write('C1', 'Present')
                           dict[str(id)]=str(id)
                           filename.close()             
            elif(id=="Swastik"):
                           worksheet.write('A2', '2')
                           worksheet.write('B2', 'Swastik')
                           worksheet.write('C2', 'Present')
                           dict[str(id)]=str(id)
                           filename.close() 
            elif(id=="Siddhi"):
                           worksheet.write('A3', '3')
                           worksheet.write('B3', 'Siddhi')
                           worksheet.write('C3', 'Present')
                           dict[str(id)]=str(id)
                           filename.close()  
            elif(id=="Akanksha"):
                          worksheet.write('A4', '4')
                          worksheet.write('B4', 'Akanksha')
                          worksheet.write('C4', 'Present')
                          dict[str(id)]=str(id)
                          filename.close()
                           
            
            #         id='Person 1'
            #         if((str(id)) not in dict):
            #               filename=xlsxwriter.output(filename,'class1',2,id,'yes')
            #               dict[str(id)]=str(id)
            # elif(id=="Priyanka"):
            #                worksheet.write('A3', '1')
            #                worksheet.write('B3', 'Priyanka')
            #                worksheet.write('C3', 'Present')
            #                dict[str(id)]=str(id)
            #                filename.close()         
            #         id='Person 2'
            #         if((str(id)) not in dict):
            #               filename=xlsxwriter.output(filename,'class1',3,id,'yes')
            #               dict[str(id)]=str(id)
            # # if(CLASS=='Shital'):
                 
            #         if((str('Shital')) not in dict):
            #              filename=xlwrite.output(filename,'class1',1,id,'yes')
            #              dict[str(id)]=str(id) 
                       
            # elif (id==2):
            #         id='Mrunalini'
            #         if((str(id)) not in dict):
            #               filename=xlwrite.output(filename,'class1',2,id,'yes')
            #               dict[str(id)]=str(id)
            # elif (id==3):
            #         id='Audumber'
            #         if((str(id)) not in dict):
            #               filename=xlwrite.output(filename,'class1',3,id,'yes')
            #               dict[str(id)]=str(id)
               
        except:
            pass
        cv2.imshow('Camera', frame)               
        cv2.imshow('Raw Webcam Feed', image)
        #combined_frame = cv2.hconcat([frame, image])
        # message_label = tk.Label(root, text="", font=("Helvetica", 18))
        # message_label.pack(padx=10, pady=10)
        # # Flag variable to track if capture is done
        # capture_done_flag = False

        # while True:
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break

        #     if not capture_done_flag:
        #         message_label.config(text="Capture done successfully!")
        #         capture_done_flag = True



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # message_label = tk.Label(root, text="Capture done successfully!", font=("Helvetica", 18))
        # message_label.pack(padx=10, pady=10)
        # message_label.config(text="Capture done successfully!")
cap.release()
cv2.destroyAllWindows()
root.mainloop()