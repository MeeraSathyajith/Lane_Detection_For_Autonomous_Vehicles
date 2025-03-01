import cv2
import numpy as np


video = cv2.VideoCapture("/Users/dhirajsolleti/Downloads/praneeth video.mp4")

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, or_frame = video.read()

    if not ret:
        print("Error: Failed to capture frame from video.")
        break


    blurred_frame = cv2.GaussianBlur(or_frame, (5, 5), 0) 
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV) 


    lower_y = np.array([18, 94, 140])  
    upper_y = np.array([48, 255, 255])  

 
    mask = cv2.inRange(hsv, lower_y, upper_y)

    edges = cv2.Canny(mask, 74, 150)

   
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    lane_instruction = "Stay on the Lane"  

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
           
            cv2.line(or_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        
        lane_instruction = "Stay on the Lane"

  
    cv2.putText(or_frame, lane_instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

 
    cv2.imshow("Lane Detection with Instructions", or_frame)

   
    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()