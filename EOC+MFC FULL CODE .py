import cv2
import numpy as np
import os


weights_path = "/Users/dhirajsolleti/Downloads/yolov3.weights"
config_path = "/Users/dhirajsolleti/Downloads/yolov3.cfg"
coco_names_path = "/Users/dhirajsolleti/Downloads/coco.names"
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"Error: The file {coco_names_path} does not exist. Please download it.")

net = cv2.dnn.readNet(weights_path, config_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("/Users/dhirajsolleti/Downloads/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


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

    
    height, width, channels = or_frame.shape

    
    blob = cv2.dnn.blobFromImage(or_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    
    class_ids = []
    confidences = []
    boxes = []

    car_detected = False  

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  
               
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

               
                if classes[class_id] == "car":
                    car_detected = True

  
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

   
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        color = (0, 255, 0)  
        cv2.rectangle(or_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(or_frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

       
        if car_detected:
            lane_instruction = "Go Slow, Traffic Ahead"

    
    cv2.putText(or_frame, lane_instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    cv2.imshow("Lane and Object Detection with Instructions", or_frame)

   
    key = cv2.waitKey(25)
    if key == 27:
        break


video.release()
cv2.destroyAllWindows()