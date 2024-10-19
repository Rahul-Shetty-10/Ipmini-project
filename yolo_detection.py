import cv2
import numpy as np
import os

# Paths to the YOLO configuration and weights files
config_path = r'C:\Users\Hp-PC\Documents\IP mini project\Cars-and-Pedestrains-Detection-main\yolov3.cfg'
weights_path = r'C:\Users\Hp-PC\Documents\IP mini project\Cars-and-Pedestrains-Detection-main\yolov3.weights'
names_path = r'C:\Users\Hp-PC\Documents\IP mini project\Cars-and-Pedestrains-Detection-main\coco.names'

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Check if the output of getUnconnectedOutLayers is a scalar or an array
if len(unconnected_out_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

# Load COCO names
with open(names_path, 'r') as f:
    classes = f.read().splitlines()

# Initiate video capture for a video file or a camera
video_path = 'video5.mp4'  # Change this path to the video file you want to use
cap = cv2.VideoCapture(video_path)

# Check if the video capture is initialized properly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
people_count = 0

# Loop until the video is successfully loaded
while cap.isOpened():
    
    # Read the next frame from the video
    ret, frame = cap.read()
    
    # Check if frame reading was successful
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width, channels = frame.shape

    # Create a blob from the frame and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []

    # Loop through each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections by ensuring the confidence is greater than a threshold
            if confidence > 0.5 and classes[class_id] == "person":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Reset people count for the current frame
    people_count = 0

    # Loop through the indexes we are keeping
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 255)  # Yellow color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            people_count += 1

            # Optionally, add a label to the bounding box
            label = f'Pedestrian {people_count}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(y, label_size[1] + 10)
            cv2.rectangle(frame, (x, label_ymin - label_size[1] - 10), (x + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (x, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the frame with the detected bodies
    cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pedestrians', frame)

    # Adjust the waitKey parameter to speed up video processing (currently set to 1 millisecond)
    if cv2.waitKey(10) & 0xFF == ord('q'):  # 'q' key
        print("Q key pressed. Exiting loop.")
        break


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Video capture released and all windows closed.")
