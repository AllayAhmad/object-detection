import cv2 as cv
import numpy as np

# Load YOLO
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO classes
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Load video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error in frame capture.")
        break

    # Detect objects
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Track objects
    conf_threshold = 0.5
    nms_threshold = 0.4

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x, y, w, h = int(center_x - width / 2), int(center_y - height / 2), int(width), int(height)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply Non-Max Suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Visualization
    cv.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
