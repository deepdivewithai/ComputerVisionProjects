from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Resource/Videos/cars.mp4")
cap.set(3,1280)
cap.set(4,720)

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()    
    if not success:
        print("Error: Could not read frame from the video source.")
        break  # Break the loop if there's an issue with the video source.

    # Perform object detection and display the frame.
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            bbox = x1,y1,w,h
            cvzone.cornerRect(img,bbox)

            # Confidence Score
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{classNames[cls].upper()} {conf}", (max(0,x1),max(20,y1)), scale=1.2,thickness=1)

    
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

# Release the video capture and close OpenCV windows when done.
cap.release()
cv2.destroyAllWindows()

# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
