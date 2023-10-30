from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture("http://192.168.0.103:4747/video")
cap.set(3,1280)
cap.set(4,720)

model = YOLO("../Yolo-Weights/yolov8l.pt")


while True:
    success, img = cap.read()
    
    if not success:
        print("Error: Could not read frame from the video source.")
        break  # Break the loop if there's an issue with the video source.

    # Perform object detection and display the frame.
    results = model(img, stream=True)
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
