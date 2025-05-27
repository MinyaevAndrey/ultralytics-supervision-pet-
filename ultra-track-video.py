import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt", )
def process_frame(frame):
    results = model.track(frame,
                          persist=True,
                          verbose=False)
    
    cv2.imshow("YOLO11 Tracking", results[0].plot())

cap = cv2.VideoCapture("output.mp4")
while cap.isOpened():
    success, frame = cap.read()

    if success:
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()