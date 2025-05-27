import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

frames_generator = sv.get_video_frames_generator('output.mp4')
for frame in frames_generator:
    results = model.track(frame,
                          persist=True,
                          verbose=False)
    cv2.imshow("YOLO11 Tracking", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


