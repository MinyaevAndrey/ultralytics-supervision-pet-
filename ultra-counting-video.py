import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("output.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
region_points = [(30, 200), (600, 200)]

counter = solutions.ObjectCounter(
    region=region_points,
    model="yolo11n.pt",
    tracker="bytetrack.yaml",
    verbose=False,
    line_width = 1,
)

def process_frame(worker, frame):
    results = worker(frame)
    cv2.imshow("YOLO11 Tracking", results.plot_im)

# Process video
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    process_frame(counter, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows