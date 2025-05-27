import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("output.mp4")

# Pass region as list
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Pass region as dictionary
region_points = {
    "region-01": [[282, 161], [289, 286], [7, 285], [163, 164]],
    "region-02": [[335, 164], [403, 156], [627, 292], [443, 313]],
}

regioncounter = solutions.RegionCounter(
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
    verbose = False,
    line_width=1,
    tracker='bytetrack.yaml',
    conf=0.5
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

    results = process_frame(regioncounter, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows