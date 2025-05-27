import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

line_zone_1  = sv.LineZone(start = sv.Point(30, 200), end = sv.Point(300, 200))
line_zone_2  = sv.LineZone(start = sv.Point(350, 200), end = sv.Point(600, 200))

box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_scale=0.3)
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.3, text_color=sv.Color.BLACK, color=sv.Color.YELLOW)

tracker = sv.ByteTrack(minimum_consecutive_frames=2,
                       track_activation_threshold=0.5)
# track_activation_threshold 0.25
# lost_track_buffer          30
# minimum_matching_threshold 0.8
# minimum_consecutive_frames 1

def process_frame(frame):
    result = model.predict(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    crossed_in, crossed_out = line_zone_1.trigger(detections)
    crossed_in, crossed_out = line_zone_2.trigger(detections)

    labels = [
        f"#{tracker_id} {result.names[class_id]} {class_conf:.2f}"
        for tracker_id, class_id, class_conf
        in zip(detections.tracker_id, detections.class_id, detections.confidence)
    ]
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone_1)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone_2)

    cv2.imshow("YOLO11 Tracking", annotated_frame)


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

# print(line_zone.in_count, line_zone.out_count)

