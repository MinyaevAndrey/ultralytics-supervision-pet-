import cv2
import time
from ultralytics import solutions

region_points = {
    "region-01": [[189, 169], [344, 94], [393, 156], [232, 240]],
    "region-02": [[240, 254], [402, 169], [449, 238], [281, 325]]
}
regioncounter = solutions.RegionCounter(
    show=True,  # display the frame
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
    line_width=1,  # Adjust the line width for bounding boxes and text display
    tracker='bytetrack.yaml',
    verbose = False
)


line_points = [[50, 220], [316, 169]]
line_counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=line_points,  # Pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    verbose = False,  # Display the output
    # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=1,  # Adjust the line width for bounding boxes and text display
    tracker='bytetrack.yaml',
    conf = 0.5
)


# np.array([[194, 270], [435, 141]]),

def display_frame(frame):
    cv2.imshow('RTSP Stream', frame)

def save_frame(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'frame_{timestamp}.jpg'
    cv2.imwrite(filename, frame)


def main(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеопоток.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: Не удалось прочитать кадр.")
            break

        # results = line_counter(frame)
        results = regioncounter(frame)
        # display_frame(results.plot_im)
        # display_frame(frame)
        # save_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main('http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard')
# main('https://www.youtube.com/watch?v=1EiC9bvVGnk&list=PLg1-gMKbNYtdQ0qDkHYR_hH5LuOfLta_m&index=9')
# main('http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg')

cv2.destroyAllWindows()  # destroy all opened windows
