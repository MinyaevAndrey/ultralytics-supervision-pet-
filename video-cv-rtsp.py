import cv2

camera_url = "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard"
# camera_url = "https://youtu.be/Ik5_Q2SHEbE"
cap = cv2.VideoCapture(camera_url)

def onFrame(frame):
    # scale_factor = 2
    # new_width = int(frame.shape[1] * scale_factor)
    # new_height = int(frame.shape[0] * scale_factor)
    # resized_frame = cv2.resize(frame, (new_width, new_height))

    # Показываем полученный кадр в окне
    cv2.imshow('Webcam Stream', frame)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при чтении кадра")
        break

    onFrame(frame)

    # Ждем нажатия кнопки ESC для завершения просмотра
    key = cv2.waitKey(1)
    if key == 27:  # Клавиша ESC имеет код 27
        break

# Закрываем окно и освобождаем ресурс захвата
cap.release()
cv2.destroyAllWindows()