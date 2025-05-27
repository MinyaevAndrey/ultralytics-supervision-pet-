import cv2

cap = cv2.VideoCapture('output.mp4')

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