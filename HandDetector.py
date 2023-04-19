import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Камера
hands = mp.solutions.hands.Hands(max_num_hands=1)  # Объект ИИ для определения ладони
draw = mp.solutions.drawing_utils  # Для рисование ладони

while True:
    # Закрытие окна
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, image = cap.read()  # Считываем изображение с камеры

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем в rgb
    results = hands.process(imageRGB)  # Работа mediapipe

    DEFAULT_OFFSET_X = 50
    DEFAULT_OFFSET_Y = 50

    main_x, main_y = 0, 0
    handPositionX, handPositionY = [], []
    h, w, c = image.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                x, y = lm.x, lm.y
                cx, cy = int(x * w), int(y * h)

                if id == 0:
                    main_x, main_y = cx, cy

                relative_x, relative_y = int(cx - main_x), int(cy - main_y)

                handPositionX.append(relative_x)
                handPositionY.append(relative_y)

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)  # Рисуем ладонь
            offset_x, offset_y = DEFAULT_OFFSET_X, h - DEFAULT_OFFSET_Y

            if (min(handPositionX) < 0):
                offset_x = DEFAULT_OFFSET_X - min(handPositionX)
            if (max(handPositionY) > 0):
                offset_y = h - DEFAULT_OFFSET_Y - max(handPositionY)

            for i in range(len(handPositionX)):
                cv2.circle(image, (handPositionX[i] + offset_x, handPositionY[i] + offset_y), 3, (0, 255, 0), -1)

            handPositionX.clear()
            handPositionY.clear()

    cv2.imshow("Hand", image)  # Отображаем картинку
