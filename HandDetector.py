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

    main_x, main_y = 0, 0
    handPositionX, handPositionY = [], []
    h, w, c = image.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                x, y = lm.x, lm.y

                if id == 0:
                    main_x, main_y = x, y

                relative_x, relative_y = x - main_x, y - main_y

                handPositionX.append(relative_x)
                handPositionY.append(abs(relative_y))

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)  # Рисуем ладонь
            offset_x, offset_y = 0, 0

            if min(handPositionX) < 0.0:
                offset_x = abs(min(handPositionX))
            if min(handPositionY) < 0.0:
                offset_y = abs(min(handPositionY))

            handPositionX = list(map(lambda it: it + offset_x, handPositionX))
            handPositionY = list(map(lambda it: it + offset_y, handPositionY))

            maxY = max(handPositionY)
            maxX = max(handPositionX)

            defaultHeight = 1
            defaultWidth = 1

            coefY = abs(defaultHeight / maxY) if maxY != 0 else 0
            coefX = abs(defaultWidth / maxX) if maxX != 0 else 0

            handPositionX = list(map(lambda it: it * coefX, handPositionX))
            handPositionY = list(map(lambda it: it * coefY, handPositionY))

            print(f"maxX {max(handPositionX)} minX {min(handPositionX)}")
            print(f"maxY {max(handPositionY)} minY {min(handPositionY)}")
            print("-----------------------")

            for i in range(len(handPositionX)):
                cv2.circle(image, (int(handPositionX[i] * w), h + int(-1 * handPositionY[i] * h)), 3, (0, 255, 0), -1)

            handPositionX.clear()
            handPositionY.clear()

    cv2.imshow("Hand", image)  # Отображаем картинку
