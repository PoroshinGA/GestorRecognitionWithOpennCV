from itertools import groupby

import cv2
import mediapipe as mp
import pickle
import time

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils


coordList = []
labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К', 'Л', 'М', 'Н', 'О', 'П',
          'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Ы', 'Э', 'Ь', 'Ю', 'Я']

sign_time = 0 # время записи буквы
space_time = 0 # время записи кадра без буквы
sign_counter = 0 # счетчик кадроы для букв
space_counter = 0 # счетчик кадров для пробела
text = []
letter = ""


while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, image = cap.read()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    DEFAULT_OFFSET_X = 50
    DEFAULT_OFFSET_Y = 50

    main_x, main_y = 0, 0
    handPositionX, handPositionY = [], []
    h, w, c = image.shape

    if results.multi_hand_landmarks and sign_counter > 20: # условие со счетчиком кадров
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                x, y = lm.x, lm.y
                cx, cy = int(x * w), int(y * h)

                if id == 0:
                    main_x, main_y = cx, cy

                relative_x, relative_y = int(cx - main_x), int(cy - main_y)

                handPositionX.append(relative_x)
                handPositionY.append(relative_y)

            offset_x, offset_y = DEFAULT_OFFSET_X, h - DEFAULT_OFFSET_Y

            if (min(handPositionX) < 0):
                offset_x = DEFAULT_OFFSET_X - min(handPositionX)

            if (max(handPositionY) > 0):
                offset_y = h - DEFAULT_OFFSET_Y - max(handPositionY)

            for i in range(len(handPositionX)):
                coordList.append(handPositionX[i])
                coordList.append(handPositionY[i])

            prediction = model.predict([coordList])

            text += labels[prediction[0]] # Добавляем букву в общий текст
            sign_time = time.time() # записываем время
            if letter == text[len(text) - 1]:
                text.pop(len(text) - 1)
            letter = labels[prediction[0]]
            sign_counter = 0

            coordList.clear()
            handPositionX.clear()
            handPositionY.clear()

    else: # если руки нет, записывает время и увеличивает счетчик кадров
        space_time = time.time()
        space_counter += 1

    # если время кадра без знака больше времени когда была записана последняя буква и, если строка текста не пустая
    # и если счетчик кадров для пробела и букв больше 15, то в текст добавляется пробел и обнуляется счетчик
    if space_time - sign_time > 1 and sign_time != 0 and space_counter > 15:
        text += " "
        space_counter = 0
        if letter == " ":
            text.pop(len(text) - 1)
        letter = " "

    sign_counter += 1
    cv2.imshow("Hand", image)
text = "".join(text)
print(text)
