import mediapipe as mp
from csv import writer
import os
import cv2

hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils


datasetPath = os.listdir("ds_split")
for directory in datasetPath:
    letter = datasetPath.index(directory)
    dataList = [letter]
    for img in os.listdir(f"ds_split/{directory}"):
        image = cv2.imread(f"ds_split/{directory}/{img}")

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

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

                # Запись в массив координат
                for i in range(len(handPositionX)):
                    dataList.append(handPositionX[i])
                    dataList.append(handPositionY[i])

                # Запись массива в виде строки в csv
                with open('data.csv', 'a', newline='') as data_csv:
                    writer_object = writer(data_csv)
                    writer_object.writerow(dataList)
                    data_csv.close()

                # Очиста унитаза
                dataList.clear()
                dataList.append(letter)
                handPositionX.clear()
                handPositionY.clear()
