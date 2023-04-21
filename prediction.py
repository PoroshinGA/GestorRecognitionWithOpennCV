import cv2
import mediapipe as mp
from csv import writer
import sklearn
import pickle

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

coordList = []
labels = ["А", "Б", "В"]

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
                coordList.append(handPositionX[i])
                coordList.append(handPositionY[i])
                cv2.circle(image, (handPositionX[i] + offset_x, handPositionY[i] + offset_y), 3, (0, 255, 0), -1)

            handPositionX.clear()
            handPositionY.clear()
            prediction = model.predict(coordList)
            coordList.clear()

        cv2.putText(image, labels[prediction], image.shape[0:1], cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255))

    cv2.imshow("Hand", image)  # Отображаем картинку

# if cv2.waitKey(1) == ord("s"):
#     for i in range(len(handPositionX)):
#         cv2.circle(image, (handPositionX[i] + offset_x, handPositionY[i] + offset_y), 3,
#                    (0, 255, 0), -1)
#         dataList.append(handPositionX[i])
#         dataList.append(handPositionY[i])
#
#     with open('data.csv', 'a', newline='') as data_csv:
#         writer_object = writer(data_csv)
#         writer_object.writerow(dataList)
#         data_csv.close()
#     dataList.clear()
#     dataList.append(letter)
#     print(counter)
#     if counter == 500:
#         exit(0)
#     counter += 1

