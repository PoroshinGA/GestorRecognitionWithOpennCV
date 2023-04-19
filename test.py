import math
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

imgSize = 600
offset = 10
counter = 0
imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
labels = os.listdir("Data") # просто метки классов засунуть
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + offset + h, x - offset:x + offset + w]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            const = imgSize / h
            wCal = math.ceil(const * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            
        else:
            const = imgSize / w
            hCal = math.ceil(const * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, f"{labels[index]} -> {math.ceil(max(prediction) * 100)}", (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255))

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
