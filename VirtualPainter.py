import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 20
folderPath = "Template"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)  # detection confidence
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. import the image and
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 2. Find Hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        x1, y1 = lmList[8][1:]  # tip of index finger
        x2, y2 = lmList[12][1:]  # tip of middle finger

        # 3. Check which finger are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode - two finger up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            # clicking
            if y1 < 125:
                header = overlayList[0]
                if 250 < x1 < 450:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                    brushThickness = 20
                elif 470 < x1 < 680:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                    brushThickness = 20
                elif 700 < x1 < 880:
                    header = overlayList[3]
                    drawColor = (255, 50, 0)
                    brushThickness = 20
                elif 900 < x1 < 1280:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                    brushThickness = 100

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If drawing mode - index finger up
        if fingers[1] and fingers[2] == False:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if drawColor != (255, 0, 255):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(imgInv, img)
    img = cv2.bitwise_or(img, imgCanvas)
    # setting the header image
    img[0:120, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
