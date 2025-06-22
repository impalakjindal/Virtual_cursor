import cv2
import numpy as np
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
smoothening = 5
plocX, plocY = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[8][1:]  # Index finger
                x2, y2 = lmList[4][1:]  # Thumb

                x3 = np.interp(x1, (100, 540), (0, screen_w))
                y3 = np.interp(y1, (100, 380), (0, screen_h))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY

                length = np.hypot(x2 - x1, y2 - y1)
                if length < 40:
                    pyautogui.click()
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
