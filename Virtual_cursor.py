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
paused = False

# Helper to detect which fingers are up
def fingers_up(lmList):
    fingers = []
    if lmList[4][1] < lmList[3][1]:  # Thumb
        fingers.append(1)
    else:
        fingers.append(0)
    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers  # [thumb, index, middle, ring, pinky]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # Help overlay
    cv2.rectangle(img, (5, 5), (440, 240), (50, 50, 50), -1)
    help_text = [
        " All Fingers Up       = Pause",
        " Fist (All Down)      = Resume",
        " Only Index Up        = Move Cursor",
        " Index + Thumb Touch  = Left Click",
        " Middle + Ring Up     = Right Click",
        " Index+Middle+Ring  = Scroll Up",
        " Index+Middle+Pinky = Scroll Down",
        "ESC                     = Exit"
    ]
    for i, txt in enumerate(help_text):
        cv2.putText(img, txt, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                fingers = fingers_up(lmList)
                x1, y1 = lmList[8][1:]  # Index
                x2, y2 = lmList[4][1:]  # Thumb
                x_middle, y_middle = lmList[12][1:]
                x_ring, y_ring = lmList[16][1:]

                #Pause on Palm
                if fingers == [1, 1, 1, 1, 1]:
                    paused = True
                    cv2.putText(img, "✋ Paused", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    continue

                # Resume on Fist
                elif fingers == [0, 0, 0, 0, 0]:
                    paused = False
                    cv2.putText(img, "✊ Resumed", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    continue

                # ✅ Actions only if not paused
                if not paused:
                    # ☝️ Move mouse (only index up)
                    if fingers[1] == 1 and fingers[2:] == [0, 0, 0]:
                        x3 = np.interp(x1, (100, 540), (0, screen_w))
                        y3 = np.interp(y1, (100, 380), (0, screen_h))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY

                    # 👉 Left Click: Index + Thumb close
                    length = np.hypot(x2 - x1, y2 - y1)
                    if fingers[0] == 1 and fingers[1] == 1 and length < 40:
                        pyautogui.click()
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, "Left Click", (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    # ✌️ Right Click: Middle + Ring up only
                    if fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                        pyautogui.rightClick()
                        cv2.circle(img, (x_middle, y_middle), 15, (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, "Right Click", (480, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    # ☝️✌️🤘 Scroll Up: Index + Middle + Ring
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                        pyautogui.scroll(20)
                        cv2.putText(img, "Scroll Up", (480, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    # ☝️✌️🤙 Scroll Down: Index + Middle + Pinky
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 1:
                        pyautogui.scroll(-20)
                        cv2.putText(img, "Scroll Down", (480, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
