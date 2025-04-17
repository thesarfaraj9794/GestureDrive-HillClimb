import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb tip and index tip
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = img.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate distance
            distance = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5

            if distance < 40:
                pyautogui.keyDown('right')  # Gas
                pyautogui.keyUp('left')
                cv2.putText(img, 'GAS', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                pyautogui.keyDown('left')   # Brake
                pyautogui.keyUp('right')
                cv2.putText(img, 'BRAKE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
