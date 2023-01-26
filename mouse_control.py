import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
from pynput.mouse import Button, Controller
import math

mouse = Controller()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands = 1,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        index_tip_x = (int)(hand_landmarks.landmark[8].x * 640)
        index_tip_y = (int)(hand_landmarks.landmark[8].y * 480)

        middle_tip_x = (int)(hand_landmarks.landmark[12].x * 640)
        middle_tip_y = (int)(hand_landmarks.landmark[12].y * 480)

        wrist_x = (int)(hand_landmarks.landmark[0].x * 640)
        wrist_y = (int)(hand_landmarks.landmark[0].y * 480)

        mouse_position_x = (int)(hand_landmarks.landmark[8].x * 1920)
        mouse_position_y = (int)(hand_landmarks.landmark[8].y * 1080)

        dist_index_wrist = math.dist((index_tip_x, index_tip_y),(wrist_x, wrist_y))

        dist_middle_wrist = math.dist((middle_tip_x, middle_tip_y),(wrist_x, wrist_y))

        index_by_middle_ratio = dist_index_wrist / dist_middle_wrist

        # print(dist_index_wrist, dist_middle_wrist)
        # print(index_by_middle_ratio)

        # 0.7 1.3
        if index_by_middle_ratio < 0.8:
            mouse.press(Button.left)

        if index_by_middle_ratio > 0.85:
            mouse.release(Button.left)

        if index_by_middle_ratio > 1.2:
            mouse.press(Button.right)
            mouse.release(Button.right)

        mouse.position = (1920 - mouse_position_x, mouse_position_y)
        #  pointing the index finger
        cv2.circle(image, (index_tip_x, index_tip_y), 1, (0, 0, 255), 5)

        #  pointing the middle finger
        cv2.circle(image, (middle_tip_x, middle_tip_y), 1, (0, 0, 255), 5)

        #  pointing the wrist
        cv2.circle(image, (wrist_x, wrist_y), 1, (0, 0, 255), 5)

        # Flip the image horizontally for a selfie-view displaqy.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(2) == ord("q"):
      break
cap.release()
