import mediapipe as mp
import cv2

def mediapipe_model(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize the hand detector
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        file = cv2.flip(cv2.imread(frame), 1)

        # Convert the frame to RGB
        image = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

        # Detecting hands on the frame
        results = hands.process(image)

        # If hands are found, return True
        if results.multi_hand_landmarks:
            return True

    # If hands are not found, return False
    return False