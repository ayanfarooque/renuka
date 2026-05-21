import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Load the hand landmark model (download required)
model_path = "hand_landmarker.task"

# Initialize HandLandmarker
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Gesture recognition function
def recognize_number(hand_landmarks):
    fingers = []
    # Landmark shortcuts
    lm = hand_landmarks
    # ---- Thumb (special case: compare x instead of y) ----
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # ---- Other 4 fingers ----
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = sum(fingers)

    # ---- Map to numbers ----
    if fingers == [0,0,0,0,0]:
        return "0"
    elif fingers == [0,1,0,0,0]:
        return "1"
    elif fingers == [0,1,1,0,0]:
        return "2"
    elif fingers == [0,1,1,1,0]:
        return "3"
    elif fingers == [0,1,1,1,1]:
        return "4"
    elif fingers == [1,1,1,1,1]:
        return "5"
    # elif fingers == [1,0,0,0,1]:
    #     return "6"
    # elif fingers == [1,1,0,0,1]:
    #     return "7"
    # elif fingers == [1,1,1,0,1]:
    #     return "8"
    # elif fingers == [1,1,1,1,0]:
    #     return "9"
    else:
        return str(total_fingers)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # flipping horizontaly
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            gesture = recognize_number(hand_landmarks)

            # Draw landmarks manually
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Get position of wrist (landmark 0)
            h, w, _ = frame.shape
            x = int(hand_landmarks[0].x * w)
            y = int(hand_landmarks[0].y * h)

            cv2.putText(frame, gesture, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()