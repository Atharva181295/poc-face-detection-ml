import cv2
import dlib
import numpy as np
import os
from datetime import datetime

shape_predictor_path = "/home/atharva/Workspace/Machine Learning/face_recognition/shape_predictor_68_face_landmarks.dat"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"The file {shape_predictor_path} does not exist.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

cap = cv2.VideoCapture(0)

capture_folder = "capture"
os.makedirs(capture_folder, exist_ok=True)

prev_angle = 0  # Initial previous angle

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        angle = np.degrees(np.arctan2(landmarks[30, 1] - landmarks[8, 1], landmarks[30, 0] - landmarks[8, 0]))

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        if abs(angle - prev_angle) > 5:  # Adjust the threshold as needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(capture_folder, f"captured_image_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image captured at {timestamp} with face angle {angle} degrees")

        # Update the previous angle
        prev_angle = angle

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
