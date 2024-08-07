import cv2 as cv
import numpy as np

# Load pre-trained models
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_region = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        if len(eyes) == 2:
            # Sort eyes based on x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            eye_1, eye_2 = eyes[0], eyes[1]

            left_eye_center = (int(eye_1[0] + eye_1[2] / 2), int(eye_1[1] + eye_1[3] / 2))
            right_eye_center = (int(eye_2[0] + eye_2[2] / 2), int(eye_2[1] + eye_2[3] / 2))

            # Draw eyes
            cv.rectangle(frame[y:y + h, x:x + w], (eye_1[0], eye_1[1]), (eye_1[0] + eye_1[2], eye_1[1] + eye_1[3]), (0, 0, 255), 2)
            cv.rectangle(frame[y:y + h, x:x + w], (eye_2[0], eye_2[1]), (eye_2[0] + eye_2[2], eye_2[1] + eye_2[3]), (0, 0, 255), 2)

            # Calculate angle between eyes
            delta_x = right_eye_center[0] - left_eye_center[0]
            delta_y = right_eye_center[1] - left_eye_center[1]

            if delta_x == 0:
                angle = 90 if delta_y > 0 else -90
            else:
                angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

            # Display tilt information
            if angle > 10:
                cv.putText(frame, 'RIGHT TILT :' + str(int(angle)) + ' degrees', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
            elif angle < -10:
                cv.putText(frame, 'LEFT TILT :' + str(int(angle)) + ' degrees', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
            else:
                cv.putText(frame, 'STRAIGHT :', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)

    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv.destroyAllWindows()
