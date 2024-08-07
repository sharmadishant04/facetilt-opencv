# Face and Eye Detection with Head Tilt Angle Measurement

This project uses OpenCV and NumPy to detect faces and eyes in a video feed (webcam or video file) and measures the head tilt angle.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sharmadishant04/facetilt-opencv.git
    cd facetilt-opencv
    ```

2. Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Make sure you have the required Haar Cascade files for face and eye detection:
    - [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
    - [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

2. Run the main code:
    ```sh
    python main.py
    ```

## Script Explanation

The script captures video from the webcam (or a video file if specified) and performs the following steps:

1. Converts each frame to grayscale.
2. Detects faces in the frame using a Haar Cascade classifier.
3. For each detected face, detects eyes within the face region.
4. Calculates the tilt angle of the head based on the relative positions of the eyes.
5. Displays the tilt angle on the video feed.

Press `Esc` to exit the video feed.

## Example

```python
import cv2 as cv 
import numpy as np 

# 0 for webcam feed; add "path to file" for detection in video file 
capture = cv.VideoCapture(0) 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml") 

while True: 
    ret, frame = capture.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) 
    x, y, w, h = 0, 0, 0, 0
    for (x, y, w, h) in faces: 
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv.circle(frame, (x + int(w * 0.5), y + int(h * 0.5)), 4, (0, 255, 0), -1) 
    eyes = eye_cascade.detectMultiScale(gray[y:(y + h), x:(x + w)], 1.1, 4) 
    index = 0
    eye_1 = [None, None, None, None] 
    eye_2 = [None, None, None, None] 
    for (ex, ey, ew, eh) in eyes: 
        if index == 0: 
            eye_1 = [ex, ey, ew, eh] 
        elif index == 1: 
            eye_2 = [ex, ey, ew, eh] 
        cv.rectangle(frame[y:(y + h), x:(x + w)], (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2) 
        index = index + 1
    if (eye_1[0] is not None) and (eye_2[0] is not None): 
        if eye_1[0] < eye_2[0]: 
            left_eye = eye_1 
            right_eye = eye_2 
        else: 
            left_eye = eye_2 
            right_eye = eye_1 
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2))) 
        right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2))) 
        left_eye_x = left_eye_center[0] 
        left_eye_y = left_eye_center[1] 
        right_eye_x = right_eye_center[0] 
        right_eye_y = right_eye_center[1] 
        delta_x = right_eye_x - left_eye_x 
        delta_y = right_eye_y - left_eye_y 
        angle = np.arctan(delta_y / delta_x) 
        angle = (angle * 180) / np.pi 
        if angle > 10: 
            cv.putText(frame, 'RIGHT TILT :' + str(int(angle))+' degrees', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4) 
        elif angle < -10: 
            cv.putText(frame, 'LEFT TILT :' + str(int(angle))+' degrees', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4) 
        else: 
            cv.putText(frame, 'STRAIGHT :', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4) 
    cv.imshow('Frame', frame) 
    if cv.waitKey(1) & 0xFF == 27: 
        break
capture.release() 
cv.destroyAllWindows() 
