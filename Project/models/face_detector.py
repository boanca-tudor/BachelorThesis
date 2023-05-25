import cv2
import numpy as np


def get_cropped_face(bytes):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    nparr = np.frombuffer(bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.03, 5)
    if len(faces) == 0:
        return 0, img
    x, y, w, h = faces[0]
    cropped = img[y:y+h, x:x+w]
    return 1, cropped
