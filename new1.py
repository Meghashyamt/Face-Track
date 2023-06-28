import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model

facetracker = load_model('facetracker1.h5')

cap = cv2.VideoCapture(0)
mtcnn = MTCNN()

while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    # Detect faces using MTCNN
    faces = mtcnn.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        x1, y1 = x, y
        x2, y2 = x + width, y + height

        face_region = frame[y1:y2, x1:x2, :]
        rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            # Controls the main rectangle
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (255, 0, 0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                          (x1, y1 - 30),
                          (x1 + 80, y1),
                          (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, 'face', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face Tracking System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
