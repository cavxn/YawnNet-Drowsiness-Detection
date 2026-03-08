import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained model
model = tf.keras.models.load_model("yawn_detection_model.keras")

IMG_SIZE = 224

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmark indices
mouth_indices = [
    61,146,91,181,84,17,314,405,321,375,
    291,308,324,318,402,317,14,87,178,88
]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            points = []

            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                points.append((x,y))

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            mouth = frame[y_min:y_max, x_min:x_max]

            if mouth.size != 0:

                img = cv2.resize(mouth,(IMG_SIZE,IMG_SIZE))
                img = img/255.0
                img = np.expand_dims(img,axis=0)

                pred = model.predict(img,verbose=0)

                if pred[0][0] > 0.5:
                    label = "YAWNING"
                    color = (0,0,255)
                else:
                    label = "NOT YAWNING"
                    color = (0,255,0)

                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,2)
                cv2.putText(frame,label,(x_min,y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,color,2)

    cv2.imshow("Yawn Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()