import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# Load model
model = tf.keras.models.load_model("yawn_detection_model4.keras")

IMG_SIZE = 224
THRESHOLD = 0.7

# Smooth predictions
pred_buffer = deque(maxlen=5)

# Yawn detection counters
yawn_counter = 0
SAVE_COOLDOWN = 5
last_save_time = 0

# Create folder for saved yawns
os.makedirs("yawn_events", exist_ok=True)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        # Resize same as training
        img = cv2.resize(face,(IMG_SIZE,IMG_SIZE))

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img/255.0
        img = np.expand_dims(img,axis=0)

        pred = model.predict(img,verbose=0)[0][0]

        pred_buffer.append(pred)
        smooth_pred = np.mean(pred_buffer)

        # Determine label
        if smooth_pred > THRESHOLD:
            label = "YAWNING"
            color = (0,0,255)
            yawn_counter += 1
        else:
            label = "NOT YAWNING"
            color = (0,255,0)
            yawn_counter = 0

        # ALERT if yawning persists
        if yawn_counter > 15:
            cv2.putText(frame,
                        "ALERT: DRIVER DROWSY!",
                        (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            # Save event image
            if time.time() - last_save_time > SAVE_COOLDOWN:
                filename = f"yawn_events/yawn_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                last_save_time = time.time()

        # Draw bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        # Show label
        cv2.putText(frame,
                    f"{label} {smooth_pred:.2f}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2)

        # Probability bar UI
        bar_length = int(smooth_pred * 200)

        cv2.rectangle(frame,(30,30),(30+bar_length,50),(0,255,0),-1)
        cv2.rectangle(frame,(30,30),(230,50),(255,255,255),2)

        cv2.putText(frame,
                    "Yawn Probability",
                    (30,25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

    cv2.imshow("Yawn Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()