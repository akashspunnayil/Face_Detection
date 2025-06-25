# apps/face_detection_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run_face_detection():
    st.title("📷 Face Detection App using Haar Cascades")

    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    img_data = st.camera_input("Capture a photo") or st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

    if img_data:
        img = Image.open(img_data)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                radius = int(round((ew + eh) * 0.25))
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, "Eye", (center[0] - 10, center[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(frame, "Smile", (x + sx, y + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Facial Features", use_column_width=True)
