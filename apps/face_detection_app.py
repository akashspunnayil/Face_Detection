# apps/face_detection_app.py
# apps/face_detection_live.py
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load Haar cascades once globally
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

class FaceProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                radius = int(round((ew + eh) * 0.25))
                cv2.circle(image, center, radius, (0, 255, 0), 2)
                cv2.putText(image, "Eye", (center[0] - 10, center[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(image, "Smile", (x + sx, y + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

def run_face_detection_live():
    st.title("📹 Live Face Detection (Haar Cascade + WebRTC)")

    webrtc_streamer(
        key="live-face-detection",
        video_processor_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

