# Haarcascade Face Detection App (Streamlit)

[View App](https://a-face-detection-app.streamlit.app/)

A lightweight face detection web app using OpenCV Haarcascades to detect faces, eyes, and smiles from camera input or uploaded images.

## Objective

To detect facial features such as faces, eyes, and smiles using traditional Haarcascade classifiers in a browser.

## Workflow

- User captures or uploads an image
- Image is processed using:
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye_tree_eyeglasses.xml`
  - `haarcascade_smile.xml`
- Detected features are annotated

## Features

- Detect faces, eyes, and smiles
- Visual overlay of detections
- Real-time webcam or file upload input

## Dependencies

- `streamlit`
- `opencv-python`
- `numpy`
- `PIL`

## Output

- Annotated image with facial feature boxes

