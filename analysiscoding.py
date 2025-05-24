# Import necessary libraries
import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np

# Path to the input image
image_path = r"C:\Users\user\OneDrive\Masaüstü\photo.jpg"

# Read the image
img = cv2.imread(image_path)

# Check if the image is valid
if img is None:
    print("Could not load the image. Please enter a valid path.")
else:
    # Keep a copy of the original image
    original_img = img.copy()

    # Initialize MediaPipe Face Detection and Face Mesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

    # Convert the image to RGB as MediaPipe works with RGB images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(img_rgb)

    # If faces are detected
    if results.detections:
        # Iterate over detected faces
        for detection in results.detections:
            # Get the bounding box for the face
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)

            # Extract the face region
            face_roi = img[y:y+h, x:x+w]

            try:
                # Analyze the facial expression of the current face using DeepFace
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Get the dominant emotion
                emotion = analysis[0]['dominant_emotion']

                # Add text overlay on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, emotion, (x, y - 10), font, 5, (0, 255, 0), 3, cv2.LINE_AA)

                # Draw a rectangle around the face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

                # Convert the face ROI to RGB for MediaPipe FaceMesh processing
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                # Get the face landmarks using MediaPipe FaceMesh
                face_landmarks = face_mesh.process(face_roi_rgb)

                if face_landmarks.multi_face_landmarks:
                    # Draw the connections (lines) between landmark points
                    for landmarks in face_landmarks.multi_face_landmarks:
                        for connection in mp_face_mesh.FACEMESH_TESSELATION:
                            start_idx = connection[0]
                            end_idx = connection[1]

                            start = landmarks.landmark[start_idx]
                            end = landmarks.landmark[end_idx]

                            # Get the x, y coordinates of each landmark
                            x1, y1 = int(start.x * w + x), int(start.y * h + y)
                            x2, y2 = int(end.x * w + x), int(end.y * h + y)

                            # Draw a line between the landmarks with the new color
                            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            except Exception as e:
                print(f"Error analyzing face at ({x}, {y}): {str(e)}")

    # If no faces are detected, print a message
    else:
        print("No faces detected in the image.")

    # Save the final result as an image
    output_path = r"C:\Users\user\OneDrive\Masaüstü\output_image.jpg"
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")
