# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import os

# # Initialize Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)

#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

#             # Crop the detected face
#             face_crop = frame[y:y+h, x:x+w]

#             try:
#                 # Recognize face using DeepFace
#                 result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)

#                 if len(result) > 0 and len(result[0]) > 0:
#                     name = result[0]["identity"][0].split("/")[-1].split(".")[0]  # Extract name from file path
#                     cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 else:
#                     cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#             except Exception as e:
#                 print("Error:", e)

#             # Draw bounding box
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import base64
import io
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@app.websocket("/detect-face")
async def detect_face(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            response = {"faces": []}

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    face_crop = frame[y:y+h, x:x+w]

                    try:
                        result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)

                        if len(result) > 0 and len(result[0]) > 0:
                            name = result[0]["identity"][0].split("/")[-1].split(".")[0]
                            response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
                        else:
                            response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

                    except Exception as e:
                        print("Error:", e)

            await websocket.send_json(response)

        except Exception as e:
            print(f"Error: {e}")
            break
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 for local, dynamic in production
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
