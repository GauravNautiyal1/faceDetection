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

# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# from PIL import Image
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# app = FastAPI()

# # ✅ CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with your frontend domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# @app.websocket("/detect-face")
# async def detect_face(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         try:
#             data = await websocket.receive_text()
#             image_bytes = base64.b64decode(data)
#             image = Image.open(io.BytesIO(image_bytes))
#             frame = np.array(image)

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(rgb_frame)

#             response = {"faces": []}

#             if results.detections:
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     h, w, _ = frame.shape
#                     x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

#                     face_crop = frame[y:y+h, x:x+w]

#                     try:
#                         result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)

#                         if len(result) > 0 and len(result[0]) > 0:
#                             name = result[0]["identity"][0].split("/")[-1].split(".")[0]
#                             response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
#                         else:
#                             response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

#                     except Exception as e:
#                         print("Error:", e)

#             await websocket.send_json(response)

#         except Exception as e:
#             print(f"Error: {e}")
#             break
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 10000))  # Default to 10000 for local, dynamic in production
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)





# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# from PIL import Image
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU if no GPU

# app = FastAPI()

# # ✅ CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with frontend domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Root Route for Health Check
# @app.get("/")
# def read_root():
#     return {"message": "Face Detection API is Running 🚀"}

# # ✅ Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# @app.websocket("/detect-face")
# async def detect_face(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         try:
#             data = await websocket.receive_text()
#             image_bytes = base64.b64decode(data)
#             image = Image.open(io.BytesIO(image_bytes))
#             frame = np.array(image)
#             print("Image received from WebSocket")

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(rgb_frame)

#             response = {"faces": []}

#             if results.detections:
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     height, width, _ = frame.shape
#                     x = int(bboxC.xmin * width)
#                     y = int(bboxC.ymin * height)
#                     w = int(bboxC.width * width)
#                     h = int(bboxC.height * height)

#                     face_crop = frame[y:y+h, x:x+w]

#                     if face_crop.size != 0:
#                         try:
#                             result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)

#                             if len(result) > 0 and len(result[0]) > 0:
#                                 name = result[0]["identity"][0].split("/")[-1].split(".")[0]
#                                 response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
#                             else:
#                                 response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

#                         except Exception as e:
#                             print("DeepFace Error:", e)
#                     else:
#                         print("Empty face crop detected.")

#             await websocket.send_json(response)

#         except Exception as e:
#             print(f"WebSocket Error: {e}")
#             await websocket.close()  # Close WebSocket on error
#             break

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)



from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import base64
import io
from PIL import Image
import os
import json

# ✅ Force TensorFlow to use CPU if GPU is unavailable
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root Route for Health Check
# @app.get("/")
# def read_root():
#     return {"message": "Face Detection API is Running 🚀"}



@app.api_route("/", methods=["GET", "HEAD"])
def read_root(request: Request):
    return {"message": "Face Detection API is Running 🚀"}



    

# ✅ Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@app.websocket("/detect-face")
async def detect_face(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket Connection Opened ✅")

    try:
        while True:
            try:
                data = await websocket.receive_text()

                # ✅ Decode Base64 image
                image_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_bytes))
                frame = np.array(image)
                print("Image received from WebSocket")

                if frame.size == 0:
                    print("⚠️ Received empty frame.")
                    continue

                # ✅ Convert to RGB for Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                print("Face detection completed")

                response = {"faces": []}

                # ✅ Face Detection with Mediapipe
                if results.detections:
                    print(f"Detected {len(results.detections)} face(s)")
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        height, width, _ = frame.shape
                        x = int(bboxC.xmin * width)
                        y = int(bboxC.ymin * height)
                        w = int(bboxC.width * width)
                        h = int(bboxC.height * height)

                        face_crop = frame[y:y+h, x:x+w]

                        if face_crop.size != 0:
                            try:
                                # ✅ DeepFace Recognition
                                result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)
                                print("DeepFace search completed")

                                if len(result) > 0 and len(result[0]) > 0:
                                    name = result[0]["identity"][0].split("/")[-1].split(".")[0]
                                    response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
                                else:
                                    response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

                            except Exception as e:
                                print("DeepFace Error:", e)
                                response["faces"].append({"name": "Recognition Error", "x": x, "y": y, "w": w, "h": h})
                        else:
                            print("⚠️ Empty face crop detected.")

                else:
                    print("No faces detected.")

                # ✅ Send response to WebSocket
                await websocket.send_json(response)
                print("Sent Response:", response)

            except Exception as e:
                print(f"WebSocket Error: {e}")
                error_response = {"error": str(e)}
                await websocket.send_json(error_response)

    except Exception as outer_error:
        print(f"WebSocket Closed due to Error: {outer_error}")
    finally:
        await websocket.close()
        print("WebSocket Connection Closed ❌")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
