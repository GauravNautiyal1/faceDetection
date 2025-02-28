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



# from fastapi import FastAPI, WebSocket, Request
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# from PIL import Image
# import os
# import json

# # ✅ Force TensorFlow to use CPU if GPU is unavailable
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

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
# # @app.get("/")
# # def read_root():
# #     return {"message": "Face Detection API is Running 🚀"}



# @app.api_route("/", methods=["GET", "HEAD"])
# def read_root(request: Request):
#     return {"message": "Face Detection API is Running 🚀"}



    

# # ✅ Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# @app.websocket("/detect-face")
# async def detect_face(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket Connection Opened ✅")

#     try:
#         while True:
#             try:
#                 data = await websocket.receive_text()

#                 # ✅ Decode Base64 image
#                 image_bytes = base64.b64decode(data)
#                 image = Image.open(io.BytesIO(image_bytes))
#                 frame = np.array(image)
#                 print("Image received from WebSocket")

#                 if frame.size == 0:
#                     print("⚠️ Received empty frame.")
#                     continue

#                 # ✅ Convert to RGB for Mediapipe
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_detection.process(rgb_frame)
#                 print("Face detection completed")

#                 response = {"faces": []}

#                 # ✅ Face Detection with Mediapipe
#                 if results.detections:
#                     print(f"Detected {len(results.detections)} face(s)")
#                     for detection in results.detections:
#                         bboxC = detection.location_data.relative_bounding_box
#                         height, width, _ = frame.shape
#                         x = int(bboxC.xmin * width)
#                         y = int(bboxC.ymin * height)
#                         w = int(bboxC.width * width)
#                         h = int(bboxC.height * height)

#                         face_crop = frame[y:y+h, x:x+w]

#                         if face_crop.size != 0:
#                             try:
#                                 # ✅ DeepFace Recognition
#                                 print("gaurav445566")
#                                 result = DeepFace.find(face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)
#                                 print("DeepFace search completed")

#                                 if len(result) > 0 and len(result[0]) > 0:
#                                     name = result[0]["identity"][0].split("/")[-1].split(".")[0]
#                                     print(f"✅ Detected Face Name: {name}") 
#                                     response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
#                                 else:
#                                     response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

#                             except Exception as e:
#                                 print("DeepFace Error:", e)
#                                 response["faces"].append({"name": "Recognition Error", "x": x, "y": y, "w": w, "h": h})
#                         else:
#                             response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
#                             print("Empty face crop detected.")

#                 else:
#                     print("No faces detected.")

#                 # ✅ Send response to WebSocket
#                 await websocket.send_json(response)
#                 print("Sent Response:", response)

#             except Exception as e:
#                 print(f"WebSocket Error: {e}")
#                 error_response = {"error": str(e)}
#                 await websocket.send_json(error_response)

#     except Exception as outer_error:
#         print(f"WebSocket Closed due to Error: {outer_error}")
#     finally:
#         await websocket.close()
#         print("WebSocket Connection Closed ❌")

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# import os
# import logging
# from datetime import datetime
# from PIL import Image
# from attendance_api import router as attendance_router
# from face_registration_api import router as face_registration_router
# from db import FACE_DB_PATH, conn, cursor

# app = FastAPI()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(attendance_router)
# app.include_router(face_registration_router)

# # Prebuild DeepFace database on startup
# def initialize_deepface_db():
#     try:
#         if os.path.exists(FACE_DB_PATH) and os.listdir(FACE_DB_PATH):
#             logger.info(f"Building DeepFace database from {FACE_DB_PATH}")
#             DeepFace.represent(img_path=os.path.join(FACE_DB_PATH, "*.jpg"), model_name="ArcFace", enforce_detection=False)
#             logger.info("DeepFace database initialized")
#         else:
#             logger.warning(f"No images found in {FACE_DB_PATH} to build database")
#     except Exception as e:
#         logger.error(f"Failed to initialize DeepFace database: {e}")

# # Initialize on startup
# initialize_deepface_db()

# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# @app.get("/")
# async def root():
#     return {"status": "OK", "message": "Face recognition server is running"}

# @app.websocket("/detect-face")
# async def detect_face(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         try:
#             data = await websocket.receive_text()
#             image_bytes = base64.b64decode(data)
#             image = Image.open(io.BytesIO(image_bytes))
#             frame = np.array(image)

#             logger.info(f"Received frame: {frame.shape}")

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(rgb_frame)

#             response = {"faces": []}
#             today_date = datetime.today().strftime('%Y-%m-%d')

#             if results.detections:
#                 logger.info(f"Detected {len(results.detections)} faces")
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     h, w, _ = frame.shape
#                     x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

#                     x, y = max(0, x), max(0, y)
#                     w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
#                     if w <= 0 or h <= 0:
#                         logger.warning(f"Invalid bounding box: x={x}, y={y}, w={w}, h={h}")
#                         continue

#                     face_crop = frame[y:y+h, x:x+w]
#                     logger.info(f"Face crop shape: {face_crop.shape}")

#                     try:
#                         # Ensure database exists before finding
#                         if not os.path.exists(os.path.join(FACE_DB_PATH, "representations_arcface.pkl")):
#                             logger.warning("No DeepFace database found, rebuilding...")
#                             initialize_deepface_db()

#                         result = DeepFace.find(face_crop, db_path=FACE_DB_PATH, model_name="ArcFace", enforce_detection=False)
#                         logger.info(f"DeepFace result: {result}")

#                         if isinstance(result, list) and len(result) > 0 and len(result[0]) > 0:
#                             name = result[0]["identity"][0].split("/")[-1].split(".")[0]
                            
#                             cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, today_date))
#                             existing_entry = cursor.fetchone()

#                             if not existing_entry:
#                                 cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (name, today_date))
#                                 conn.commit()
#                                 logger.info(f"Attendance marked for {name} on {today_date}")

#                             response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
#                         else:
#                             logger.info("No match found in DeepFace database")
#                             response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

#                     except Exception as e:
#                         logger.error(f"Error in face detection: {e}")
#                         response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})  # Fallback

#             else:
#                 logger.info("No faces detected by MediaPipe")

#             logger.info(f"Sending response: {response}")
#             await websocket.send_json(response)

#         except Exception as e:
#             logger.error(f"WebSocket error: {e}")
#             break

# @app.get("/registered-faces/")
# async def list_registered_faces():
#     logger.info(f"Listing files in {FACE_DB_PATH}")
#     try:
#         files = os.listdir(FACE_DB_PATH)
#         logger.info(f"Found files: {files}")
#         return {"registered_faces": files}
#     except Exception as e:
#         logger.error(f"Error listing files: {e}")
#         return {"registered_faces": [], "error": str(e)}



# .............................................................

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import base64
import io
import logging
from datetime import datetime
from PIL import Image
from attendance_api import router as attendance_router
from face_registration_api import router as face_registration_router
from db import FACE_DB_PATH, conn, cursor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices([], 'GPU')
    logger.info("GPU disabled successfully")
else:
    logger.info("No GPU detected, running on CPU")
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attendance_router)
app.include_router(face_registration_router)

def initialize_deepface_db():
    try:
        dir_path = FACE_DB_PATH
        files = os.listdir(dir_path)
        logger.info(f"Files in {dir_path}: {files}")
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        logger.info(f"Valid image files: {image_files}")

        if not image_files:
            logger.warning(f"No valid image files found in {dir_path} to build database")
            return

        # Pre-build representations for all images
        logger.info(f"Building DeepFace database from {dir_path}")
        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                continue
            DeepFace.represent(img, model_name="ArcFace", enforce_detection=False)  # Precompute representations
        DeepFace.find(np.zeros((100, 100, 3)), db_path=dir_path, model_name="ArcFace", enforce_detection=False)  # Trigger database save
        logger.info("DeepFace database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DeepFace database: {e}")

# Run at startup (optional, may not find images yet)
initialize_deepface_db()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@app.get("/")
async def root():
    return {"status": "OK", "message": "Face recognition server is running"}

@app.get("/rebuild-deepface-db/")
async def rebuild_deepface_db():
    initialize_deepface_db()
    return {"message": "DeepFace database rebuild attempted"}
@app.websocket("/detect-face")
async def detect_face(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection opened")
    while True:
        try:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)

            logger.info(f"Received frame: {frame.shape}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            response = {"status": "success", "faces": []}
            today_date = datetime.today().strftime('%Y-%m-%d')

            if results.detections:
                logger.info(f"Detected {len(results.detections)} faces")
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    x, y = max(0, x), max(0, y)
                    w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                    if w <= 0 or h <= 0:
                        logger.warning(f"Invalid bounding box: x={x}, y={y}, w={w}, h={h}")
                        continue

                    face_crop = frame[y:y+h, x:x+w]
                    logger.info(f"Face crop shape: {face_crop.shape}")

                    try:
                        db_files = [f for f in os.listdir(FACE_DB_PATH) if f.endswith(('.jpg', '.png', '.jpeg'))]
                        if not db_files:
                            logger.warning("No registered faces in database")
                            response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})
                        else:
                            result = DeepFace.find(face_crop, db_path=FACE_DB_PATH, model_name="ArcFace", enforce_detection=False)
                            logger.info(f"DeepFace result: {result}")

                            if isinstance(result, list) and result and not result[0].empty:
                                name = result[0]["identity"].iloc[0].split("/")[-1].split(".")[0]
                                cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, today_date))
                                if not cursor.fetchone():
                                    cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (name, today_date))
                                    conn.commit()
                                    logger.info(f"Attendance marked for {name} on {today_date}")
                                response["faces"].append({"name": name, "x": x, "y": y, "w": w, "h": h})
                            else:
                                logger.info("No match found in DeepFace database")
                                response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

                    except Exception as e:
                        logger.error(f"Recognition error: {e}")
                        response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": w, "h": h})

            else:
                logger.info("No faces detected by MediaPipe")
                response["status"] = "success"
                response["faces"] = []

            logger.info(f"Sending response: {response}")
            await websocket.send_json(response)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_json({"status": "error", "message": str(e)})
            break
@app.get("/registered-faces/")
async def list_registered_faces():
    logger.info(f"Listing files in {FACE_DB_PATH}")
    try:
        files = os.listdir(FACE_DB_PATH)
        logger.info(f"Found files: {files}")
        return {"registered_faces": files}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"registered_faces": [], "error": str(e)}
