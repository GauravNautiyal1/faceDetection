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


# this is working code --main code --

# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# from PIL import Image

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


# Update code to applied database to store user data --

# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np
# import base64
# import io
# import sqlite3
# from datetime import datetime
# from PIL import Image
# import os
# from attendance_api import router as attendance_router
# from face_registration_api import router as face_registration_router

# app = FastAPI()

# # ✅ CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with your frontend domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(attendance_router)
# app.include_router(face_registration_router)

# # ✅ Initialize Database
# conn = sqlite3.connect("attendance.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute(
#     """CREATE TABLE IF NOT EXISTS attendance (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT,
#         date TEXT
#     )"""
# )
# conn.commit()

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

#             # ✅ Ensure RGB format
#             frame = np.array(image.convert("RGB"))

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             results = face_detection.process(rgb_frame)

#             response = {"faces": []}
#             today_date = datetime.today().strftime('%Y-%m-%d')

#             if results.detections:
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     h, w, _ = frame.shape
#                     x = max(0, int(bboxC.xmin * w))
#                     y = max(0, int(bboxC.ymin * h))
#                     width = max(0, int(bboxC.width * w))
#                     height = max(0, int(bboxC.height * h))

#                     face_crop = frame[y:y+height, x:x+width]

#                     if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
#                         print("⚠️ Skipping face - invalid crop size")
#                         continue  # Ignore invalid face regions

#                     try:
#                         # ✅ Debug: Check registered faces
#                         if not os.path.exists("registered_faces/"):
#                             os.makedirs("registered_faces/")

#                         registered_faces = os.listdir("registered_faces/")
#                         print(f"📂 Registered Faces: {registered_faces}")
#                         print("🔍 Searching for face in database...")

#                         result = DeepFace.find(
#                             face_crop, db_path="registered_faces/", model_name="ArcFace", enforce_detection=False
#                         )
#                         print(f"📌 DeepFace result: {result}")

#                         if len(result) > 0 and not result[0].empty:
#                             name = result[0]["identity"][0].split("/")[-1].split(".")[0]
#                             print(f"✅ Recognized as: {name}")
                            
#                             # ✅ Check if already marked today
#                             print("2345678")
#                             cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, today_date))
#                             existing_entry = cursor.fetchone()

#                             if not existing_entry:
#                                 cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (name, today_date))
#                                 conn.commit()
#                                 print(f"✅ Attendance marked for {name} on {today_date}")

#                             response["faces"].append({"name": name, "x": x, "y": y, "w": width, "h": height})
#                         else:
#                             response["faces"].append({"name": "Unknown", "x": x, "y": y, "w": width, "h": height})

#                     except Exception as e:
#                         print(f"⚠️ Face Recognition Error: {e}")

#             await websocket.send_json(response)

#         except Exception as e:
#             print(f"⚠️ WebSocket Error666221: {e}")
#             await websocket.send_json({"error": str(e)})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import base64
import io
import sqlite3
from datetime import datetime
from PIL import Image
import cloudinary
import cloudinary.api
import requests
import json
from attendance_api import router as attendance_router
from face_registration_api import router as face_registration_router
from database import conn, cursor  # ✅ Import database connection


app = FastAPI()

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(attendance_router)
app.include_router(face_registration_router)

# # ✅ Initialize Database
# conn = sqlite3.connect("attendance.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute(
#     """CREATE TABLE IF NOT EXISTS attendance (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT,
#         date TEXT
#     )"""
# )
# conn.commit()
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS students (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT,
#         branch TEXT,
#         semester TEXT,
#         image_url TEXT
#     )
# """)
# conn.commit()


# ✅ Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ✅ Configure Cloudinary
cloudinary.config(
    cloud_name="dpciu7mp5",
    api_key="364559298773626",
    api_secret="Q9wTwNqswx9M8YxsbLxjpfyAONA"
)
@app.get("/")
@app.head("/")
async def root():
    return {"message": "Face Recognition API is running!"}
# ✅ Get all registered faces from Cloudinary
def get_registered_faces(branch, semester):
    try:
        print(branch,"  ",semester)
        # ✅ Fetch only images stored in the correct branch & semester folder
        prefix = f"registered_faces/{branch}/{semester}"
        response = cloudinary.api.resources(type="upload", prefix=prefix)

        return {res["public_id"].split("/")[-1]: res["secure_url"] for res in response["resources"]}
    except Exception as e:
        print(f"⚠️ Error fetching registered faces from Cloudinary: {e}")
        return {}

# @app.websocket("/detect-face/{branch}/{semester}")
# async def detect_face(websocket: WebSocket, branch: str, semester: str):
#     await websocket.accept()
#     while True:
#         try:
#             data = await websocket.receive_text()
#             image_bytes = base64.b64decode(data)
#             image = Image.open(io.BytesIO(image_bytes))

#             frame = np.array(image.convert("RGB"))
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             results = face_detection.process(rgb_frame)

#             face_response = {"faces": []}
#             today_date = datetime.today().strftime('%Y-%m-%d')

#             if results.detections:
#                 # ✅ Fetch only the relevant branch & semester data
#                 registered_faces = get_registered_faces(branch, semester)
#                 print(f"📂 Cloudinary Registered Faces ({branch} - {semester}): {registered_faces}")

#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     h, w, _ = frame.shape
#                     x = max(0, int(bboxC.xmin * w))
#                     y = max(0, int(bboxC.ymin * h))
#                     width = max(0, int(bboxC.width * w))
#                     height = max(0, int(bboxC.height * h))

#                     face_crop = frame[y:y+height, x:x+width]

#                     if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
#                         print("⚠️ Skipping face - invalid crop size")
#                         continue

#                     try:
#                         # ✅ Compare only against filtered student faces
#                         matched_name = "Unknown"
#                         for name, image_url in registered_faces.items():
#                             ref_image_response = requests.get(image_url)
#                             if ref_image_response.status_code == 200:
#                                 ref_image = np.array(Image.open(io.BytesIO(ref_image_response.content)).convert("RGB"))
#                                 ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)

#                                 verification = DeepFace.verify(face_crop, ref_image, model_name="ArcFace", enforce_detection=False)

#                                 if isinstance(verification, dict) and verification.get("verified", False):
#                                     matched_name = name
#                                     print(f"✅ Recognized as: {matched_name}")
#                                     break

#                         # ✅ Mark attendance if recognized
#                         if matched_name != "Unknown":
#                             cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (matched_name, today_date))
#                             existing_entry = cursor.fetchone()

#                             if not existing_entry:
#                                 cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (matched_name, today_date))
#                                 conn.commit()
#                                 print(f"✅ Attendance marked for {matched_name} on {today_date}")

#                         face_response["faces"].append({
#                             "name": matched_name,
#                             "x": int(x),
#                             "y": int(y),
#                             "w": int(width),
#                             "h": int(height)
#                         })

#                     except Exception as e:
#                         print(f"⚠️ Face Recognition Error: {e}")

#             await websocket.send_text(json.dumps(face_response, default=str))

#         except Exception as e:
#             print(f"⚠️ WebSocket Error: {e}")
#             await websocket.send_text(json.dumps({"error": str(e)}, default=str))

@app.websocket("/detect-face/{branch}/{semester}")
async def detect_face(websocket: WebSocket, branch: str, semester: str):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            try:
                image_bytes = base64.b64decode(data)
            except base64.binascii.Error as e:
                print(f"⚠️ Invalid base64 data: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid image data"}))
                continue

            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image.convert("RGB"))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = face_detection.process(rgb_frame)

            face_response = {"faces": []}
            today_date = datetime.today().strftime('%Y-%m-%d')

            if results.detections:
                registered_faces = get_registered_faces(branch, semester)
                print(f"📂 Cloudinary Registered Faces ({branch} - {semester}): {registered_faces}")

                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = max(0, int(bboxC.xmin * w))
                    y = max(0, int(bboxC.ymin * h))
                    width = max(0, int(bboxC.width * w))
                    height = max(0, int(bboxC.height * h))

                    face_crop = frame[y:y+height, x:x+width]
                    if face_crop.size == 0:
                        print("⚠️ Skipping face - invalid crop size")
                        continue

                    matched_name = "Unknown"
                    for name, image_url in registered_faces.items():
                        ref_image_response = requests.get(image_url)
                        if ref_image_response.status_code == 200:
                            ref_image = np.array(Image.open(io.BytesIO(ref_image_response.content)).convert("RGB"))
                            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)

                            if ref_image.size > 0 and face_crop.size > 0:
                                try:
                                    verification = DeepFace.verify(face_crop, ref_image, model_name="ArcFace", enforce_detection=False)
                                    if isinstance(verification, dict) and verification.get("verified", False):
                                        matched_name = name
                                        print(f"✅ Recognized as: {matched_name}")
                                        break
                                except Exception as e:
                                    print(f"⚠️ DeepFace verification failed for {name}: {e}")

                    if matched_name != "Unknown":
                        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (matched_name, today_date))
                        if not cursor.fetchone():
                            cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (matched_name, today_date))
                            conn.commit()
                            print(f"✅ Attendance marked for {matched_name} on {today_date}")

                    face_response["faces"].append({
                        "name": matched_name,
                        "x": int(x),
                        "y": int(y),
                        "w": int(width),
                        "h": int(height)
                    })

            await websocket.send_text(json.dumps(face_response, default=str))
            await asyncio.sleep(0.1)  # Rate limit to ~10 FPS

        except Exception as e:
            print(f"⚠️ WebSocket Error: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
