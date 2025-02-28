from fastapi import UploadFile, File
from . import logger, FACE_DB_PATH
from face_recognition import initialize_deepface_db  # Import from main file

@router.post("/register-face/")
async def register_face(file: UploadFile = File(...)):
    save_path = os.path.join(FACE_DB_PATH, file.filename)
    logger.info(f"Attempting to save face to: {save_path}")
    with open(save_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"Face saved successfully to {save_path}")
    initialize_deepface_db()  # Rebuild database
    return {"message": "Face registered successfully"}
