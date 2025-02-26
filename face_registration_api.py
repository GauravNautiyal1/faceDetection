from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil
import logging
from db import FACE_DB_PATH

router = APIRouter()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/register-face/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    file_path = os.path.join(FACE_DB_PATH, f"{name}.jpg")
    
    logger.info(f"Attempting to save face to: {file_path}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        logger.info(f"Face saved successfully to {file_path}")
    
    return {"message": f"Face saved successfully as {file_path}"}
