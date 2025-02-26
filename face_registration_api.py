from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil

router = APIRouter()

# Ensure the folder exists
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

@router.post("/register-face/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    file_path = f"registered_faces/{name}.jpg"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": f"Face saved successfully as {file_path}"}
