from fastapi import APIRouter, UploadFile, File, Form
import cloudinary
import cloudinary.uploader
import sqlite3  # ✅ Import SQLite
from database import conn, cursor  # ✅ Import database connection


cloudinary.config(
    cloud_name="dpciu7mp5",
    api_key="364559298773626",
    api_secret="Q9wTwNqswx9M8YxsbLxjpfyAONA"
)
router = APIRouter()

# # Ensure the folder exists
# if not os.path.exists("registered_faces"):
#     os.makedirs("registered_faces")

# @router.post("/register-face/")
# async def register_face(name: str = Form(...), file: UploadFile = File(...)):
#     file_path = f"registered_faces/{name}.jpg"
    
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     return {"message": f"Face saved successfully as {file_path}"}




# @router.post("/register-face/")
# async def register_face(name: str = Form(...), branch: str = Form(...), semester: str = Form(...), file: UploadFile = File(...)):
#     try:
#         upload_result = cloudinary.uploader.upload(file.file, folder="registered_faces/{branch}/{semester}", public_id=name)
#         image_url = upload_result.get("secure_url")
#         # cursor.execute("INSERT INTO students (name, branch, semester) VALUES (?, ?, ?)", (name, branch, semester))
#         # conn.commit()

#         return {"message": "Face uploaded successfully", "image_url": image_url}

#     except Exception as e:
#         return {"error": f"Failed to upload image: {str(e)}"}

@router.post("/register-face/")
async def register_face(
    name: str = Form(...), 
    branch: str = Form(...), 
    semester: str = Form(...), 
    file: UploadFile = File(...)
):
    try:
        # ✅ Correct folder path formatting
        folder_path = f"registered_faces/{branch}/{semester}"
        print(f"📂 Uploading to Cloudinary in folder: {folder_path}")

        # ✅ Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(file.file, folder=folder_path, public_id=name)
        image_url = upload_result.get("secure_url")
        print(f"✅ Image uploaded: {image_url}")

        # ✅ Store in the database
        cursor.execute(
            "INSERT INTO students (name, branch, semester, image_url) VALUES (?, ?, ?, ?)", 
            (name, branch, semester, image_url)
        )
        conn.commit()
        print(f"✅ Data inserted into database for {name}")

        return {"message": "Face uploaded successfully", "image_url": image_url}

    except sqlite3.Error as db_error:
        print(f"❌ Database Error: {db_error}")
        return {"error": f"Database Error: {str(db_error)}"}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": f"Failed to upload image: {str(e)}"}
