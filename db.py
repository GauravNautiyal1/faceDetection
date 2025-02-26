# db.py
import os
import sqlite3

BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")
FACE_DB_PATH = os.path.join(BASE_DIR, "data", "registered_faces")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT
        )"""
    )
    conn.commit()

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(FACE_DB_PATH, exist_ok=True)
