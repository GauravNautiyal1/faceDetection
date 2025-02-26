# db.py
import os
import sqlite3
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")
FACE_DB_PATH = os.path.join(BASE_DIR, "data", "registered_faces")

def init_db():
    try:
        # Ensure directories exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        logger.info(f"Created directories: {os.path.dirname(DB_PATH)}, {FACE_DB_PATH}")

        # Test writability
        test_file = os.path.join(BASE_DIR, "data", "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Confirmed {BASE_DIR}/data is writable")

        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                date TEXT
            )"""
        )
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
        return conn, cursor
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# Initialize at module level
conn, cursor = init_db()
