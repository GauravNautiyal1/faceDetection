import sqlite3
# ✅ Initialize Database
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT
    )"""
)
conn.commit()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        branch TEXT,
        semester TEXT,
        image_url TEXT
    )
""")
conn.commit()
