from fastapi import APIRouter, HTTPException
import sqlite3

router = APIRouter()

conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

@router.get("/recognition/api/monthly-attendance/{year}/{month}/")
async def get_monthly_attendance(year: int, month: int):
    try:
        cursor.execute(
            "SELECT name, date FROM attendance WHERE strftime('%Y', date) = ? AND strftime('%m', date) = ?",
            (str(year), f"{month:02d}")
        )
        attendance_records = cursor.fetchall()

        attendance_data = {}
        for name, date in attendance_records:
            day = int(date.split("-")[2])
            if name not in attendance_data:
                attendance_data[name] = {}
            attendance_data[name][day] = "P"  # P for Present

        formatted_data = [
            {"name": name, "attendance": attendance}
            for name, attendance in attendance_data.items()
        ]

        return {"attendance": formatted_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
