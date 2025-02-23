import cv2
import os

# Create a folder for registered faces if not exists
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

# Open webcam
cap = cv2.VideoCapture(0)
name = input("Enter your name: ").strip()

if name == "":
    print("Name cannot be empty!")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 's' to save the face, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        file_path = f"registered_faces/{name}.jpg"
        cv2.imwrite(file_path, frame)
        print(f"Face saved as {file_path}")
        break
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
