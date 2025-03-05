# import cv2
# import os
# from tkinter import Tk, filedialog

# # Create a folder for registered faces if not exists
# if not os.path.exists("registered_faces"):
#     os.makedirs("registered_faces")

# # Hide Tkinter main window
# Tk().withdraw()

# # Open file dialog to select image
# file_path = filedialog.askopenfilename(title="Select Face Image", 
#                                        filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

# if not file_path:
#     print("No file selected!")
#     exit()

# # Enter the name for the face
# name = input("Enter the name for the face: ").strip()

# if name == "":
#     print("Name cannot be empty!")
#     exit()

# # Read and save the selected image
# image = cv2.imread(file_path)
# if image is None:
#     print("Invalid image file!")
#     exit()

# # Save image to the registered_faces folder
# save_path = f"registered_faces/{name}.jpg"
# cv2.imwrite(save_path, image)
# print(f"Face saved as {save_path}")

# # Show the saved image (Optional)
# cv2.imshow("Saved Face", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# from deepface import DeepFace

# result = DeepFace.find(img_path="debug_face.jpg", db_path="registered_faces/", model_name="ArcFace", enforce_detection=False)
# print(result)

import keras as tf
print(tf.__version__)
