import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import subprocess

# Define global variables
path = 'C:/coding/coding/python/face recognisation by sasank/images'
images = []
classNames = []
markedNames = set()

# Function to add images
def add_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            file_name = os.path.basename(file_path)
            save_path = os.path.join(path, file_name)
            cv2.imwrite(save_path, img)
            classNames.append(os.path.splitext(file_name)[0])
            images.append(img)
            print(f"Image {file_name} added successfully.")
        else:
            print("Error loading image.")

# Function to open the attendance CSV
def open_attendance_sheet():
    current_date = datetime.now().strftime('%d-%m-%y_%H-%M')
    csv_file_path = f'C:/coding/coding/python/face recognisation by sasank/attendance_{current_date}.csv'
    
    # Check if the file exists and open it
    if os.path.exists(csv_file_path):
        subprocess.Popen(['notepad.exe', csv_file_path])  # Open in Notepad (Windows)
    else:
        print(f"Attendance file {csv_file_path} not found.")

# Function to find encodings for the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

# Function to mark attendance
def markAttendance(name):
    global markedNames
    current_date = datetime.now().strftime('%d-%m-%y_%H-%M')
    csv_file_path = f'C:/coding/coding/python/face recognisation by sasank/attendance_{current_date}.csv'
    
    with open(csv_file_path, 'a+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        # Check if the name is already marked
        if name not in markedNames and name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}\n')
            markedNames.add(name)

# Create the GUI interface
def create_gui():
    window = tk.Tk()
    window.title("Face Recognition Attendance System")

    # Add Image Button
    add_image_button = tk.Button(window, text="Add Image", command=add_image)
    add_image_button.pack(pady=10)

    # Open Attendance Sheet Button
    open_attendance_button = tk.Button(window, text="Open Attendance Sheet", command=open_attendance_sheet)
    open_attendance_button.pack(pady=10)

    # Start the video capture
    start_button = tk.Button(window, text="Start Recognition", command=start_recognition)
    start_button.pack(pady=20)

    # Run the GUI
    window.mainloop()

# Function to start facial recognition
def start_recognition():
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the GUI
create_gui()