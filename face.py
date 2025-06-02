import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import threading
import face_recognition

def get_model_path():
    if hasattr(sys, "_MEIPASS"):
        # If running from the bundled .exe, use the temp folder path
        return os.path.join(sys._MEIPASS, 'face_recognition_models', 'models', 'shape_predictor_5_face_landmarks.dat')
    else:
        # If running in the development environment, use the path on disk
        return 'path_to_model/shape_predictor_5_face_landmarks.dat'

# Use the function to get the correct model path
model_path = get_model_path()

# Now use model_path to load the model in face_recognition or other relevant functions
try:
    face_recognition.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

class AttendanceSystemApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("700x500")

        # UI Elements
        self.start_button = tk.Button(root, text="Start", command=self.start_system, bg="green", fg="white", font=("Arial", 14))
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_system, bg="red", fg="white", font=("Arial", 14), state="disabled")
        self.stop_button.pack(pady=10)

        self.update_db_button = tk.Button(root, text="Update Photo Database", command=self.update_photo_database, bg="blue", fg="white", font=("Arial", 14))
        self.update_db_button.pack(pady=10)

        self.view_logs_button = tk.Button(root, text="View Attendance Logs", command=self.view_logs, bg="orange", fg="black", font=("Arial", 14))
        self.view_logs_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Stopped", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.attendance_log = ttk.Treeview(root, columns=("Name", "Time"), show="headings", height=10)
        self.attendance_log.heading("Name", text="Name")
        self.attendance_log.heading("Time", text="Time")
        self.attendance_log.pack(pady=10)

        # Variables
        self.running = False
        self.video_capture = None
        self.thread = None

        # Load known faces
        self.known_face_encodings, self.known_faces_names = self.load_known_faces()
        self.students = self.known_faces_names.copy()

    def load_known_faces(self, image_folder="photos"):
        """
        Load known face encodings and names from a folder.
        """
        known_face_encodings = []
        known_faces_names = []

        jpeg_files = [file for file in os.listdir(image_folder) if file.endswith(".jpg")]

        for jpeg_file in jpeg_files:
            name = os.path.splitext(jpeg_file)[0]
            image_path = os.path.join(image_folder, jpeg_file)

            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_faces_names.append(name)
            else:
                print(f"No face detected in {jpeg_file}")

        return known_face_encodings, known_faces_names

    def start_system(self):
        """
        Start the attendance system.
        """
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Running")
        self.thread = threading.Thread(target=self.run_system)
        self.thread.start()

    def stop_system(self):
        """
        Stop the attendance system.
        """
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")
        if self.video_capture:
            self.video_capture.release()
            cv2.destroyAllWindows()

    def update_photo_database(self):
        """
        Capture a new photo using the webcam when the user presses the spacebar or Enter,
        and save it with the user's preferred name in the photo database.
        """
        # Ask the user for a name
        name = simpledialog.askstring("Enter Name", "Enter name for the new photo:")
        if not name:
            messagebox.showerror("Error", "No name provided. Photo will not be saved.")
            return

        # Open the webcam for capturing the photo
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            return

        while True:
            # Capture a frame from the camera
            ret, frame = video_capture.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                video_capture.release()
                return

            # Display the frame with live feed
            cv2.putText(frame, "Press 'Space' or 'Enter' to take a photo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Photo", frame)

            # Wait for the user to press 'Space' or 'Enter' to capture the photo
            key = cv2.waitKey(1) & 0xFF
            if key == 32 or key == 13:  # Spacebar (32) or Enter (13)
                # Save the captured image in the 'photos' folder
                photo_path = os.path.join("photos", f"{name}.jpg")
                cv2.imwrite(photo_path, frame)

                messagebox.showinfo("Photo Saved", f"Photo saved as {name}.jpg in the photo database.")

                # Update the photo database
                self.known_face_encodings, self.known_faces_names = self.load_known_faces()
                self.students = self.known_faces_names.copy()

                break  # Exit the loop after taking the photo

        # Release the camera and close the window
        video_capture.release()
        cv2.destroyAllWindows()

    def view_logs(self):
        """
        View attendance logs in a new window.
        """
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Attendance Logs")
        logs_window.geometry("500x400")

        logs_tree = ttk.Treeview(logs_window, columns=("Name", "Time"), show="headings", height=20)
        logs_tree.heading("Name", text="Name")
        logs_tree.heading("Time", text="Time")
        logs_tree.pack(fill=tk.BOTH, expand=True)

        # Load logs from CSV
        log_files = [file for file in os.listdir() if file.endswith(".csv")]
        if log_files:
            for log_file in log_files:
                with open(log_file, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)  # Skip header
                    for row in reader:
                        logs_tree.insert("", "end", values=row)
        else:
            messagebox.showinfo("View Logs", "No logs found.")

    def run_system(self):
        """
        Main loop for the attendance system.
        """
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            self.stop_system()
            return

        now = datetime.now()
        csv_filename = now.strftime("%Y-%m-%d") + ".csv"
        with open(csv_filename, "w+", newline="") as csv_file:
            lnwriter = csv.writer(csv_file)
            lnwriter.writerow(["Name", "Time"])

            while self.running:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_detected = False # Track if any face is detected in the frame

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    face_detected = True
                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(distances)

                    if distances[best_match_index] < 0.6:
                        name = self.known_faces_names[best_match_index]
                    else:
                        name = "Unknown"

                    # Drawing the box around the face and showing name
                    top, right, bottom, left = face_location
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4 # scale back up
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    if name in self.students:
                        self.students.remove(name)
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        lnwriter.writerow([name, current_time])
                        self.attendance_log.insert("", "end", values=(name, current_time))

                if not face_detected:
                    cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the frame
                cv2.imshow("Attendance System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.video_capture.release()
        cv2.destroyAllWindows()

# Create the Tkinter app
root = tk.Tk()
app = AttendanceSystemApp(root)
root.mainloop()
