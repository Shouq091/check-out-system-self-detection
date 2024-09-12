
import cv2
import json
import numpy as np
from datetime import datetime
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import supervision as sv
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

class App:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Object Detection App")
        self.root.geometry("800x600")
        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.frame, width=640, height=480)
        self.canvas.pack()

        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.upload_video_button = tk.Button(self.frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.live_button = tk.Button(self.frame, text="Live Stream", command=self.live_stream)
        self.live_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = tk.Button(self.frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.total_object_count = 0  # Initialize total object count
        self.video_cap = None
        self.live_cap = None
        self.image_processing = False
        self.running_video = False
        self.running_live = False
        self.frame_queue = Queue()
        self.frame_count = 0  # Initialize frame count
        self.counted_objects = set()  # Set to keep track of counted objects
        self.frame_counter_for_total = 0  # Initialize counter for adding to total objects

        # Setup annotators
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

        # Thread pool executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

    def process_frame(self, frame):
        # Convert frame to the format expected by the model
        result = self.model.predict(frame, confidence=30, overlap=20).json()
        detections = sv.Detections.from_inference(result)

        labels = [
            f"{detection['class']} {detection['confidence']:0.2f}"
            for detection in result['predictions']
        ]

        # Annotate bounding boxes
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=frame, 
            detections=detections
        )

        # Annotate labels
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # Calculate mirrored points for the red line
        pt1 = (20, 50)  # Starting point of the line
        pt2 = (600, 500)  # Ending point of the line
        mirrored_pt1 = (annotated_frame.shape[1] - pt1[0], pt1[1])
        mirrored_pt2 = (annotated_frame.shape[1] - pt2[0], pt2[1])
        cv2.line(annotated_frame, mirrored_pt1, mirrored_pt2, (0, 0, 255), 2)

        object_count = 0
        for detection in result['predictions']:
            obj_id = detection['id'] if 'id' in detection else f"{detection['x']}_{detection['y']}"
            x1, y1, x2, y2 = detection['x'] - detection['width'] // 2, detection['y'] - detection['height'] // 2, detection['x'] + detection['width'] // 2, detection['y'] + detection['height'] // 2
            if obj_id not in self.counted_objects and self.is_crossing_line(y1, y2, mirrored_pt1, mirrored_pt2):
                self.counted_objects.add(obj_id)
                self.frame_counter_for_total += 1
                if self.frame_counter_for_total >= 30:  # Check if 30 frames have passed
                    self.total_object_count += 1
                    self.frame_counter_for_total = 0  # Reset counter after adding to total
                object_count += 1

        # Draw object count on the frame
        cv2.putText(annotated_frame, f'Objects in Frame: {object_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f'Total Objects: {self.total_object_count}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated_frame, object_count

    def is_crossing_line(self, y1, y2, pt1, pt2):
        # Custom logic to determine if an object is crossing the line
        return (pt1[1] <= y1 <= pt2[1]) or (pt1[1] <= y2 <= pt2[1])

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Resize the image to fit within the canvas while maintaining the aspect ratio
        img.thumbnail((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

    def upload_image(self):
        if not self.image_processing:
            self.image_processing = True
            self.reset_total_object_count()
            file_path = filedialog.askopenfilename()
            if file_path:
                frame = cv2.imread(file_path)
                frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
                frame, object_count = self.process_frame(frame)
                self.display_frame(frame)
                print(f"Objects in image: {object_count} | Total Objects: {self.total_object_count}")
            self.image_processing = False

    def upload_video(self):
        if not self.running_video:
            self.reset_total_object_count()
            file_path = filedialog.askopenfilename()
            if file_path:
                self.video_cap = cv2.VideoCapture(file_path)
                self.running_video = True
                self.video_thread = threading.Thread(target=self.update_video_frame)
                self.video_thread.start()

    def live_stream(self):
        if not self.running_live:
            self.reset_total_object_count()
            self.live_cap = cv2.VideoCapture(0)
            self.running_live = True
            self.live_thread = threading.Thread(target=self.update_live_frame)
            self.live_thread.start()

    def update_video_frame(self):
        while self.running_video and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                break

            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Process every 3rd frame for faster processing
                frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
                self.executor.submit(self.process_and_display, frame)

            # Sleep briefly to avoid overloading the CPU
            cv2.waitKey(1)

        if self.video_cap:
            self.video_cap.release()
        self.running_video = False

    def update_live_frame(self):
        while self.running_live and self.live_cap.isOpened():
            ret, frame = self.live_cap.read()
            if not ret:
                break

            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Process every 3rd frame for faster processing
                frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
                self.executor.submit(self.process_and_display, frame)

            # Sleep briefly to avoid overloading the CPU
            cv2.waitKey(1)

        if self.live_cap:
            self.live_cap.release()
        self.running_live = False

    def process_and_display(self, frame):
        frame, object_count = self.process_frame(frame)
        self.display_frame(frame)

    def stop_video(self):
        self.running_video = False
        self.running_live = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join()
        if self.live_thread and self.live_thread.is_alive():
            self.live_thread.join()

    def reset_total_object_count(self):
        self.total_object_count = 0
        self.frame_count = 0
        self.frame_counter_for_total = 0  # Reset the counter for total objects
        self.counted_objects.clear()  # Reset counted objects

def main(model=None):
    if model is None:
        rf = Roboflow(api_key="Lal8NmqQraw3Vcg2rxuG")
        project = rf.workspace().project("gidfn")
        model = project.version("2").model

    root = tk.Tk()
    app = App(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
