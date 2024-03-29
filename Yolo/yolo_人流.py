import threading
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import csv
import os
import datetime
import pickle
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

class Person():
    def  __init__(self,idx, position):
        self.idx = [idx]
        self.same_person = []
        self.image = []
        self.path = []
        self.previous_position = position
        self.current_position = position
    def add_image(self, image):
        self.image.append(image)
    def add_path(self, path):
        self.path.append(path)
    def update_position(self, position):
        self.previous_position = self.current_position
        self.current_position = position
    def connect_same_person(self, person_idx):
        self.same_person.append(person_idx)

class Region():
    def __init__(self):
        self.people_in_region = set()
        self.num_people = 0
    def add_person(self,people_idx):
        for person_idx in people_idx:
            if person_idx not in self.people_in_region:
                self.people_in_region.add(person_idx)
                self.num_people += 1
    def delete_person(self,people_idx):
        for person_idx in people_idx:
            if person_idx in self.people_in_region:
                self.people_in_region.remove(person_idx)
                self.num_people -= 1

class Frame():
    def __init__(self):
        self.ret = False
        self.frame = 0
    def read(self):
        return self.ret, self.frame


def extract_images_from_box(frame, box):
    """
    Extracts images from the bounding boxes in the frame.

    Args:
        frame (numpy.ndarray): The input frame.
        boxes (list): List of bounding boxes.

    Returns:
        list: List of cropped images.
    """
    x1, y1, x2, y2 = box[:4]  # Extract bounding box coordinates
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer

    # Crop the region of interest (ROI) from the frame
    roi = frame[y1:y2, x1:x2]

    return roi
# Global variables for storing time and object counts
def write_to_csv(filename, time_data, path, left_region_name, right_region_name):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Time', 'Path', left_region_name, right_region_name])
        writer.writerow([time_data, path, region[left_region_name].num_people, region[right_region_name].num_people])

# Function to run tracker in thread
def run_tracker_in_thread(filename, model_name, left_region_name, right_region_name, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
    """
    #global people, region
    model = YOLO(model_name)
    video = cv2.VideoCapture(filename)  # Read the video file
    person_id = set()
    number0 = 0
    model2 = YOLO("yolov8n-pose.pt")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1200, 400))
    
    # Set up Matplotlib figure and canvas for the object count plot
    fig_object_count = Figure()
    canvas_object_count = FigureCanvas(fig_object_count)
    ax_object_count = fig_object_count.add_subplot(111)
    times = []
    object_counts = []
    real_times = []
    t0 = time.perf_counter()
    paths = []
    frame_count = 0
    frame_index = 0
    detection_interval=5
    # Create directory for saving frames
    frame_dir = 'frames'
    os.makedirs(frame_dir, exist_ok=True)
    folder_name = "person"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    while True:
        ret, frame = video.read()  # Read the video frames
        if not ret:
            print("none")
            continue
        frame_count += 1
        if frame_count % detection_interval != 0:
            continue  # Skip this frame if it's not for detection
        
        results = model.track(frame, classes=[0], persist=True)
        results_pose = model2.track(frame, classes=[0], persist=True)
        #results = model.track(frame, persist=True)
        #results_pose = model2.track(frame, persist=True)
        boxes = results[0].numpy().boxes
        for box in boxes:
            if box.id is not None:
                img = extract_images_from_box(frame, box.xyxy[0])
                b = int(box.id.tolist()[0])
                x_center = (box.xyxy[0][0]+box.xyxy[0][2])/2
                y_center = (box.xyxy[0][1]+box.xyxy[0][3])/2
                if b not in people:
                    people[b] = Person(b,[x_center,y_center])
                people[b].add_image(img)
                people[b].update_position([x_center,y_center])
                gate_x = len(frame[0,:])*0.5
                if people[b].current_position[0]<gate_x:
                    region[left_region_name].add_person([b])
                    region[right_region_name].delete_person([b])
                    same_person = people[b].same_person
                    region[left_region_name].delete_person(same_person)
                if people[b].current_position[0]>gate_x:
                    region[right_region_name].add_person([b])
                    region[left_region_name].delete_person([b])
                    same_person = people[b].same_person
                    region[right_region_name].delete_person(same_person)
                filename = os.path.join(folder_name, str(b) + ".pickle")  # 构建完整的文件路径
                with open(filename, "wb") as f:
                    pickle.dump(people[b], f)

        #number_total = len(people)
        #number = number_total-number0
        #number0 = number_total
        print(len(region))
        number = len(boxes)
        number_in_left = region[left_region_name].num_people
        res_plotted = results[0].plot()
        res_plotted_pose = results_pose[0].plot()

        # Store the time and object count
        times.append(time.perf_counter()-t0)
        object_counts.append(number_in_left)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")  # Generate timestamp
        path = os.path.join(frame_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(path, res_plotted)
        frame_index += 1
        write_to_csv('output_paths.csv', timestamp, path,left_region_name,right_region_name)

        # Draw the number of tracked objects on the frame
        cv2.putText(res_plotted, f'total_number: {region[left_region_name].num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        # Plot object count over time
        ax_object_count.clear()
        print(len(times),len(object_counts))
        ax_object_count.plot(np.array(times)/60, object_counts)
        ax_object_count.set_xlabel('Time')
        ax_object_count.set_ylabel('Object Count')
        ax_object_count.set_title('Object Count Over Time')

        # Draw the figure on a separate OpenCV frame
        canvas_object_count.draw()
        buf = canvas_object_count.buffer_rgba()
        X, Y = buf.shape[1], buf.shape[0]
        image = np.asarray(buf)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        hight = 400
        weight = 400
        res_plotted = cv2.resize(res_plotted,(hight,weight))
        res_plotted_pose = cv2.resize(res_plotted_pose,(hight,weight))
        image = cv2.resize(image,(hight,weight))
        all_image = np.hstack((res_plotted,res_plotted_pose,image))
        if file_index == 1:
            frame_for_window[0].ret = True
            frame_for_window[0].frame = all_image
        cv2.imshow(f"Tracking_Stream_{file_index}", all_image)
        out.write(all_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()



class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Camera Viewer')

        self.label = QLabel(self)
        self.label.setScaledContents(True)  # Ensure image fills the label
        self.label.setAlignment(Qt.AlignCenter)  # Center image in label
        self.setMinimumSize(320, 240)  # Set minimum size for the widget

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Open the camera
        #self.cap = cv2.VideoCapture("c.mp4")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(30)  # Update frame every 30 milliseconds

    def display_frame(self):
        #ret, frame = self.cap.read()
        ret, frame = frame_for_window[0].read()
        if ret:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to QImage
            img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            # Display image on label
            self.label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_image_size()

    def adjust_image_size(self):
        # Get the current size of the label
        label_size = self.label.size()
        # Check if a pixmap is set for the label
        if self.label.pixmap():
            # Resize the image to fit the label, maintaining aspect ratio
            self.label.setPixmap(self.label.pixmap().scaled(label_size, Qt.KeepAspectRatio))

def show_window():
    time.sleep(10)
    app = QApplication(sys.argv)
    camera = CameraWidget()
    camera.show()
    sys.exit(app.exec_())

people = dict()
region = {"A":Region(), "Outside":Region(), "B":Region(), "C":Region(), "D":Region(), "E":Region()}
frame_for_window = {0:Frame()}
# Define the video file for the tracker
video_file1 = R"H:\yolo\door1.MOV"  # Path to video file, 0 for webcam
video_file2 = R"H:\yolo\door2.mp4"
video_file3 = R"H:\yolo\door3.MOV"
#run_tracker_in_thread(video_file1,model1,"A","Outside")
#run_tracker_in_thread(video_file2,model1,"A","Outside")
#run_tracker_in_thread(video_file3,model1,"A","Outside")
# Create the tracker thread
tracker_thread1 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file1, 'yolov8n.pt',"A","Outside",1), daemon=True)
tracker_thread2 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file2, 'yolov8n.pt',"A","D",2), daemon=True)
tracker_thread3 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file3, 'yolov8n.pt',"B","Outside",3), daemon=True)
tracker_thread4 = threading.Thread(
    target=show_window, daemon=True)
# Start the tracker thread
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()
tracker_thread4.start()
# Wait for the tracker thread to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()
tracker_thread4.join()

# Clean up and close windows
cv2.destroyAllWindows()
