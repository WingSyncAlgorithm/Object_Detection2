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
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton
from PyQt5.QtGui import QPainter, QImage, QColor, QPolygon
from shapely.geometry import Point, Polygon


class Person():
    def __init__(self, idx, position):
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

    def add_person(self, people_idx):
        for person_idx in people_idx:
            if person_idx not in self.people_in_region:
                self.people_in_region.add(person_idx)
                self.num_people += 1

    def delete_person(self, people_idx):
        for person_idx in people_idx:
            if person_idx in self.people_in_region:
                self.people_in_region.remove(person_idx)
                self.num_people -= 1


class Frame():
    def __init__(self):
        self.ret = False
        self.frame = None
        self.frame_processed = None
        self.start_detection = False
        self.quit = False
        self.polygons = []

    def read_processed_frame(self):
        return self.ret, self.frame_processed

    def read_frame(self):
        return self.frame

    def start_detect(self):
        self.start_detection = True


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
            writer.writerow(
                ['Time', 'Path', left_region_name, right_region_name])
        writer.writerow([time_data, path, region[left_region_name].num_people,
                        region[right_region_name].num_people])

# Function to run tracker in thread


def run_tracker_in_thread(filename, model_name, left_region_name, right_region_name, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
    """
    # global people, region
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
    detection_interval = 5
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
        # results = model.track(frame, persist=True)
        # results_pose = model2.track(frame, persist=True)
        if frame_for_window[file_index].start_detection == False:
            # frame_for_window[file_index].ret = True
            frame_for_window[file_index].frame = frame
            continue
        boxes = results[0].numpy().boxes
        for box in boxes:
            if box.id is not None:
                img = extract_images_from_box(frame, box.xyxy[0])
                b = int(box.id.tolist()[0])
                x_center = (box.xyxy[0][0]+box.xyxy[0][2])/2
                y_center = (box.xyxy[0][1]+box.xyxy[0][3])/2
                if b not in people:
                    people[b] = Person(b, [x_center, y_center])
                people[b].add_image(img)
                people[b].update_position([x_center, y_center])
                w = len(frame[0, :])
                h = len(frame[:, 0])
                polygon = Polygon([(x // w, y // h) for x, y in frame_for_window[file_index].polygons])
                point = Point(people[b].current_position[0],people[b].current_position[1])
                if point.within(polygon) == True:
                    region[left_region_name].add_person([b])
                    region[right_region_name].delete_person([b])
                    same_person = people[b].same_person
                    region[left_region_name].delete_person(same_person)
                if point.within(polygon) == False:
                    region[right_region_name].add_person([b])
                    region[left_region_name].delete_person([b])
                    same_person = people[b].same_person
                    region[right_region_name].delete_person(same_person)
                filename = os.path.join(
                    folder_name, str(b) + ".pickle")  # 构建完整的文件路径
                with open(filename, "wb") as f:
                    pickle.dump(people[b], f)

        # number_total = len(people)
        # number = number_total-number0
        # number0 = number_total
        print(len(region))
        number = len(boxes)
        number_in_left = region[left_region_name].num_people
        res_plotted = results[0].plot()
        res_plotted_pose = results_pose[0].plot()
        print(frame_for_window[file_index].polygons)
        # Store the time and object count
        times.append(time.perf_counter()-t0)
        object_counts.append(number_in_left)
        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S%f")  # Generate timestamp
        path = os.path.join(frame_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(path, res_plotted)
        frame_index += 1
        write_to_csv('output_paths.csv', timestamp, path,
                     left_region_name, right_region_name)

        # Draw the number of tracked objects on the frame
        cv2.putText(res_plotted, f'total_number: {region[left_region_name].num_people}', (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        # Plot object count over time
        ax_object_count.clear()
        print(len(times), len(object_counts))
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
        res_plotted = cv2.resize(res_plotted, (hight, weight))
        res_plotted_pose = cv2.resize(res_plotted_pose, (hight, weight))
        image = cv2.resize(image, (hight, weight))
        all_image = np.hstack((res_plotted, res_plotted_pose, image))
        frame_for_window[file_index].ret = True
        frame_for_window[file_index].frame_processed = all_image
        # cv2.imshow(f"Tracking_Stream_{file_index}", all_image)
        out.write(all_image)
        if frame_for_window[file_index].quit == True:
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
        
        self.frame = None
        self.points = []
        self.polygons = {}  # Dictionary to store drawn polygons for each video
        self.width = 600
        self.height = 800
        self.timer0 = QTimer(self)
        self.timer0.timeout.connect(self.updateFrame)
        self.timer0.start(30)  # Update frame rate in milliseconds

        self.video_selector = QComboBox()
        for index in frame_for_window:
            self.video_selector.addItem(f"Frame {index}")
        self.video_selector.currentIndexChanged.connect(self.changeVideo)

        self.redraw_button = QPushButton('Redraw', self)
        self.redraw_button.clicked.connect(self.clearPoints)

        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.confirmPolygon)
        self.selected_frame_index0 = 0
        layout.addWidget(self.video_selector)
        layout.addWidget(self.redraw_button)
        layout.addWidget(self.confirm_button)
        #layout.addStretch(1)

        #self.setLayout(layout)

        # Add a button to show menu and image
        self.show_menu_button = QPushButton('Show Menu and Image')
        self.show_menu_button.clicked.connect(self.show_menu_and_image)
        layout.addWidget(self.show_menu_button)

        # Create a combo box for selecting frame_for_window
        self.comboBox = QComboBox(self)
        layout.addWidget(self.comboBox)
        self.comboBox.hide()  # Initially hide the combo box
        for index in frame_for_window:
            self.comboBox.addItem(f"Frame {index}")
        self.setLayout(layout)
        self.comboBox.currentIndexChanged.connect(self.select_frame_for_window)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)
        # Open the camera
        # self.cap = cv2.VideoCapture("c.mp4")
        
        
        

    def updateFrame(self):
        ret, frame = True, frame_for_window[self.selected_frame_index0].read_frame()
        if frame is not None:
            self.frame = cv2.resize(frame,(self.width,self.height))
            self.update()

    def show_menu_and_image(self):
        for i in range(len(frame_for_window)):
            frame_for_window[i].start_detect()
        # Set the initial selected frame index
        self.selected_frame_index = 0
        self.timer0.stop()
        self.confirm_button.hide()
        self.redraw_button.hide()
        self.video_selector.hide()
        self.comboBox.show()  # Show the combo box
        self.show_menu_button.hide()  # Hide the button
        self.timer.start(30)  # Start the timer to display frames

    def select_frame_for_window(self, index):
        self.selected_frame_index = index

    def display_frame(self):
        # ret, frame = self.cap.read()
        ret, frame = frame_for_window[self.selected_frame_index].read_processed_frame(
        )
        if ret:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to QImage
            img = QImage(
                rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            # Display image on label
            self.label.setPixmap(pixmap)
    

    def changeVideo(self, index):
        self.selected_frame_index0 = index
        self.points = []
        self.update()


    def clearPoints(self):
        self.points = []
        self.update()

    def confirmPolygon(self):
        if self.selected_frame_index0 not in self.polygons:
            self.polygons[self.selected_frame_index0] = []
        self.polygons[self.selected_frame_index0].append(self.points.copy())
        frame_for_window[self.selected_frame_index0].polygons = [(point.x()/self.width, point.y()/self.height) for point in self.points]
        print("Polygon Confirmed for", self.selected_frame_index0, ":", self.points)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def paintEvent(self, event):
        if self.frame is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Convert OpenCV BGR format to QImage
            image = QImage(
                self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_BGR888)
            painter.drawImage(0, 0, image)

            # Draw existing polygons
            if self.selected_frame_index0 in self.polygons:
                for polygon_points in self.polygons[self.selected_frame_index0]:
                    if len(polygon_points) >= 3:
                        polygon = QPolygon(polygon_points)
                        painter.drawPolygon(polygon)

            # Draw current polygon
            pen = painter.pen()
            pen.setWidth(2)
            pen.setColor(QColor(0, 0, 255))
            painter.setPen(pen)

            brush = painter.brush()
            brush.setStyle(Qt.NoBrush)
            painter.setBrush(brush)

            if len(self.points) >= 3:
                polygon = QPolygon(self.points)
                painter.drawPolygon(polygon)

            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    painter.drawLine(self.points[i], self.points[i + 1])
    def closeEvent(self, event):
        # Terminate all threads before closing the window
        for i in range(len(frame_for_window)):
            frame_for_window[i].quit = True
        print("ttttt",self.polygons)
        event.accept()

        # Clean up and close windows
        # cv2.destroyAllWindows()
        # QApplication.quit()


def show_window():
    # Add frame indices to the combo box
    app = QApplication(sys.argv)
    camera = CameraWidget()
    # Run the application
    camera.show()
    sys.exit(app.exec_())


people = dict()
region = {"A": Region(), "Outside": Region(), "B": Region(),
          "C": Region(), "D": Region(), "E": Region()}
frame_for_window = {0: Frame(), 1: Frame(), 2: Frame()}
# Define the video file for the tracker
# video_file1 = R"H:\yolo\door1.MOV"  # Path to video file, 0 for webcam
# video_file2 = R"H:\yolo\door2.mp4"
# video_file3 = R"H:\yolo\door3.MOV"
video_file1 = "c.mp4"
video_file2 = "e.mp4"
# run_tracker_in_thread(video_file1,model1,"A","Outside")
# run_tracker_in_thread(video_file2,model1,"A","Outside")
# run_tracker_in_thread(video_file3,model1,"A","Outside")
# Create the tracker thread
tracker_thread1 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file1, 'yolov8n.pt', "A", "Outside", 0), daemon=True)
tracker_thread2 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file2, 'yolov8n.pt', "A", "D", 1), daemon=True)
# tracker_thread3 = threading.Thread(
#    target=run_tracker_in_thread, args=(video_file3, 'yolov8n.pt',"B","Outside",3), daemon=True)
tracker_thread4 = threading.Thread(
    target=show_window, daemon=True)
# Start the tracker thread
tracker_thread1.start()
tracker_thread2.start()
# tracker_thread3.start()
tracker_thread4.start()
# Wait for the tracker thread to finish
tracker_thread1.join()
tracker_thread2.join()
# tracker_thread3.join()
tracker_thread4.join()
# Clean up and close windows
cv2.destroyAllWindows()
