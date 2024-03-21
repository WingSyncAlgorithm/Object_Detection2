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

# Global variables for storing time and object counts
object_counts = []
# Function to write data to CSV
def write_to_csv(filename, times, pixels):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Pixels'])
        for i in range(len(times)):
            writer.writerow([times[i], pixels[i]])

# Function to run tracker in thread
def run_tracker_in_thread(filename, model, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
    """
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
    t0 = time.perf_counter()
    pixels_data = []
    while True:
        ret, frame = video.read()  # Read the video frames
        if not ret:
            break

        
        results = model.track(frame, classes=[0], persist=True)
        results_pose = model2.track(frame, classes=[0], persist=True)
        #results = model.track(frame, persist=True)
        #results_pose = model2.track(frame, persist=True)
        boxes = results[0].numpy().boxes
        for box in boxes:
            if box.id is not None:
                b = int(box.id.tolist()[0])
                person_id.add(b)
        number_total = len(person_id)
        #number = number_total-number0
        #number0 = number_total
        number = len(boxes)
        res_plotted = results[0].plot()
        res_plotted_pose = results_pose[0].plot()

        # Store the time and object count
        times.append(time.perf_counter()-t0)
        object_counts.append(number)

        # Store pixel values of the frame
        pixels_data.append(frame.flatten())

        # Write time and pixel data to CSV
        write_to_csv('output_data.csv', times, pixels_data)

        # Draw the number of tracked objects on the frame
        cv2.putText(res_plotted, f'total_number: {number_total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        # Plot object count over time
        ax_object_count.clear()
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
        cv2.imshow("Object_Count_Plot", all_image)
        out.write(all_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()

# Load the model
model1 = YOLO('yolov8n.pt')

# Define the video file for the tracker
video_file1 = "c.mp4"  # Path to video file, 0 for webcam
run_tracker_in_thread(video_file1,model1,1)
# Create the tracker thread
#tracker_thread1 = threading.Thread(
#    target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)

# Start the tracker thread
#tracker_thread1.start()

# Wait for the tracker thread to finish
#tracker_thread1.join()

# Clean up and close windows
cv2.destroyAllWindows()
