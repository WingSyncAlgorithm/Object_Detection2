import threading
import cv2
from ultralytics import YOLO
import torch


def run_tracker_in_thread(filename, model_name, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    model = YOLO(model_name)
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, classes=[0], persist=True)
        boxes = results[0].numpy().boxes
        #person_id = set()
        person_id = []
        for box in boxes:
            if box.id != None:
                print(box.id[0])
                person_id.append(box.id.tolist())
                print(person_id)
                #person_id = person_id | set(box.id)

            #print("id: ", box.id, len(person_id))
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


# Load the models
#model1 = YOLO('yolov8n.pt')
# model2 = YOLO('yolov8n-seg.pt')

# Define the video files for the trackers
video_file1 = R"H:\yolo\door1.MOV"  # Path to video file, 0 for webcam
video_file2 = R"H:\yolo\door2.mp4"

# Create the tracker threads
tracker_thread1 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file1, 'yolov8n.pt', 1), daemon=True)
tracker_thread2 = threading.Thread(
    target=run_tracker_in_thread, args=(video_file2,'yolov8n.pt', 2), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
