from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QComboBox
import sys
import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Camera Viewer')

        self.label = QLabel(self)
        self.label.setScaledContents(True)  # Ensure image fills the label
        self.label.setAlignment(Qt.AlignCenter)  # Center image in label
        self.setMinimumSize(320, 240)  # Set minimum size for the widget

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.create_menu()

        # Open the camera
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)

    def create_menu(self):
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("Video 1")  # Add pre-set video files to the combo box
        self.comboBox.addItem("Video 2")
        self.comboBox.activated[str].connect(self.select_video)  # Connect the activated signal to select_video method
        layout = QVBoxLayout()
        layout.addWidget(self.comboBox)
        self.setLayout(layout)

    def select_video(self, video_name):
        # Define dictionary mapping video names to file paths
        video_files = {
            "Video 1": "c.mp4",
            "Video 2": "d.mp4"
        }
        # Get the file path corresponding to the selected video name
        file_path = video_files.get(video_name)
        if file_path:
            self.open_video(file_path)

    def open_video(self, file_name):
        # Release previous video capture if exists
        if self.cap is not None:
            self.cap.release()

        # Open new video capture
        self.cap = cv2.VideoCapture(file_name)
        if self.cap.isOpened():
            self.timer.start(30)  # Update frame every 30 milliseconds

    def display_frame(self):
        ret, frame = self.cap.read()
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
    app = QApplication(sys.argv)
    camera = CameraWidget()
    camera.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    show_window()
