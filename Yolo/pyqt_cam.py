import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
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

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Open the camera
        self.cap = cv2.VideoCapture("output.avi")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)
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
    app = QApplication(sys.argv)
    camera = CameraWidget()
    camera.show()
    sys.exit(app.exec_())
