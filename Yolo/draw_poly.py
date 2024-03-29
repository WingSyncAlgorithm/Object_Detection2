import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QComboBox
from PyQt5.QtGui import QPainter, QImage, QColor, QPolygon
from PyQt5.QtCore import Qt, QTimer, QPoint

class VideoBackgroundWithPolygonDrawer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Video Background with Polygon Drawer')
        self.setGeometry(100, 100, 600, 400)

        self.video_paths = ['c.mp4', 'e.mp4']  # Add your video paths here
        self.current_video_path = self.video_paths[0]
        self.video_capture = cv2.VideoCapture(self.current_video_path)
        self.frame = None
        self.points = []
        self.polygons = {}  # Dictionary to store drawn polygons for each video
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # Update frame rate in milliseconds

        self.video_selector = QComboBox()
        self.video_selector.addItems(self.video_paths)
        self.video_selector.currentIndexChanged.connect(self.changeVideo)

        self.redraw_button = QPushButton('Redraw', self)
        self.redraw_button.clicked.connect(self.clearPoints)

        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.confirmPolygon)

        layout = QVBoxLayout()
        layout.addWidget(self.video_selector)
        layout.addWidget(self.redraw_button)
        layout.addWidget(self.confirm_button)
        layout.addStretch(1)

        self.setLayout(layout)

    def changeVideo(self, index):
        self.current_video_path = self.video_paths[index]
        self.video_capture = cv2.VideoCapture(self.current_video_path)
        self.points = []
        self.update()

    def updateFrame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.frame = frame
            self.update()

    def clearPoints(self):
        self.points = []
        self.update()

    def confirmPolygon(self):
        if self.current_video_path not in self.polygons:
            self.polygons[self.current_video_path] = []
        self.polygons[self.current_video_path].append(self.points.copy())
        print("Polygon Confirmed for", self.current_video_path, ":", self.points)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def paintEvent(self, event):
        if self.frame is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Convert OpenCV BGR format to QImage
            image = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_BGR888)
            painter.drawImage(0, 0, image)

            # Draw existing polygons
            if self.current_video_path in self.polygons:
                for polygon_points in self.polygons[self.current_video_path]:
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoBackgroundWithPolygonDrawer()
    window.show()
    sys.exit(app.exec_())
