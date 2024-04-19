import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
import threading
import cv2
import numpy as np
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

def write_to_csv(filename, time_data, path, left_region_name, right_region_name):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(
                ['Time', 'Path', left_region_name, right_region_name])
        writer.writerow([time_data, path, region[left_region_name].num_people,
                        region[right_region_name].num_people])

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", nargs='?', help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    # Return parser
    return parser



def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

def run_tracker_in_thread(exp, args, filename, left_region_name, right_region_name, file_index):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    logger.info("\tFusing model...")
    model = fuse_model(model)

    model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BoTSORT(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []
    
    person_id = set()
    number0 = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output_{file_index}.avi', fourcc, 20.0, (800, 400)) #形狀要跟影片加圖表的大小一樣
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
    folder_name = "person_"+str(file_index)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if frame_for_window[file_index].start_detection == False:
            # frame_for_window[file_index].ret = True
            cv2.waitKey(25) #讀影片時需要延遲，以降低影片速度
            frame_for_window[file_index].frame = frame
            continue
        if ret_val:
            # Detect objects
            outputs, img_info = predictor.inference(frame, timer)
            scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale

                # Run tracker
                online_targets = tracker.update(detections, img_info["raw_img"])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                for i, tlwh in enumerate(online_tlwhs):
                    x1, y1, w, h = tlwh
                    img = extract_images_from_box(frame, [x1,y1,x1+w,y1+h])
                    b = int(online_ids[i])
                    x_center = x1 + w/2
                    y_center = y1 + w/2
                    if b not in people:
                        people[b] = Person(b, [x_center, y_center])
                    people[b].add_image(img)
                    people[b].update_position([x_center, y_center])
                    w = len(frame[0, :])
                    h = len(frame[:, 0])
                    polygon = Polygon([(x * w, y * h) for x, y in frame_for_window[file_index].polygons])
                    point = Point(people[b].current_position[0],people[b].current_position[1])
                    print(point.within(polygon))
                    if point.within(polygon) == False:
                        print("bbb",b,people[b].current_position[0],people[b].current_position[1],region[left_region_name].people_in_region)
                        print("kkk",[(x * w, y * h) for x, y in frame_for_window[file_index].polygons])
                        region[left_region_name].add_person([b])
                        region[right_region_name].delete_person([b])
                        same_person = people[b].same_person
                        region[left_region_name].delete_person(same_person)
                    if point.within(polygon) == True:
                        region[right_region_name].add_person([b])
                        region[left_region_name].delete_person([b])
                        same_person = people[b].same_person
                        region[right_region_name].delete_person(same_person)
                    filename = os.path.join(
                        folder_name, str(b) + ".pickle")  # 构建完整的文件路径
                    with open(filename, "wb") as f:
                        pickle.dump(people[b], f)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            number = len(online_ids)
            number_in_left = region[left_region_name].num_people
            res_plotted = online_im
            # Store the time and object count
            times.append(time.perf_counter()-t0)
            object_counts.append(number_in_left)
            timestamp = datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S%f")  # Generate timestamp
            path = os.path.join(frame_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(path, res_plotted)
            frame_index += 1
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
            image = cv2.resize(image, (hight, weight))
            all_image = np.hstack((res_plotted, image))
            frame_for_window[file_index].ret = True
            frame_for_window[file_index].frame_processed = all_image
            # cv2.imshow(f"Tracking_Stream_{file_index}", all_image)
            out.write(all_image)
            if frame_for_window[file_index].quit == True:
                break
            if args.save_result:
                cv2.imshow('oxxostudio',online_im)
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    cap.release()                           # 所有作業都完成後，釋放資源
    '''
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    '''

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

if __name__ == "__main__":
    # Set specific argument values
    args_dict = {
        'demo': 'webcam',
        'path': "c.mp4",
        'exp_file': 'yolox/exps/example/mot/yolox_x_mix_det.py',
        'ckpt': 'pretrained/bytetrack_x_mot17.pth.tar',
        'with_reid': True,
        'fuse_score': True,
        'fp16': True,
        'fuse': True,
        'save_result': True
}

    # Parse arguments using defaults and update with specific values
    args = make_parser().parse_args([])
    args.__dict__.update(args_dict)
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score
    
    video_file1 = "c.mp4.MOV"  # Path to video file, 0 for webcam
    video_file2 = "d.mp4"
    video_file3 = "door3.MOV"
    #video_file2 = "c.mp4"
    people = dict()
    region = {"A": Region(), "Outside": Region(), "B": Region(),
            "C": Region(), "D": Region(), "E": Region()}
    frame_for_window = {0: Frame(), 1: Frame(), 2: Frame()}
    
    tracker_thread1 = threading.Thread(
        target=run_tracker_in_thread, args=(exp, args, video_file1, "A", "Outside", 0), daemon=True)
    
    args_dict2 = {
        'demo': 'webcam',
        'path': "d.mp4",
        'exp_file': 'yolox/exps/example/mot/yolox_x_mix_det.py',
        'ckpt': 'pretrained/bytetrack_x_mot17.pth.tar',
        'with_reid': True,
        'fuse_score': True,
        'fp16': True,
        'fuse': True,
        'save_result': True
}

    # Parse arguments using defaults and update with specific values
    args2 = make_parser().parse_args([])
    args2.__dict__.update(args_dict2)
    exp2 = get_exp(args2.exp_file, args2.name)

    args2.ablation = False
    args2.mot20 = not args2.fuse_score
    
    tracker_thread2 = threading.Thread(
        target=run_tracker_in_thread, args=(exp2, args2, video_file2, "A", "D", 1), daemon=True)
    '''
    args_dict3 = {
        'demo': 'webcam',
        'path': R"H:\yolo\door3.MOV",
        'exp_file': 'yolox/exps/example/mot/yolox_x_mix_det.py',
        'ckpt': 'pretrained/bytetrack_x_mot17.pth.tar',
        'with_reid': True,
        'fuse_score': True,
        'fp16': True,
        'fuse': True,
        'save_result': True
}

    # Parse arguments using defaults and update with specific values
    args3 = make_parser().parse_args([])
    args3.__dict__.update(args_dict3)
    exp3 = get_exp(args3.exp_file, args3.name)

    args3.ablation = False
    args3.mot20 = not args3.fuse_score
    
    tracker_thread3 = threading.Thread(
        target=run_tracker_in_thread, args=(exp3, args3, video_file3, "B", "Outside", 2), daemon=True)
    '''
    tracker_thread4 = threading.Thread(
        target=show_window, daemon=True)
    tracker_thread1.start()
    tracker_thread2.start()
    #tracker_thread3.start()
    tracker_thread4.start()
    # Wait for the tracker thread to finish
    tracker_thread1.join()
    tracker_thread2.join()
    #tracker_thread3.join()
    tracker_thread4.join()
    # Clean up and close windows
    cv2.destroyAllWindows()
