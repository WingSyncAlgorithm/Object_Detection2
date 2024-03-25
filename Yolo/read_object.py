import pickle
import cv2
import os

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

folder_name = "person"
filename = "46.pickle"  # 假设要加载的文件名为1.pickle

filepath = os.path.join(folder_name, filename)
with open(filepath, "rb") as f:
    retrieved_obj = pickle.load(f)
    print(f"Loaded object from '{filename}': {retrieved_obj}")

images = retrieved_obj.image
for i, image in enumerate(images):
    cv2.imshow(f"Image {i+1}", image)

# 等待用户按下任意键后关闭图像窗口
cv2.waitKey(0)
cv2.destroyAllWindows()