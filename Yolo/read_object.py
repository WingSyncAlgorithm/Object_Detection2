import pickle
import cv2
import os

class Person():
    def  __init__(self, idx, position):
        self.idx = idx
        self.same_person = []
        self.image = []   # Storing images
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

folder_name = "person_1"
output_folder = "person_picture"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍歷所有pickle文件
for filename in os.listdir(folder_name):
    if filename.endswith(".pickle"):
        filepath = os.path.join(folder_name, filename)
        try:
            with open(filepath, "rb") as f:
                retrieved_obj = pickle.load(f)
                print(f"Loaded object from '{filename}': {retrieved_obj}")

                # Accessing images from the retrieved object
                images = retrieved_obj.image
                idx = retrieved_obj.idx[0]  # Assuming idx is a single value
                person_folder = os.path.join(output_folder, f"person_{idx}")

                # Create person's folder if it doesn't exist
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                for i, image in enumerate(images):
                    # 構造要保存的圖片文件名
                    img_filename = f"{filename}_{i+1}.jpg"
                    img_filepath = os.path.join(person_folder, img_filename)

                    # 將圖片保存為jpg文件
                    cv2.imwrite(img_filepath, image)
                    print(f"Image {i+1} saved as {img_filepath}")
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

print("All images saved successfully.")
