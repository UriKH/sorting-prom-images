import os
import shutil
import threading
import torch
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image, ImageTk
import numpy as np
from tqdm import tqdm
from colored import fg, attr
from math import floor
import tkinter as tk
import config
from initializer import Init


class Sorter(Init):
    def __init__(self, images_root: str, anchor_dir: str):
        super().__init__()
        self.root = images_root
        self.anchor = anchor_dir

        self.anchors = {}
        self.load_anchors()

    @classmethod
    def extract_faces(cls, path: str) -> tuple[list, list]:
        """
        Extract faces from image in format BGR using MTCNN
        :param path: path to the image in disk
        :returns: list of cropped faces and their coordinates
        """
        image = cv.imread(path)
        h, w, _ = image.shape

        if image is None:
            Sorter.log_simple_info(f'image was None - check it! {path}')
            return [], []
        else:
            image = cv.resize(image, (w // 2, h // 2))  # downscale image for faster execution
        h, w, _ = image.shape

        faces_coord, conf = Sorter.mtcnn.detect(image)  # detect faces using MTCNN

        if faces_coord is None or len(faces_coord) == 0:
            return [], []

        faces_coord = faces_coord.astype(int)
        faces_coord = [face for face, c in zip(faces_coord, conf) if c >= 0.9]
        faces = []
        for face in faces_coord:
            x1, y1, x2, y2 = face
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            margin = int((x2 - x1) * 0.1)  # calculate margin
            x1 = max(x1 - margin, 0)
            x2 = min(x2 + margin, w - 1)
            y1 = max(y1 - margin * 2, 0)  # capture forehead
            y2 = min(y2 + int(margin * 1.2), h - 1)  # capture chin

            faces.append(image[y1:y2, x1:x2])
        return faces, faces_coord

    @classmethod
    def create_image(cls, image, size: int = 200):
        """
        Align image on canvas
        :param image: image to align
        :param size: size of the canvas
        :returns: the aligned image
        """
        height, width, _ = image.shape
        prop = width / height

        if width > height:
            width = size
            height = int(size / prop)
        else:
            width = int(prop * size)
            height = size
        image = cv.resize(image, (width, height))

        # create a white canvas
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        start_x = (size - width) // 2
        start_y = (size - height) // 2

        # position the cropped face in the middle of the canvas
        canvas[start_y: start_y + height, start_x: start_x + width] = image
        return canvas

    @classmethod
    def image_to_embedding(cls, image):
        """
        Preprocess image and generate embedding using InceptonResnetV1
        :param image: image to preprocess
        :returns: embedding tensor
        """
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Sorter.create_image(image, 160)
        image = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the image
        ])

        preprocessed_image = transform(image)

        Sorter.resnet.classify = True
        embedding = Sorter.resnet(preprocessed_image.unsqueeze(0).to(config.DEVICE))
        return embedding

    @classmethod
    def compare_pair(cls, em1, em2):
        """
        Compute the cosine similarity between the embedding vectors
        :param em1: first embedding
        :param em2: second embedding
        """
        dist = torch.nn.functional.cosine_similarity(em1, em2)
        return dist.item()

    def load_anchors(self):
        """
        Load faces from the anchor images
        """
        image_names = os.listdir(self.anchor)

        def show_face(_face):
            choose_name = tk.Tk()
            choose_name.title('choose name')
            image = cv.cvtColor(_face, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.thumbnail((300, 300))
            image = ImageTk.PhotoImage(image)
            label = tk.Label(choose_name)
            label.config(image=image)
            label.image = image
            label.pack(pady=10)
            name_entry = tk.Entry(choose_name)
            name_entry.pack(pady=10)
            submit_button = tk.Button(choose_name, text='submit', command=lambda: get_name(name_entry, choose_name))
            submit_button.pack(pady=10)
            choose_name.mainloop()

        def get_name(name_entry, choose_name):
            name = name_entry.get()
            self.anchors[name] = [Sorter.image_to_embedding(face), {"60": [], "70": [], "80": [], "90": []}, face]
            choose_name.destroy()

        for img_name in image_names:
            img_name = os.path.join(self.anchor, img_name)
            faces = Sorter.extract_faces(img_name)[0]
            for face in faces:
                show_face(face)
        Sorter.log_simple_info('anchors loaded')

    def sort(self, path: str):
        """
        sort in each folder the images and check if they are similar to the anchors images
        :param path: path to the folder to sort
        """
        Sorter.log_simple_info(f'current folder: {path} - sorting...')
        image_paths = os.listdir(os.path.join(self.root, path))
        for image_path in (pbar := tqdm(image_paths, position=0, leave=True)):
            pbar.set_description(f'current image: {image_path}')
            image_path = os.path.join(self.root, path, image_path)
            temp_f, face_cord = Sorter.extract_faces(image_path)
            faces = [Sorter.image_to_embedding(face) for face in temp_f]

            for index, face in enumerate(faces):
                for name, (em, _, _) in self.anchors.items():
                    if (dist := Sorter.compare_pair(face, em)) >= config.COS_THRESH:
                        self.anchors[name][1][(str(floor((dist * 10)) * 10) if dist < 1 else '90')] \
                            .append((image_path, face_cord[index]))
                        Sorter.log_simple_info('match!')
                        break

        Sorter.log_simple_info(f'folder {path} sorted')

    def sort_wrapper(self):
        threads = []
        """
                Sort images to respective persons in the anchor images asynchronously
                using format:

                root folder
                        |__ sub dir 1
                        |__ sub dir 2
                        |__ ...
                anchor images
                """
        folder_paths = os.listdir(self.root)
        for path in folder_paths:
            threads.append(threading.Thread(target=self.sort, args=(path,)))
            threads[-1].start()
        # wait for all threads to finish
        for thread in threads:
            thread.join()
        # save the results
        for key, (_, paths_dict, _) in self.anchors.items():
            anc_path = os.path.join(self.root, str(key))
            os.makedirs(anc_path, exist_ok=True)
            for quality in paths_dict.items():
                os.makedirs(os.path.join(anc_path, quality[0]), exist_ok=True)
                for _path, _face_cord in quality[1]:
                    file_name = os.path.join(anc_path, quality[0], os.path.basename(_path))
                    shutil.copy(_path, file_name)
                    x1, y1, x2, y2 = _face_cord
                    image = cv.imread(_path)
                    cv.rectangle(image, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 255, 0), 2)
                    cv.imwrite(file_name + '_marked.jpg',
                               image)
        Sorter.log_simple_info('sorting done')

    @staticmethod
    def log_simple_info(msg):
        print(f'{fg("light_blue")}[INFO] {attr("reset")}{msg}')
