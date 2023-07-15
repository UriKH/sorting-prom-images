import os
import shutil
import threading
import torch
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
import numpy as np
from tqdm import tqdm
from colored import fg, attr

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
    def extract_faces(cls, path: str) -> list:
        """
        Extract faces from image in format BGR using MTCNN
        :param path: path to the image in disk
        :returns: list of cropped faces
        """
        image = cv.imread(path)
        h, w, _ = image.shape

        if image is None:
            Sorter.log_simple_info(f'image was None - check it! {path}')
            return []
        else:
            image = cv.resize(image, (w // 2, h // 2))  # downscale image for faster execution
        h, w, _ = image.shape

        faces_coord, conf = Sorter.mtcnn.detect(image)  # detect faces using MTCNN

        if faces_coord is None or len(faces_coord) == 0:
            return []

        faces_coord = faces_coord.astype(int)
        faces_coord = [face for face, c in zip(faces_coord, conf) if c >= 0.9]
        faces = []
        for face in faces_coord:
            x1, y1, x2, y2 = face
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            margin = int((x2 - x1) * 0.1)               # calculate margin
            x1 = max(x1 - margin, 0)
            x2 = min(x2 + margin, w - 1)
            y1 = max(y1 - margin * 2, 0)                # capture forehead
            y2 = min(y2 + int(margin * 1.2), h - 1)     # capture chin

            faces.append(image[y1:y2, x1:x2])
        return faces

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
            transforms.ToTensor(),                                              # Convert the image to a tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),    # Normalize the image
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

        def show_face(face):
            cv.imshow('choose name', face)
            cv.waitKey(0)

        def get_name():
            name = input('choose person name: ')
            self.anchors[name] = [Sorter.image_to_embedding(face), [], face]
            Sorter.load_anchors.name_chosen_flag = True
            return name

        for img_name in image_names:
            img_name = os.path.join(self.anchor, img_name)
            faces = Sorter.extract_faces(img_name)
            for face in faces:
                t1 = threading.Thread(target=show_face, args=(face,))
                t2 = threading.Thread(target=get_name)

                t1.start()
                t2.start()

                t1.join()
                t2.join()
        Sorter.log_simple_info('anchors loaded')

    def sort(self):
        """
        Sort images to respective persons in the anchor images
        using format:

        root folder
                |__ sub dir 1
                |__ sub dir 2
                |__ ...
        anchor images
        """
        folder_paths = os.listdir(self.root)

        for path in folder_paths:
            Sorter.log_simple_info(f'current folder: {path}')
            image_paths = os.listdir(os.path.join(self.root, path))

            for image_path in tqdm(image_paths):
                image_path = os.path.join(self.root, path, image_path)
                temp_f = Sorter.extract_faces(image_path)
                faces = [Sorter.image_to_embedding(face) for face in temp_f]

                for index, face in enumerate(faces):
                    for name, (em, _, _) in self.anchors.items():
                        if (dist := Sorter.compare_pair(face, em)) >= config.COS_THRESH:
                            self.anchors[name][1].append(image_path)
                            Sorter.log_simple_info('match!')

                            # save image in the folder
                            cv.imwrite(os.path.join(self.root, name, f'{dist}.jpg'), temp_f[index])
                            Sorter.log_simple_info(f"saved to {os.path.join(self.root, name, f'{dist}.jpg')}")
                            break

            for key, (_, paths, _) in self.anchors.items():
                anc_path = os.path.join(self.root, str(key))
                os.makedirs(anc_path, exist_ok=True)
                for _path in paths:
                    shutil.copy(_path, anc_path)
                self.anchors[key][1] = []

    @staticmethod
    def log_simple_info(msg):
        print(f'{fg("light_blue")}[INFO] {attr("reset")}{msg}')
