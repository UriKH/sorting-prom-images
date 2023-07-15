import os
import shutil
import threading
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2 as cv
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create face detector
mtcnn = MTCNN(select_largest=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
anchors = {}
THRESH = 0.6


def extract_faces(path):
    """
    Load image from disk
    """
    image = cv.imread(path)
    h, w, _ = image.shape

    if image is None:
        print(f'image was None - check it! {path}')
        return []
    else:
        image = cv.resize(image, (w // 2, h // 2))
    h, w, _ = image.shape

    faces_coord, conf = mtcnn.detect(image)

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

        margin = int((x2 - x1) * 0.1)
        x1 = x1 - margin if x1 - margin >= 0 else 0
        x2 = x2 + margin if x2 - margin < w else w - 1
        y1 = y1 - margin * 2 if y1 - margin * 2 >= 0 else 0  # capture forehead
        y2 = y2 + int(margin * 1.2) if y2 - int(margin * 1.2) < h else h - 1  # capture chin

        faces.append(image[y1:y2, x1:x2])
    return faces


def load_anchors(path):
    """
    Load friend faces as embedding into anchors list
    """
    image_names = os.listdir(path)

    def show_face(face):
        cv.imshow('choose name', face)
        cv.waitKey(0)

    def get_name():
        name = input('choose person name: ')
        anchors[name] = (image_to_embedding(face), [], face)
        load_anchors.name_chosen_flag = True
        return name

    for img_name in image_names:
        img_name = os.path.join(path, img_name)
        faces = extract_faces(img_name)
        for face in faces:
            t1 = threading.Thread(target=show_face, args=(face,))
            t2 = threading.Thread(target=get_name)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
    print('loaded')


def create_image(cut, size=200):
    height, width, _ = cut.shape
    prop = width / height

    if width > height:
        width = size
        height = int(size / prop)
    else:
        width = int(prop * size)
        height = size
    cut = cv.resize(cut, (width, height))

    # create a white canvas
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    start_x = (size - width) // 2
    start_y = (size - height) // 2

    # position the cropped face in the middle of the canvas
    canvas[start_y: start_y + height, start_x: start_x + width] = cut
    return canvas


def image_to_embedding(image):
    """
    Create embedding from processed image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = create_image(image, 160)
    image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the image
    ])

    preprocessed_image = transform(image)

    resnet.classify = True
    embedding = resnet(preprocessed_image.unsqueeze(0).to(device))
    return embedding


def compare_pair(em1, em2):
    dist = torch.nn.functional.cosine_similarity(em1, em2)  # cosine_sim = 1 - cosine
    # print(f'dist {dist.item()}')
    return dist.item()


def compare_all(folders_path):
    """
    Sort folder to anchors
    """
    folder_paths = os.listdir(folders_path)
    for path in folder_paths:
        print(f'current folder: {path}')
        image_paths = os.listdir(os.path.join(folders_path, path))
        for image_path in image_paths:
            image_path = os.path.join(folders_path, path, image_path)
            temp_f = extract_faces(image_path)
            faces = [image_to_embedding(face) for face in temp_f]
            # print(f'current img: {image_path}')

            for index, face in tqdm(enumerate(faces)):
                for name, (em, _, _) in anchors.items():
                    if (dist := compare_pair(face, em)) >= THRESH:
                        anchors[name][1].append(image_path)
                        print('match!')
                        # save image in the folder
                        cv.imwrite(os.path.join(folders_path, name, f'{dist}.jpg'), temp_f[index])
                        print(f"saved to {os.path.join(folders_path, name, f'{dist}.jpg')}")
                        break

    for key, (_, paths, _) in anchors.items():
        anc_path = os.path.join(folders_path, str(key))
        os.makedirs(anc_path, exist_ok=True)
        for path in paths:
            shutil.copy(path, anc_path)


load_anchors(r'')
compare_all(r'')
