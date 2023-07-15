import config
from facenet_pytorch import MTCNN, InceptionResnetV1


class Init:
    mtcnn = None
    resnet = None

    def __init__(self):
        if not Init.mtcnn:
            Init.mtcnn = MTCNN(select_largest=False, device=config.DEVICE)

        if not Init.resnet:
            Init.resnet = InceptionResnetV1(pretrained='vggface2', device=config.DEVICE).eval()
