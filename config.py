import torch

COS_THRESH = 0.6
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
