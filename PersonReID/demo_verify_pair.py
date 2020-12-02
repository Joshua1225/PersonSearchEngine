# demo program, input two image and recognize whether two images belong to the same person.
import numpy as np
import torch
from PIL import Image
from PersonReID.reid import ReID
from torch.nn import functional as F

threshold = 0.4 # need to change

def cosine_distance(input1, input2):
    """Computes cosine distance for tensor.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

def cosine_distance_np(vector1, vector2):
    # vector1 = np.array([1, 2, 3])
    # vector2 = np.array([4, 7, 5])
    dist = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    return dist

if __name__ == '__main__':

    reid = ReID()

    image_path_1 = 'data/0096_c2s1_015101_00.jpg'
    imgs = []
    im = Image.open(image_path_1)
    img = reid.transform_test(im)
    imgs.append(img)
    imgs = torch.stack(imgs)
    features_1 = reid.cal_features(imgs)

    image_path_2 = 'data/0096_c3s1_015551_00.jpg'
    imgs = []
    im = Image.open(image_path_2)
    img = reid.transform_test(im)
    imgs.append(img)
    imgs = torch.stack(imgs)
    features_2 = reid.cal_features(imgs)

    distance = np.array(cosine_distance(features_1, features_2))[0][0]
    print(str(distance))

    if distance < threshold:
        print("same")
    else:
        print("different")