# demo program, get the features of the given batch of images
import numpy as np
import torch
from PIL import Image
from PersonReID.reid import ReID

if __name__ == '__main__':
    # read the img_list.npy
    img_list = np.load('img_list.npy')
    img_list = img_list.tolist()

    reid = ReID()

    imgs = []
    for item in img_list:
        img = Image.fromarray(item)
        img = reid.transform_test(img)
        imgs.append(img)
    imgs = torch.stack(imgs)

    features = reid.cal_features(imgs)

    print("finish")