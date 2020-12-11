import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn

from PersonReID.torchreid import transforms as T
from PersonReID.torchreid import models
from PersonReID.reid_config import config as conf

class ReID():
    def __init__(self, training=False):
        # load parameters
        self.args = conf
        self._load_parameters()

        torch.manual_seed(self.args.seed)
        if not self.args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_devices
        use_gpu = torch.cuda.is_available()
        if self.args.use_cpu: use_gpu = False

        if use_gpu:
            print("Currently using GPU {}".format(self.args.gpu_devices))
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            print("Currently using CPU (GPU is highly recommended)")

        self.transform_test = T.Compose([
            T.Resize((self.args.height, self.args.width)),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Normalize(mean=[0.3495, 0.3453, 0.3941], std=[0.2755, 0.2122, 0.2563]),
        ])

        # pin_memory = True if use_gpu else False

        self.model = models.init_model(name=self.args.arch,
                                  num_classes=19658,  # 30874 or 20906 or 29626 or 34394 or 29906 or 29626 or 19658
                                  # num_classes=19658,
                                  isFinal=True,
                                  global_branch=self.args.global_branch,
                                  arch="resnet50")

        checkpoint = torch.load(self.args.model_weight)
        pretrain_dict = checkpoint['state_dict']
        self.model.load_state_dict(pretrain_dict)

        if use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        self.model.eval()


    def _load_parameters(self):
        """load the visible parameters"""
        self.args.gpu_devices = '0'
        self.args.test_batch = 100
        self.args.root = './'
        self.args.model_weight = './PersonReID/log/reid-model.pth.tar'
        self.args.height = 256
        self.args.width = 128
        self.args.dist_metric = 'cosine'


    def cal_features(self, imgs, use_gpu=True):
        """
        input a batch of imgs and returns the corresponding features
        imgs: size [batch, 3, height, width]
        use_gpu: True or False
        return: features: size [batch, 4096]
        """
        with torch.no_grad():
            if use_gpu:
                imgs = imgs.cuda()

            features = self.model(imgs)
            features = features.data.cpu()
        return features


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
