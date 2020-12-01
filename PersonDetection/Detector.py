from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import BaseTransform, VOC_Config
from layers.functions import Detect, PriorBox
from models.RFB_Net_vgg import build_net
from utils.nms_wrapper import nms
from utils.timer import Timer
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('--img_dir', default='images', type=str,
                    help='Dir to save results')
parser.add_argument('-m', '--trained_model', default='weights/epoches_112.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()


class PedestrianDetectionResultDTO(object):
    def __init__(self, img, bbox_list):
        '''
        Args:
            img: ndarray  [x,y,3]
            bbox_list: list[np.ndarray] [x_0,y_0,x_1,y_1,confidence)
        '''
        self.img = img
        self.bbox_list = bbox_list

    def get_img_list(self):
        '''
        Returns: list[np.ndarray] 截出来行人的的图
        '''
        img_list = [self._crop_to_img(bbox) for bbox in self.bbox_list]
        return img_list

    def _crop_to_img(self, bbox):
        img = self.img
        image_cliped = img[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        return image_cliped

    def get_bbox_img(self):
        '''
        Returns: 加上bbox的整张图
        '''
        img = self.img
        for bbox in self.bbox_list:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        return img

    def save_img(self, path):
        '''
        保存整张图到本地
        Args:
            path: 保存的文件名
        '''
        img = self.get_bbox_img()
        cv2.imwrite(path, img)


class PedestrianDetector(object):
    def __init__(self, weight_path, cuda, cpu):
        '''
        Args:
            weight_path: 训练好的参数位置
            cuda: 模型是否使用cuda
            cpu:  nms是否使用CPU
        '''
        img_dim = 300
        num_classes = 2
        rgb_means = (104, 117, 123)
        net = build_net('test', img_dim, num_classes)  # initialize detector
        state_dict = torch.load(weight_path)
        cfg = VOC_Config
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward()
            if cuda:
                priors = priors.cuda()

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')

        if cuda:
            net = net.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()
        detector = Detect(num_classes, 0, cfg)
        transform = BaseTransform(img_dim, rgb_means, (2, 0, 1))
        self.object_detector = ObjectDetector(net, detector, transform, priors, cuda=cuda, cpu=cpu)

    def detect(self, img, confidence=0.6):
        '''
        Args:
            img: cv2 img 待检测图像
            confidence: 框框的置信度大于confidence才会出现结果
        Returns:
        '''
        detect_bboxes, tim = self.object_detector.predict(img)
        bbox_list = []
        for class_id, class_collection in enumerate(detect_bboxes):
            if len(class_collection) > 0:
                for k in range(class_collection.shape[0]):
                    if class_collection[k, -1] > confidence:
                        pt = class_collection[k]
                        bbox_list.append(pt)

        pedestrianDetectionResultDTO = PedestrianDetectionResultDTO(img, bbox_list)
        return pedestrianDetectionResultDTO


class ObjectDetector:
    def __init__(self, net, detection, transform, priors, num_classes=2, thresh=0.1, cuda=True, cpu=False):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.num_classes = num_classes
        self.thresh = thresh
        self.cuda = cuda
        self.cpu = cpu
        self.priors = priors

    def predict(self, img):
        _t = {'im_detect': Timer(), 'misc': Timer()}
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = self.net(x)  # forward pass
        boxes, scores = self.detection.forward(out, self.priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        _t['misc'].tic()
        all_boxes = [[] for _ in range(self.num_classes)]

        for j in range(1, self.num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            # print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = nms(c_dets, 0.2, force_cpu=self.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        nms_time = _t['misc'].toc()
        total_time = detect_time + nms_time

        # print('total time: ', total_time)
        return all_boxes, total_time
