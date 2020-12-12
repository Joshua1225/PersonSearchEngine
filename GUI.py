###################################################################
# This GUI is based on PyQt 4, the ui is designed with qt designer
# the .mp4 format is not support in the video loading
###################################################################

from sys import argv, exit
import cv2
import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import splitext
import random
from PyQt4 import QtGui, uic
from PyQt4.phonon import Phonon


import sys

sys.path.append("./PersonAttribute/")
sys.path.append('./PersonDetection/')
sys.path.append("./PersonReID/")

from PersonDetection.Detector import PedestrianDetectionResultDTO, PedestrianDetector
from PersonAttribute.PAR_sdk import PedestrianAtrributeRecognizer
from PersonReID.reid import ReID

attr_map = {0: ["所有", "男性", "女性"],
            1: ["所有", "短发", "长发"],
            2: ["所有", "长袖", "短袖"],
            3: ["所有", "长下装", "短下装"],
            4: ["所有", "裙子", "裤子"],
            5: ["所有", "无帽子", "有帽子"],
            6: ["所有", "无背包", "有背包"],
            7: ["所有", "无带包", "有带包"],
            8: ["所有", "无手包", "有手包"],
            9: ["所有", "小孩", "青年", "成年", "老年"],
            10: ["所有", "上装黑色", "上装白色", "上装红色", "上装紫色", "上装黄色", "上装灰色", "上装蓝色", "上装绿色"],
            11: ["所有", "下装黑色", "下装白色", "下装粉色", "下装紫色", "下装黄色", "下装灰色", "下装蓝色", "下装绿色", "下装棕色"]}
convert_map = {
    "男性": "male",
    "女性": "female",
    "所有": "all",
    "长袖": "long sleeve",
    "短袖": "short sleeve",
    "长下装": "long lower body clothing",
    "短下装": "short",
    "裙子": "dress",
    "裤子": "pants",
    "无帽子": "no",
    "有帽子": "yes",
    "无背包": "no", "有背包": "yes",
    "无带包": "yes", "有带包": "yes",
    "无手包": "no", "有手包": "yes",
    "小孩": "young", "青年": "teenager", "成年": "adult", "老年": "old",
    "上装黑色": "black", "上装白色": "white", "上装红色": "red", "上装紫色": "purple", "上装黄色": "yellow", "上装灰色": "gray", "上装蓝色": "blue",
    "上装绿色": "green",
    "下装黑色": "black", "下装白色": "white", "下装粉色": "pink", "下装紫色": "purple", "下装黄色": "yellow", "下装灰色": "gray",
    "下装蓝色": "blue", "下装绿色": "green", "下装棕色": "brown"
}
attr_list = ["gender",
             "hair length",
             "sleeve length",
             "length of lower-body clothing",
             "type of lower-body clothing",
             "wearing hat",
             "carrying backpack",
             "carrying bag",
             "carrying handbag",
             "age",
             "color of upper-body clothing",
             "color of lower-body clothing"]

qtCreatorFile = "layout.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.origin_video_prepared = False  # whether the origin video is prepared
        self.result_video_prepared = False  # whether the result video is prepared

        # init the drop-down box
        self.init_comboBox()

        # init the model and parameters, unfinished!
        self.person_detector = PedestrianDetector('./PersonDetection/weights/epoches_112.pth', cuda=True, cpu=False)
        self.person_recognizer = PedestrianAtrributeRecognizer(
            './PersonAttribute/checkpoints/market/resnet50_nfc/net_last.pth')
        # self.person_ReID = ReID()

        self.frame_list = []

        # trigger the funtion when clicking the button
        self.load_image_button.clicked.connect(
            self.load_image)  # when clicking load_image_button trigger the function "load_image"
        self.load_video_button.clicked.connect(
            self.load_video)  # when clicking load_video_button trigger the function "load_video"
        self.search_button.clicked.connect(self.search)  # when clicking search_button trigger the function "search"
        self.clear_button.clicked.connect(self.clear)  # when clicking clear_button trigger the function "clear"
        self.origin_video_pushButton.clicked.connect(self.change_status_origin_video)
        self.result_video_pushButton.clicked.connect(self.change_status_result_video)

    # unfinished!
    def load_image(self):
        """load an image from directories and show the image to the label "image_show" """
        img_path = QtGui.QFileDialog.getOpenFileName(self, self.load_image_button.text())
        if img_path.split('.')[-1] in (
                "jpg", "JPG", "jpeg", "JPEG"):  # jpg is incompatible, need to change to png and save it
            tmp_root = '.tmp/'
            if not os.path.exists(tmp_root):
                os.mkdir(tmp_root)
            img = Image.open(img_path)
            save_path = os.path.join(tmp_root, str(random.randint(10000, 100000)) + '.png')
            img.save(save_path)
            img_path = save_path
        self.image_path = img_path
        pix = QtGui.QPixmap(img_path)
        self.image_show.setPixmap(pix)
        self.image_show.setScaledContents(True)

        # need to read the image and change to Tensor

        pass

    def load_video(self):
        """load a video from directories."""

        # when the media is on, need to turn off it.
        if self.origin_videoPlayer.isPlaying():
            self.origin_videoPlayer.stop()
            self.origin_video_pushButton.setText("播放")
        path = QtGui.QFileDialog.getOpenFileName(self, self.load_video_button.text())
        self.origin_videoPlayer.load(Phonon.MediaSource(path))
        self.origin_video_prepared = True

        # extract some of the frames in the chosen video
        sample_interval = 20  # the interval of each sample, can be changed
        cap = cv2.VideoCapture(path)
        rate = cap.get(5)  # 帧速率
        if rate == 0:  # the empty video
            return
        FrameNumber = cap.get(7)  # 视频文件的帧数

        frame_list = []  # save the sampled frames
        for idx in range(0, int(FrameNumber - 1), sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
            frame_list.append(frame)

        if len(self.frame_list) > 0:  # clear the current frame_list
            self.frame_list = []
        self.frame_list = frame_list  # (sample_size,width,height,3)

    # unfinished!
    def search(self):
        """
            search the results according to the person attributes and input image,
                and show the results to the Phonon "result_videoPlayer" """

        def check_attr(attr_target, attr_res):
            '''
            :param attr_target: chinese context list in gui
            :param attr_res:model's results dict
            :return: bool
            '''
            for idy, val in enumerate(attr_list):

                if convert_map[attr_target[idy]] == "all" or attr_res[val] == 'unknown':
                    continue
                elif convert_map[attr_target[idy]] != attr_res[val]:
                    return False
            return True

        # demo for the use of QCheckBox
        use_attri = self.attri_checkBox.isChecked()  # use attribute for searching
        use_img = self.image_checkBox.isChecked()  # use image for searching

        if use_attri:
            attr_target = []
            items = [self.comboBox_1, self.comboBox_2, self.comboBox_3, self.comboBox_4, self.comboBox_5,
                     self.comboBox_6, self.comboBox_7, self.comboBox_8, self.comboBox_9, self.comboBox_10,
                     self.comboBox_11, self.comboBox_12]
            for idx in range(len(items)):
                attr_target.append(items[idx].currentText())
        if use_img:
            image_target = np.array(Image.open(self.image_path) )

        rectangle_frame_list = []
        for frame in self.frame_list:

            pdrDTO = self.person_detector.detect(frame)
            img_list = pdrDTO.get_img_list()

            attr_result = self.person_recognizer.infer_img_list(img_list)

            for idx, bbox in enumerate(pdrDTO.bbox_list):
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)

                if use_attri:
                    if check_attr(attr_target, attr_result[idx]):
                        print(attr_result[idx])
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                elif use_img:
                    if self.person_ReID.verify_pair(image_target,img_list[idx]) :
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)



            rectangle_frame_list.append(frame)

        # change to the video and save it
        fps = 10  # 保存视频的帧率
        tmp_root = '.tmp/'
        if not os.path.exists(tmp_root):
            os.mkdir(tmp_root)
        save_path = os.path.join(tmp_root, str(random.randint(10000, 100000)) + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # videoWriter = cv2.VideoWriter(save_path, apiPreference=0, fourcc=fourcc, fps=fps, frameSize=(self.frame_list[0], self.frame_list[1]))
        videoWriter = cv2.VideoWriter(save_path, fourcc, fps,
                                      (self.frame_list[0].shape[1], self.frame_list[0].shape[0]))
        for img in rectangle_frame_list:
            videoWriter.write(img)
        videoWriter.release()
        # load the video and show it
        # when the media is on, need to turn off it.
        if self.result_videoPlayer.isPlaying():
            self.result_videoPlayer.stop()
            self.result_video_pushButton.setText("播放")
        self.result_videoPlayer.load(Phonon.MediaSource(save_path))
        self.result_video_prepared = True



    # unfinished!
    def clear(self):
        """clear all of the things"""

        self.init_comboBox()  # clear the comboBox
        self.image_show.setText("请加载行人图像")  # clear the shown image

        # stop all of the playing videos
        if self.origin_videoPlayer.isPlaying():
            # stop it
            self.origin_videoPlayer.stop()
            self.origin_video_pushButton.setText("播放")
            self.origin_video_prepared = False
        if self.result_videoPlayer.isPlaying():
            # stop it
            self.result_videoPlayer.stop()
            self.result_video_pushButton.setText("播放")
            self.result_video_prepared = False

        pass

    def change_status_origin_video(self):
        """
        If the original video is playing, stop it.
        If the original video is stopped, play it."""
        if not self.origin_video_prepared:
            pass
        elif self.origin_videoPlayer.isPlaying():
            # stop it
            self.origin_videoPlayer.stop()
            self.origin_video_pushButton.setText("播放")
        else:
            self.origin_videoPlayer.play()
            self.origin_video_pushButton.setText("停止")

    # unfinished!
    def change_status_result_video(self):
        """
        If the result video is playing, stop it.
        If the result video is stopped, play it."""

        if not self.result_video_prepared:
            pass
        elif self.result_videoPlayer.isPlaying():
            # stop it
            self.result_videoPlayer.stop()
            self.result_video_pushButton.setText("播放")
        else:
            self.result_videoPlayer.play()
            self.result_video_pushButton.setText("停止")

    def init_comboBox(self):
        """init the drop-down box"""
        items = [self.comboBox_1, self.comboBox_2, self.comboBox_3, self.comboBox_4, self.comboBox_5,
                 self.comboBox_6, self.comboBox_7, self.comboBox_8, self.comboBox_9, self.comboBox_10,
                 self.comboBox_11, self.comboBox_12]
        for idx in range(len(items)):
            items[idx].clear()
            items[idx].addItems(attr_map[idx])

    # #清空
    # def clear(self):
    #     self.clear_num()
    #     self.clear_image()
    #
    # #加载图片
    # def loadImage(self):
    #     mcnn.reloadData()
    #     #self.image_1.setPixmap(data_loader.blob_list[0]['data'])
    #     self.changeJPGtoPNG()#将JPG转换为PNG
    #
    #     self.pix = QtGui.QPixmap(data_path_base+category[0]+'/1.png') #QPixmap不可读入jpg
    #     self.res_image_1.setPixmap(self.pix)
    #     self.res_image_1.setScaledContents(True) #使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[0] + '/2.png')  # QPixmap不可读入jpg
    #     self.res_image_2.setPixmap(self.pix)
    #     self.res_image_2.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[0] + '/3.png')  # QPixmap不可读入jpg
    #     self.res_image_3.setPixmap(self.pix)
    #     self.res_image_3.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[0] + '/4.png')  # QPixmap不可读入jpg
    #     self.res_image_4.setPixmap(self.pix)
    #     self.res_image_4.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[0] + '/5.png')  # QPixmap不可读入jpg
    #     self.res_image_5.setPixmap(self.pix)
    #     self.res_image_5.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[0] + '/6.png')  # QPixmap不可读入jpg
    #     self.res_image_6.setPixmap(self.pix)
    #     self.res_image_6.setScaledContents(True)  # 使图像匹配label大小
    #
    #     self.pix = QtGui.QPixmap(data_path_base + category[1] + '/1.png')  # QPixmap不可读入jpg
    #     self.out_image_1.setPixmap(self.pix)
    #     self.out_image_1.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[1] + '/2.png')  # QPixmap不可读入jpg
    #     self.out_image_2.setPixmap(self.pix)
    #     self.out_image_2.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[1] + '/3.png')  # QPixmap不可读入jpg
    #     self.out_image_3.setPixmap(self.pix)
    #     self.out_image_3.setScaledContents(True)  # 使图像匹配label大小
    #
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/1.png')  # QPixmap不可读入jpg
    #     self.class_image_1.setPixmap(self.pix)
    #     self.class_image_1.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/2.png')  # QPixmap不可读入jpg
    #     self.class_image_2.setPixmap(self.pix)
    #     self.class_image_2.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/3.png')  # QPixmap不可读入jpg
    #     self.class_image_3.setPixmap(self.pix)
    #     self.class_image_3.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/4.png')  # QPixmap不可读入jpg
    #     self.class_image_4.setPixmap(self.pix)
    #     self.class_image_4.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/5.png')  # QPixmap不可读入jpg
    #     self.class_image_5.setPixmap(self.pix)
    #     self.class_image_5.setScaledContents(True)  # 使图像匹配label大小
    #     self.pix = QtGui.QPixmap(data_path_base + category[2] + '/6.png')  # QPixmap不可读入jpg
    #     self.class_image_6.setPixmap(self.pix)
    #     self.class_image_6.setScaledContents(True)  # 使图像匹配label大小
    #
    # #将data_path文件夹下的JPG转为PNG
    # def changeJPGtoPNG(self):
    #     for i in category:
    #         path=data_path_base+i+'/'
    #         for filename in listdir(path):
    #             if splitext(filename)[1] == '.jpg':
    #                 img = imread(path + filename)
    #                 newfilename = filename.replace(".jpg", ".png")
    #                 imwrite(path + newfilename, img)
    #
    # #计算人数
    # def calculate_num(self):
    #     num_mcnn=mcnn.returnNum("restaurant",1)# 用mcnn获取第1张图片的总人数
    #     num_retinanet=retinanet.returnNum("restaurant",1)#用RetinaNet获取第1张图片的总人数
    #     num_yolo=yolo.returnNum("restaurant",1)
    #     num=p*num_retinanet+q*num_mcnn+r*num_yolo
    #     self.res_result_1.setText(str(format(num,".2f")))
    #     num_mcnn = mcnn.returnNum("restaurant", 2)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("restaurant", 2)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("restaurant", 2)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.res_result_2.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("restaurant", 3)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("restaurant", 3)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("restaurant", 3)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.res_result_3.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("restaurant", 4)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("restaurant", 4)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("restaurant", 4)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.res_result_4.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("restaurant", 5)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("restaurant", 5)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("restaurant", 5)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.res_result_5.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("restaurant", 6)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("restaurant", 6)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("restaurant", 6)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.res_result_6.setText(str(format(num, ".2f")))
    #
    #     num_mcnn = mcnn.returnNum("outdoor", 1)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("outdoor", 1)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("outdoor", 1)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.out_result_1.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("outdoor", 2)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("outdoor", 2)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("outdoor", 2)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.out_result_2.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("outdoor", 3)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("outdoor", 3)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("outdoor", 3)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.out_result_3.setText(str(format(num, ".2f")))
    #
    #     num_mcnn = mcnn.returnNum("classroom", 1)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 1)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 1)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_1.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("classroom", 2)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 2)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 2)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_2.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("classroom", 3)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 3)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 3)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_3.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("classroom", 4)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 4)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 4)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_4.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("classroom", 5)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 5)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 5)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_5.setText(str(format(num, ".2f")))
    #     num_mcnn = mcnn.returnNum("classroom", 6)  # 用mcnn获取第1张图片的总人数
    #     num_retinanet = retinanet.returnNum("classroom", 6)  # 用RetinaNet获取第1张图片的总人数
    #     num_yolo = yolo.returnNum("classroom", 6)
    #     num = p * num_retinanet + q * num_mcnn + r * num_yolo
    #     self.class_result_6.setText(str(format(num, ".2f")))
    #
    # #清空人数
    # def clear_num(self):
    #     self.res_result_1.setText("0")
    #     self.res_result_2.setText("0")
    #     self.res_result_3.setText("0")
    #     self.res_result_4.setText("0")
    #     self.res_result_5.setText("0")
    #     self.res_result_6.setText("0")
    #     self.out_result_1.setText("0")
    #     self.out_result_2.setText("0")
    #     self.out_result_3.setText("0")
    #     self.class_result_1.setText("0")
    #     self.class_result_2.setText("0")
    #     self.class_result_3.setText("0")
    #     self.class_result_4.setText("0")
    #     self.class_result_5.setText("0")
    #     self.class_result_6.setText("0")
    #
    # #清空图像
    # def clear_image(self):
    #     self.res_image_1.setText("请单击“加载图像”")
    #     self.res_image_2.setText("请单击“加载图像”")
    #     self.res_image_3.setText("请单击“加载图像”")
    #     self.res_image_4.setText("请单击“加载图像”")
    #     self.res_image_5.setText("请单击“加载图像”")
    #     self.res_image_6.setText("请单击“加载图像”")
    #     self.out_image_1.setText("请单击“加载图像”")
    #     self.out_image_2.setText("请单击“加载图像”")
    #     self.out_image_3.setText("请单击“加载图像”")
    #     self.class_image_1.setText("请单击“加载图像”")
    #     self.class_image_2.setText("请单击“加载图像”")
    #     self.class_image_3.setText("请单击“加载图像”")
    #     self.class_image_4.setText("请单击“加载图像”")
    #     self.class_image_5.setText("请单击“加载图像”")
    #     self.class_image_6.setText("请单击“加载图像”")


if __name__ == "__main__":
    app = QtGui.QApplication(argv)
    window = MyApp()
    window.show()
    exit(app.exec_())
