import os
from net import get_model
from PIL import Image
from torchvision import transforms as T
import torch
import json


class predict_decoder(object):

    def __init__(self, dataset):
        with open('./PersonAttribute/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./PersonAttribute/doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        '''

        :param pred:
        :return:
        {
        "age":"young","teenager","adult","old",
        "carrying backpack":"yes","no"
        "carrying bag":"yes","no"
        "length of lower-body clothing": "long lower body clothing","short"
        "sleeve length":"long sleeve","short sleeve"
        "type of lower-body clothing": "dress","pants"
        "color of upper-body clothing"
        "color of lower-body clothing"
        "hair length": "short hair","long hair"
        "wearing hat":"yes","no"
        "gender":"male","female"
        }
        '''

        pred = pred.squeeze(dim=0)
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
        attr = {}
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                attr[name] = chooce[pred[idx]]
                # print('{}: {}'.format(name, chooce[pred[idx]]))
        for a in attr_list:
            if a not in attr:
                attr[a] = "unknown"
        return attr


class PedestrianAtrributeRecognizer(object):
    dataset_dict = {
        'market': 'Market-1501',
        'duke': 'DukeMTMC-reID',
    }
    num_cls_dict = {'market': 30, 'duke': 23}
    num_ids_dict = {'market': 751, 'duke': 702}

    transforms = T.Compose([
        T.Resize(size=(288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, weigth_path):
        self.backbone = 'resnet50'
        self.dataset = 'market'

        model_name = '{}_nfc_id'.format(self.backbone)
        num_label, num_id = self.num_cls_dict[self.dataset], self.num_ids_dict[self.dataset]
        self.model = get_model(model_name, num_label, use_id=False, num_id=num_id)

        self.model.load_state_dict(torch.load(weigth_path))
        print('Resume model from {}'.format(weigth_path))
        self.model.eval()

    def infer(self, image_path):
        def load_image(path):
            src = Image.open(path)
            src = self.transforms(src)
            src = src.unsqueeze(dim=0)
            return src

        src = load_image(image_path)
        out = self.model.forward(src)
        pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5

        dec = predict_decoder(self.dataset)
        return dec.decode(pred)

    def infer_img_list(self, img_list):
        attrs = []

        dec = predict_decoder(self.dataset)

        for nd in img_list:
            img = Image.fromarray(nd)
            img = self.transforms(img).unsqueeze(dim=0)

            out = self.model.forward(img)
            pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5

            attrs.append(dec.decode(pred))
        return attrs


'''
model = PedestrianAtrributeRecognizer('./checkpoints/market/resnet50_nfc/net_last.pth')
model.infer(model, './test_sample/test_market.jpg')
model.infer(model, './test_sample/test_duke.jpg')
'''
