import cv2
import sys
sys.path.append('./PersonDetection/')
sys.path.append('./PersonAttribute/')

from PersonDetection.Detector import PedestrianDetector, PedestrianDetectionResultDTO
from PersonAttribute.PAR_sdk import PedestrianAtrributeRecognizer

if __name__ == '__main__':
    detector = PedestrianDetector('./PersonDetection/weights/epoches_112.pth', cuda=True, cpu=False)
    img = cv2.imread('./PersonDetection/images/1.jpg')
    pdrDTO = detector.detect(img)
    img_list = pdrDTO.get_img_list()
    
    model = PedestrianAtrributeRecognizer('./PersonAttribute/checkpoints/market/resnet50_nfc/net_last.pth')
    model.infer_img_list(img_list)
    
    
    
#    for i, img in enumerate(img_list):
#        cv2.imwrite(f"{i}_.jpg", img)

