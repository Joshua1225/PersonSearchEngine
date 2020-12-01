import cv2
import sys
sys.path.append('./pedestrain_detection/')
sys.path.append('./PAR/')

from pedestrain_detection.Detector import PedestrianDetector, PedestrianDetectionResultDTO
from PAR.PAR_sdk import PedestrianAtrributeRecognizer

if __name__ == '__main__':
    detector = PedestrianDetector('./pedestrain_detection/weights/epoches_112.pth', cuda=True, cpu=False)
    img = cv2.imread('./pedestrain_detection/images/1.jpg')
    pdrDTO = detector.detect(img)
    img_list = pdrDTO.get_img_list()
    
    model = PedestrianAtrributeRecognizer('./PAR/checkpoints/market/resnet50_nfc/net_last.pth')
    model.infer_img_list(img_list)
    
    
    
#    for i, img in enumerate(img_list):
#        cv2.imwrite(f"{i}_.jpg", img)

