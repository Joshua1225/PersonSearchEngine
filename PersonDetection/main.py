from Detector import PedestrianDetector, PedestrianDetectionResultDTO
import cv2

if __name__ == '__main__':
    detector = PedestrianDetector('./weights/epoches_112.pth', cuda=True, cpu=False)
    img = cv2.imread('./images/1.jpg')
    pdrDTO = detector.detect(img)
    img_list = pdrDTO.get_img_list()
    for i, img in enumerate(img_list):
        cv2.imwrite(f"{i}_.jpg", img)

