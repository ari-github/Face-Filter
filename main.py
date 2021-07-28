from utils import FaceMesh
from utils.image_on_image import draw_img

import imutils
import cv2

sunglasses = cv2.imread("images/sunglasses1.png", -1)
sunglasses = imutils.resize(sunglasses, width=500)

fm = FaceMesh()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # flip the image because this is selfie camera
    image = cv2.flip(image, 1)
    fm.process(image)

    glasses_landmark = fm.get_glasses_landmarks()
    if glasses_landmark:
        landmarks = glasses_landmark[0]
        h, w = image.shape[:2]
        left, left_d, right, right_d, center = [(int(lms.y * h), int(lms.x * w)) for lms in landmarks]

        image.flags.writeable = True
        image = draw_img(image, sunglasses, center, left, left_d, right, right_d)

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
