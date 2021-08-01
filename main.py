from utils import FaceMesh
from utils.image_on_image import draw_img

import imutils
import cv2

glasses = cv2.imread("images/glasses1.png", -1)
glasses = imutils.resize(glasses, width=500)

moustache = cv2.imread("images/moustache1.png", -1)
moustache = imutils.resize(moustache, width=500)

fm = FaceMesh()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # flip the image because this is selfie camera
    image = cv2.flip(image, 1)
    fm.process(image)

    # if fm.get_multi_face_landmarks():
    #     image = fm.draw_face_landmarks()
    #     image = fm.draw_landmark_index()

    glasses_landmark = fm.get_glasses_landmarks()
    moustache_landmark = fm.get_mustache_landmarks()
    if glasses_landmark:

        image.flags.writeable = True

        g_landmarks = glasses_landmark[0]
        h, w = image.shape[:2]
        left, left_d, right, right_d, center = [(int(lms.y * h), int(lms.x * w)) for lms in g_landmarks]
        image = draw_img(image, glasses, center, left, left_d, right, right_d)

        m_landmarks = moustache_landmark[0]
        h, w = image.shape[:2]
        left, left_d, right, right_d, center = [(int(lms.y * h), int(lms.x * w)) for lms in m_landmarks]
        image = draw_img(image, moustache, center, left, left_d, right, right_d)

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
