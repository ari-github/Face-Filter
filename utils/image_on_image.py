import cv2
import imutils
import numpy as np


def perspective_transform(left, left_d, right, right_d, mask):
    l_h, r_h = left_d[0] - left[0], right_d[0] - right[0]
    rel = r_h / l_h

    h, w = mask.shape[:2]

    pst1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pst2 = np.float32([[0, 0], [0, h], [w, 0], [w, h * rel]])

    mat = cv2.getPerspectiveTransform(pst1, pst2)
    result = cv2.warpPerspective(mask, mat, (w, max(int(h * rel), h)))

    return result


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def adjust_image(mask, base_shape, center):
    back_h, back_w = base_shape

    front_h, front_w = mask.shape[:2]

    y = center[0] - (front_h // 2)
    x = center[1] - (front_w // 2)

    if x + front_w < 0 or y + front_h < 0 or x > back_w or y > back_h:
        return None, None, None

    crop_x = crop_y = 0
    if x < 0:
        crop_x = x * -1
        x = 0

    if y < 0:
        crop_y = y * -1
        y = 0

    return y, x, mask[crop_y: back_h - y, crop_x: back_w - x]


def img_on_img(base, mask, x, y):
    h, w, c = mask.shape

    resize = base[y:y + h, x:x + w]

    result = np.zeros((h, w, 3), np.uint8)

    alpha = mask[:, :, 3] / 255.0
    result[:, :, 0] = (1. - alpha) * resize[:, :, 0] + alpha * mask[:, :, 0]
    result[:, :, 1] = (1. - alpha) * resize[:, :, 1] + alpha * mask[:, :, 1]
    result[:, :, 2] = (1. - alpha) * resize[:, :, 2] + alpha * mask[:, :, 2]

    base[y:y + h, x:x + w] = result

    return base


def draw_img(base, mask, center, left, left_d, right, right_d):
    mask_width = right[1] - left[1]
    mask = imutils.resize(mask, width=mask_width)

    mask = perspective_transform(left, left_d, right, right_d, mask)

    rotation = angle_between((0, 0), (right[1] - left[1], right[0] - left[0]))
    mask = rotate_bound(mask, rotation)

    y, x, mask = adjust_image(mask, base.shape[:2], center)
    if y is None:
        return base

    base = img_on_img(base, mask, x, y)

    return base
