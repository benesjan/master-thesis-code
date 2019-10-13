import cv2
import numpy as np
from skimage import transform as trans


def _bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    xSide = (xB - xA + 1)
    ySide = (yB - yA + 1)

    if xSide <= 0 or ySide <= 0:
        return 0

    interArea = xSide * ySide

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area

    if boxAArea + boxBArea - interArea == 0:
        return 0

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox_i_by_IoU(bboxes, orig_bbox, threshold=0.4):
    # Select the bounding box with biggest intersection over union
    bbox_i, max_IoU = -1, threshold
    for i, cur_bbox in enumerate(bboxes):
        cur_IoU = _bb_intersection_over_union(cur_bbox, orig_bbox)
        if cur_IoU >= max_IoU:
            bbox_i, max_IoU = i, cur_IoU
    return bbox_i


def frontalize_face(img, landmarks):
    target_res = (128, 128)

    # Convert landmarks of shape (1, 10) to array of coordinates of 5 facial points (shape (5, 2))
    dst = landmarks.astype(np.float32)
    facial5points = np.array([[dst[j], dst[j + 5]] for j in range(5)])

    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    src[:, 0] *= (target_res[0] / 96)
    src[:, 1] *= (target_res[1] / 112)

    tform = trans.SimilarityTransform()
    tform.estimate(facial5points, src)
    M = tform.params[0:2, :]

    # Applying the transformation matrix M to the original image
    img_warped = cv2.warpAffine(img, M, target_res, borderValue=0.0)

    return img_warped
