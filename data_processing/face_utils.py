import sys

import cv2
from PIL import Image
from mtcnn.detector import detect_faces
import numpy as np
from skimage import transform as trans


def _intersects(self, other):
    # Collision detection using the separating axis theorem
    return not (self[2] < other[0] or self[0] > other[2] or self[3] < other[1] or self[1] > other[3])


def _transform(img, landmarks):
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


def get_aligned_face(img_cv2, orig_bbox):
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    bounding_boxes, landmarks = detect_faces(img)

    for i, bbox in enumerate(bounding_boxes):
        if _intersects(bbox, orig_bbox):
            return _transform(img_cv2, landmarks[i])

    return None
