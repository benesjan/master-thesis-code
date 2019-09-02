import json
from os import listdir

import cv2
from PIL import Image
import numpy as np
from mtcnn.detector import detect_faces
from skimage import transform as trans

from config import Config
from data_processing.standardize_fav_dataset import get_frame


def get_next_image(video, annotations):
    num_of_names = len(annotations.keys())
    for i, name in enumerate(annotations.keys()):
        print(f"\t{i + 1}/{num_of_names} {name}")
        try:
            for detection in annotations[name]['detections']:
                # Get the cropped image
                frame = detection['frame']

                # Load the frame
                im = get_frame(video, frame)

                return name, frame, im
        except Exception as e:
            print(f"An error occurred when processing image of {name}\n{e}")


if __name__ == '__main__':
    conf = Config()
    target_res = (128, 128)

    # Load the image from video
    # Get video names
    videos = listdir(conf.VIDEO_PATH)
    video_name = videos[5]

    with open(conf.ANNOTATIONS_PATH + video_name + "_people.json", "r") as f:
        annotations = json.load(f)

    # load the video
    video = cv2.VideoCapture(conf.VIDEO_PATH + video_name)

    name, frame, img_cv2 = get_next_image(video, annotations)
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    # img = Image.open('jf.jpg')
    # img_cv2 = np.array(img)[..., ::-1]

    bounding_boxes, landmarks = detect_faces(img)

    # Convert landmarks of shape (1, 10) to array of coordinates of 5 facial points (shape (5, 2))
    dst = landmarks[0].astype(np.float32)
    facial5points = np.array([[dst[j], dst[j + 5]] for j in range(5)])

    # Computation of the transformation matrix M
    # ??? Desirable position of facial points ???
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    # ??? Scale the source facial points according to the size of the image ???
    src[:, 0] *= (target_res[0] / 96)
    src[:, 1] *= (target_res[1] / 112)

    tform = trans.SimilarityTransform()
    tform.estimate(facial5points, src)
    M = tform.params[0:2, :]

    # Applying the transformation matrix M to the original image
    # warped = cv2.warpAffine(img_cv2, M, (img.size[0], img.size[1]), borderValue=0.0)
    warped = cv2.warpAffine(img_cv2, M, target_res, borderValue=0.0)

    img_processed = Image.fromarray(warped[..., ::-1])

    img_processed.show()
