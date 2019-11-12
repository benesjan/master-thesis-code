import json
from os import listdir, path
from sys import exit

import cv2
import numpy as np
from mtcnn import detect_faces

from config import Config
from data_processing.mtcnn_utils import get_bbox_i_by_IoU


# video - the opened video object
# frame_id - the frame number
def get_frame(video, frame_id):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    video.set(cv2.COLOR_BGR2GRAY, frame_id)
    # read the image from the video
    _, im = video.read()
    return im


def get_file_map(dir_path):
    file_map = {}
    if not path.exists(dir_path):
        return file_map
    files = listdir(dir_path)
    for file in files:
        file_map[file.split('&region')[0]] = file
    return file_map


if __name__ == '__main__':
    conf = Config()

    if path.exists(conf.MTCNN_PREDS):
        print(f"Target file containing values already exists on path {conf.MTCNN_PREDS}.\nEXITING")
        exit(0)

    with open(conf.MTCNN_PREDS, 'w') as mtcnn_file:
        for video_name in listdir(conf.VIDEO_PATH):
            print(f'Processing {video_name}')

            # Used in file name
            video_date = video_name.split('_')[2]

            # 2) Load the annotations
            with open(conf.ANNOTATIONS_PATH + video_name + '_people.json', 'r') as f:
                annotations = json.load(f)

            # 3) load the video
            video = cv2.VideoCapture(conf.VIDEO_PATH + video_name)

            # 4) check if the video file opened successfully, if not continue with another one
            if not video.isOpened():
                print(f'The videofile {video_name} could not be opened!')
                continue

            num_of_names = len(annotations.keys())
            # 5) Iterate over names
            for i, name in enumerate(annotations.keys()):
                print(f'{video_name}, {i + 1}/{num_of_names} {name}')
                name_formatted = name.replace(' ', '_')
                name_dir = path.join(conf.DATASET, name_formatted)
                file_map = get_file_map(name_dir)

                try:
                    # 6) Iterate over detections which belong to the name
                    for detection in annotations[name]['detections']:
                        detection_dict = {}

                        frame = detection['frame']
                        rect = detection['rect']

                        image_key = f'name={name_formatted}&video_date={video_date}&frame={frame}'

                        if image_key in file_map:
                            # Image already exists
                            image_path = path.join(conf.DATASET, name_formatted, file_map[image_key])
                            region = image_path.split('&region=')[1][:-4].split('_')
                            bbox = [float(x) for x in region]
                        else:
                            # Load the frame
                            im = get_frame(video, frame)

                            # 8) Get bboxes and landmarks
                            bboxes, _ = detect_faces(im)
                            bbox_i = get_bbox_i_by_IoU(bboxes, rect, threshold=0)
                            if bbox_i == -1:
                                bbox = []
                            else:
                                bbox = bboxes[bbox_i]
                                bbox = np.round(bbox, decimals=2)
                                bbox = bbox.tolist()

                        detection_dict['orig'] = rect
                        detection_dict['mtcnn'] = bbox
                        mtcnn_file.write(json.dumps(detection_dict) + '\n')

                except Exception as e:
                    print(f'An error occurred when processing image of {name}\n{e}')
