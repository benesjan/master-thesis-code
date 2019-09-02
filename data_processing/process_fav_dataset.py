import json
import sys
from os import listdir, path
import cv2

from config import Config
from data_processing.face_utils import get_aligned_face
from utils import create_dir, strip_accents


# video - the opened video object
# frame_id - the frame number
def get_frame(video, frame_id):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    video.set(cv2.COLOR_BGR2GRAY, frame_id)
    # read the image from the video
    _, im = video.read()
    return im


def get_next_image(video, annotations):
    resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    num_of_names = len(annotations.keys())
    for i, name in enumerate(annotations.keys()):
        print(f"\t{i + 1}/{num_of_names} {name}")
        try:
            for detection in annotations[name]['detections']:
                # Get the cropped image
                frame = detection['frame']
                rect = detection['rect']

                # Load the frame
                im = get_frame(video, frame)

                im = get_aligned_face(im, rect)
                if im is None:
                    print(f"Intersection of bounding boxes not found. Skipping the image\nName: {name}, Frame: {frame}")
                    continue

                yield name, frame, im
        except Exception as e:
            print(f"An error occurred when processing image of {name}\n{e}")


if __name__ == '__main__':
    conf = Config()

    if path.exists(conf.DATASET):
        print(f'Dataset folder {conf.DATASET} already exists. Aborting')
        sys.exit()

    # 2) Get video names
    videos = listdir(conf.VIDEO_PATH)

    for video_name in videos:
        print(f"Processing {video_name}")

        # Used in file name
        video_date = video_name.split('_')[2]

        # 3) Load the annotations
        with open(conf.ANNOTATIONS_PATH + video_name + "_people.json", "r") as f:
            annotations = json.load(f)

        # 4) load the video
        video = cv2.VideoCapture(conf.VIDEO_PATH + video_name)

        # 5) check if the video file opened successfully, if not continue with another one
        if not video.isOpened():
            print(f'The videofile {video_name} could not be opened!')
            continue

        for name, frame, image in get_next_image(video, annotations):
            name_formatted = strip_accents(name.replace(' ', '_'))
            image_name = f'{name_formatted}_{frame}_{video_date}.jpg'
            dir_path = path.join(conf.DATASET, name_formatted)

            # Create directory in the dataset folder if it doesn't exist
            create_dir(dir_path)

            # Save the image
            cv2.imwrite(path.join(dir_path, image_name), image)
