import json
import cv2
from os import listdir, path
from mtcnn import detect_faces

from config import Config
from data_processing.mtcnn_utils import get_bbox_i_by_IoU, frontalize_face
from utils import create_dir


# video - the opened video object
# frame_id - the frame number
def get_frame(video, frame_id):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    video.set(cv2.COLOR_BGR2GRAY, frame_id)
    # read the image from the video
    _, im = video.read()
    return im


if __name__ == '__main__':
    conf = Config()

    # 1) Get video names
    videos = listdir(conf.VIDEO_PATH)

    processed_counter, skipped_counter, not_found_counter = 0, 0, 0
    for video_name in videos:
        print(f"Processing {video_name}")

        # Used in file name
        video_date = video_name.split('_')[2]

        # 2) Load the annotations
        with open(conf.ANNOTATIONS_PATH + video_name + "_people.json", "r") as f:
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
            print(f"\t{i + 1}/{num_of_names} {name}, num. of frames processed: {processed_counter}, "
                  f"skipped: {skipped_counter}, not found: {not_found_counter},"
                  f" total: {processed_counter + skipped_counter + not_found_counter}")
            try:
                # 6) Iterate over detections which belong to the name
                for detection in annotations[name]['detections']:
                    frame = detection['frame']
                    rect = detection['rect']

                    # 7) Format names and create paths
                    name_formatted = name.replace(' ', '_')
                    image_name = f'{name_formatted}_{frame}_{video_date}.jpg'
                    dir_path = path.join(conf.DATASET, name_formatted)
                    image_path = path.join(dir_path, image_name)

                    if path.isfile(image_path):
                        skipped_counter += 1
                        continue

                    create_dir(dir_path)

                    # Load the frame
                    im = get_frame(video, frame)

                    # 8) Get bboxes and landmarks
                    bounding_boxes, landmarks = detect_faces(im)
                    bbox_i = get_bbox_i_by_IoU(bounding_boxes, rect)

                    # 9) Skip the image if no bbox was selected
                    if bbox_i == -1:
                        not_found_counter += 1
                        continue

                    processed_counter += 1
                    # 10) Frontalize
                    im = frontalize_face(im, landmarks[bbox_i])

                    # 9) Save the image
                    cv2.imwrite(image_path, im)
            except Exception as e:
                print(f"An error occurred when processing image of {name}\n{e}")