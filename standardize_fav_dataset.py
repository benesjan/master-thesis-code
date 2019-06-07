import json
import sys
from os import listdir, path
import cv2

from config import Config
from utils import create_dir


def get_square(rect, im_res):
    selection_height = rect[3] - rect[1]
    # make sure the selection height is even
    if selection_height & 1:
        selection_height -= 1
        rect[3] -= 1
    width_middle = (rect[0] + rect[2]) / 2
    left = int(width_middle - (selection_height / 2))
    right = int(width_middle + (selection_height / 2))
    # make sure the selection is within the image boundaries
    if left < 0:
        left = 0
        right = selection_height
    elif right > im_res[0]:
        left = im_res[0] - selection_height
        right = im_res[0]
    return left, rect[1], right, rect[3]


# returns an image with selected face
# video - the opened video object
# frame - the frame number
# rect - the rectangle of the face
def get_face(video, frame, rect):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    video.set(cv2.COLOR_BGR2GRAY, frame)
    # read the image from the video
    ret, im = video.read()

    square = get_square(rect, im.shape)

    return im[square[1]:square[3], square[0]:square[2], :]


def move_selection(selection, resolution):
    # selection = [x1, y1, x2, y2], resolution= (width, height)
    if selection[0] < 0:
        selection[2] -= selection[0]
        selection[0] = 0
    elif selection[2] > resolution[0]:
        selection[0] -= (selection[2] - resolution[0])
        selection[2] = resolution[0]
    if selection[1] < 0:
        selection[3] -= selection[1]
        selection[1] = 0
    elif selection[3] > resolution[1]:
        selection[1] -= (selection[3] - resolution[1])
        selection[3] = resolution[1]
    return selection


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

                # Check if the selection crosses the image border
                if rect[0] < 0 or rect[2] > resolution[0] or rect[1] < 0 or rect[3] > resolution[1]:
                    if conf.MOVE_SELECTION:
                        rect = move_selection(rect, resolution)
                    else:
                        # Ignore the frame
                        continue

                im = get_face(video, frame, rect)
                im_rsz = cv2.resize(im, (128, 128))

                yield name, frame, im_rsz
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
            name_formated = name.replace(' ', '_')
            image_name = f'{name_formated}_{frame}.jpg'
            dir_path = path.join(conf.DATASET, name_formated)

            # Create directory in the dataset folder if it doesn't exist
            create_dir(dir_path)

            # Save the image
            cv2.imwrite(path.join(dir_path, image_name), image)