import cv2
import json
from os import listdir

video_path = "/media/honza/My Passport/Faces/videos/"
annotations_path = "/media/honza/My Passport/Faces/CEMI-annotations-Udalosti/"


# returns an image with selected face
# video - the opened video object
# frame - the frame number
# rect - the rectangle of the face
def getFace(video, frame, rect, border=0.0):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    # read the image from the video
    ret, im = video.read()

    # obtain the crop from the image
    if ret != False:
        im_ret = im[rect[1]:rect[3], rect[0]:rect[2], :]

    # return the cropped image
    return im_ret


if __name__ == "__main__":
    # 1) Get video names
    videos = listdir(video_path)

    for video_name in videos:
        # 2) Load the annotations
        with open(annotations_path + video_name + "_people.json", "r") as f:
            data = json.load(f)

        # 3) load the video
        video = cv2.VideoCapture(video_path + video_name)

        # 4) check if the video file opened successfully, if not continue with another one
        if not video.isOpened():
            print(f'The videofile {video_name} could not be opened!')
            continue

        for name in data.keys():
            # 5) Get the cropped image
            frame = data[name]['detections'][0]['frame']
            rect = data[name]['detections'][0]['rect']

            im = getFace(video, frame, rect)

            print(im)

        break
