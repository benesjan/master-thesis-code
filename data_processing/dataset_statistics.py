import json
from os import listdir
from os.path import join

from config import Config

if __name__ == '__main__':
    conf = Config()

    detections_count, identities_count = 0, 0
    for video_name in listdir(conf.VIDEO_PATH):
        with open(conf.ANNOTATIONS_PATH + video_name + '_people.json', 'r') as f:
            annotations = json.load(f)

        num_of_names = len(annotations.keys())
        identities_count += num_of_names
        for i, name in enumerate(annotations.keys()):
            detections_count += len(annotations[name]['detections'])

    print(f'Original dataset:\nNumber of identities: {identities_count}\nNumber of detections: {detections_count}')

    # Processed dataset
    names = listdir(conf.DATASET)
    names_len = len(names)
    names.sort()

    detections_count, identities_count = 0, 0
    for label, name in enumerate(names):
        identities_count += 1
        detections_count += len(listdir(join(conf.DATASET, name)))

    print(f'Processed dataset:\nNumber of identities: {identities_count}\nNumber of detections: {detections_count}')
