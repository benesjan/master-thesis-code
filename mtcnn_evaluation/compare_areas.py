"""This script computes the ratio of area selected by MTCNN detector with those selected by humans"""
import json

from config import Config
from data_processing.mtcnn_utils import bb_intersection_over_union


def compute_area(bbox: list) -> float:
    """A function which returns area size"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


if __name__ == '__main__':
    conf = Config()

    human_area_sum, detector_area_sum = 0, 0

    with open(conf.MTCNN_PREDS, 'r') as mtcnn_file:
        for line in mtcnn_file:
            boxes = json.loads(line)
            if len(boxes['mtcnn']) != 0:
                IoU = bb_intersection_over_union(boxes['mtcnn'], boxes['orig'])
                if IoU > 0.5:
                    human_area_sum += compute_area(boxes['orig'])
                    detector_area_sum += compute_area(boxes['mtcnn'])

    print('Ratio of average area size selected by human to that selected by detector: '
          f'{human_area_sum / detector_area_sum}')
