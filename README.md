## Introduction

A facial recognition pipeline implemented in Python and PyTorch as a part of my
[master thesis](https://github.com/benesjan/master-thesis).

## Setup

1. Install the dependencies with the following command:
    ```bash
    pip install -r requirements.txt 
    ```
2. Edit paths in the config.py file. Make sure corresponding directories exist
(if it's not the case results of computations might not be saved)

## Execution

It's important to note that all these scripts have to be executed in the same order.
In case the dataset is already in the desired format of *dataset/name/image_128x128.jpg*
the first step can be skipped.

1. If the dataset is unprocessed (raw video files and json files with) add the project root to the python 
path
    ```bash
    export PYTHONPATH="${PYTHONPATH}:/path/to/project/root"
    ```
    and run the process_dataset.py script:
    ```bash
    python data_processing/process_dataset.py
    ```
   The names of json files are expected to be in the following format:
   ```bash
    ${video_name}_people.json
    ```
   and the content itself has to have the following properties:
    ```json
    {
    "Name": {
        "detections": [
            {
                "frame": 28877, 
                "rect": [
                    495, 
                    118, 
                    834, 
                    494
                ]
            } 
        ]
     }   
   }
    ```
   If you want to grab every Nth frame out of the video add N{number} at the end of the dataset name. For example:
    ```
    dataset_name_N10
    ```

2. Once the dataset is ready we can compute the feature vectors:
    ```bash
    python compute_features.py
    ```
3. To compute the true positives, true negatives, false positives and false negatives for thresholds in the range
<0, 2> with step of size 0.005 run:
    ```bash
    python find_threshold.py
    ```