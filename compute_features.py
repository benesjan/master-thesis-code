from os import listdir, path

import cv2
import h5py
import numpy as np
import torch
from torch.nn import DataParallel

from config import Config
from models.resnet import resnet_face18


# Process image so that it can be fed to NN
def process_face(im):
    # The following lines are copied from arcface-pytorch project
    # Stack image and it's flipped version. Output dimensions: (128, 128, 2)
    im = np.dstack((im, np.fliplr(im)))
    # Transpose. Output dimensions: (2, 128, 128)
    im = im.transpose((2, 0, 1))
    # Add dimension. Output dimensions: (2, 1, 128, 128)
    im = im[:, np.newaxis, :, :]
    im = im.astype(np.float32, copy=False)
    # Normalize to <-1, 1>
    im -= 127.5
    im /= 127.5
    return im


# Get features for 1 batch
def predict(model, images):
    data = torch.from_numpy(images)
    data = data.to(conf.DEVICE)
    output = model(data)
    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    features = np.hstack((fe_1, fe_2))

    return features


if __name__ == '__main__':
    conf = Config()

    batch_size = 100

    # 1) Load model
    model = resnet_face18(False)
    model = DataParallel(model)

    model.load_state_dict(torch.load(conf.MODEL_PATH, map_location=conf.DEVICE))
    model.to(conf.DEVICE)

    model.eval()

    # 2) Get video names
    names = listdir(conf.DATASET)
    names_len = len(names)

    # 3) Open the h5 file
    with h5py.File(conf.FEATURES, 'w') as h5f:
        try:
            labels, images, features = [], [], []
            for label, name in enumerate(names):
                print(f'{label + 1}/{names_len} - {name}')

                for image_name in listdir(path.join(conf.DATASET, name)):
                    image = cv2.imread(path.join(conf.DATASET, name, image_name), cv2.IMREAD_GRAYSCALE)
                    processed_image = process_face(image)

                    labels.append(label)
                    images.append(processed_image)

                if len(images) > batch_size:
                    print('Computing the features')
                    images_array = np.vstack(images)
                    feature_batch = predict(model, images_array)
                    features.append(feature_batch)
                    images = []

            features = np.vstack(features)

            # 7) save the features and names
            h5f.create_dataset('features', data=features)
            h5f.create_dataset('labels', data=labels)
            h5f.flush()

        except KeyboardInterrupt as e:
            print('KeyboardInterrupt has been caught. Exiting...')
            h5f.flush()
