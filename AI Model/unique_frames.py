import os
from model import CnnEncoder
import cv2
from torch.nn import CosineSimilarity
import numpy as np
import torchvision.transforms as transform


def extract_unique_frames(path_in, embed_size, hidden_size, threshold, path_out, temp_path=None):  # func to convert video to frames
    vid = cv2.VideoCapture(path_in)

    count = 0
    feature_previous = np.zeros((2, 2))
    while vid.isOpened():
        success, image = vid.read()

        while success:
            if temp_path is not None:
                cv2.imwrite(os.path.join(temp_path, f'frame{count}.jpg'), image)

            feature = get_feature(CnnEncoder(embed_size, hidden_size), image)
            if count == 0:
                feature_previous = feature
                simi_score = 0.
            else:
                simi_score = get_similarity(feature_previous, feature)

            if float(simi_score) < threshold:

                cv2.imwrite(os.path.join(path_out, f'frame{count}.jpg'), image)

            success, image = vid.read()
            count += 1

        vid.release()


def get_feature(pretrained_model, image_name):  # get frame representation
    transforms = transform.Compose(
        [
            transform.ToPILImage(),
            transform.Resize([256, 256]),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    image = transforms(image_name).float()
    image = image.unsqueeze(0)

    feature = pretrained_model(image)

    return feature


def get_similarity(feature1, feature2):  # get frame similarity
    cos = CosineSimilarity(dim=-1, eps=1e-6)
    output = cos(feature1, feature2)

    return output
