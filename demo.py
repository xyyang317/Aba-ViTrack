import glob
import os
import time
from typing import List

import numpy as np
import torch
import cv2 as cv
from Tracker import Tracker
from model.AbaViTrack import AbaViTrack
from model.head import CenterPredictor
from model.AbaViT import abavit_patch16_224


def build_box_head(in_channel, out_channel, search_size, stride):
    feat_sz = search_size / stride
    center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                  feat_sz=feat_sz, stride=stride)
    return center_head


def build_model():
    search_size = 256
    stride = 16
    backbone = abavit_patch16_224()
    box_head = build_box_head(backbone.embed_dim, 256, search_size, stride)

    model = AbaViTrack(
        backbone,
        box_head
    )

    return model


def read_image(image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    else:
        raise ValueError("type of image_file should be str or list")


def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')


def save_time(file, data):
    exec_times = np.array(data).astype(float)
    np.savetxt(file, exec_times, delimiter='\t', fmt='%f')


def main(
        seq: List = [],
        initial_bbox: List[int] = [499, 421, 102, 179],
        output_path: str = "outputs",
        bbox_file='bbox.txt',
        time_file='time.txt',
        weights_path: str = "checkpoints/ckpt.pth",
):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

    tracker = Tracker(model)

    pred_box = []
    times = []

    image = read_image(seq[0])
    pred_box.append(initial_bbox)
    tracker.initialize(image, initial_bbox)
    for frame_num, frame_path in enumerate(seq[1:], start=1):
        image = read_image(frame_path)
        start_time = time.time()
        out = tracker.track(image)
        times.append(time.time() - start_time)
        pred_box.append(out)

    if not os.path.isdir(output_path):
        os.mkdir(output_path, mode=0o777)

    bbox_file = os.path.join(output_path, bbox_file)
    time_file = os.path.join(output_path, time_file)
    save_bb(bbox_file, pred_box)
    save_time(time_file, times)


if __name__ == '__main__':
    seq = sorted(glob.glob(os.path.join('frames', '*.jpg')))
    initial_bbox = [410, 106, 483, 613]

    main(seq, initial_bbox)
