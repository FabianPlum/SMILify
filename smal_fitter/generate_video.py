
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import cv2
import argparse

import matplotlib.pyplot as plt
from smal_fitter import SMALFitter

import torch
import imageio

from data_loader import load_badja_sequence, load_data_from_npz
import time

import pickle as pkl

class ImageExporter():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        imageio.imsave(os.path.join(self.output_dir, "{0:04}.png".format(global_id)), collage_np)

        with open(os.path.join(self.output_dir, "{0:04}.pkl".format(global_id)), 'wb') as f:
            pkl.dump(img_parameters, f)

def main():
    BADJA_PATH = "smal_fitter/BADJA"
    SHAPE_FAMILY = [1]
    CHECKPOINT_NAME = "cosker-maggie"
    # CHECKPOINT_NAME = "20190531-174847"
    EPOCH_NAME = "st10_ep0"
    # EPOCH_NAME = "st0_ep10"
    OUTPUT_DIR = os.path.join("smal_fitter", "exported", CHECKPOINT_NAME, EPOCH_NAME)
    WINDOW_SIZE = 5
    CROP_SIZE = 256
    GPU_IDS = "0"

    INPUT_PATH = "/data/cvfs/bjb56/data/smal_data/smal_joints/hg/24_04/prediction/"
    CLEANED_NAME = "20190522-140530_rocky_rl6_pop256" 


    image_exporter = ImageExporter(OUTPUT_DIR)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    # data, filenames = load_badja_sequence(BADJA_PATH, "rs_dog", CROP_SIZE)
    data, filenames = load_data_from_npz(os.path.join(INPUT_PATH, CHECKPOINT_NAME, "cleaned_skeleton", CLEANED_NAME))

    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    plt.figure()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = SMALFitter(data, WINDOW_SIZE, SHAPE_FAMILY)
    model.load_checkpoint(os.path.join("smal_fitter", "checkpoints", CHECKPOINT_NAME), EPOCH_NAME)
    model.generate_visualization(image_exporter) # Final stage

if __name__ == '__main__':
    main()