import sys

sys.path.append('../')

import numpy as np
import cv2
import torch
import imageio
from tqdm import tqdm
from utils import crop_to_silhouette
import os
import json
from pathlib import Path

from csv import DictReader
from PIL import Image, ImageFilter
from pycocotools.mask import decode as decode_RLE
from copy import copy
from skimage.draw import polygon

import config


def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, image_range=None):
    file_names = []
    rgb_imgs = []
    sil_imgs = []
    joints = []
    visibility = []

    annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
    json_path = os.path.join(annotations_path, "{0}.json".format(sequence_name))
    with open(json_path) as json_data:
        sequence_annotation = np.array(json.load(json_data))
        if image_range is not None:
            sequence_annotation = sequence_annotation[image_range]
        for image_annotation in tqdm(sequence_annotation):
            file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
            seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

            if os.path.exists(file_name) and os.path.exists(seg_name):
                landmarks = np.array(image_annotation['joints'])[config.BADJA_ANNOTATED_CLASSES]
                visibility.append(np.array(image_annotation['visibility'])[config.BADJA_ANNOTATED_CLASSES])

                rgb_img = imageio.imread(file_name) / 255.0
                sil_img = imageio.imread(seg_name)[:, :, 0] / 255.0

                rgb_h, rgb_w, _ = rgb_img.shape
                sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

                sil_img, rgb_img, landmarks = crop_to_silhouette(sil_img, rgb_img, landmarks, crop_size)

                rgb_imgs.append(rgb_img)
                sil_imgs.append(sil_img)
                joints.append(landmarks)
                file_names.append(os.path.basename(image_annotation['image_path']))

            elif os.path.exists(file_name):
                print("BADJA SEGMENTATION file path: {0} is missing".format(seg_name))
            else:
                print("BADJA IMAGE file path: {0} is missing".format(file_name))

    rgb = torch.FloatTensor(np.stack(rgb_imgs, axis=0)).permute(0, 3, 1, 2)
    sil = torch.FloatTensor(np.stack(sil_imgs, axis=0))[:, None, :, :]
    joints = torch.FloatTensor(np.stack(joints, axis=0))
    visibility = torch.FloatTensor(np.stack(visibility, axis=0).astype(np.float))

    ## Sets invalid joints (i.e. not labelled) as invisible
    invalid_joints = np.array(config.BADJA_ANNOTATED_CLASSES) == -1
    visibility[:, invalid_joints] = 0.0

    return (rgb, sil, joints, visibility), file_names


def load_stanford_sequence(STANFORD_EXTRA, image_name, crop_size):
    # edit this to the location of the extracted StanfordDogs tar file (e.g. /.../Images).
    img_dir = os.path.join(STANFORD_EXTRA, "sample_imgs")

    # edit this to the location of the downloaded full dataset .json
    json_loc = os.path.join(STANFORD_EXTRA, "StanfordExtra_sample.json")

    # load json into memory
    with open(json_loc) as infile:
        json_data = json.load(infile)

    # convert json data to a dictionary of img_path : all_data, for easy lookup
    json_dict = {i['img_path']: i for i in json_data}

    def get_seg_from_entry(entry):
        """Given a .json entry, returns the binary mask as a numpy array"""

        rle = {
            "size": [entry['img_height'], entry['img_width']],
            "counts": entry['seg']
        }

        decoded = decode_RLE(rle)
        return decoded

    def get_dog(name):
        data = json_dict[name]

        # load img
        img_data = imageio.imread(os.path.join(img_dir, data['img_path']))

        # load seg
        seg_data = get_seg_from_entry(data)

        # add to output
        data['img_data'] = img_data
        data['seg_data'] = seg_data

        return data

    loaded_data = get_dog(image_name)

    # add an extra dummy invisble joint for tail_mid which wasn't annotated in Stanford-Extra
    raw_joints = np.concatenate([
        np.array(loaded_data['joints']), [[0.0, 0.0, 0.0]]], axis=0)

    sil_img, rgb_img, landmarks = crop_to_silhouette(
        loaded_data['seg_data'], loaded_data['img_data'] / 255.0,
        raw_joints[:, [1, 0]], crop_size)

    rgb = torch.FloatTensor(rgb_img)[None, ...].permute(0, 3, 1, 2)
    sil = torch.FloatTensor(sil_img)[None, None, ...]
    joints = torch.FloatTensor(landmarks)[:, :2].unsqueeze(0)
    visibility = torch.FloatTensor(raw_joints)[:, -1].unsqueeze(0)
    file_names = [os.path.basename(loaded_data['img_path'])]

    return (rgb, sil, joints, visibility), file_names


def load_SMIL_sequence(SMIL_COCO, image_name, crop_size,
                       alt_seg=True,
                       use_crop=False):
    """
    Loads custom configuration data and image.

    :param SMIL_COCO: Directory containing the configuration files and images.
    :param image_name: Name of the image file to load.
    :param crop_size: The size to which the image should be cropped.
    THIS PARAMETER IS FUNCTIONLESS AT THE MOMENT
    :return: A tuple containing the loaded image and associated segmentation mask.
    """
    # Define paths to images and JSON files
    img_dir = os.path.join(SMIL_COCO, "data")
    json_loc = os.path.join(SMIL_COCO, "labels.json")

    # Load JSON data into memory
    with open(json_loc) as infile:
        json_data = json.load(infile)

    # Convert JSON data to a dictionary of img_path: all_data for easy lookup
    images_dict = {entry['file_name']: entry for entry in json_data['images']}

    # Load the annotation data
    annotations_dict = {ann['image_id']: ann for ann in json_data['annotations']}

    def get_seg_from_entry(entry_seg, entry_img):
        """Given a JSON entry, returns the binary mask as a numpy array."""
        height, width = entry_img['height'], entry_img['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        # Process each segmentation polygon
        for segmentation in entry_seg['segmentation']:
            coords = np.array(segmentation).reshape(-1, 2)
            rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
            mask[rr, cc] = 1

        return mask

    def get_alt_seg(entry_img):
        """Get segmentation from ID pass instead of polygon approximation from COCO dataset"""
        mask_name = entry_img['file_name'][:-9] + "ID.png"
        print(mask_name)
        mask_data = imageio.imread(os.path.join(Path(img_dir).parent.parent.absolute(),"SMIL", mask_name))

        # only the red chanel contains ID data in the current convention
        mask_gray = mask_data[:,:,0]

        return mask_gray

    def get_image_and_mask(name):
        image_entry = images_dict[name]
        annotation = annotations_dict[image_entry['id']]

        # Load image
        img_data = imageio.imread(os.path.join(img_dir, image_entry['file_name']))

        # convert value range to 0 - 1
        img_data = img_data / 255.0

        # Load segmentation mask
        if alt_seg:
            seg_data = get_alt_seg(image_entry)
        else:
            seg_data = get_seg_from_entry(annotation, image_entry)

        return img_data, seg_data

    img_data, seg_data = get_image_and_mask(image_name)

    raw_joints = np.array(annotations_dict[images_dict[image_name]['id']]['keypoints']).reshape(-1, 3)
    joint_locs = raw_joints[:, :2]
    visibility = raw_joints[:, 2]

    print(joint_locs.shape)
    print(visibility.shape)

    # map joint names with correct ids regardless of order
    new_joint_locs = np.zeros((len(config.dd["J_names"]), 2), int)
    new_visibility = np.zeros((len(config.dd["J_names"])), int)
    for o, orig_joint in enumerate(config.dd["J_names"]):
        for m, mapped_joints in enumerate(json_data["categories"][0]["keypoints"]):
            if orig_joint == mapped_joints:
                new_joint_locs[o] = [joint_locs[m][1], joint_locs[m][0]]  # flip x and y
                new_visibility[o] = visibility[m]

    if config.DEBUG:
        img_copy = img_data.copy()
        for k, key_point in enumerate(new_joint_locs):
            if k in config.TORSO_JOINTS:
                img_copy = cv2.circle(img_copy, (key_point[1], key_point[0]),
                                      radius=3, color=(255, 0, 255), thickness=-1)

        cv2.imshow("test img", img_copy)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    if use_crop:
        sil_img, rgb_img, landmarks = crop_to_silhouette(
            seg_data, img_data,
            new_joint_locs, crop_size)
    else:
        sil_img, rgb_img, landmarks = seg_data, img_data, new_joint_locs

    rgb = torch.FloatTensor(rgb_img)[None, ...].permute(0, 3, 1, 2)
    sil = torch.FloatTensor(sil_img)[None, None, ...]
    joints = torch.FloatTensor(landmarks)[:, :2].unsqueeze(0)
    visibility = torch.FloatTensor(new_visibility).unsqueeze(0)
    file_names = [os.path.basename(image_name)]

    return (rgb, sil, joints, visibility), file_names
