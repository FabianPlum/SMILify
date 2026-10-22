import numpy as np
import cv2
import torch
import imageio
from smal_fitter.utils import crop_to_silhouette
import os
import json
from pathlib import Path

from skimage.draw import polygon

import config


def load_SMIL_sequence(SMIL_COCO, image_name, crop_size, alt_seg=True, use_crop=False):
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
    images_dict = {entry["file_name"]: entry for entry in json_data["images"]}

    # Load the annotation data
    annotations_dict = {ann["image_id"]: ann for ann in json_data["annotations"]}

    def get_seg_from_entry(entry_seg, entry_img):
        """Given a JSON entry, returns the binary mask as a numpy array."""
        height, width = entry_img["height"], entry_img["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Process each segmentation polygon
        for segmentation in entry_seg["segmentation"]:
            coords = np.array(segmentation).reshape(-1, 2)
            rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
            mask[rr, cc] = 1

        return mask

    def get_alt_seg(entry_img):
        """Get segmentation from ID pass instead of polygon approximation from COCO dataset"""
        mask_name = entry_img["file_name"][:-9] + "ID.png"
        print(mask_name)
        mask_data = imageio.v2.imread(os.path.join(Path(img_dir).parent.parent.absolute(), "SMIL", mask_name))

        # only the red chanel contains ID data in the current convention
        mask_gray = mask_data[:, :, 0]

        return mask_gray

    def get_image_and_mask(name):
        image_entry = images_dict[name]
        annotation = annotations_dict[image_entry["id"]]

        # Load image
        img_data = imageio.v2.imread(os.path.join(img_dir, image_entry["file_name"]))

        # convert value range to 0 - 1
        img_data = img_data / 255.0

        # Load segmentation mask
        if alt_seg:
            seg_data = get_alt_seg(image_entry)
        else:
            seg_data = get_seg_from_entry(annotation, image_entry)

        return img_data, seg_data

    img_data, seg_data = get_image_and_mask(image_name)

    raw_joints = np.array(annotations_dict[images_dict[image_name]["id"]]["keypoints"]).reshape(-1, 3)
    joint_locs = raw_joints[:, :2]
    visibility = raw_joints[:, 2]

    # map joint names with correct ids regardless of order
    new_joint_locs = np.zeros((len(config.joint_names), 2), int)
    new_visibility = np.zeros((len(config.joint_names)), int)
    for o, orig_joint in enumerate(config.joint_names):
        for m, mapped_joints in enumerate(json_data["categories"][0]["keypoints"]):
            if orig_joint == mapped_joints:
                if orig_joint in config.IGNORE_JOINTS:
                    new_visibility[o] = 0  # set ignored joints to be invisible
                else:
                    new_visibility[o] = visibility[m]
                new_joint_locs[o] = [joint_locs[m][1], joint_locs[m][0]]  # flip x and y

    if config.DEBUG:
        img_copy = cv2.convertScaleAbs(img_data.copy() * 255)
        for k, (key_point, name) in enumerate(zip(new_joint_locs, config.joint_names)):
            print(k, key_point, name)
            if k in config.TORSO_JOINTS:
                img_copy = cv2.circle(
                    img_copy, (key_point[1], key_point[0]), radius=3, color=(255, 0, 255), thickness=-1
                )

        cv2.imshow("test img", cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    if use_crop:
        sil_img, rgb_img, landmarks = crop_to_silhouette(seg_data, img_data, new_joint_locs, crop_size)
    else:
        sil_img, rgb_img, landmarks = seg_data, img_data, new_joint_locs

    rgb = torch.FloatTensor(rgb_img)[None, ...].permute(0, 3, 1, 2)
    sil = torch.FloatTensor(sil_img)[None, None, ...]
    joints = torch.FloatTensor(landmarks)[:, :2].unsqueeze(0)
    visibility = torch.FloatTensor(new_visibility).unsqueeze(0)
    file_names = [os.path.basename(image_name)]

    return (rgb, sil, joints, visibility), file_names
