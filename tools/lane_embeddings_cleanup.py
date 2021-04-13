import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob
from argparse import ArgumentParser


def lane_markings_from_embeddings_cleanup(image_rgb):
    result_image = image_rgb
    hls_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    hue_layer = hls_image[:, :, 0]
    lightness_layer = hls_image[:, :, 1]
    saturation_layer = hls_image[:, :, 2]

    # Masks
    hue_mask_l1 = (15 <= hue_layer) & (hue_layer <= 35)
    hue_mask_l2 = (60 <= hue_layer) & (hue_layer <= 90)
    hue_mask_l3 = (160 <= hue_layer) & (hue_layer <= 175)
    hue_mask_l4 = (135 <= hue_layer) & (hue_layer <= 170)
    lightness_mask_l4 = (lightness_layer >= 170)
    saturation_mask = saturation_layer >= 160

    mask_lane1 = saturation_mask & hue_mask_l1
    mask_lane2 = saturation_mask & hue_mask_l2
    mask_lane3 = saturation_mask & hue_mask_l3
    mask_lane4 = saturation_mask & lightness_mask_l4  # & hue_mask_l4
    rgb_mask = mask_lane1 | mask_lane2 | mask_lane3 | mask_lane4

    result_image[:, :, :][np.logical_not(rgb_mask)] = 0
    return result_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dir", help="The directory in which 3-embedding intermediate results are saved as jpg images.")
    args = parser.parse_args()

    folder = Path(args.dir).resolve()
    image_paths = glob(str(folder) + "/*.jpg")

    for filepath in image_paths:
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(lane_markings_from_embeddings_cleanup(image_rgb))
        plt.show()
