#!/usr/bin/env python3
# Run inference using the LaneNet model on a folder of images.
import argparse
import os
import os.path as ops
import time
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

from lane_embeddings_cleanup import lane_markings_from_embeddings_cleanup

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_process_images')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir_path', type=str, help='The images directory path')
    parser.add_argument('weights_path', type=str, help='The model weights path')
    parser.add_argument('--debug', action="store_true", help='Display debugging output')

    return parser.parse_args()


def plot_grid_images(content, rows=1, cols=1, titles="Element {item}/{count}"):
    elem_count = len(content)
    if elem_count > 1 and rows == cols == 1:
        rows = 2 + elem_count % 2
        cols = elem_count // 2
    figure, axs = plt.subplots(rows, cols)
    for i in range(elem_count):
        row = i // cols
        col = (i - cols * row)
        axs[row, col].imshow(content[i])
        axs[row, col].set_title(titles.format(item=i + 1, count=elem_count))
    plt.show()


def save_image_with_suffix_in_folder(image,
                                     image_filepath: str, subfolder_paths: str = "",
                                     image_filename_suffix: str = "",
                                     string_mappings: dict = dict(),
                                     ):
    resolved_image_filepath = Path(image_filepath).resolve()
    image_filename = resolved_image_filepath.stem + image_filename_suffix.format(**string_mappings) + \
        resolved_image_filepath.suffix
    saved_path = resolved_image_filepath.parent / subfolder_paths / image_filename
    if not ops.exists(saved_path.parent):
        os.mkdir(saved_path.parent)
    cv2.imwrite(str(saved_path), image)


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr


def prepare_frame(image):
    horizontal_centerpoint = image.shape[1] // 2
    vertical_centerpoint = image.shape[0] // 2
    height_cropped = 1100  # Experimentally confirmed
    vertical_bias = 150
    width_cropped = height_cropped * 2
    cutout_side_len = horizontal_centerpoint - width_cropped // 2
    hor_start_crop = cutout_side_len
    hor_end_crop = image.shape[1] - cutout_side_len
    ver_start_crop = vertical_centerpoint - height_cropped // 2 + vertical_bias
    ver_end_crop = height_cropped + ver_start_crop

    return image[ver_start_crop:ver_end_crop, hor_start_crop:hor_end_crop]


def run_lanenet(images_folder, weights_path, debug=False):
    """
    :param images_folder: The path to the directory containing images to be processed.
    :param weights_path: The neural network model weights (checkpoint) location.
    :param debug: Display debugging information and images.
    """
    images_folder = Path(images_folder).resolve()
    assert ops.exists(images_folder), '{:s} not exist'.format(images_folder)

    LOG.info('Start video ingest and preprocessing')
    t_start = time.time()

    images_paths = glob("{folder}/*.jpg".format(folder=images_folder))
    inference_batch_size = 10  # default: 1

    # TODO: REFACTOR!
    image_path = images_paths[0]
    image_for_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
    horizontal_centerpoint = image_for_data.shape[1] // 2
    vertical_centerpoint = image_for_data.shape[0] // 2
    height_cropped = 1100  # Experimentally confirmed
    vertical_bias = 150
    width_cropped = height_cropped * 2
    cutout_side_len = horizontal_centerpoint - width_cropped // 2
    hor_start_crop = cutout_side_len
    hor_end_crop = image_for_data.shape[1] - cutout_side_len
    ver_start_crop = vertical_centerpoint - height_cropped // 2 + vertical_bias
    ver_end_crop = height_cropped + ver_start_crop
    # TODO: END REFACTOR! ##########################################

    if debug:
        LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[inference_batch_size, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()

        # crop input images in a 'TuSimple fashion' (no hood in the image)
        # input images are 512x256 (2:1), so I try to make use of that aspect ratio
        while len(images_paths) > 0:
            try:
                # IMAGE PREPROCESSING START #################################################
                batch_images_for_nn = list()
                batch_images_originals = list()
                batch_images_vis = list()
                for i in range(inference_batch_size):
                    if i < len(images_paths):
                        image_original = cv2.imread(images_paths[i], cv2.IMREAD_COLOR)
                        image_vis = prepare_frame(image_original)
                        image_for_nn = cv2.resize(image_vis, (512, 256), interpolation=cv2.INTER_AREA)
                        image_for_nn = image_for_nn / 127.5 - 1.0
                    else:  # Batch padding to ensure proper calls to the NN
                        image_original = np.zeros(batch_images_originals[-1].shape, dtype=np.uint8)
                        image_vis = np.zeros(batch_images_vis[-1].shape, dtype=np.uint8)
                        image_for_nn = np.zeros(batch_images_for_nn[-1].shape, dtype=np.float64)

                    # Bundle images in batches (30 images per batch for example)
                    batch_images_for_nn.append(image_for_nn)
                    batch_images_originals.append(image_original)
                    batch_images_vis.append(image_vis)

                # IMAGE PREPROCESSING END #################################################

                binary_seg_images, instance_seg_images = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: batch_images_for_nn}
                )

                # NN RESULT POSTPROCESSING START ##############################################
                for nn_res_index in range(binary_seg_images.shape[0]):
                    if nn_res_index < len(images_paths):
                        current_image_filepath = images_paths[nn_res_index]
                    else:
                        current_image_filepath = ""

                    for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                        instance_seg_images[nn_res_index][:, :, i] = minmax_scale(
                            instance_seg_images[nn_res_index][:, :, i])
                    embedding_image = np.array(instance_seg_images[nn_res_index], np.uint8)

                    # TODO: Let users choose whether the output resolution is kept as-is
                    # or are upscaled to match the input images

                    # Get each embedding layer in its own image/mask
                    embeddings_as_separate_images = list()
                    embeddings_as_separate_images_scaled_to_input_dims = list()
                    for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                        laneline_embedding = np.dstack((
                            instance_seg_images[nn_res_index][:, :, i],
                            instance_seg_images[nn_res_index][:, :, i],
                            instance_seg_images[nn_res_index][:, :, i]
                        )).astype(np.uint8)
                        embeddings_as_separate_images.append(laneline_embedding)

                        # TODO: Make sure you add sufficient padding around the image so that the dimensions fit!
                        embeddings_as_separate_images_scaled_to_input_dims.append(
                            cv2.resize(laneline_embedding, (batch_images_vis[nn_res_index].shape[1],
                                                            batch_images_vis[nn_res_index].shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                        )
                    if debug and current_image_filepath is not "":
                        plot_grid_images(content=embeddings_as_separate_images, rows=2, cols=2,
                                         titles="Lane segmentation image for lane {item}/{count}, grayscale")

                        embedding_as_triplets_bgr = [
                            np.dstack((
                                instance_seg_images[nn_res_index][:, :, 0],
                                instance_seg_images[nn_res_index][:, :, 1],
                                instance_seg_images[nn_res_index][:, :, 2]
                            )).astype(np.uint8),
                            np.dstack((
                                instance_seg_images[nn_res_index][:, :, 0],
                                instance_seg_images[nn_res_index][:, :, 1],
                                instance_seg_images[nn_res_index][:, :, 3],
                            )).astype(np.uint8),
                            np.dstack((
                                instance_seg_images[nn_res_index][:, :, 0],
                                instance_seg_images[nn_res_index][:, :, 2],
                                instance_seg_images[nn_res_index][:, :, 3],
                            )).astype(np.uint8),
                            np.dstack((
                                instance_seg_images[nn_res_index][:, :, 1],
                                instance_seg_images[nn_res_index][:, :, 2],
                                instance_seg_images[nn_res_index][:, :, 3],
                            )).astype(np.uint8),
                        ]
                        for item in range(len(embedding_as_triplets_bgr)):
                            save_image_with_suffix_in_folder(embedding_as_triplets_bgr[item],
                                                             image_filepath=current_image_filepath,
                                                             subfolder_paths="debug_embeddings_rgb",
                                                             image_filename_suffix="_embedding_{i}",
                                                             string_mappings={"i": item}
                                                             )
                        plot_grid_images(content=embedding_as_triplets_bgr,
                                         titles="Pairs of embeddings as RGB, pair {item}/{count}")

                    instance_segmentation_fused_output_rgb = np.dstack((
                        instance_seg_images[nn_res_index][:, :, 2],
                        instance_seg_images[nn_res_index][:, :, 1],
                        instance_seg_images[nn_res_index][:, :, 0]
                    )).astype(np.uint8)

                    binary_color = np.dstack((binary_seg_images[nn_res_index] * 255,
                                              binary_seg_images[nn_res_index] * 255,
                                              binary_seg_images[nn_res_index] * 255)).astype(np.uint8)
                    instance_segmentation_color_for_visualization = cv2.cvtColor(
                        lane_markings_from_embeddings_cleanup(instance_segmentation_fused_output_rgb),
                        cv2.COLOR_RGB2BGR)
                    scaled_output_for_visualization = cv2.resize(instance_segmentation_color_for_visualization,
                                                                 (batch_images_vis[nn_res_index].shape[1],
                                                                  batch_images_vis[nn_res_index].shape[0]),
                                                                 interpolation=cv2.INTER_LINEAR)
                    scaled_binary_output_for_visualization = cv2.resize(binary_color,
                                                                        (batch_images_vis[nn_res_index].shape[1],
                                                                         batch_images_vis[nn_res_index].shape[0]),
                                                                        interpolation=cv2.INTER_LINEAR)
                    binary_additive_mask_1 = scaled_output_for_visualization > 0
                    # binary_additive_mask_1 = scaled_output_for_visualization[:, :, 0] > 0
                    # binary_additive_mask_2 = scaled_output_for_visualization[:, :, 1] > 0
                    # binary_additive_mask_3 = scaled_output_for_visualization[:, :, 2] > 0
                    binary_additive_mask_4 = scaled_binary_output_for_visualization > 0
                    binary_additive_mask = np.logical_not(binary_additive_mask_1) & \
                                           (scaled_binary_output_for_visualization > 0)
                    scaled_output_for_visualization[binary_additive_mask] = 255
                    if debug and current_image_filepath is not "":
                        plt.title("Binary segmentation image")
                        plt.imshow(binary_color)
                        plt.show()

                    # Rescale the masks and plot on top of the input images
                    result_image_vis = batch_images_originals[nn_res_index]
                    binary_detection_overlay = cv2.addWeighted(batch_images_vis[nn_res_index], 0.75,
                                                               scaled_output_for_visualization, 0.25, 0)
                    result_image_vis[ver_start_crop:ver_end_crop,
                                     hor_start_crop:hor_end_crop] = binary_detection_overlay
                    result_image_vis = cv2.rectangle(result_image_vis, (hor_start_crop, ver_start_crop),
                                                     (hor_end_crop, ver_end_crop),
                                                     (255, 255, 255), 3, cv2.LINE_4)
                    scale_factor = 1.5
                    scaled_embeddings_output = cv2.resize(instance_segmentation_color_for_visualization,
                                                          (int(embedding_image.shape[1] * scale_factor),
                                                           int(embedding_image.shape[0] * scale_factor)),
                                                          interpolation=cv2.INTER_LINEAR)
                    scaled_binary_color_output = cv2.resize(binary_color, (int(binary_color.shape[1] * scale_factor),
                                                                           int(binary_color.shape[0] * scale_factor)),
                                                            interpolation=cv2.INTER_LINEAR)
                    result_image_vis[0:scaled_embeddings_output.shape[0],
                                     0:scaled_embeddings_output.shape[1]] = scaled_embeddings_output
                    result_image_vis[scaled_embeddings_output.shape[0]:scaled_embeddings_output.shape[0] +
                                     scaled_binary_color_output.shape[0],
                                     0:scaled_binary_color_output.shape[1]] = scaled_binary_color_output

                    if debug and current_image_filepath is not "":
                        plt.title("Resulting image, visualized")
                        plt.imshow(result_image_vis[:, :, (2, 1, 0)])  # Lazy BGR2RGB
                        plt.show()

                    # Create and save the input image - mind the batch padding!
                    if current_image_filepath is not "":
                        save_image_with_suffix_in_folder(binary_color,
                                                         image_filepath=current_image_filepath,
                                                         subfolder_paths="binary_segmentation",
                                                         image_filename_suffix="_binary_mask"
                                                         )
                        # Save embeddings as separate images
                        if debug:
                            for current_embedding_no in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                                save_image_with_suffix_in_folder(embeddings_as_separate_images[current_embedding_no],
                                                                 image_filepath=current_image_filepath,
                                                                 subfolder_paths="lane_segmentation",
                                                                 image_filename_suffix="_embedding_{i}",
                                                                 string_mappings={"i": current_embedding_no}
                                                                 )
                        save_image_with_suffix_in_folder(instance_segmentation_color_for_visualization,
                                                         image_filepath=current_image_filepath,
                                                         subfolder_paths="lane_segmentation",
                                                         image_filename_suffix="_lanes_instance_segmentation_mask"
                                                         )
                        # Save visualization image
                        save_image_with_suffix_in_folder(result_image_vis,
                                                         image_filepath=current_image_filepath,
                                                         subfolder_paths="visualized",
                                                         image_filename_suffix="_visualized"
                                                         )
                # END NN POSTPROCESSING LOOP ######################################
                if len(images_paths) > inference_batch_size:
                    images_paths = images_paths[inference_batch_size:]
                    print("{count} images left to process...".format(count=len(images_paths)))
                else:
                    images_paths = []
            except Exception as e:
                print("Error processing image!")
                print(e)
    t_cost = time.time() - t_start
    LOG.info('Images processing time: {:.5f}s'.format(t_cost))
    sess.close()


if __name__ == '__main__':
    args = init_args()
    run_lanenet(args.image_dir_path, args.weights_path, debug=args.debug)
