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

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_process_images')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir_path', type=str, help='The images directory path')
    parser.add_argument('weights_path', type=str, help='The model weights path')
    parser.add_argument('--debug', action="store_true", help='Display debugging output')

    return parser.parse_args()


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
                        figure, axs = plt.subplots(2,2)
                        axs[0, 0].imshow(embeddings_as_separate_images[0])
                        axs[0, 0].set_title("Lane segmentation image for lane {lane}/{count}, grayscale".format(
                                                                                lane=1,
                                                                                count=CFG.MODEL.EMBEDDING_FEATS_DIMS))
                        axs[0, 1].imshow(embeddings_as_separate_images[1])
                        axs[0, 1].set_title("Lane segmentation image for lane {lane}/{count}, grayscale".format(
                                                                                lane=2,
                                                                                count=CFG.MODEL.EMBEDDING_FEATS_DIMS))
                        axs[1, 0].imshow(embeddings_as_separate_images[2])
                        axs[1, 0].set_title("Lane segmentation image for lane {lane}/{count}, grayscale".format(
                                                                                lane=3,
                                                                                count=CFG.MODEL.EMBEDDING_FEATS_DIMS))
                        axs[1, 1].imshow(embeddings_as_separate_images[3])
                        axs[1, 1].set_title("Lane segmentation image for lane {lane}/{count}, grayscale".format(
                                                                                lane=4,
                                                                                count=CFG.MODEL.EMBEDDING_FEATS_DIMS))
                        plt.show()

                    # TODO: Make sure you add sufficient padding around the image so that the dimensions fit!
                    binary_color = np.dstack((binary_seg_images[nn_res_index] * 255,
                                              binary_seg_images[nn_res_index] * 255,
                                              binary_seg_images[nn_res_index] * 255)).astype(np.uint8)
                    scaled_binary_output = cv2.resize(binary_color, (batch_images_vis[nn_res_index].shape[1],
                                                                     batch_images_vis[nn_res_index].shape[0]),
                                                      interpolation=cv2.INTER_LINEAR)

                    if debug and current_image_filepath is not "":
                        plt.title("Binary segmentation image")
                        plt.imshow(binary_color)
                        plt.show()

                    # Rescale the masks and plot on top of the input images
                    result_image_vis = batch_images_originals[nn_res_index]
                    binary_detection_overlay = cv2.addWeighted(batch_images_vis[nn_res_index], 0.75,
                                                               scaled_binary_output, 0.25, 0)
                    result_image_vis[ver_start_crop:ver_end_crop,
                                     hor_start_crop:hor_end_crop] = binary_detection_overlay
                    result_image_vis = cv2.rectangle(result_image_vis, (hor_start_crop, ver_start_crop),
                                                     (hor_end_crop, ver_end_crop),
                                                     (255, 255, 255), 3, cv2.LINE_4)
                    scale_factor = 1.25
                    scaled_embeddings_output = cv2.resize(embedding_image[:, :, (0, 1, 2)],
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
                        resolved_image_filepath = Path(current_image_filepath).resolve()

                        binary_seg_image_filename = resolved_image_filepath.stem + \
                                                   "_binary_mask{suffix}".format(suffix=resolved_image_filepath.suffix)
                        saved_path = resolved_image_filepath.parent/"binary_segmentation"/binary_seg_image_filename
                        if not ops.exists(saved_path.parent):
                            os.mkdir(saved_path.parent)
                        cv2.imwrite(str(saved_path), binary_color)
                        # Save embeddings as separate images
                        for current_embedding_no in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                            embedding_image_filename = resolved_image_filepath.stem + \
                                                       "_embedding_{i}{suffix}".format(i=current_embedding_no,
                                                                                        suffix=resolved_image_filepath.suffix)
                            saved_path = resolved_image_filepath.parent/"lane_segmentation"/embedding_image_filename
                            if not ops.exists(saved_path.parent):
                                os.mkdir(saved_path.parent)
                            cv2.imwrite(str(saved_path), embeddings_as_separate_images[current_embedding_no])
                        # Save visualization image
                        visualization_image_filename = resolved_image_filepath.stem + \
                                                   "_visualized{suffix}".format(suffix=resolved_image_filepath.suffix)
                        saved_path = resolved_image_filepath.parent/"visualized"/visualization_image_filename
                        if not ops.exists(saved_path.parent):
                            os.mkdir(saved_path.parent)
                        cv2.imwrite(str(saved_path), result_image_vis)
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
