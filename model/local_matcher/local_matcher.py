
import numpy as np
import torch
import cv2
import model.local_matcher.superpoint as superpoint


def detectAndcompute(image, matcher_method='superpoint',):
    if matcher_method == 'superpoint':
        return des_spp_compute(image)
    elif matcher_method == 'sift':
        return des_sift_compute(image)
    else:
        raise ValueError('unknown matcher descriptor')


def des_spp_compute(query_image):
    interp = cv2.INTER_AREA
    weights_path = './model/local_matcher/superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = True
    fe = superpoint.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda)

    # extract feature point and compute descriptors
    q_image_path = query_image
    q_image = cv2.imread(q_image_path, 0)
    # for aerial dataset, the query images need to rotate from 90 degrees to match with the database
    # -1 denotes counter-clockwise
    # q_image = np.clip(np.rot90(q_image, -1), 0, 255).astype(np.uint8)

    # resize the image  if necessary
    q_image = cv2.resize(q_image, (500, 500), interpolation=interp)
    # q_image_color = cv2.resize(q_image_color, (640, 480), interpolation=interp)
    q_image = (q_image.astype('float32') / 255.)
    q_pts, q_des, _ = fe.run(q_image)
    q_des = np.swapaxes(q_des, 0, 1)
    q_kpts = []
    q_pts = np.swapaxes(q_pts, 0, 1)
    for keypoint in q_pts:
        q_kpts.append(cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1))

    return q_kpts, q_des


def des_sift_compute(query_image):
    sift = cv2.SIFT_create()

    q_image_path = query_image
    q_image = cv2.imread(q_image_path, 0)
    q_image = (q_image.astype('uint8'))

    q_kpts, q_des = sift.detectAndCompute(q_image, None)

    return q_kpts, q_des