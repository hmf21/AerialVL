
import numpy as np
import torch
import cv2
import model.local_matcher.superpoint as superpoint
import torch.nn.functional as F


def dense_feature_PV(warp_destine_image, query_image):
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
    d_image = cv2.cvtColor(warp_destine_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # resize the image  if necessary
    q_image = cv2.resize(q_image, (500, 500), interpolation = interp)
    # q_image_color = cv2.resize(q_image_color, (640, 480), interpolation=interp)
    q_image = (q_image.astype('float32') / 255.)
    q_fm = fe.extract_feature_map(q_image)
    d_fm = fe.extract_feature_map(d_image)

    # similarity_map = F.cosine_similarity(q_fm, d_fm, dim = 1)
    # similarity = ((similarity_map > 0.5) * (similarity_map < 0.9)).sum().item()
    similarity = torch.sqrt(torch.sum(torch.pow(q_fm - d_fm, 2))).item()

    return similarity