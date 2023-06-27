
import numpy as np
import torch
import cv2


def torch_nn(x, y):
    mul = torch.matmul(x, y)

    dist = 2 - 2 * mul + 1e-9
    dist = torch.sqrt(dist)

    _, fw_inds = torch.min(dist, 0)
    bw_inds = torch.argmin(dist, 1)

    return fw_inds, bw_inds


def calc_receptive_boxes(height, width):
    """Calculate receptive boxes for each feature point.
    Modified from
    https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/delf/delf/python/feature_extractor.py

    Args:
      height: The height of feature map.
      width: The width of feature map.
      rf: The receptive field size.
      stride: The effective stride between two adjacent feature points.
      padding: The effective padding size.

    Returns:
      rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
      Each box is represented by [xmin, ymin, xmax, ymax].
    """

    rf, stride, padding = [196.0, 16.0, 90.0]  # hardcoded for vgg-16 conv5_3

    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    coordinates = torch.reshape(torch.stack([y, x], dim=2), [-1, 2])
    # [y,x,y,x]
    point_boxes = torch.cat([coordinates, coordinates], 1)
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    rf_boxes = stride * point_boxes + torch.FloatTensor(bias)
    return rf_boxes


def calc_keypoint_centers_from_patches(patch_size_h, patch_size_w, stride_h, stride_w):
    '''Calculate patch positions in image space

    Args:
        patch_size_h: height of patches
        patch_size_w: width of patches
        stride_h: stride in vertical direction between patches
        stride_w: stride in horizontal direction between patches

    :returns
        keypoints: patch positions back in image space for RANSAC
        indices: 2-D patch positions for rapid spatial scoring
    '''

    H = int(int(480) / 16)  # 16 is the vgg scaling from image space to feature space (conv5)
    W = int(int(640) / 16)
    padding_size = [0, 0]
    patch_size = (int(patch_size_h), int(patch_size_w))
    stride = (int(stride_h), int(stride_w))

    Hout = int((H + (2 * padding_size[0]) - patch_size[0]) / stride[0] + 1)
    Wout = int((W + (2 * padding_size[1]) - patch_size[1]) / stride[1] + 1)

    boxes = calc_receptive_boxes(H, W)

    num_regions = Hout * Wout

    k = 0
    indices = np.zeros((2, num_regions), dtype=int)
    keypoints = np.zeros((2, num_regions), dtype=int)
    # Assuming sensible values for stride here, may get errors with large stride values
    for i in range(0, Hout, stride_h):
        for j in range(0, Wout, stride_w):
            keypoints[0, k] = ((boxes[j + (i * W), 0] + boxes[(j + (patch_size[1] - 1)) + (i * W), 2]) / 2)
            keypoints[1, k] = ((boxes[j + (i * W), 1] + boxes[j + ((i + (patch_size[0] - 1)) * W), 3]) / 2)
            indices[0, k] = j
            indices[1, k] = i
            k += 1

    return keypoints, indices


def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score


def compare_two_spatial(all_indices, qfeats, dbfeats):
    scores = []
    for qfeat, dbfeat, indices in zip(qfeats, dbfeats, all_indices):
        fw_inds, bw_inds = torch_nn(qfeat, dbfeat)

        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

        if len(mutuals) > 0:
            index_keypoints = indices[:, mutuals]
            query_keypoints = indices[:, fw_inds[mutuals]]

            spatial_dist = index_keypoints - query_keypoints # manhattan distance works reasonably well and is fast
            mean_spatial_dist = np.mean(spatial_dist, axis=1)

            # residual between a spatial distance and the mean spatial distance. Smaller is better
            s_dists_x = spatial_dist[0, :] - mean_spatial_dist[0]
            s_dists_y = spatial_dist[1, :] - mean_spatial_dist[1]
            s_dists_x = np.absolute(s_dists_x)
            s_dists_y = np.absolute(s_dists_y)

            # anchor to the maximum x and y axis index for the patch "feature space"
            xmax = np.max(indices[0, :])
            ymax = np.max(indices[1, :])

            # find second-order residual, by comparing the first residual to the respective anchors
            # after this step, larger is now better
            # add non-linearity to the system to excessively penalise deviations from the mean
            s_score = (xmax - s_dists_x)**2 + (ymax - s_dists_y)**2

            scores.append(-s_score.sum() / qfeat.shape[0])
            # we flip to negative such that best match is the smallest number, to be consistent with vanilla NetVlad
            # we normalise by patch count to remove biases in the scoring between different patch sizes (so that all
            # patch sizes are weighted equally and that the only weighting is from the user-defined patch weights)
        else:
            scores.append(0.)

    return scores, None, None
