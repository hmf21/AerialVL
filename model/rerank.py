import os
import time

import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
from PIL import Image

import model.resizeimage as resizeimage
from model.local_matcher import local_matcher
from model.patch_matcher import patch_matcher
from model.patch_matcher.patchnetvlad.models import models_generic


def LocalMatcher(predictions, eval_ds, args):
    database_paths = eval_ds.database_paths
    queries_paths = eval_ds.queries_paths
    reranked_preds = []

    for q_idx, pred in enumerate(tqdm(predictions, leave=False, desc='Local matcher comparing prediction: ')):
        query_image = queries_paths[q_idx]
        q_kpts, q_des = local_matcher.detectAndcompute(image=query_image, matcher_method='superpoint')

        # dual circulation
        score_matcher_db = np.zeros((len(pred)))
        for d_idx, candidate in enumerate(pred):
            destine_image = database_paths[candidate]
            d_kpts, d_des = local_matcher.detectAndcompute(image=destine_image, matcher_method='superpoint')

            correct_matched_kp = ''

            if (len(q_kpts) >= 10) and (len(d_kpts) >= 10):
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(q_des, d_des, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.95 * n.distance:
                        good.append(m)

                # use mask in findHomography to get the correct matched point
                if len(good) > 10:
                    src_pts = np.float32([q_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([d_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    correct_matched_kp = [src_pts[i] for i in range(len(good)) if mask[i]]
                else:
                    correct_matched_kp = ''

            score_matcher_db[d_idx] = float(len(correct_matched_kp) / len(d_kpts))

        # rerank the prediction by their matched point pairs
        cand_sorted = np.argsort(score_matcher_db)[::-1]
        reranked_preds.append(pred[cand_sorted])

    return np.array(reranked_preds)


def PatchMatcher(predictions, eval_ds, args):
    database_paths = eval_ds.database_paths
    queries_paths = eval_ds.queries_paths
    reranked_preds = []
    it = resizeimage.input_transform((480, 640))
    args_strides = '1,1,1'
    args_patch_sizes = '2, 5, 8'
    args_patchweights2use = '0.45, 0.15, 0.4'
    args_pool_size = 4096

    for q_idx, pred in enumerate(tqdm(predictions, leave = False, desc = 'Patch matcher comparing prediction: ')):
        query_image_path = queries_paths[q_idx]
        query_image = cv2.imread(query_image_path, -1)
        query_image_pil = Image.fromarray(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        query_image_pil = it(query_image_pil).unsqueeze(0)

        # dual circulation
        score_matcher_db = np.zeros((len(pred)))
        for d_idx, candidate in enumerate(pred):
            destine_image_path = database_paths[candidate]
            destine_image = cv2.imread(destine_image_path, -1)
            destine_image_pil = Image.fromarray(cv2.cvtColor(destine_image, cv2.COLOR_BGR2RGB))
            destine_image_pil = it(destine_image_pil).unsqueeze(0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            encoder_dim, encoder = models_generic.get_backend()

            # must resume to do extraction
            resume_ckpt = './model/patch_matcher/patchnetvlad/pretrained_models/mapillary_WPCA4096.pth.tar'

            if os.path.isfile(resume_ckpt):
                checkpoint = torch.load(resume_ckpt, map_location = lambda storage, loc: storage)
                args_num_clusters = str(checkpoint['state_dict']['pool.centroids'].shape[0])
                model = models_generic.get_model(encoder, encoder_dim, args_num_clusters, args_patch_sizes, args_strides, append_pca_layer = True)
                model.load_state_dict(checkpoint['state_dict'])
                model = model.to(device)
            else:
                raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

            model.eval()

            input_data = torch.cat((query_image_pil.to(device), destine_image_pil.to(device)), 0)

            with torch.no_grad():
                image_encoding = model.encoder(input_data)

                vlad_local, _ = model.pool(image_encoding)
                # global_feats = get_pca_encoding(model, vlad_global).cpu().numpy()

                local_feats_one = []
                local_feats_two = []
                for this_iter, this_local in enumerate(vlad_local):
                    this_local_feats = models_generic.get_pca_encoding(
                        model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))).\
                        reshape(this_local.size(2), this_local.size(0), args_pool_size).permute(1, 2, 0)
                    local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
                    local_feats_two.append(this_local_feats[1, :, :])

            patch_sizes = [int(s) for s in args_patch_sizes.split(",")]
            strides = [int(s) for s in args_strides.split(",")]
            patch_weights = np.array(args_patchweights2use.split(",")).astype(float)

            all_keypoints = []
            all_indices = []

            tqdm.write('====> Matching Local Features')
            for patch_size, stride in zip(patch_sizes, strides):
                # we currently only provide support for square patches, but this can be easily modified for future works
                keypoints, indices = patch_matcher.calc_keypoint_centers_from_patches(patch_size, patch_size, stride, stride)
                all_keypoints.append(keypoints)
                all_indices.append(indices)

            matcher = patch_matcher.compare_two_spatial(local_feats_one, local_feats_two, all_indices)

            scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
            score = patch_matcher.apply_patch_weights(scores, len(patch_sizes), patch_weights)

            torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the

            score_matcher_db[d_idx] = scores

            # rerank the prediction by their matched point pairs
            cand_sorted = np.argsort(score_matcher_db)[::-1]
            reranked_preds.append(pred[cand_sorted])

        return np.array(reranked_preds)


def RandomSelect(predictions, eval_ds, args):
    reranked_preds = []

    for _, pred in enumerate(tqdm(predictions, leave = False, desc = 'Random select comparing prediction: ')):
        random_sorted = np.random.default_rng()
        sorted_result = random_sorted.permutation((np.arange(20)))
        reranked_preds.append(pred[sorted_result])

    return reranked_preds


def rerank(predictions, eval_ds, args):
    # eval_ds contains database and the

    rerank_method = args.add_rerank

    if rerank_method == 'local_match':
        reranked_preds = LocalMatcher(predictions, eval_ds, args)
    elif rerank_method == 'patch_match':
        reranked_preds = PatchMatcher(predictions, eval_ds, args)
    elif rerank_method == 'dense_match':
        pass
    elif rerank_method == 'semantic_match':
        pass
    elif rerank_method == 'e2e':
        pass
    elif rerank_method == 'random':
        reranked_preds = RandomSelect(predictions, eval_ds, args)
    else:
        print("Rerank method error, please give the right methods")

    return 12reranked_preds