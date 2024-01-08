import os
import sys
import torch
import arg_parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
from tqdm import tqdm
from glob import glob
import faiss
import util
import commons
import datasets_ws
from model import network
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
from model.rerank import rerank

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    'resnet50conv5'  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}


if __name__ == '__main__':
    # source code path in remote computer is '/home/cloudam/Sourcecode/deep-visual-geo-localization-benchmark'

    ######################################### SETUP #########################################
    args = arg_parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    # remove unwanted logging print
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # save viz results
    args.viz_save_dir = join("result", args.dataset_name)
    if not os.path.exists(args.viz_save_dir):
        os.mkdir(args.viz_save_dir)

    ######################################### MODEL #########################################
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    if args.aggregation in ["netvlad", "crn"]:
        args.features_dim *= args.netvlad_clusters

    if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
        if args.off_the_shelf.startswith("radenovic"):
            pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
            url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
            state_dict = load_url(url, model_dir=join("data", "off_the_shelf_nets"))
        else:
            # This is a hacky workaround to maintain compatibility
            sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
            zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
            if not os.path.exists(zip_file_path):
                gdd.download_file_from_google_drive(file_id=OFF_THE_SHELF_NAVER[args.backbone],
                                                    dest_path=zip_file_path, unzip=True)
            if args.backbone == "resnet50conv5":
                state_dict_filename = "Resnet50-AP-GeM.pt"
            elif args.backbone == "resnet101conv5":
                state_dict_filename = "Resnet-101-AP-GeM.pt"
            state_dict = torch.load(join("data", "off_the_shelf_nets", state_dict_filename))
        state_dict = state_dict["state_dict"]
        model_keys = model.state_dict().keys()
        renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
        model.load_state_dict(renamed_state_dict)
    elif args.resume is not None:
        logging.info(f"Resuming model from {args.resume}")
        model = util.resume_model(args, model)
    # Enable DataParallel after loading checkpoint, otherwise doing it before
    # would append "module." in front of the keys of the state dict triggering errors
    model = torch.nn.DataParallel(model)

    if args.pca_dim is None:
        pca = None
    else:
        full_features_dim = args.features_dim
        args.features_dim = args.pca_dim
        pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

    ######################################### DATASETS #########################################
    # use the subdir 'eval'
    test_ds = datasets_ws.TestDataset(args, args.querys_folder, args.datasets_folder)
    logging.info(f"Test set: {test_ds}")

    database_dataloader = DataLoader(dataset = test_ds, num_workers = args.num_workers,
                                     batch_size = args.infer_batch_size, pin_memory = (args.device == "cuda"), shuffle=False)

    query_feature = np.empty((len(test_ds), args.features_dim), dtype = "float32")

    for inputs, indices in tqdm(database_dataloader, ncols = 100):
        features = model(inputs.to(args.device))
        features = features.cpu().detach().numpy()
        if pca is not None:
            features = pca.transform(features)
        query_feature[indices.numpy(), :] = features
    features_dim = query_feature.shape[1]

    database_features = np.load('./resource/global_feature_{}.npy'.format(args.dataset_name))
    dataset_images_index = glob(join(args.datasets_folder, "**", args.img_ext), recursive = True)
    query_images_index = glob(join(args.querys_folder, "**", args.img_ext), recursive = True)

    faiss_index = faiss.IndexFlatL2(features_dim)
    faiss_index.add(database_features)
    del database_features
    distances, predictions_ = faiss_index.search(query_feature, 10)

    predictions_ = rerank(predictions_, test_ds, args)

    positives_per_query = test_ds.get_positives()

    recalls = np.zeros(len(args.recall_values))
    for q_index, predictions in enumerate(predictions_):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(predictions[:n], positives_per_query[q_index])):
                recalls[i:] += 1

        # save the viz result
        retrieved_tile_name_0 = dataset_images_index[predictions[0]]
        retrieved_tile_name_1 = dataset_images_index[predictions[1]]
        retrieved_tile_name_2 = dataset_images_index[predictions[2]]
        retrieved_tile_name_3 = dataset_images_index[predictions[3]]
        retrieved_tile_name_4 = dataset_images_index[predictions[4]]
        query_image_name = query_images_index[q_index]
        retrieved_tile_0 = mpimg.imread(retrieved_tile_name_0)
        retrieved_tile_1 = mpimg.imread(retrieved_tile_name_1)
        retrieved_tile_2 = mpimg.imread(retrieved_tile_name_2)
        retrieved_tile_3 = mpimg.imread(retrieved_tile_name_3)
        retrieved_tile_4 = mpimg.imread(retrieved_tile_name_4)
        query_image = mpimg.imread(query_image_name)
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.imshow(query_image)
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.imshow(retrieved_tile_0)
        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.imshow(retrieved_tile_1)
        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.imshow(retrieved_tile_2)
        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.imshow(retrieved_tile_3)
        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.imshow(retrieved_tile_4)
        sub_dir_save_viz_result = './result/{}/{}'.format(args.dataset_name, str(q_index).zfill(5))
        if not os.path.exists(sub_dir_save_viz_result):
            os.mkdir(sub_dir_save_viz_result)
        plt.savefig('./result/{}/{}/query_{}.png'.format(args.dataset_name, str(q_index).zfill(5), str(q_index).zfill(5)))
        # plt.show()

        # for pred_idx, prediction in enumerate(predictions[:5]):
        #     shutil.copyfile(dataset_images_index[prediction], sub_dir_save_viz_result+'\\'+str(pred_idx).zfill(2)+'.png')

    recalls = recalls / test_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    print("Recalls: ", recalls_str)
