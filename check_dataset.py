
"""
With this script you can evaluate your self-built dataset to see whether it is in a right format
"""

import os
import sys
import torch
import arg_parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
import shutil

import commons
import datasets_ws
from model import network


if __name__ == '__main__':
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

    ######################################### MODEL #########################################
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "train")
    logging.info(f"Test set: {test_ds}")

    positives_per_query = test_ds.get_positives()

    database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                     batch_size=1, pin_memory=(args.device == "cuda"), shuffle=True)

    queries_subset_ds = Subset(test_ds, list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=1, pin_memory=(args.device == "cuda"), shuffle=True)

    for inputs, indices in tqdm(queries_dataloader, ncols=100):
        positive_database_index = positives_per_query[indices[0] - test_ds.database_num]
        paired_database_path_1 = test_ds.database_paths[positive_database_index[0]]
        paired_database_path_2 = test_ds.database_paths[positive_database_index[1]]
        paired_database_path_3 = test_ds.database_paths[positive_database_index[2]]
        paired_database_path_4 = test_ds.database_paths[positive_database_index[3]]
        paired_database_path_5 = test_ds.database_paths[positive_database_index[4]]
        paired_queries_path = test_ds.queries_paths[indices[0] - test_ds.database_num]
        _ = shutil.copy(paired_database_path_1, './result/check_dataset/db_pic_1.png')
        _ = shutil.copy(paired_database_path_2, './result/check_dataset/db_pic_2.png')
        _ = shutil.copy(paired_database_path_3, './result/check_dataset/db_pic_3.png')
        _ = shutil.copy(paired_database_path_4, './result/check_dataset/db_pic_4.png')
        _ = shutil.copy(paired_database_path_5, './result/check_dataset/db_pic_5.png')
        _ = shutil.copy(paired_queries_path, './result/check_dataset/qr_pic.png')
        pass
