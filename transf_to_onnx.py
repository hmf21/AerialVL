import torch.onnx
import os
import sys
import torch
import arg_parser
import logging
import sklearn
from os.path import join
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd

import util
from model import network

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}


# Function to Convert to ONNX
def Convert_ONNX():

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    input_size = (1, 3, 560, 940)
    dummy_input = torch.randn(input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "./pretrained_models/test.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


def define_model():
    args = arg_parser.parse_arguments()
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    if args.aggregation in ["netvlad", "crn"]:
        args.features_dim *= args.netvlad_clusters

    if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
        if args.off_the_shelf.startswith("radenovic"):
            pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
            url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
            state_dict = load_url(url, model_dir = join("data", "off_the_shelf_nets"))
        else:
            # This is a hacky workaround to maintain compatibility
            sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
            zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
            if not os.path.exists(zip_file_path):
                gdd.download_file_from_google_drive(file_id = OFF_THE_SHELF_NAVER[args.backbone],
                                                    dest_path = zip_file_path, unzip = True)
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

    return model


if __name__ == "__main__":
    # Let's load the model we just created and test the accuracy per label
    model = define_model()
    model.to('cpu')
    # Conversion to ONNX
    Convert_ONNX()