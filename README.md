# Deep Visual Geo-localization Benchmark
This repository is the revised version from the original Deep VGL benchmark [Paper](https://arxiv.org/abs/2204.03444) and [Code](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

We add some customer modules to meet the requirement of our own datasets. The details and full datasets will be provided if our paper is accepted. 

## quick start
```
python eval.py
--datasets_folder=YourDatasetDir
--dataset_name=DatasetName
--backbone=resnet101conv4
--aggregation=gem
--resume=./pretrained_models/msls_r101l3_gem_partial.pth
--split_folder=SpiltFolderName
--add_rerank=local_match
```