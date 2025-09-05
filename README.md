# AerialVL Dataset

This repo covers the supplementary materials of the following paper: 

[AerialVL: A Dataset, Baseline and Algorithm Framework for Aerial-Based Visual Localization With Reference Map](https://ieeexplore.ieee.org/abstract/document/10632587)

<p align="center">
  <img width="600" src="asset/AerialVL.png">
</p>

## Updates
 - **[2025/9/5]** 🎉 We have released the evaluated ros bags as well as the map resource of our new T-RO paper GeoVINS (the final version is ready for publication).  [Data Link](https://pan.baidu.com/s/1iJfCyyVSwmiKTZkcwNRCKQ?pwd=d7mc)  [Demo Link](https://www.bilibili.com/video/BV1UFa6z4Eks/?vd_source=dff151c0c5eee4ac7993c1d019aa0aff) 

## Dataset

### Download

Two parts of the dataset can be downloaded from the [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/68c3a4ed24cc40f1a7da/) or [Baidu Netdisk](https://pan.baidu.com/share/init?surl=GsqOeb8Eo8bcMN1TYQv16Q&pwd=j0no), including the captured frames and satellite imageries for training.

### Description

- Flight data covering about 70 km trajectories with various terrains, multiple heights and illumination changes.
- For sequence-based visual localization, 11 frame sequences with different paths ranging from the shortest one of 3.7 km to the longest up to 11 km.
- For visual place recognition, 18361 separate aerial-based images with 14096 cropped corresponding map patches are provided.

### Sensor Setup

RGB camera (FLIR BFS-U3-31S4C-C) is attached with a gimbal to the UAV. The GNSS (NovAtel OEM718D) has 1.5 m accuracy in RMS with the single point mode.

<p align="center">
  <img width="400" src="asset/collect_pltfm_v2.png">
</p>


## Evaluation

### VAL

In the sequence-based visual localization part, we have prepared the evaluation data as follows.

```
+--- geo_referenced_map
|   +--- @large_map@120.42114259488751@36.604504047017464@120.48431398161503@36.573629616877625@.tif
|   +--- @small_map@120.42114259488751@36.604504047017464@120.4568481612987@36.586863027841225@.tif
+--- long_trajtr
|   +--- 2023-03-16-18-04-01
|   +--- 2023-03-18-12-18-25
|   +--- 2023-03-18-12-47-05
|   +--- 2023-03-18-14-38-32
|   +--- 2023-03-18-15-01-14
|   +--- 2023-03-18-15-40-18
+--- short_trajtr
|   +--- 2023-03-11-11-48-35
|   +--- 2023-03-16-16-58-43
|   +--- 2023-03-18-16-30-27
|   +--- 2023-03-18-16-43-16
|   +--- 2023-03-18-16-55-37

```

There are two geo-referenced map with different size for flight sequences with different length. They are also renamed as the the following formats (maps are heading north):

```
@map_name@LeftTopLongitude@LeftTopLatitude@RightBottomLongitude@RightBottomLatitude@.tif
```

And the captured frames are also re-organized as `@UTCTimeStamp@Longitude@Latitude@.png` (frames are heading east).

### VPR

The visual place recognition part are presented as follows:

```
+--- map_database
|   +--- level_1
|   +--- level_2
|   +--- level_3
+--- query_images
|   +--- query_images_1
|   +--- query_images_2
|   +--- query_images_3
|   +--- query_images_4
+--- raw_satellite_imagery
|   +--- @map@120.42251588590332@36.60395282621937@120.48225404509132@36.573629616877625@.tif
```

The map tiles in the `map_database` folder are sampled from the satellite imagery, which is downloaded from the [Google Earth](https://earth.google.com/). The different levels in the `map_database` present the tiles with different size. These tiles are re-organized as follows (tiles are heading east to be consistent with the captured frame):

```
@map@LeftBottomLongitude@LeftBottomLatitude@RightTopLongitude@RightTopLatitude@.png
```

It is worth mentioning that the definition here is different from the VAL part because we adjust the heading of these tiles to make the VPR task more easier.

The `query_image`folder contains the capture four parts of captured frames with names as  `@Longitude@Latitude@.png`.

### Training Data

We have also provided a lot of satellite imageries collected from different years using [USGS](https://earthexplorer.usgs.gov/). As the training data, these imageries can help you to get a new VPR model designed for the aerial-based platform.

## Citation

If you find this dataset useful for your research, please consider citing the paper

```
@article{he2024aerialvl,
  author={He, Mengfan and Chen, Chao and Liu, Jiacheng and Li, Chunyu and Lyu, Xu and Huang, Guoquan and Meng, Ziyang},
  journal={IEEE Robotics and Automation Letters}, 
  title={AerialVL: A Dataset, Baseline and Algorithm Framework for Aerial-Based Visual Localization With Reference Map}, 
  year={2024},
  volume={9},
  number={10},
  pages={8210-8217},
  publisher={IEEE}
}
```

