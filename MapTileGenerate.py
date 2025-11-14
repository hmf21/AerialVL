import os
import re
import math
import random
import numpy as np
from PIL import Image
from rasterio.windows import Window
from haversine import haversine, Unit
import glob


def parse_coordinates(filename):
    """解析TIF文件名获取经纬度坐标"""
    base = os.path.splitext(filename)[0]
    parts = base.split('@')
    if len(parts) < 6:
        raise ValueError(f"文件名格式错误: {filename}")

    try:
        lon1 = float(parts[1])
        lat1 = float(parts[2])
        lon2 = float(parts[3])
        lat2 = float(parts[4])
    except ValueError:
        raise ValueError(f"坐标解析失败: {filename}")

    return lon1, lat1, lon2, lat2


def meters_to_degrees(meters, latitude):
    """将米数转换为经纬度变化量（简化模型）"""
    lat_deg = meters / 111000  # 纬度方向：1度≈111公里
    lon_deg = meters / (111000 * math.cos(math.radians(latitude)))
    return lon_deg, lat_deg


def get_sampled_path(image_path, output_dir, input_index):
    """生成均匀采样切割的地图样本"""
    # 打开图像文件
    try:
        Image.MAX_IMAGE_PIXELS = 1000000000
        img = Image.open(image_path)
    except:
        print("File Not Found")
        pass
    else:
        width, height = img.size
        cityname = image_path.split('\\')[3] + '_' + str(input_index).zfill(3)
        filename = os.path.basename(image_path)

        # 解析坐标范围
        lon_min, lat_max, lon_max, lat_min = parse_coordinates(filename)
        avg_lat = (lat_min + lat_max) / 2  # 使用平均纬度计算经度缩放

        # 计算总经纬度范围
        total_lon = lon_max - lon_min
        total_lat = lat_max - lat_min
        patch_size_list = [i for i in range(200, 900, 50)]  # 变宽度采样
        # patch_size_list = [400]                             # 确定采样宽度
        for patch_size_ in patch_size_list:
            patch_size = patch_size_
            # 转换为经纬度变化
            delta_lon, delta_lat = meters_to_degrees(patch_size, avg_lat)
            sampled_interval = 100   # 设定采样宽度为X米
            sampled_lon, sampled_lat = meters_to_degrees(sampled_interval, avg_lat)
            start_lon = lon_min
            while start_lon < lon_max:
                start_lon = start_lon + sampled_lon
                start_lat = lat_max
                while start_lat > lat_min:
                    start_lat = start_lat - sampled_lat
                    try:
                        # 转换为像素坐标（假设左上角为原点）
                        pixel_lon_step = total_lon / width
                        pixel_lat_step = total_lat / height

                        x_start = int((start_lon - lon_min) / pixel_lon_step)
                        y_start = int((lat_max - start_lat) / pixel_lat_step)  # 修正Y轴方向

                        patch_width = int(delta_lon / pixel_lon_step)
                        patch_height = int(delta_lat / pixel_lat_step)

                        # 边界检查
                        if x_start + patch_width > width:
                            continue
                        if y_start + patch_height > height:
                            continue

                        # 执行切割
                        patch = img.crop((
                            x_start,
                            y_start,
                            x_start + patch_width,
                            y_start + patch_height
                        ))

                        # 生成输出路径
                        output_name = f"@{cityname}@{start_lon + delta_lon / 2}@{start_lat - delta_lat / 2}@{patch_size}.tif"
                        output_path = os.path.join(output_dir, output_name)

                        # 保存样本
                        patch.resize((500, 500)).save(output_path)
                        print(f"生成样本: {output_path} ({patch_size}米)")



                    except Exception as e:
                        print(f"处理 {filename} 失败: {str(e)}")




def get_random_patch(image_path, output_dir, input_index, num_samples=50):
    """生成随机切割样本"""
    # 打开图像文件
    try:
        Image.MAX_IMAGE_PIXELS = 1000000000
        img = Image.open(image_path)
    except:
        print("File Not Found")
        pass
    else:
        width, height = img.size
        cityname = image_path.split('\\')[3]+'_'+str(input_index).zfill(3)
        filename = os.path.basename(image_path)

        # 解析坐标范围
        lon_min, lat_max, lon_max, lat_min = parse_coordinates(filename)
        avg_lat = (lat_min + lat_max) / 2  # 使用平均纬度计算经度缩放

        # 计算总经纬度范围
        total_lon = lon_max - lon_min
        total_lat = lat_max - lat_min

        # 生成样本
        patch_size_list = [i for i in range(50, 900, 50)]
        for patch_size_ in range(4):
            for patch_idx in range(num_samples):
                try:
                    # 随机选择边长（米）
                    # patch_size = random.randint(50, 700)
                    patch_size = random.choice(patch_size_list)
                    # patch_size = patch_size_list[patch_idx]
                    # patch_size = patch_size_

                    # 转换为经纬度变化
                    delta_lon, delta_lat = meters_to_degrees(patch_size, avg_lat)

                    # 随机选择起始点（确保不越界）
                    start_lon = random.uniform(
                        lon_min + delta_lon,
                        lon_max - delta_lon
                    )
                    start_lat = random.uniform(
                        lat_min + delta_lat,
                        lat_max - delta_lat
                    )

                    # 转换为像素坐标（假设左上角为原点）
                    pixel_lon_step = total_lon / width
                    pixel_lat_step = total_lat / height

                    x_start = int((start_lon - lon_min) / pixel_lon_step)
                    y_start = int((lat_max - start_lat) / pixel_lat_step)  # 修正Y轴方向

                    patch_width = int(delta_lon / pixel_lon_step)
                    patch_height = int(delta_lat / pixel_lat_step)

                    # 边界检查
                    if x_start + patch_width > width:
                        x_start = width - patch_width
                    if y_start + patch_height > height:
                        y_start = height - patch_height

                    # 执行切割
                    patch = img.crop((
                        x_start,
                        y_start,
                        x_start + patch_width,
                        y_start + patch_height
                    ))

                    # 生成输出路径
                    output_name = f"{cityname}_{str(patch_idx).zfill(4)}_{patch_size}_.tif"
                    output_path = os.path.join(output_dir, output_name)

                    # 保存样本
                    patch.resize((500, 500)).save(output_path)
                    print(f"生成样本: {output_path} ({patch_size}米)")

                except Exception as e:
                    print(f"处理 {filename} 失败: {str(e)}")


def main(input_dir, output_dir, input_index):
    """主处理函数"""
    os.makedirs(output_dir, exist_ok=True)

    if input_dir.lower().endswith('.tif'):
        try:
            get_sampled_path(input_dir, output_dir, input_index)
        except ValueError:
            raise ValueError(f"图像切分失败失败: {input_dir}")


if __name__ == "__main__":
    # 配置路径
    INPUT_DIR_TMPLT = "K:\\Dataset\\GoogleMapTilesDownload\\raw_tif_2\\"
    OUTPUT_DIR = "F:\\SourceCode\\AeroCities\\HE-VLOC\\height_database_large\\"

    # 执行处理
    INPUT_DIR_LIST = glob.glob(os.path.join(INPUT_DIR_TMPLT, "*.tif"))
    for input_idx, INPUT_DIR in enumerate(INPUT_DIR_LIST):
        main(INPUT_DIR, OUTPUT_DIR, input_idx)