import os
import sys
import random

from func import rle

# import rle # for debug

import imageio
import pandas as pd
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage



ROOT = "./data"
image_path = os.path.join(ROOT, "train/images")
label_path = os.path.join(ROOT, "train/labels")
aug_path = os.path.join(ROOT, "aug")

rle_df = pd.read_csv(os.path.join(ROOT, "train_ship_segmentations_v2.csv")).dropna()

def aug(image_name):
    # 이미지 읽기
    image = cv2.imread(os.path.join(image_path, image_name))
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # 라벨 파일 읽기
    gts = rle_df[rle_df['ImageId'] == image_name]['EncodedPixels'].values
    if np.nan in gts:
        return
    segmap = np.zeros(image.shape[:2], dtype=np.uint8)  # (H, W)
    # print(gts)
    for gt in gts:
        # print(type(gt))
        # print(gt.shape, image.shape)
        segmap += rle.rle_decode(gt, shape=image.shape[:2])  # RLE 디코딩

    # SegmentationMapsOnImage에 사용하기 위해 차원 조정
    segmap = np.clip(segmap, 0, 1)  # 이진화
    segmap = segmap[..., np.newaxis]  # (H, W, 1) 형태로 변환
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

    # 데이터 증강 파이프라인 정의
    seq = iaa.Sequential(random.sample([
        iaa.Dropout([0.05, 0.2]),
        iaa.Sharpen((0.0, 1.0)),
        iaa.Multiply((0.7, 1.5)),
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.Clouds(),
    ], random.randint(1, 5)), random_order=True)

    # 이미지 및 세그멘테이션 맵 증강
    images_aug, segmaps_aug = seq(image=image, segmentation_maps=segmap)

    # 바운딩 박스 저장
    txt_file_path = os.path.join(aug_path + "/labels", f"{os.path.splitext(image_name)[0]}_aug.txt")
    with open(txt_file_path, 'w') as f:
        bboxes = rle.get_bbox(segmaps_aug.get_arr().astype(np.uint8))
        for bbox in bboxes:
            # print(bbox)
            f.write(f"0 {bbox[0][0]} {bbox[0][1]} {bbox[1][0]} {bbox[1][1]} {bbox[2][0]} {bbox[2][1]} {bbox[3][0]} {bbox[3][1]}\n")

    cv2.imwrite(os.path.join(aug_path + "/images", f"{os.path.splitext(image_name)[0]}_aug.jpg"), images_aug)

    return images_aug, segmaps_aug

if __name__ == '__main__':
    image = cv2.imread(os.path.join(image_path, '0a3b48a9c.jpg'))
    images_aug, segmaps_aug = aug('0a3b48a9c.jpg')

    # 결과를 그리드 형태로 생성
    cells = []
    cells.append(image)                                           # 원본 이미지
    cells.append(images_aug)                                     # 증강된 이미지
    cells.append(segmaps_aug.draw_on_image(images_aug)[0])     # 증강된 이미지에 세그멘테이션 맵
    cells.append(segmaps_aug.draw(size=images_aug.shape[:2])[0])  # 증강된 세그멘테이션 맵

    # 그리드 이미지 생성 및 저장
    grid_image = ia.draw_grid(cells, cols=5)
    imageio.imwrite("example_segmaps.jpg", grid_image)
