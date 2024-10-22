import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

from typing import Tuple


# def rle_decode(mask_rle: str, shape: Tuple=(768, 768)) -> np.array:
#     '''
#     Convert RLE to mask
#     '''
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths

#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T

import numpy as np

def rle_decode(mask_rle, shape=(768, 768)):
    """ RLE decode. Returns a binary mask.
    
    mask_rle: RLE string
    shape: (height, width) of the mask
    """
    # RLE decoding
    s = np.fromstring(mask_rle, sep=' ', dtype=int)
    starts, lengths = s[::2] - 1, s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape).T  # Transpose to match the original image shape


def get_bbox(mask: np.array) -> Tuple:
    '''
    Get bbox((x, y), (w, h), a) from RLE mask
    '''
    boxes = []
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # not copying here will throw an error
    for contour in contours:
        boxes.append(cv2.boxPoints(cv2.minAreaRect(contour))) # basically you can feed this rect into your classifier

    return boxes

def reshape_box(box):
    result = []
    for x, y, in box:
        result.append(str(x/768))
        result.append(str(y/768))
    return result