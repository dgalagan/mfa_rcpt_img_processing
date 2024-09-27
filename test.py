import os
import cv2
import pillow_heif
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import io

src_img_folder = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\1_rcpt_img_heic'
src_imgs_names = os.listdir(src_img_folder)

crop_img_folder = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\8_croped'
crop_imgs_names = os.listdir(crop_img_folder)

rcpt_areas = []

for crop_img_name in crop_imgs_names:
    croped_img_path = os.path.join(crop_img_folder, crop_img_name)
    croped_img = cv2.imread(croped_img_path)
    croped_img_size = np.size(croped_img) / 3
    orgnl_img_size = 3024 * 4032
    rcpt_area = croped_img_size / orgnl_img_size
    rcpt_areas.append(rcpt_area)

print(sorted(rcpt_areas))