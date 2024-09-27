from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import cv2
import pillow_heif
import os
import logging
import torch
import io

def obtain_input_points(img, quad_size=10):
    
    img_array = img.get_img_array()

    # Define img shape
    width = img.get_img_width()
    height = img.get_img_height()

    # Calculate img center
    width_cntr = width // 2
    height_cntr = height // 2
    img_cntr = [width_cntr, height_cntr]
    
    # Define a block slice located in the center of the image to check the intensity level
    start_row = height_cntr - quad_size
    end_row = height_cntr + quad_size + 1
    start_col = width_cntr - quad_size
    end_col = width_cntr + quad_size + 1

    # Check if block located within img
    if start_row < 0 or end_row > height:
        decrease_quad_height = end_row - height
        print(f'Your input point is outside the image, please deacrease the quad_size at least to {quad_size - decrease_quad_height}')
        # Make img center a default input point
        input_points = np.array([[width_cntr, height_cntr], [width_cntr, height_cntr + 10]])
    elif start_col < 0 or end_col > width:
        decrease_quad_width = end_col - width
        print(f'Your input point is outside the image, please deacrease the quad_size at least to {quad_size - decrease_quad_width}')
        # Make img center a default input point
        input_points = np.array([[width_cntr, height_cntr], [width_cntr, height_cntr + 10]])
    else:
        block_slice = img_array[start_row:end_row, start_col:end_col,...]
        
        # Define quads within block
        top_left_quad = block_slice[0:quad_size, 0:quad_size,...]
        top_right_quad = block_slice[0:quad_size, quad_size:,...]
        bottom_right_quad = block_slice[quad_size:, quad_size:,...]
        bottom_left_quad = block_slice[quad_size:, 0:quad_size,...]

        # Check intensity surrounding of the picked input
        # Define color intensity within quads
        top_left_intensity = int(np.mean(top_left_quad))
        top_right_intensity = int(np.mean(top_right_quad))
        bottom_right_intensity = int(np.mean(bottom_right_quad))
        bottom_left_intensity = int(np.mean(bottom_left_quad))
        
        # Define quad with maximum intensity
        max_intensity = max(top_left_intensity, top_right_intensity, bottom_left_intensity, bottom_right_intensity)
    
        if max_intensity == top_left_intensity:
            input_points = np.array([img_cntr, [start_col, start_row]])
        
        elif max_intensity == top_right_intensity:
            input_points = np.array([img_cntr, [end_col, start_row]])

        elif max_intensity == bottom_right_intensity:
            input_points = np.array([img_cntr, [end_col, end_row]])
        
        elif max_intensity == bottom_left_intensity:
            input_points = np.array([img_cntr, [start_col, end_row]])
    
    return input_points











def obtain_corner_points(img, blockSize=3, ksize=5, k=0.04, percentile=70):

    # Input parameters description
    # blockSize - neighbourhood considered for corner detection
    # ksize - aperture parameter of the Sobel derivative used
    # k - Harris detector free parameter in the equation
    # percentile - filters the stronges corner points
    
    # Find corners
    corner_map_img = cv2.cornerHarris(img, blockSize, ksize, k)
    # Filter values that represents corners
    positive_cornerness = corner_map_img[corner_map_img > 0.0]
    # Calculate the threshold to identify the strongest corner values
    percentile_threshold = np.percentile(positive_cornerness, percentile) # 70th percentile by default
    # Identify potential corner coords
    corner_points = np.where(corner_map_img >= percentile_threshold)
    
    return corner_points, corner_map_img

def cluster_corner_points(img, corner_points):
    
    # STEP 0 DEFINE IMG SHAPE AND UNPACK CORNER POINTS
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_width_cntr = img_width / 2
    img_height_cntr = img_height / 2
    
    x_points = corner_points[1]
    y_points = corner_points[0]

    x_y_coords = [[x_points, y_points] for x_points, y_points in zip(x_points, y_points)]
    
    ## STEP 1 SPLIT CORNER COORDINATES INTO TOP AND BOTTOM ##
    
    # Identify y coordinates location with respect to height dimension
    top_bottom = []
    
    for y_point in y_points:
        y_loc = round(y_point / img_height, 2)
        top_bottom.append(y_loc)

    # Check if list is not empty prior to calculate distance and center
    if top_bottom:
        # Calculate distance between top and bottom coordinates
        distance_bottom_top = round(max(top_bottom) - min(top_bottom), 2)
        # Calculate the border between top and bottom
        top_bottom_cntr = round((max(top_bottom) + min(top_bottom)) / 2, 2)
    else:
        # Calculate distance between top and bottom coordinates
        distance_bottom_top = 0.00
        # Use img height center to split between left and right
        top_bottom_cntr = round(img_height_cntr / img_height, 2)

    # Split x, y coordinates to the top and bottom ones
    top_coords = []
    bottom_coords = []
    
    for x_y_coord in x_y_coords:
      y = x_y_coord[1]
      y_loc = round(y / img_height, 2)
    
      if y_loc < top_bottom_cntr:
        top_coords.append(x_y_coord)
      else:
        bottom_coords.append(x_y_coord)

    ## STEP 2 SPLIT CORNER COORDINATES INTO LEFT AND RIGHT FOR TOP AND BOTTOM SEPARATELY ##
    
    # Identify x coordinates location with respect to width dimension
    top_right_left = []
    for top_coord in top_coords:
        x = top_coord[0]
        x_loc = round(x / img_width, 2)
        top_right_left.append(x_loc)
    
    bottom_right_left = []
    for bottom_coord in bottom_coords:
        x = bottom_coord[0]
        x_loc = round(x / img_width, 2)
        bottom_right_left.append(x_loc)

    # Check if top points has been identified
    if top_right_left:
        # Calculate distance between right and left for top coordinates
        distance_top_r_l = round(max(top_right_left) - min(top_right_left), 2)
    else:
        # Use img width center to split between left and right
        distance_top_r_l = 0.00
    
    # Check if bottom points has been identified
    if bottom_right_left:
        # Calculate distance between right and left for bottom coordinates
        distance_bottom_r_l = round(max(bottom_right_left) - min(bottom_right_left), 2)
    else:
        # Use img width center to split between left and right
        distance_bottom_r_l = 0.00

    # Calculate the border between left and right for top and bottom separately
    if distance_top_r_l >= 0.05 and distance_bottom_r_l >= 0.05:
        status = 0
        top_right_left_cntr = round((max(top_right_left) + min(top_right_left)) / 2, 2)
        bottom_right_left_cntr = round((max(bottom_right_left) + min(bottom_right_left)) / 2, 2)
    
    elif distance_top_r_l >= 0.05 and distance_bottom_r_l <= 0.05:
        status = 1
        top_right_left_cntr = round((max(top_right_left) + min(top_right_left)) / 2, 2)
        bottom_right_left_cntr = round((max(top_right_left) + min(top_right_left)) / 2, 2)
    
    elif distance_top_r_l <= 0.05 and distance_bottom_r_l >= 0.05:
        status = 1
        top_right_left_cntr = round((max(bottom_right_left) + min(bottom_right_left)) / 2, 2)
        bottom_right_left_cntr = round((max(bottom_right_left) + min(bottom_right_left)) / 2, 2)
        
    else:
        status = 2
        top_right_left_cntr = round(img_width_cntr / img_width, 2)
        bottom_right_left_cntr = round(img_width_cntr / img_width, 2)

    # Split corner coordinates into top left, top right, bottom right, bottom left
    top_left_points = []
    top_right_points = []
    bottom_right_points = []
    bottom_left_points = []
    
    for top_coord in top_coords:
        x = top_coord[0]
        x_loc = round(x / img_width, 2)

        if x_loc < top_right_left_cntr:
            top_left_points.append(top_coord)
        else:
            top_right_points.append(top_coord)
    
    for bottom_coord in bottom_coords:
        x = bottom_coord[0]
        x_loc = round(x / img_width, 2)
        
        if x_loc < bottom_right_left_cntr:
            bottom_left_points.append(bottom_coord)
        else:
            bottom_right_points.append(bottom_coord)

    # Calculate receipt aspect ratio to identify cases where width bigger than height, that is not usual for receipt shape 
    ratio_1 = distance_top_r_l/distance_bottom_top
    ratio_2 = distance_bottom_r_l/distance_bottom_top

    if ratio_1 > 1 or ratio_2 > 1:
        status = 3
    
    clustered_points = [top_left_points, top_right_points, bottom_right_points, bottom_left_points]
   
    return clustered_points, status

def extract_bbox_coords(clustered_points):

    # Check how many clustered points is missing, if more then 2 
    count = sum(1 for clustered_point in clustered_points if not clustered_point)
    
    if count < 2:

        # Unpack clustered points
        top_left_points, top_right_points, bottom_right_points, bottom_left_points = clustered_points
        
        if not bottom_right_points:
            width = top_right_points[0][0] - top_left_points[0][0]
            bottom_right_points = [[bottom_left_points[0][0] + width, bottom_left_points[0][1]]]
    
        if not bottom_left_points:
            width = top_right_points[0][0] - top_left_points[0][0]
            bottom_left_points = [[bottom_right_points[0][0] - width, bottom_right_points[0][1]]]
        
        if not top_right_points:
            width = bottom_right_points[0][0] - bottom_left_points[0][0]
            top_right_points = [[top_left_points[0][0] + width, top_left_points[0][1]]]
        
        if not top_left_points:
            width = bottom_right_points[0][0] - bottom_left_points[0][0]
            top_left_points = [[top_right_points[0][0] - width, top_right_points[0][1]]]
    
        # Define points with maximum distance betweem  top left and bottom right
        coord_dict_0 = {}
    
        for idx_1, top_left_point in enumerate(top_left_points):
            for idx_2, bottom_right_point in enumerate(bottom_right_points):
                
                x_left = top_left_point[0]
                y_left = top_left_point[1]
                x_right = bottom_right_point[0]
                y_right = bottom_right_point[1]
                
                distance = (x_left - x_right) ** 2 + (y_left - y_right) ** 2
                
                coord_dict_0[idx_1, idx_2] = distance
        
        # Define points with maximum distance betweem  top right and bottom left
        coord_dict_1 = {}
    
        for idx_1, top_right_point in enumerate(top_right_points):
            for idx_2, bottom_left_point in enumerate(bottom_left_points):
                
                x_right = top_right_point[0]
                y_right = top_right_point[1]
                x_left = bottom_left_point[0]
                y_left = bottom_left_point[1]
                
                distance = (x_left - x_right) ** 2 + (y_left - y_right) ** 2
                
                coord_dict_1[idx_1, idx_2] = distance
    
        # Get points
        max_key_0 = max(coord_dict_0, key=coord_dict_0.get)
        max_key_1 = max(coord_dict_1, key=coord_dict_1.get)
        
        top_left_point = top_left_points[max_key_0[0]]
        top_right_point = top_right_points[max_key_1[0]]
        bottom_right_point = bottom_right_points[max_key_0[1]]
        bottom_left_point = bottom_left_points[max_key_1[1]]
    
        bbox = np.array([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
    else:
        bbox =  np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    return bbox

def get_rectangular_bbox(bbox):
    
    # Calculate the convex hull of the points
    hull = cv2.convexHull(bbox)
    # Calculate the rotated bounding box of the convex hull
    rect = cv2.minAreaRect(hull)
    # Extract the center, size, and angle of the bounding box
    center, (width, height), angle = rect
    # Compute the four corners of the bounding box
    box_rect = cv2.boxPoints(rect)
    box_rect = np.asarray(box_rect, dtype=int)
    # Ensure the box points are in the correct order
    box_rect = box_rect[np.argsort(np.arctan2(box_rect[:,1] - center[1], box_rect[:,0] - center[0]))]
    
    return box_rect, angle

def get_coord_for_orgnl_size(bbox_rszd, inverse_coefs, expand_by = 60):
    
    inverse_coef_height, inverse_coef_width = inverse_coefs

    top_left_rszd, top_right_rszd, bottom_right_rszd, bottom_left_rszd = bbox_rszd
    top_left_orgnl = [int(top_left_rszd[0] * inverse_coef_width)-expand_by, int(top_left_rszd[1] * inverse_coef_height)-expand_by]
    top_right_orgnl = [int(top_right_rszd[0] * inverse_coef_width)+expand_by, int(top_right_rszd[1] * inverse_coef_height)-expand_by]
    bottom_right_orgnl = [int(bottom_right_rszd[0] * inverse_coef_width)+expand_by, int(bottom_right_rszd[1] * inverse_coef_height)+expand_by]
    bottom_left_orgnl = [int(bottom_left_rszd[0] * inverse_coef_width)-expand_by, int(bottom_left_rszd[1] * inverse_coef_height)+expand_by]
    bbox = [top_left_orgnl, top_right_orgnl, bottom_right_orgnl, bottom_left_orgnl]
    
    return bbox

def draw_bbox(img, bbox):
    
    img_w_bbox = np.copy(img)
    # img_w_bbox = cv2.cvtColor(img_w_bbox, cv2.COLOR_BGR2RGB)
    bbox_array = np.array(bbox, np.int32)
    bbox_array_reshaped = bbox_array.reshape((-1, 1, 2))
    img_w_bbox = cv2.polylines(img_w_bbox, [bbox_array_reshaped], isClosed = True, color = (255,0,0), thickness = 5)
    
    return img_w_bbox

def get_src_dst(bbox):
    
    # Define src points
    src = np.float32(bbox)

    top_left_orgnl, top_right_orgnl, bottom_right_orgnl, bottom_left_orgnl = bbox

    # Define dst points
    width_top_x = top_right_orgnl[0] - top_left_orgnl[0]
    width_top_y = top_right_orgnl[1] - top_left_orgnl[1]
    width_bottom_x = bottom_right_orgnl[0] - bottom_left_orgnl[0]
    width_bottom_y = bottom_right_orgnl[1] - bottom_left_orgnl[1]
    height_left_x = bottom_left_orgnl[0] - top_left_orgnl[0]
    height_left_y = bottom_left_orgnl[1] - top_left_orgnl[1]
    height_right_x = bottom_right_orgnl[0] - top_right_orgnl[0]
    height_right_y = bottom_right_orgnl[1] - top_right_orgnl[1]
    
    width_1 = np.sqrt((width_top_x ** 2) + ((width_top_y) ** 2))
    width_2 = np.sqrt(((width_bottom_x) ** 2) + ((width_bottom_y) ** 2))
    maxWidth = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt((height_left_x ** 2) + ((height_left_y) ** 2))
    height_2 = np.sqrt(((height_right_x) ** 2) + ((height_right_y) ** 2))
    maxHeight = max(int(height_1), int(height_2))
    
    max_W_H = (maxWidth, maxHeight)
    
    dst = np.float32([[0, 0],
                      [maxWidth - 1, 0],
                      [maxWidth - 1, maxHeight - 1],
                      [0, maxHeight - 1]
                     ]
                    )
    return src, dst, max_W_H

def apply_warp_perspective(img, src, dst, frame):
    # Compute the homography matrix (you'll have to use getPerspectiveTransform function from OpenCV here)
    M = cv2.getPerspectiveTransform(src, dst)
    # Build the rectified image using the computed matrix (you'll have to use warpPerspective function from OpenCV)
    warped_img = cv2.warpPerspective(img, M, frame)

    return warped_img

def crop_imgs(input_dict, sam):

    # Initiate dictionaries to store interim imgs
    resized_imgs_dict = {}
    resized_imgs_w_roi_dict = {}
    masked_imgs_dict = {}
    corner_map_imgs_dict = {}
    imgs_with_bbox_dict = {}
    croped_imgs_dict = {}

    # Initiate dictionaries to store interim processing parameters
    input_point_dict = {}
    corner_points_dict = {}
    corner_status_dict = {}
    bbox_for_warp_dict = {}

    # Count images in source dictionary
    total_imgs = len(input_dict.values())

    counter = 0
    for img_name, img in input_dict.items():
        
        try:
            #-----#
            current_step = 'resize' 
            resized_img, inverse_coefs = resize_img(img)
            resized_imgs_dict[img_name] = resized_img

            #-----#
            current_step = 'obtain_points'
            input_point, resized_img_w_roi = obtain_input_point(resized_img)
            input_point_dict[img_name] = input_point
            resized_imgs_w_roi_dict[img_name] = resized_img_w_roi
            
            #-----#
            masked_img = mask_image(resized_img, sam, input_point=input_point)
            masked_imgs_dict[img_name] = masked_img
            
            #-----#
            corner_points, corner_map_img = obtain_corner_points(masked_img)
            corner_points_dict[img_name] = corner_points
            corner_map_imgs_dict[img_name] = corner_map_img
            
            #-----#
            # Cluster corner points into top left, top right, bottom right and bottom left
            clustered_points, status = cluster_corner_points(resized_img, corner_points)
            corner_status_dict[img_name] = status
            # Extract initial bbox coords
            bbox_initial = extract_bbox_coords(clustered_points)
            # Align bbox coordinates to match a rectangle
            bbox_rect, angle = get_rectangular_bbox(bbox_initial)
            # Inverse bbox coords to the original JPEG image size
            bbox_final = get_coord_for_orgnl_size(bbox_rect, inverse_coefs)
            bbox_for_warp_dict[img_name] = bbox_final
            # Draw bbox on the original image
            img_with_bbox = draw_bbox(img, bbox_final)
            imgs_with_bbox_dict[img_name] = img_with_bbox
            
            #-----#
            # Obtain bbox to apply warp
            bbox_for_warp = bbox_for_warp_dict[img_name]
            # Obtain src and dst point to apply warp perspective
            src, dst, frame = get_src_dst(bbox_for_warp) # custom function
            # Crop image
            croped_img = apply_warp_perspective(img, src, dst, frame) # custom function
            # Save croped image
            croped_imgs_dict[img_name] = croped_img
            
            #-----#
            counter += 1
        
            # Calculate and display the percentage progress
            processing_progress = int((counter / total_imgs) * 100)
            print(f'Processing progress: {processing_progress}%', end="\r", flush=True)
        
        except Exception as e:
            logging.error(f'Error processing {img_name} at step {current_step}: {str(e)}')
        
        current_step = 'done'
    print(f'\nProcessing completed successfully. {counter} files processed')
    return croped_imgs_dict