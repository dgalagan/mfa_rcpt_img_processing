# class Image() 
import cv2
import pillow_heif
import io
import os
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

class MyImage():
    
    """Open image file"""
    def __init__(self, folder_path, img_name, trgt_ext):
        self.folder_path = folder_path
        self.img_name = img_name
        self.img_path = os.path.join(self.folder_path, self.img_name)
        self.img_basename, self.img_ext_src = os.path.splitext(self.img_name)
        self.img_ext_check = cv2.haveImageWriter(self.img_name)
        self.img_ext_trgt = trgt_ext
        # self.img_ext_check = cv2.haveImageReader(self.img_name)
        self.img = None
    
    def open_img(self):
        if self.img_ext_check == False:
            self.img = self.convert_img()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            self.img = cv2.imread(self.img_path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return self.img
    
    def convert_img(self):

        self.img_ext_src = self.img_ext_src.lower()

        if self.img_ext_src == '.heic':
            
            # Open the HEIC file using pyheif
            with open(self.img_path, 'rb') as heic_file:
                heic_file = pillow_heif.open_heif(heic_file)
                pil_img = Image.frombytes(
                    heic_file.mode,
                    heic_file.size,
                    heic_file.data,
                    'raw',
                    heic_file.mode,
                    heic_file.stride
                )
            
            # Convert the PIL image to JPEG in memory
            with io.BytesIO() as output:
                pil_img.save(output, format=self.img_ext_trgt)
                jpeg_data = output.getvalue()

            # Load the JPEG data into OpenCV format
            self.img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        
        else:
            raise ValueError("Not supported file extension to convert")

        return self.img
    
    def get_img_path(self):
        return self.img_path

    def get_img_basename(self):
        return self.img_basename

    def get_img_ext_src(self):
        return self.img_ext_src
    
    def get_img_ext_trgt(self):
        return self.img_ext_trgt

    def get_img_ext_check(self):
        return self.img_ext_check

class MyImageShape():

    def __init__(self, img):
        self.orgnl_img = img
        self.orgnl_img_h = self.orgnl_img.shape[0]
        self.orgnl_img_w = self.orgnl_img.shape[1]
        self.orgnl_img_AR = self.orgnl_img_h / self.orgnl_img_w

        if self.orgnl_img_AR < 1:
            self.orgnl_img = cv2.rotate(self.orgnl_img, cv2.ROTATE_90_CLOCKWISE)
    
    def get_orgnl_img_height(self):
        return  self.orgnl_img_h
    
    def get_orgnl_img_width(self):
        return  self.orgnl_img_w
    
    def get_orgnl_img_AR(self):
        return self.orgnl_img_AR

class MyImageResizer(MyImageShape):
    
    """Resizes image"""
    def __init__(self, img, trgt_height):
        super().__init__(img)
        self.rszd_img_h = trgt_height
        self.scale_ratio = self.rszd_img_h / self.orgnl_img_h
        self.rszd_img_w = int(self.orgnl_img_w * self.scale_ratio)
        self.inverse_scale_ratio = 1 / self.scale_ratio

        if self.orgnl_img_AR < 1:
            self.orgnl_img = cv2.rotate(self.orgnl_img, cv2.ROTATE_90_CLOCKWISE)
            self.rszd_img = cv2.resize(self.orgnl_img, (self.rszd_img_w, self.rszd_img_h), interpolation = cv2.INTER_AREA)
        else:
            self.rszd_img = cv2.resize(self.orgnl_img, (self.rszd_img_w, self.rszd_img_h), interpolation = cv2.INTER_AREA)

    def get_rszd_img(self):
        return self.rszd_img
    
    def get_inverse_scale_ratio(self):
        return self.inverse_scale_ratio
    
class ActivateModel():

    """Activate Segment Anything Model"""
    def __init__(self, model_folder, model_files_names):
        self.model_path = model_folder
        self.model_files_names = model_files_names
    
    def activate_large_model(self):
        model_file_name = "sam_vit_h_4b8939.pth"
        model_name = "vit_h"
        
        if model_file_name in self.model_files_names:
            model_path = os.path.join(self.model_path, model_file_name)
            sam = sam_model_registry[model_name](checkpoint=model_path)
        else:
            Exception("Checkpoint file for large model is missing in the folder")
        
        return sam

    def activate_small_model(self):
        model_file_name = "sam_vit_b_01ec64.pth"
        model_name = "vit_b"
        
        if model_file_name in self.model_files_names:
            model_path = os.path.join(self.model_path, model_file_name)
            sam = sam_model_registry[model_name](checkpoint=model_path)
        else:
            Exception("Checkpoint file for small model is missing in the folder")
        
        return sam

class ReceiptMask():

    """Extract receipt mask"""
    def __init__(self, img, model, model_params):
        self.img = img
        self.model = model
        self.input_point = model_params['input_point']
        self.input_label = model_params['input_label']
        self.multimask_output = model_params['multimask_output']
    
    def mask_receipt(self):
        
        # Initiate SAM predictor
        predictor = SamPredictor(self.model)

        # Embed image
        predictor.set_image(self.img)
        
        # Check input availability
        if self.input_point is None:
            raise ValueError('Input point has not been provided')
        
        # Segment image
        masks, scores, logits = predictor.predict(
            point_coords=self.input_point,
            point_labels=self.input_label,
            multimask_output=self.multimask_output
        )

        # Define the mask with the highest score
        max_score_idx = np.argmax(scores)
        masked_img = masks[max_score_idx]
        masked_img = masked_img.astype(np.float32)
        
        return masked_img

class ReceiptCorners():
    def __init__(self, img, params):
        self.img = img
        self.block_size = params['blockSize']
        self.k_size = params['ksize']
        self.k = params['k']
        self.percentile = params['percentile']
        self.corner_map_img = cv2.cornerHarris(self.img, self.block_size, self.k_size, self.k)

    def get_corner_map(self):

        return self.corner_map_img

    def identify_corner_points(self):
        positive_cornerness = self.corner_map_img[self.corner_map_img > 0.0]
        # Calculate the threshold to identify the strongest corner values
        percentile_threshold = np.percentile(positive_cornerness, self.percentile) # 70th percentile by default
        # Identify potential corner coords
        self.corner_points = np.where(self.corner_map_img >= percentile_threshold)  
        
        return self.corner_points

class ReceiptBbox():

    def __init__(self, corner_points):

        self.masked_img_height = masked_img.shape[0]
        self.masked_img_width = masked_img.shape[1]
        self.masked_img_height_cntr = masked_img.shape[0] / 2
        self.masked_img_width_cntr = masked_img.shape[1] / 2
        
        self.corner_points = corner_points
        self.clustered_corner_points = {}
        self.bbox = np.array([])
    
    def cluster_corner_points(self):
        
        x_points = self.corner_points[1]
        y_points = self.corner_points[0]

        x_y_coords = [[x_points, y_points] for x_points, y_points in zip(x_points, y_points)]

        ## STEP 1 SPLIT CORNER COORDINATES INTO TOP AND BOTTOM ##
        
        # Identify y coordinates location with respect to height dimension
        top_bottom = []
        
        for y_point in y_points:
            y_loc = round(y_point / self.masked_img_height, 2)
            top_bottom.append(y_loc)

        # Check if list is not empty prior to calculate distance and center
        if top_bottom:
            # Calculate the border between top and bottom
            top_bottom_cntr = round((max(top_bottom) + min(top_bottom)) / 2, 2)
        else:
            # Use img height center to split between top and bottom
            top_bottom_cntr = round(self.masked_img_height_cntr / self.masked_img_height, 2)

        # Split x, y coordinates to the top and bottom ones
        top_coords = []
        bottom_coords = []
        
        for x_y_coord in x_y_coords:
            y = x_y_coord[1]
            y_loc = round(y / self.masked_img_height, 2)
        
            if y_loc < top_bottom_cntr:
                top_coords.append(x_y_coord)
            else:
                bottom_coords.append(x_y_coord)

        ## STEP 2 SPLIT CORNER COORDINATES INTO LEFT AND RIGHT FOR TOP AND BOTTOM SEPARATELY ##
        
        # Identify x coordinates location with respect to width dimension
        top_right_left = []
        for top_coord in top_coords:
            x = top_coord[0]
            x_loc = round(x / self.masked_img_width, 2)
            top_right_left.append(x_loc)
        
        bottom_right_left = []
        for bottom_coord in bottom_coords:
            x = bottom_coord[0]
            x_loc = round(x / self.masked_img_width, 2)
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
            bottom_right_left_cntr = top_right_left_cntr
        
        elif distance_top_r_l <= 0.05 and distance_bottom_r_l >= 0.05:
            status = 1
            bottom_right_left_cntr = round((max(bottom_right_left) + min(bottom_right_left)) / 2, 2)
            top_right_left_cntr = bottom_right_left_cntr
        
        else:
            status = 2
            top_right_left_cntr = round(self.masked_img_height_cntr / self.masked_img_width, 2)
            bottom_right_left_cntr = top_right_left_cntr

        # Split corner coordinates into top left, top right, bottom right, bottom left
        top_left_points = []
        top_right_points = []
        bottom_right_points = []
        bottom_left_points = []
        
        for top_coord in top_coords:
            x = top_coord[0]
            x_loc = round(x / self.masked_img_width, 2)

            if x_loc < top_right_left_cntr:
                top_left_points.append(top_coord)
            else:
                top_right_points.append(top_coord)
        
        for bottom_coord in bottom_coords:
            x = bottom_coord[0]
            x_loc = round(x / self.masked_img_width, 2)
            
            if x_loc < bottom_right_left_cntr:
                bottom_left_points.append(bottom_coord)
            else:
                bottom_right_points.append(bottom_coord)
        
        self.clustered_corner_points[status] = [top_left_points, top_right_points, bottom_right_points, bottom_left_points]
    
        return self.clustered_corner_points

    def identify_bbox(self):

        status = list(self.clustered_corner_points.keys())[0]
        clustered_points = list(self.clustered_corner_points.values())[0]

        if status < 2:

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
        
            self.bbox = np.array([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
        else:
            self.bbox =  np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

        return self.bbox

