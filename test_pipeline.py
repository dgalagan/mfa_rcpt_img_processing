import matplotlib.pyplot as plt
from rcpt_img_preprocessing.filters import ImageFileLoader, MyImage, ImageResizer, ActivateModel, ObjectSegmentation, ObjectDetection
from rcpt_img_preprocessing.utils import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Folder with images
    img_files_dir = r'C:\Users\dmitr\OneDrive\Desktop\1_img'
    img_files_names = os.listdir(img_files_dir)
    
    # Folder with Segment Anything Model checkpoints
    SAM_dir = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\SAM_checkpoint'
    SAM_checkpoints_names = os.listdir(SAM_dir)

    # Define SAM parameters
    sam_params = {
        'input_point': None,
        'input_label': [1, 1],
        'multimask_output': True
        }
    
    # corner_harris_params = {
    #     'blockSize': 3,
    #     'ksize': 5,
    #     'k': 0.04,
    #     'percentile': 70
    #     }   

    for img_file_name in img_files_names:
        
        # Open image
        img_file = ImageFileLoader(img_files_dir, img_file_name)
        img_array = img_file.open_img_file()
        
        # Create Image object
        img = MyImage(img_array)
        print(img.get_img_AR())
        if img.get_img_AR() < 1:
              img.rotate_img()
        print(img.get_img_AR())
        # Resize original image
        img_resizer = ImageResizer(img, trgt_height=320)
        rszd_img = img_resizer.get_rszd_img()

        # Activate Segment Anything model        
        sam = ActivateModel(SAM_dir, SAM_checkpoints_names)
        sam = sam.activate_large_model()
        
        # Try to run model on GPU, if not on CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam.to(device)
        
        # Define input point
        sam_params['input_point'] = obtain_input_points(rszd_img)
        
        # Segment receipt on the image
        object_segmentation = ObjectSegmentation(rszd_img.get_img_array(), sam, sam_params)
        masked_img = object_segmentation.mask_receipt()

        # Check the segmented area size
        masked_area = np.sum(masked_img)/rszd_img.get_img_size()

        if masked_area <= 0.05:
            print(f'{img_file.get_img_file_basename()} sefmentation failed')

        # Define receipt bounding box
        receipt_detection = ObjectDetection(masked_img, img_resizer.get_inverse_scale_ratio())
        bbox = receipt_detection.get_scaled_bbox()
        
        # Crop receipt
        pil_img = Image.fromarray(img.get_img_array())
        croped_pil_img = pil_img.crop(bbox)

        # Save croped receipt image
        save_dir = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\8_croped'
        filename = img_file.get_img_file_basename() + ".jpg"
        save_path = os.path.join(save_dir, filename)
        croped_pil_img.save(save_path)
        print(f'{img_file.get_img_file_basename()} saved')

        # Get receipt bounding box 
        # bbox = ReceiptBbox(masked_img)
        # bbox = bbox.get_bbox()
        # bbox_inversed = np.round(inverse_scale_ratio * bbox).astype(int)

        # print(top_left_x)
        # bboxing = ReceiptBbox(masked_img, corner_harris_params)
        # corner_points = bboxing.identify_corner_points()
        # clustered_corner_points = bboxing.cluster_corner_points()
        # print(clustered_corner_points)
        # bbox = bboxing.identify_bbox()
        # print(bbox)
