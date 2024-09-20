import matplotlib.pyplot as plt
from rcpt_img_preprocessing.filters import MyImage, MyImageShape, MyImageResizer, ActivateModel, ReceiptMask, ReceiptBbox

from rcpt_img_preprocessing.utils import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Folder with images
    src_img_folder = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\1_rcpt_img_heic'
    src_imgs_names = os.listdir(src_img_folder)
    
    # Folder with Segment Anything Model checkpoints
    SAM_folder = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\SAM_checkpoint'
    SAM_checkpoints_names = os.listdir(SAM_folder)

    # Define SAM parameters
    sam_params = {
        'input_point': None,
        'input_label': [1],
        'multimask_output': True
        }
    
    corner_harris_params = {
        'blockSize': 3,
        'ksize': 5,
        'k': 0.04,
        'percentile': 70
        }
    
    img_dict = {}

    for src_img_name in src_imgs_names:
        
        # Open image
        img_file = MyImage(src_img_folder, src_img_name, trgt_ext="jpeg")
        img_file_basename = img_file.get_img_basename()
        orgnl_img = img_file.open_img()
        
        # Get shape of the original image
        img_shape = MyImageShape(orgnl_img)
        
        # Resize original image
        img_resizer = MyImageResizer(orgnl_img, trgt_height=320)
        rszd_img = img_resizer.get_rszd_img()
        img_dict[img_file_basename] = rszd_img
        
        # sam = ActivateModel(SAM_folder, SAM_checkpoints_names)
        # sam = sam.activate_large_model()
        
        # # Try to run model on GPU, if not on CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # sam.to(device)
        # # Define input point
        # sam_params['input_point'] = obtain_input_point(resized_img)
        
        # masking = ReceiptMask(resized_img, sam, sam_params)
        # masked_img = masking.mask_receipt()

        # bboxing = ReceiptBbox(masked_img, corner_harris_params)
        # corner_points = bboxing.identify_corner_points()
        # clustered_corner_points = bboxing.cluster_corner_points()
        # print(clustered_corner_points)
        # bbox = bboxing.identify_bbox()
        # print(bbox)

print(img_dict)