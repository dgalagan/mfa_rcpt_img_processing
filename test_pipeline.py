import matplotlib.pyplot as plt
from rcpt_img_preprocessing.helpers import SingleImageLoader, MyImage, ImageResizer, ImageSlicer, ImageIntensityMap
from rcpt_img_preprocessing.utils import *
from scipy.ndimage import label

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Folder with images
    img_files_dir = r'C:\Users\dmitr\OneDrive\Desktop\1_img'
    # img_files_dir = r'C:\Users\dmitr\OneDrive\Desktop\several_imgs'
    img_files_names = os.listdir(img_files_dir)
    
    # Folder with Segment Anything Model checkpoints
    SAM_dir = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\SAM_checkpoint'
    SAM_checkpoints_names = os.listdir(SAM_dir)

    # Folder for JPEG images
    jpeg_img_files_dir = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\2_rcpt_img_jpeg'

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
        file_loader = SingleImageLoader(img_files_dir, img_file_name)
        
        pil_img = file_loader.get_pil_img_obj()
        pil_img_width, pil_img_height = pil_img.size
        pil_img_AR = pil_img_height / pil_img_width

        # Rotate Image
        if pil_img_AR < 1:
            pil_img = pil_img.rotate(angle=90.0, expand=True)

        # # Resize Image
        # rszd_img = pil_img.resize((240, 320))
  
        # Resize original image
        img_resizer = ImageResizer(pil_img, trgt_height=320)
        rszd_img = img_resizer.call()
        print(rszd_img)

        # Slice original image
        sliding_window_size = 7
        img_slicer = ImageSlicer(rszd_img, sliding_window_size)
        slices_dict = img_slicer.call()

        # Get intensities map
        intensity = ImageIntensityMap(rszd_img, slices_dict, percentile=80)
        cluster = intensity.get_clustered_slices()
        
        # # Draw bbox around the biggest clusters
        bboxA = cluster.getbbox()
        orinialrszd = np.copy(np.asarray(rszd_img))
        cv2.rectangle(orinialrszd, (bboxA[0], bboxA[1]), (bboxA[2], bboxA[3]), (0,255,0), 1)

        plt.imshow(orinialrszd)
        plt.show()

        ############ Commented zone #############

        # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # ss.setBaseImage(rszd_img.get_img_copy_array())
        # ss.switchToSelectiveSearchFast()
        # rects = ss.process()
        # areas = [rect[2]*rect[3] for rect in rects]
        # bboxes = [(rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]) for rect in rects]

        # best_bboxes = []
        # for bboxB in bboxes:
        #     iou = bb_intersection_over_union(bboxA, bboxB)
            
        #     if iou > 0.8:
        #         print(img_file.get_img_file_basename(), iou)
        #         # cv2.rectangle(orinialrszd, (bboxB[0], bboxB[1]), (bboxB[2], bboxB[3]), (255,0,0), 1)
        #         best_bboxes.append(bboxB)

        # pick = non_maximum_suppression(np.array(best_bboxes))
        # print(pick)

        # cv2.rectangle(orinialrszd, (pick[0][0], pick[0][1]), (pick[0][2], pick[0][3]), (255,0,0), 1)

        # # for rect in rects:
        # #     area = rect[2]*rect[3]
        # #     bbox = 
        # #     if area > 30000:
        # #         cv2.rectangle(orinialrszd, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (255,0,0), 1)

        # # Plot results
        # plt.imshow(labeled_matrix)
        # plt.figure()
        # plt.imshow(orinialrszd)
        # # plt.hist(list_of_intensities, bins=50)
        # plt.show()
        
        # Save image
        # image = Image.fromarray(img_array)
        # save_path = os.path.join(jpeg_img_files_dir, img_file.get_img_file_basename() + '.jpeg')
        # # cv2.imwrite(save_path, img.get_img_array())
        # image.save(save_path)

        # # Activate Segment Anything model        
        # sam = ActivateModel(SAM_dir, SAM_checkpoints_names)
        # sam = sam.activate_large_model()
        
        # # Try to run model on GPU, if not on CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # sam.to(device)
        
        # # Define input point
        # sam_params['input_point'] = obtain_input_points(rszd_img)

        # # Segment receipt on the image
        # object_segmentation = ObjectSegmentation(rszd_img.get_img_array(), sam, sam_params)
        # masked_img = object_segmentation.mask_receipt()

        # # Check the segmented area size
        # masked_area = np.sum(masked_img)/rszd_img.get_img_size()

        # if masked_area <= 0.05:
        #     print(f'{img_file.get_img_file_basename()} sefmentation failed')

        # # Define receipt bounding box
        # receipt_detection = ObjectDetection(masked_img, img_resizer.get_inverse_scale_ratio())
        # scaled_bbox = receipt_detection.get_scaled_bbox()
        
        # # Crop receipt
        # receipt_crop = ObjectCrop(img, scaled_bbox)
        # croped_img = receipt_crop.call()

        # # Save croped receipt image
        # save_dir = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\8_croped'
        # filename = img_file.get_img_file_basename() + ".jpg"
        # save_path = os.path.join(save_dir, filename)
        # croped_img.save(save_path)
        # print(f'{img_file.get_img_file_basename()} saved')

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
