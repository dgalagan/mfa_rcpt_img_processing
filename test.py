from rcpt_img_preprocessing.utils import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    folder_path = r'C:\Users\dmitr\computer_vision_course\receipt_img_processing\1_rcpt_img_heic'
    converted_img = convert_heic_to_jpeg(folder_path)

