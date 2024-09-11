from PIL import Image
import numpy as np
import cv2
import pillow_heif
import os
import logging
import io

def convert_heic_to_jpeg(heic_dir, file_prefix='.heic'):

    # Dictionary to store JPEG images
    jpeg_imgs_dict = {}
    
    # List all the HEIC files
    heic_imgs = [file for file in os.listdir(heic_dir) if file.lower().endswith(file_prefix)]
    total_imgs = len(heic_imgs)

    # Convert each HEIC file to JPEG
    num_converted = 0
    for heic_img_index, heic_img_name in enumerate(heic_imgs, start=1):
        
        img_name = os.path.splitext(heic_img_name)[0]
        heic_path = os.path.join(heic_dir, heic_img_name)

        try:
            # Open the HEIC file using pyheif
            with open(heic_path, 'rb') as heic_file:
                heif_file = pillow_heif.open_heif(heic_file)
                pil_img = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    'raw',
                    heif_file.mode,
                    heif_file.stride
                )
            
            # Convert the PIL image to JPEG in memory
            with io.BytesIO() as output:
                pil_img.save(output, format='JPEG')
                jpeg_data = output.getvalue()

            # Load the JPEG data into OpenCV format
            jpeg_img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            jpeg_img = cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB)
            
            # Define JPEG img shape
            height_jpeg_img = jpeg_img.shape[0]
            width_jpeg_img = jpeg_img.shape[1]
            
            # Check orientation of the JPEG image
            if width_jpeg_img / height_jpeg_img > 1:
                jpeg_img = cv2.rotate(jpeg_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            jpeg_imgs_dict[img_name] = jpeg_img
            num_converted += 1
            
            # Calculate and display the percentage progress
            conversion_progress = int((num_converted / total_imgs) * 100)
            print(f'Conversion progress: {conversion_progress}%',  end="\r", flush=True)

        except Exception as e:
            logging.error(f'Error converting {heic_img_name}: {str(e)}')
    if heic_img_index == num_converted:
        print(f'\nConversion completed successfully. {num_converted} files processed.')
    else:
        print(f'Some images failed to be converted')

    return jpeg_imgs_dict