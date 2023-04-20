import cv2
import os
def process_image(input_image):
    img = cv2.imread(input_image, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3
if __name__== '__main__':
    # Set input and output directories
    input_dir = '01_init\input_handmade'
    output_dir = '01_init\output\cefazolin'

    # Loop through all images in the input directory
    for i, filename in enumerate(os.listdir(input_dir)):
        # Load image
        input_image = os.path.join(input_dir, filename)
        
        img = process_image(input_image)
        
        # Save image to output directory with an ordered name
        output_image_name = f"cefazolin_{i+1}.png"
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, img)
