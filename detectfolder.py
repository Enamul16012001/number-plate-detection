from PIL import Image
import os
from ultralytics import YOLO
import numpy as np

model_path = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(model_path)

def process_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image = Image.open(img_path)
            
            results = model(image)
            result = results[0]
            
            # Get the result image as a numpy array (with boxes, etc.)
            result_image = result.plot()  # This will return a NumPy array
            
            # Convert the numpy array to a PIL Image
            result_image_pil = Image.fromarray(result_image)
            
            # Save the result image to the output folder
            output_path = os.path.join(output_folder, filename)
            result_image_pil.save(output_path)
            print(f"Processed image saved at: {output_path}")

# Example usage
input_folder = "./val_resize"
output_folder = "./val_out"
process_and_save_images(input_folder, output_folder)
