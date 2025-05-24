from PIL import Image
import os
from ultralytics import YOLO
import time

model_path = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(model_path)

def benchmark_inference(input_folder):
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    print(f"Found {len(image_files)} images to process...")

    start_time = time.time()

    for img_path in image_files:
        image = Image.open(img_path)
        _ = model(image)  # Run inference only, discard results

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(image_files) if image_files else 0

    print(f"\nTotal inference time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds")

# Example usage
input_folder = "./val_resize"
benchmark_inference(input_folder)
