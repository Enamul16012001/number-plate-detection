import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import time
import cv2

# Load ONNX session
onnx_path = "./license_plate_detection/yolov11_training/weights/best.onnx"
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [1, 3, 640, 640]

def preprocess(image_path, input_shape):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))  # Resize to (W, H)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def benchmark_onnx_inference(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Found {len(image_files)} images")

    start = time.time()
    for image_path in image_files:
        img = preprocess(image_path, input_shape)
        _ = session.run(None, {input_name: img})
    end = time.time()

    total_time = end - start
    avg_time = total_time / len(image_files) if image_files else 0

    print(f"\nONNX Runtime Inference Time:")
    print(f"Total: {total_time:.2f} seconds")
    print(f"Average per image: {avg_time:.4f} seconds")

# Run benchmark
benchmark_onnx_inference("./val_resize")
