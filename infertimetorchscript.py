import torch
import numpy as np
import os
import time
import cv2

# Load TorchScript model
torchscript_model_path = "./license_plate_detection/yolov11_training/weights/best.torchscript"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(torchscript_model_path).to(device)
model.eval()

# Assuming input shape similar to ONNX model
input_shape = [1, 3, 640, 640]

def preprocess(image_path, input_shape):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))  # Resize to (W, H)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img

def benchmark_torchscript_inference(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Found {len(image_files)} images")

    # Warm-up (optional, especially for GPU)
    warmup_img = preprocess(image_files[0], input_shape)
    _ = model(warmup_img)

    start = time.time()
    with torch.no_grad():
        for image_path in image_files:
            img = preprocess(image_path, input_shape)
            _ = model(img)
    end = time.time()

    total_time = end - start
    avg_time = total_time / len(image_files) if image_files else 0

    print(f"\nTorchScript Inference Time:")
    print(f"Total: {total_time:.2f} seconds")
    print(f"Average per image: {avg_time:.4f} seconds")

# Run benchmark
benchmark_torchscript_inference("./val_resize")
