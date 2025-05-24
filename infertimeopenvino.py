from openvino.runtime import Core
import numpy as np
import os
import cv2
import time

# Load OpenVINO IR model
core = Core()
model_path = "./license_plate_detection/yolov11_training/weights/best_openvino_model/best.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")

# Get input info
input_layer = compiled_model.input(0)
input_shape = input_layer.shape  # e.g. [1, 3, 640, 640]

def preprocess(image_path, input_shape):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def benchmark_openvino_inference(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(folder_path, f))
    ]

    print(f"Found {len(image_files)} images.")

    start_time = time.time()

    for image_path in image_files:
        input_tensor = preprocess(image_path, input_shape)
        _ = compiled_model([input_tensor])

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / len(image_files) if image_files else 0

    print(f"\n✅ OpenVINO Inference Timing:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Avg time per image: {avg_time:.4f} seconds")

# Run benchmark
benchmark_openvino_inference("./val_resize")