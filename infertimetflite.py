import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite  # Use this if you have tflite_runtime installed
# import tensorflow as tf  # Uncomment if you prefer using TensorFlow’s interpreter

# Model paths
tflite_f32_path = "./license_plate_detection/yolov11_training/weights/best_saved_model/best_float32.tflite"
tflite_f16_path = "./license_plate_detection/yolov11_training/weights/best_saved_model/best_float16.tflite"

def preprocess_image(path, target_size):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_image_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

def report_time(label, total_time, num_images):
    avg_time = total_time / num_images if num_images else 0
    print(f"{label} Inference Time:")
    print(f"  Total: {total_time:.2f} seconds")
    print(f"  Average per image: {avg_time:.4f} seconds")

def benchmark_tflite(folder_path, model_path, label):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]

    image_files = get_image_files(folder_path)
    print(f"\n[{label}] Found {len(image_files)} images")

    # Warm-up
    warmup_img = preprocess_image(image_files[0], (w, h))
    interpreter.set_tensor(input_details[0]['index'], warmup_img)
    interpreter.invoke()

    start = time.time()
    for path in image_files:
        img = preprocess_image(path, (w, h))
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()

    report_time(label, end - start, len(image_files))

if __name__ == "__main__":
    folder = "./val_resize"

    benchmark_tflite(folder, tflite_f32_path, "TFLite Float32")
    benchmark_tflite(folder, tflite_f16_path, "TFLite Float16")
