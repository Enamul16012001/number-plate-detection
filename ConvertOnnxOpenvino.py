from ultralytics import YOLO

model_path = "./license_plate_detection/yolov11_training/weights/best.pt"
model = YOLO(model_path)

model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
ov_model = YOLO("./license_plate_detection/yolov11_training/weights/best_openvino_model")

model.export(format="onnx")
onnx_model = YOLO("./license_plate_detection/yolov11_training/weights/best.onnx")

model.export(format="tflite")
tflite_model = YOLO("yolo11n_float32.tflite")


results = onnx_model("./val_resize/290.jpg")
results = ov_model("./val_resize/290.jpg")

# Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
#results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")