For detect:

# Basic usage with just the image path
python3 detect.py --image ./truck_ocr_dataset_new/Bangla_License_Plate/images/val/1479.jpg

# More advanced usage with all options
python3 detect.py \
  --image ./truck_ocr_dataset_new/Bangla_License_Plate/images/val/1479.jpg \
  --model ./license_plate_detection/yolov11_training/weights/best.pt \
  --output ./my_prediction.jpg \
  --conf 0.3


# Model Inference Benchmarking Tool

This tool benchmarks the inference performance of different model formats for license plate detection. It supports multiple model formats including YOLO, TorchScript, ONNX, OpenVINO, and TensorFlow Lite.

## Supported Model Formats

- **YOLO**: Original YOLOv11 PyTorch model
- **TorchScript**: Optimized PyTorch model for production
- **ONNX**: Open Neural Network Exchange format
- **OpenVINO**: Intel's optimization toolkit format
- **TensorFlow Lite Float32**: Quantized TFLite model (32-bit)
- **TensorFlow Lite Float16**: Quantized TFLite model (16-bit)

## Usage

### Running All Models (Default)

To benchmark all available model formats:

```bash
python3 infertime_comparison_preprocessed.py
```

or explicitly:

```bash
python infertime_comparison_preprocessed.py --model ALL
```

### Running Individual Models

You can benchmark specific model formats using the `--model` parameter:

#### YOLO Model
```bash
python3 infertime_comparison_preprocessed.py --model YOLO
```

#### TorchScript Model
```bash
python3 infertime_comparison_preprocessed.py --model TorchScript
```

#### ONNX Model
```bash
python3 infertime_comparison_preprocessed.py --model ONNX
```

#### OpenVINO Model
```bash
python3 infertime_comparison_preprocessed.py --model OpenVINO
```

#### TensorFlow Lite Float32
```bash
python3 infertime_comparison_preprocessed.py --model "TFLite Float32"
```

#### TensorFlow Lite Float16
```bash
python3 infertime_comparison_preprocessed.py --model "TFLite Float16"
```

## Input Requirements

### Image Format Support
The script supports the following image formats:
- `.jpg` / `.jpeg`
- `.png`

### Image Preprocessing
All images are automatically:
- Resized to 640x640 pixels (or model-specific input size)
- Normalized to [0, 1] range
- Converted to RGB format
- Formatted according to each model's requirements

## Output

The script provides detailed timing information:

```
[YOLO] Found 150 images
[TorchScript] Found 150 images
[ONNX] Found 150 images
[OpenVINO] Found 150 images
[TFLite Float32] Found 150 images
[TFLite Float16] Found 150 images

========= Inference Summary =========
YOLO: Total = 70.63s, Avg per image = 0.1320s
TorchScript: Total = 57.77s, Avg per image = 0.1080s
ONNX: Total = 41.74s, Avg per image = 0.0780s
OpenVINO: Total = 33.54s, Avg per image = 0.0627s
TFLite Float32: Total = 112.38s, Avg per image = 0.2100s
TFLite Float16: Total = 127.95s, Avg per image = 0.2392s
```
