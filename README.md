For detect:

# Basic usage with just the image path
python3 detect.py --image ./truck_ocr_dataset_new/Bangla_License_Plate/images/val/1479.jpg

# More advanced usage with all options
python3 detect.py \
  --image ./truck_ocr_dataset_new/Bangla_License_Plate/images/val/1479.jpg \
  --model ./license_plate_detection/yolov11_training/weights/best.pt \
  --output ./my_prediction.jpg \
  --conf 0.3