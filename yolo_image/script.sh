yolo detect train model=yolov8n.pt data=data.yaml epochs=200 imgsz=640 batch=16

yolo detect train model=yolov8n.pt data=data.yaml epochs=500 imgsz=640 batch=16

yolo detect train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640 batch=16 lr0=0.005

yolo detect train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640 batch=16 lr0=0.005 degrees=30 scale=0.8 hsv_h=0.02

yolo detect train model=yolov8n.pt data=data.yaml epochs=200 imgsz=640 batch=16 \
    hsv_h=0.02 hsv_s=0.8 hsv_v=0.4 \
    degrees=45 translate=0.2 scale=0.9 \
    fliplr=0.5 mosaic=1.0 mixup=0.1

yolo detect train model=yolov8n.pt data=data.yaml epochs=200 imgsz=640 batch=16 lr0=0.005

yolo detect train model=yolov8n.pt data=data.yaml epochs=200 imgsz=640 batch=64