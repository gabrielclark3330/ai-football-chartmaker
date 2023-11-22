from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='./datasets/Football Player Detection.v1-originals.yolov8/data.yaml',
   imgsz=640,#1280,
   epochs=50,
   batch=8,
   name='yolov8n_v8_50e'
)
