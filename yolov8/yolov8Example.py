from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
   # Training.
   results = model.train(
      #data='./datasets/Football Player Detection Just Number NA.v9i.yolov8/data.yaml',
      data='./datasets/Football Player Detection (off deff ref qb).v1i.yolov8 (1)/data.yaml',
      imgsz=1280,
      epochs=150,
      batch=14,
      name='yolov8m_justnumbers_150e'
   )
