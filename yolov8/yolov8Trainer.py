from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
   # Training.
   results = model.train(
      #data='./datasets/Football Player Detection Just Number NA.v9i.yolov8/data.yaml',
      data='./datasets/Football Player Detection (players reff).v2i.yolov8/data.yaml',
      imgsz=1280,
      epochs=150,
      batch=14,
      name='yolov8m_playersreffsonly_150e'
   )
