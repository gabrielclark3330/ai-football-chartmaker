from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
   # Training.
   results = model.train(
      #data='./datasets/Football Player Detection.v1-originals.yolov8/data.yaml',
      #data='./datasets/Football Player Detection Robo Augment.v8i.yolov8/data.yaml',
      #data='./datasets/Football Player Detection Robo Augment.v8i.yolov8/',
      data='./datasets/Football Player Detection No Augment.v7i.yolov8/data.yaml',
      imgsz=1280,
      epochs=150,
      batch=16,
      name='yolov8m_roboaugment_150e'
   )
