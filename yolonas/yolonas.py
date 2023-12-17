import cv2
import torch
from super_gradients.training import models
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import math
from numpy import random

device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model=models.get('yolo_nas_s', pretrained_weights="coco").to(device)

out=cv2.VideoWriter('output1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
count=0
while True:
  ret, frame=cap.read()
  count+=1
  if ret:
    result=list(model.predict(frame, conf=0.5))[0]
    bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
    confidences=result.prediction.confidence
    labels=result.prediction.labels.tolist()
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
      bbox=np.array(bbox_xyxy)
      x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      classname=int(cls)
      class_name=names[classname]
      conf=math.ceil((confidence*100))/100
      label=f'{class_name}{conf}'
      print("Frame N", count, "", x1, y1,x2, y2)
      cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255, 0.6),thickness=2, lineType=cv2.LINE_AA)
      t_size=cv2.getTextSize(str(label), 0, fontScale=1/2, thickness=1)[0]
      c2=x1+t_size[0], y1-t_size[1]-3
      cv2.rectangle(frame, (x1, y1), c2, color=(0, 0, 255, 0.6), thickness=-1, lineType=cv2.LINE_AA)
      cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    out.write(frame)
  else:
    break
out.release()
cap.release()


cfg_deep = get_config()
cfg_deep.merge_from_file("./deep_sort.yaml")
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1,  x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        cv2.rectangle(img, (x1, y1), (x2, y2), color= compute_color_for_labels(cat),thickness=2, lineType=cv2.LINE_AA)
        label = str(id) + ":" + classNames[cat]
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
        c2=x1+t_size[0], y1-t_size[1]-3
        cv2.rectangle(frame, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img

while True:
    ret, frame = cap.read()
    if ret:
      result = list(model.predict(frame, conf=0.5))[0]
      bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
      confidences = result.prediction.confidence
      labels = result.prediction.labels.tolist()
      for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
          bbox = np.array(bbox_xyxy)
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((confidence*100))/100
          cx, cy = int((x1+x2)/2), int((y1+y2)/2)
          bbox_width = abs(x1-x2)
          bbox_height = abs(y1-y2)
          xcycwh = [cx, cy, bbox_width, bbox_height]
          xywh_bboxs.append(xcycwh)
          confs.append(conf)
          oids.append(int(cls))
      xywhs = torch.tensor(xywh_bboxs)
      confss= torch.tensor(confs)
      outputs = deepsort.update(xywhs, confss, oids, frame)
      if len(outputs)>0:
          bbox_xyxy = outputs[:,:4]
          identities = outputs[:, -2]
          object_id = outputs[:, -1]
          draw_boxes(frame, bbox_xyxy, identities, object_id)
      output.write(frame)
    else:
        break

tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)
count=0
while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        detections = np.empty((0,6))
        result = list(model.predict(frame, conf=0.40))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100
            currentArray = np.array([x1, y1, x2, y2, conf, cls])
            detections = np.vstack((detections, currentArray))
        tracker_dets = tracker.update(detections)
        if len(tracker_dets) >0:
            bbox_xyxy = tracker_dets[:,:4]
            identities = tracker_dets[:, 8]
            categories = tracker_dets[:, 4]
            draw_boxes(frame, bbox_xyxy, identities, categories)
        out.write(frame)

