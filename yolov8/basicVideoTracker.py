import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
import numpy as np

import cv2

from norfair.camera_motion import (
    HomographyTransformationGetter,
    MotionEstimator,
    TranslationTransformationGetter,
)
from norfair import (
    Detection,
    draw_absolute_grid,
)

def yolo_detections_to_norfair_detections(yolo_detections):
    norfair_detections = []
    boxes = []
    for detection in yolo_detections:
        detection_as_xyxy = detection[0]
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        boxes.append(bbox)
        points = bbox
        scores = np.array([detection[2], detection[2]])

        norfair_detections.append(
            Detection(points=points, scores=scores, label=detection_as_xyxy[-1].item())
        )

    return norfair_detections, boxes

def create_mask(boxes, frame):
    # create a mask of ones
    mask = np.ones(frame.shape[:2], frame.dtype)
    # set to 0 all detections
    for b in boxes:
        i = b.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    return mask

def process_video(
    numbers_source_weights_path: str,
    players_source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    number_model = YOLO(numbers_source_weights_path)
    player_model = YOLO(players_source_weights_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    transformations_getter = HomographyTransformationGetter()
    motion_estimator = MotionEstimator(
                max_points=100, #Max points sampled to calculate camera motion
                min_distance=7, #Min distance between points sampled to calculate camera motion
                transformations_getter=transformations_getter,
                draw_flow=False, #visual flow in the image idk how it does this????
            )

    tracker = Tracker(
            distance_function="euclidean",
            detection_threshold=.15, #Confidence threshold of detections
            distance_threshold=.8, #Max distance to consider when matching detections and tracked objects
            initialization_delay=3, #Min detections needed to start the tracked object,
            hit_counter_max=30, #Max iteration the tracked object is kept after when there are no detections
        )

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            number_results = number_model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            number_detections = sv.Detections.from_ultralytics(number_results)

            player_results = player_model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            player_detections = sv.Detections.from_ultralytics(player_results)
            #detections = tracker.update_with_detections(detections)

            # don't use the players on the field to do key points
            # make sure you filter their detection boxes out
            norfair_detections, boxes = yolo_detections_to_norfair_detections(player_detections)
            mask = create_mask(boxes, frame)
            coord_transformations = motion_estimator.update(frame, mask)

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=player_detections)#number_detections

            annotated_labeled_frame = label_annotator.annotate(scene=annotated_frame, detections=player_detections)#number_detections

            # grid to show camera motion
            draw_absolute_grid(annotated_frame, coord_transformations)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_labeled_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            #sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    input_path = "../oldMethod/footballss.mp4"
    output_path = "./footballssOut.mp4"
    number_weights_path = "./runs/detect/yolov8m_justnumbers_150e/weights/best.pt"
    player_weights_path = "./runs/detect/yolov8m_playersreffsonly_150e/weights/best.pt"
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack"
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.4,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        numbers_source_weights_path=number_weights_path,
        players_source_weights_path=player_weights_path,
        source_video_path=input_path,
        target_video_path=output_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    cv2.destroyAllWindows()