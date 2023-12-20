import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
import numpy as np
import copy

import cv2

from norfair.camera_motion import (
    HomographyTransformationGetter,
    MotionEstimator,
    TranslationTransformationGetter,
)

from norfair import (
    AbsolutePaths,
    Detection,
    draw_absolute_grid,
    Tracker,
    Color,
)
from norfair.drawing import draw_tracked_objects
from norfair.drawing import draw_points
from norfair.drawing.drawer import Drawer

# for TraceAnnotator
from typing import List, Optional, Tuple, Union
from supervision.annotators.utils import ColorLookup, Trace, resolve_color
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position

class TraceAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        position: Position = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.trace = Trace(max_size=trace_length, anchor=position)
        self.thickness = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
        transformation_function = None,
    ) -> np.ndarray:
        self.trace.put(detections)

        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            xy = self.trace.get(tracker_id=tracker_id)
            if len(xy) > 1:
                if transformation_function:
                    xy = transformation_function(xy)
                scene = cv2.polylines(
                    scene,
                    [xy.astype(np.int32)],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return scene


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

def centroid_from_xyxy(xyxy):
    """
    Calculate the centroid from an array of coordinates in XYXY format.

    :param xyxy_array: Array of coordinates in the format [x1, y1, x2, y2]
    :return: (x_center, y_center) tuple representing the centroid
    """
    x1, y1, x2, y2 = xyxy
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    return (x_center, y_center)

def convert_xyxy_array_np(xyxy_array, transformation_function):
    """
    Convert an array of XYXY coordinates to a different reference frame using NumPy.

    :param xyxy_array: NumPy array of XYXY coordinates
    :param transformation_function: Function to apply to each [x, y] pair
    :return: Transformed NumPy array of XYXY coordinates
    """
    # Reshape and split the array into two [x, y] pair arrays
    xy1 = xyxy_array[:, :2]
    xy2 = xyxy_array[:, 2:]

    # Apply the transformation function
    transformed_xy1 = transformation_function(xy1)
    transformed_xy2 = transformation_function(xy2)

    # Combine the transformed pairs back into XYXY format
    transformed_xyxy = np.hstack((transformed_xy1, transformed_xy2))

    return transformed_xyxy

def process_video(
    numbers_weights_path: str,
    players_weights_path: str,
    players_class_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    number_model = YOLO(numbers_weights_path)
    player_model = YOLO(players_weights_path)
    player_class_model = YOLO(players_class_weights_path)

    tracker = sv.ByteTrack()
    trace_annotator = TraceAnnotator(trace_length=500)
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    transformations_getter = HomographyTransformationGetter()
    motion_estimator = MotionEstimator(
                max_points=100, #Max points sampled to calculate camera motion
                min_distance=7, #Min distance between points sampled to calculate camera motion
                transformations_getter=transformations_getter,
                draw_flow=False, #visual flow in the image
            )

    detections_in_initial_reference_frame = {} # key is a track id and the value is an array of detections in global reference frame ((pointx, pointy), class_id) class_id=[player(0), ref(1)]
    initial_frame_classes = {} # key is a track id and value is a detections in global reference frame ((pointx, pointy), class_id) class_id=[defense(0), oline(1), qb(2), ref(3), skill(4)]
    is_first_frame = True # index 5 is where we get our class labels

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

            player_class_results = player_class_model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            player_class_detections = sv.Detections.from_ultralytics(player_class_results)

            # don't use the players on the field to do key points
            # make sure you filter their detection boxes out
            norfair_player_detections, player_boxes = yolo_detections_to_norfair_detections(player_detections)
            mask = create_mask(player_boxes, frame)
            coord_transformations = motion_estimator.update(frame, mask)

            # convert detections into initial reference frame
            player_detections = tracker.update_with_detections(player_detections)
            player_detections.xyxy = convert_xyxy_array_np(player_detections.xyxy.copy(), coord_transformations.rel_to_abs)
            #player_detections.xyxy = convert_xyxy_array_np(player_detections.xyxy.copy(), coord_transformations.abs_to_rel)

            # transform all of our detections into the initial reference frame and add them to a dict
            for detection_index, track_id in enumerate(player_detections.tracker_id):
                centroid_in_current_reference_frame = centroid_from_xyxy(player_detections.xyxy[detection_index])
                centroid_in_initial_reference_frame = coord_transformations.rel_to_abs(np.array([centroid_in_current_reference_frame]))
                detection = (centroid_in_initial_reference_frame, player_detections.class_id[detection_index]) # a point is structured as ((pointx, pointy), class_id)

                if track_id not in detections_in_initial_reference_frame:
                    detections_in_initial_reference_frame[track_id] = [detection]
                else:
                    detections_in_initial_reference_frame[track_id].append(detection)

            # player paths adjusted for camera movement
            frame = trace_annotator.annotate(scene=frame.copy(), detections=player_detections, transformation_function=coord_transformations.abs_to_rel)

            # player frames and labels in the current reference frame
            '''
            temp_player_detections = copy.deepcopy(player_detections)
            temp_player_detections.xyxy = convert_xyxy_array_np(player_detections.xyxy.copy(), coord_transformations.abs_to_rel)
            frame = box_annotator.annotate(scene=frame, detections=temp_player_detections)
            frame = label_annotator.annotate(scene=frame, detections=temp_player_detections)
            '''

            # specific player classes frames and labels in the current reference frame
            frame = box_annotator.annotate(scene=frame, detections=player_class_detections)
            frame = label_annotator.annotate(scene=frame, detections=player_class_detections)

            # number frames and labels
            frame = box_annotator.annotate(scene=frame, detections=number_detections)
            #frame = label_annotator.annotate(scene=frame, detections=number_detections)

            # grid to show camera motion
            #draw_absolute_grid(frame, coord_transformations)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            sink.write_frame(frame=frame)


if __name__ == "__main__":
    #input_path = "../oldMethod/footballss.mp4"
    #input_path = "./BillsExample.mp4"
    #input_path = "./CardsExample.mp4"
    input_path = "./SteelersExample.mp4"
    output_path = "./tempNoSync.mp4"
    number_weights_path = "./runs/detect/yolov8m_justnumbers_150e/weights/best.pt"
    player_weights_path = "./runs/detect/yolov8m_playersreffsonly_150e/weights/best.pt"
    player_class_weights_path = "./runs/detect/yolov8m_footballplayersoffdeffreffqb_150e3/weights/best.pt"
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
        numbers_weights_path=number_weights_path,
        players_weights_path=player_weights_path,
        players_class_weights_path=player_class_weights_path,
        source_video_path=input_path,
        target_video_path=output_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    cv2.destroyAllWindows()