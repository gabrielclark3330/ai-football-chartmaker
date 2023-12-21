import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
import numpy as np
#from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import copy
import json  

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

'''
{"label": "t10l","x": 77.12,"y": 71.28,},
{"label": "t20l","x": 76.91,"y": 201.75,},
{"label": "t30l","x": 76.91,"y": 336.19,},
{"label": "t40l","x": 77.33,"y": 468.95,},
{"label": "50l","x": 76.70,"y": 606.10,},
{"label": "b40l","x": 77.12,"y": 739.07,},
{"label": "b30l","x": 76.70,"y": 873.92,},
{"label": "b20l","x": 76.91,"y": 1009.19,},
{"label": "b10l","x": 76.70,"y": 1144.45,},

{"label": "t10r","x": 513.97,"y": 66.90,},
{"label": "t20r","x": 514.18,"y": 201.34,},
{"label": "t30r","x": 514.60,"y": 336.19,},
{"label": "t40r","x": 514.18,"y": 470.41,},
{"label": "50r","x": 514.60,"y": 602.76,},
{"label": "b40r","x": 514.39,"y": 739.07,},
{"label": "b30r","x": 513.97,"y": 873.71,},
{"label": "b20r","x": 514.60,"y": 1008.98,},
{"label": "b10r","x": 513.56,"y": 1141.12,},
'''
# note these are from the perspective of the actual chart which is rotated 90 counter clockwise from almost all of our footage
end_result_diagram_info = {
    "top_boxes": [
        {"label": "b10l","x": 76.70,"y": 1144.45,},
        {"label": "b20l","x": 76.91,"y": 1009.19,},
        {"label": "b30l","x": 76.70,"y": 873.92,},
        {"label": "b40l","x": 77.12,"y": 739.07,},
        {"label": "50l","x": 76.70,"y": 606.10,},
        {"label": "t40l","x": 77.33,"y": 468.95,},
        {"label": "t30l","x": 76.91,"y": 336.19,},
        {"label": "t20l","x": 76.91,"y": 201.75,},
        {"label": "t10l","x": 77.12,"y": 71.28,},
    ],
    "bottom_boxes": [
        {"label": "b10r","x": 513.56,"y": 1141.12,},
        {"label": "b20r","x": 514.60,"y": 1008.98,},
        {"label": "b30r","x": 513.97,"y": 873.71,},
        {"label": "b40r","x": 514.39,"y": 739.07,},
        {"label": "50r","x": 514.60,"y": 602.76,},
        {"label": "t40r","x": 514.18,"y": 470.41,},
        {"label": "t30r","x": 514.60,"y": 336.19,},
        {"label": "t20r","x": 514.18,"y": 201.34,},
        {"label": "t10r","x": 513.97,"y": 66.90,},
    ],
    "width": 590,
    "height": 1215,
    "key": "footballFieldDiagram.png",
}

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

def check_and_add_new_number_detection_set(best_number_detection_set, new_number_detection_set):
    # TODO: validation on if the numbers are correctly matched
    if len(new_number_detection_set) > len(best_number_detection_set):
        return new_number_detection_set
    elif len(new_number_detection_set) == len(best_number_detection_set) and np.sum(new_number_detection_set.confidence) > np.sum(best_number_detection_set.confidence):
        return new_number_detection_set
    else:
        return best_number_detection_set

def find_lines(number_detections):
    points = number_detections.get_anchors_coordinates(Position.CENTER)
    number_classes = number_detections.class_id

    points_array = np.array(points)
    n_points = len(points)
    best_error = float('inf')
    best_lines = None
    best_classes = None

    # Iterate over all possible ways to split points into two groups
    for n in range(1, n_points // 2 + 1):
        for subset in combinations(range(n_points), n):
            set1 = points_array[list(subset)]
            set2 = points_array[list(set(range(n_points)) - set(subset))]

            # Fit a line to each set and calculate the error
            error = 0
            for point_set in [set1, set2]:
                if len(point_set) > 1:  # Need at least 2 points to fit a line
                    reg = LinearRegression().fit(point_set[:, 0].reshape(-1, 1), point_set[:, 1])
                    predictions = reg.predict(point_set[:, 0].reshape(-1, 1))
                    error += mean_squared_error(point_set[:, 1], predictions)

            # Check if this division has a lower error than the best found so far
            if error < best_error:
                best_error = error
                best_lines = (set1, set2)
                classes = np.zeros(n_points)
                classes[list(subset)] = 1  # Classify points in the subset as 1, others as 0
                best_classes = classes

    number_points_and_classes = [(points[i].tolist(), int(number_classes[i])) for i in range(len(number_classes))]

    '''
    '''
    x1, y1 = zip(*best_lines[0].tolist())
    x2, y2 = zip(*best_lines[1].tolist())
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, color='blue', label='Line 1')
    ax.scatter(x2, y2, color='red', label='Line 2')
    #label each point with its class
    for point, number_class in number_points_and_classes:
        ax.annotate(number_class, point)
    # Display the plot
    plt.show()

    bottom_line = None
    top_line = None
    if np.mean(best_lines[0][0, :]) < np.mean(best_lines[1][0, :]): # see which line on average is on top
        top_line = [number_points_and_classes[i] for i in range(len(best_classes)) if best_classes[i]==1]
        bottom_line = [number_points_and_classes[i] for i in range(len(best_classes)) if best_classes[i]==0]
    else:
        top_line = [number_points_and_classes[i] for i in range(len(best_classes)) if best_classes[i]==0]
        bottom_line = [number_points_and_classes[i] for i in range(len(best_classes)) if best_classes[i]==1]

    return (bottom_line, top_line)

def find_subsequence(seq, subseq):
    # This function finds a subsequence in a sequence using NumPy
    target_strides = np.lib.stride_tricks.sliding_window_view(seq, len(subseq))
    matches = np.all(target_strides == subseq, axis=1)
    return np.where(matches)[0]

def warp_point(detection, homography_matrix):
    point = np.array(detection[0])
    point_homogeneous = np.append(point, 1) # Convert to homogeneous coordinates
    transformed_point = np.dot(homography_matrix, point_homogeneous)
    transformed_point /= transformed_point[2] # Convert back to Cartesian coordinates
    return (transformed_point[:2].tolist(), detection[1])

def invert_tracking_data_over_y_ax(detections_in_initial_reference_frame, lines_bottom_top):
    for key in detections_in_initial_reference_frame:
        for i in range(len(detections_in_initial_reference_frame[key])):
            detections_in_initial_reference_frame[key][i] = ([detections_in_initial_reference_frame[key][i][0][0], -detections_in_initial_reference_frame[key][i][0][1]], detections_in_initial_reference_frame[key][i][1])
    for top_or_bottom in [0,1]:
        for detection_index in range(len(lines_bottom_top[top_or_bottom])):
            lines_bottom_top[top_or_bottom][detection_index] = ([lines_bottom_top[top_or_bottom][detection_index][0], -lines_bottom_top[top_or_bottom][detection_index][1]], lines_bottom_top[top_or_bottom][detection_index][1])
    tmp = (lines_bottom_top[1], lines_bottom_top[0])
    lines_bottom_top = tmp
    return lines_bottom_top



def cast_data_to_end_diagram(detections_in_initial_reference_frame, lines_bottom_top, end_result_diagram_info):
    # TODO: probably need to find a more robust solution than sorting by x values
    ideal_sideline = [0,1,2,3,4,3,2,1,0] # [10,20,30,40,50,40,30,20,10]
    film_number_positions = []
    chart_number_positions = []

    bottom_line = lines_bottom_top[0]
    bottom_line = sorted(bottom_line, key=lambda x : x[0][0])
    bottom_line_classes = [x[1] for x in bottom_line]
    bottom_line_start_index = find_subsequence(ideal_sideline, bottom_line_classes)
    for i in range(len(bottom_line_classes)):
        film_number_positions.append(bottom_line[i][0])

        number_index = i+bottom_line_start_index[0]
        x = end_result_diagram_info["bottom_boxes"][number_index]["x"]
        y = end_result_diagram_info["bottom_boxes"][number_index]["y"]
        chart_number_positions.append((x,y))

    top_line = lines_bottom_top[1]
    top_line = sorted(top_line, key=lambda x : x[0][0])
    top_line_classes = [x[1] for x in top_line]
    top_line_start_index = find_subsequence(ideal_sideline, top_line_classes)
    for i in range(len(top_line_classes)):
        film_number_positions.append(top_line[i][0])

        number_index = i+top_line_start_index[0]
        x = end_result_diagram_info["top_boxes"][number_index]["x"]
        y = end_result_diagram_info["top_boxes"][number_index]["y"]
        chart_number_positions.append((x,y))

    h, status = cv2.findHomography(np.array(film_number_positions), np.array(chart_number_positions))

    for key in detections_in_initial_reference_frame:
        for i in range(len(detections_in_initial_reference_frame[key])):
            detections_in_initial_reference_frame[key][i] = warp_point(detections_in_initial_reference_frame[key][i], h)

def match_points(set1, set2):
    """
    Match each point in set1 to the closest point in set2.

    :param set1: NumPy array of points (shape Nx2).
    :param set2: NumPy array of points (shape Mx2).
    :return: List of tuples where each tuple contains a point from set1
             and its closest point in set2.
    """
    matches = []
    for set1_index, point in enumerate(set1):
        # Calculate distances to all points in set2
        distances = np.sqrt(np.sum((set2 - point) ** 2, axis=1))

        # Find the closest point in set2
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        #matches.append((point, set2[min_distance_index]))
        matches.append((set1_index, min_distance_index))

    return matches

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
    number_tracker = sv.ByteTrack()
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
    first_frame_trackid_classes = {} # key is a track id and the value is class_id=[defense(0), oline(1), qb(2), ref(3), skill(4)]
    frame_index = 0 # index 5 is where we get our class labels
    best_number_detections = []

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            number_results = number_model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            number_detections = sv.Detections.from_ultralytics(number_results)
            #number_detections = number_tracker.update_with_detections(number_detections)

            player_results = player_model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            player_detections = sv.Detections.from_ultralytics(player_results)

            # we shouldn't use the players on the field to do key points so we filter their detection boxes out
            norfair_player_detections, player_boxes = yolo_detections_to_norfair_detections(player_detections)
            mask = create_mask(player_boxes, frame)
            coord_transformations = motion_estimator.update(frame, mask)

            # use tracker's Kalman filter to stabilize detections
            player_detections = tracker.update_with_detections(player_detections)

            # right before the snap record classes for each player
            frame_wait_time = 5
            if frame_index==frame_wait_time:
                player_class_results = player_class_model(
                    frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
                )[0]
                player_class_detections = sv.Detections.from_ultralytics(player_class_results)

                player_class_centroids = player_class_detections.get_anchors_coordinates(Position.CENTER)
                player_class_labels = player_class_detections.class_id

                player_centroids = player_detections.get_anchors_coordinates(Position.CENTER)
                player_labels = player_detections.class_id
                player_track_ids = player_detections.tracker_id

                if len(player_class_centroids)!=len(player_centroids):
                    print("ERROR detection amount mismatch")

                # match each class point to its closest player counter part
                matches = match_points(player_class_centroids, player_centroids) # where matches are (index in set 1, index in set 2)

                for class_index, player_index in matches:
                    player_track_id = player_track_ids[player_index]
                    first_frame_trackid_classes[int(player_track_id)] = int(player_class_labels[class_index])
            if frame_index<=frame_wait_time:
                frame_index+=1

            # convert detections into initial reference frame
            player_detections.xyxy = convert_xyxy_array_np(player_detections.xyxy.copy(), coord_transformations.rel_to_abs)

            # add the player detections to a dict
            player_centroids = player_detections.get_anchors_coordinates(Position.CENTER)
            for detection_index, track_id in enumerate(player_detections.tracker_id):
                detection = (player_centroids[detection_index].tolist(), int(player_detections.class_id[detection_index])) # a point is structured as ((pointx, pointy), class_id)

                if track_id not in detections_in_initial_reference_frame:
                    detections_in_initial_reference_frame[int(track_id)] = [detection]
                else:
                    detections_in_initial_reference_frame[int(track_id)].append(detection)

            temp_number_detections = copy.deepcopy(number_detections)
            temp_number_detections.xyxy = convert_xyxy_array_np(number_detections.xyxy.copy(), coord_transformations.rel_to_abs)
            best_number_detections = check_and_add_new_number_detection_set(best_number_detections, temp_number_detections)

            # player paths adjusted for camera movement
            frame = trace_annotator.annotate(scene=frame.copy(), detections=player_detections, transformation_function=coord_transformations.abs_to_rel)

            # player frames and labels in the current reference frame
            temp_player_detections = copy.deepcopy(player_detections)
            temp_player_detections.xyxy = convert_xyxy_array_np(player_detections.xyxy.copy(), coord_transformations.abs_to_rel)
            frame = box_annotator.annotate(scene=frame, detections=temp_player_detections)
            frame = label_annotator.annotate(scene=frame, detections=temp_player_detections)
            # specific player classes frames and labels in the current reference frame
            #frame = box_annotator.annotate(scene=frame, detections=player_class_detections)
            #frame = label_annotator.annotate(scene=frame, detections=player_class_detections)

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

    #print(first_frame_trackid_classes)
    #print(best_number_detections)
    #print(find_lines(best_number_detections))
    #print(detections_in_initial_reference_frame)


    lines_bottom_top = find_lines(best_number_detections)
    #steelers fix
    '''
    del_index = None
    for detection_index in range(len(lines_bottom_top[1])):
        if lines_bottom_top[1][detection_index][0][0]<0 and lines_bottom_top[1][detection_index][1]==2:
            del_index = detection_index
    lines_bottom_top[1].pop(del_index)
    '''

    #cards fix
    '''
    del_index = None
    for detection_index in range(len(lines_bottom_top[1])):
        if lines_bottom_top[1][detection_index][0][0]<500 and lines_bottom_top[1][detection_index][1]==1:
            del_index = detection_index
    lines_bottom_top[1].pop(del_index)
    for detection_index in range(len(lines_bottom_top[0])):
        if lines_bottom_top[0][detection_index][0][0]<600 and lines_bottom_top[0][detection_index][1]==2:
            lines_bottom_top[0][detection_index] = (lines_bottom_top[0][detection_index][0], 1)
    '''

    #bills fix
    print(lines_bottom_top)



    #lines_bottom_top = invert_tracking_data_over_y_ax(detections_in_initial_reference_frame, lines_bottom_top)
    cast_data_to_end_diagram(detections_in_initial_reference_frame, lines_bottom_top, end_result_diagram_info)

    '''
    fig, ax = plt.subplots()
    for track_key in first_frame_trackid_classes:
        if first_frame_trackid_classes[track_key]==4:
            xs = [x[0][0] for x in detections_in_initial_reference_frame[track_key]]
            ys = [x[0][1] for x in detections_in_initial_reference_frame[track_key]]
            ax.scatter(xs, ys, color='blue', label='Line 1')
    # Display the plot
    plt.show()
    '''


    with open(f'../react_play_viewer/src/{source_video_path}_player_detections_json_string.json', 'w') as fp:
        json.dump(detections_in_initial_reference_frame, fp, indent=4)
    with open(f'../react_play_viewer/src/{source_video_path}_player_classes_json_string.json', 'w') as fp:
        json.dump(first_frame_trackid_classes, fp, indent=4)
    #with open(f'{source_video_path}_number_detections_json_string.json', 'w') as fp:
    #    json.dump(lines_bottom_top, fp, indent=4)


if __name__ == "__main__":
    #input_path = "../oldMethod/footballss.mp4"
    input_path = "./BillsExample.mp4"
    #input_path = "./CardsExample.mp4"
    #input_path = "./SteelersExample.mp4"
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