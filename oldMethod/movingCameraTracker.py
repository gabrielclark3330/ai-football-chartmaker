import argparse
from functools import partial

import numpy as np
import torch

import cv2

from norfair import (
    AbsolutePaths,
    Detection,
    FixedCamera,
    Tracker,
    Video,
    draw_absolute_grid,
    draw_tracked_boxes,
    Palette,
)
from norfair.camera_motion import (
    HomographyTransformationGetter,
    MotionEstimator,
    TranslationTransformationGetter,
)
from norfair.drawing import draw_tracked_objects
from norfair.drawing import draw_points
from norfair.drawing.drawer import Drawer

def root_homography_draw(self, frame, tracked_objects, coord_transform=None):
    frame_scale = frame.shape[0] / 100
    if self.radius is None:
        self.radius = int(max(frame_scale * 0.7, 1))
    if self.thickness is None:
        self.thickness = int(max(frame_scale / 7, 1))
    for obj in tracked_objects:
        if not obj.live_points.any():
            continue
            #pass

        if self.color is None:
            color = Palette.choose_color(obj.id)
        else:
            color = self.color

        points_to_draw = self.get_points_to_draw(obj.get_estimate(absolute=True))

        '''
        for point in coord_transform.abs_to_rel(points_to_draw):
            Drawer.circle(
                frame,
                position=tuple(point.astype(int)),
                radius=self.radius,
                color=color,
                thickness=self.thickness,
            )
        '''

        last = points_to_draw
        for i, past_points in enumerate(self.past_points[obj.id]):
            overlay = frame.copy()
            #last = coord_transform.abs_to_rel(last)
            for j, point in enumerate(past_points):#coord_transform.abs_to_rel(past_points)):
                Drawer.line(
                    overlay,
                    tuple(last[j].astype(int)),
                    tuple(point.astype(int)),
                    color=color,
                    thickness=self.thickness,
                )
            last = past_points

            alpha = self.alphas[i]
            frame = Drawer.alpha_blend(overlay, frame, alpha=alpha)
        self.past_points[obj.id].insert(0, points_to_draw)
        self.past_points[obj.id] = self.past_points[obj.id][: self.max_history]
    return frame

def sticky_update(
        self,
        detections = None,
        period = 1,
        coord_transformations = None,
    ):
        """
        Process detections found in each frame.

        The detections can be matched to previous tracked objects or new ones will be created
        according to the configuration of the Tracker.
        The currently alive and initialized tracked objects are returned

        Parameters
        ----------
        detections : Optional[List[Detection]], optional
            A list of [`Detection`][norfair.tracker.Detection] which represent the detections found in the current frame being processed.

            If no detections have been found in the current frame, or the user is purposely skipping frames to improve video processing time,
            this argument should be set to None or ignored, as the update function is needed to advance the state of the Kalman Filters inside the tracker.
        period : int, optional
            The user can chose not to run their detector on all frames, so as to process video faster.
            This parameter sets every how many frames the detector is getting ran,
            so that the tracker is aware of this situation and can handle it properly.

            This argument can be reset on each frame processed,
            which is useful if the user is dynamically changing how many frames the detector is skipping on a video when working in real-time.
        coord_transformations: Optional[CoordinatesTransformation]
            The coordinate transformation calculated by the [MotionEstimator][norfair.camera_motion.MotionEstimator].

        Returns
        -------
        List[TrackedObject]
            The list of active tracked objects.
        """
        if coord_transformations is not None:
            for det in detections:
                det.update_coordinate_transformation(coord_transformations)

        # Remove stale trackers and make candidate object real if the hit counter is positive
        alive_objects = []
        dead_objects = []
        if self.reid_hit_counter_max is None:
            self.tracked_objects = [
                o for o in self.tracked_objects if o.hit_counter_is_positive
            ]
            alive_objects = self.tracked_objects
        else:
            tracked_objects = []
            for o in self.tracked_objects:
                if o.reid_hit_counter_is_positive:
                    tracked_objects.append(o)
                    if o.hit_counter_is_positive:
                        alive_objects.append(o)
                    else:
                        dead_objects.append(o)
                        alive_objects.append(o)
            self.tracked_objects = tracked_objects

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()
            obj.update_coordinate_transformation(coord_transformations)

        # Update initialized tracked objects with detections
        (
            unmatched_detections,
            _,
            unmatched_init_trackers,
        ) = self._update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if not o.is_initializing],
            detections,
            period,
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        (
            unmatched_detections,
            matched_not_init_trackers,
            _,
        ) = self._update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if o.is_initializing],
            unmatched_detections,
            period,
        )

        if self.reid_distance_function is not None:
            # Match unmatched initialized tracked objects with not yet initialized tracked objects
            _, _, _ = self._update_objects_in_place(
                self.reid_distance_function,
                self.reid_distance_threshold,
                unmatched_init_trackers + dead_objects,
                matched_not_init_trackers,
                period,
            )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                self._obj_factory.create(
                    initial_detection=detection,
                    hit_counter_max=self.hit_counter_max,
                    initialization_delay=self.initialization_delay,
                    pointwise_hit_counter_max=self.pointwise_hit_counter_max,
                    detection_threshold=self.detection_threshold,
                    period=period,
                    filter_factory=self.filter_factory,
                    past_detections_length=self.past_detections_length,
                    reid_hit_counter_max=self.reid_hit_counter_max,
                    coord_transformations=coord_transformations,
                )
            )

        return self.get_active_objects()

AbsolutePaths.draw = root_homography_draw
#Tracker.update = sticky_update


def yolo_detections_to_norfair_detections(yolo_detections, track_boxes):
    norfair_detections = []
    boxes = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        detection_as_xyxy = detection_as_xyxy.cpu().numpy()
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        boxes.append(bbox)
        if track_boxes:
            points = bbox
            scores = np.array([detection_as_xyxy[4], detection_as_xyxy[4]])
        else:
            points = bbox.mean(axis=0, keepdims=True)
            scores = detection_as_xyxy[[4]]

        norfair_detections.append(
            Detection(points=points, scores=scores, label=detection_as_xyxy[-1].item())
        )

    return norfair_detections, boxes


def run():
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n",
        help="YOLO model to use, possible values are yolov5n, yolov5s, yolov5m, yolov5l, yolov5x",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold of detections",
        default=0.15,
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.8,
        help="Max distance to consider when matching detections and tracked objects",
    )
    parser.add_argument(
        "--initialization-delay",
        type=float,
        default=3,
        help="Min detections needed to start the tracked object",
    )
    parser.add_argument(
        "--track-boxes",
        dest="track_boxes",
        action="store_true",
        help="Pass it to track bounding boxes instead of just the centroids",
    )
    parser.add_argument(
        "--hit-counter-max",
        type=int,
        default=30,
        help="Max iteration the tracked object is kept after when there are no detections",
    )
    parser.add_argument(
        "--iou-threshold", type=float, help="Iou threshold for detector", default=0.15
    )
    parser.add_argument(
        "--image-size", type=int, help="Size of the images for detector", default=480
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0], help="Classes to track"
    )
    parser.add_argument(
        "--transformation",
        default="homography",
        help="Type of transformation, possible values are homography, translation, none",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Max points sampled to calculate camera motion",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=7,
        help="Min distance between points sampled to calculate camera motion",
    )
    parser.add_argument(
        "--no-mask-detections",
        dest="mask_detections",
        action="store_false",
        default=True,
        help="By default we don't sample regions where objects were detected when estimating camera motion. Pass this flag to disable this behavior",
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Pass this flag to save the video instead of showing the frames",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        "--downsample-ratio",
        type=int,
        default=1,
        help="Downsample ratio when showing frames",
    )
    parser.add_argument(
        "--fixed-camera-scale",
        type=float,
        default=0,
        help="Scale of the fixed camera, set to 0 to disable. Note that this only works for translation",
    )
    parser.add_argument(
        "--draw-absolute-grid",
        dest="absolute_grid",
        action="store_true",
        help="Pass this flag to draw absolute grid for reference",
    )
    parser.add_argument(
        "--draw-objects",
        dest="draw_objects",
        action="store_true",
        help="Pass this flag to draw tracked object as points or as boxes if --track-boxes is used.",
    )
    parser.add_argument(
        "--draw-paths",
        dest="draw_paths",
        action="store_true",
        help="Pass this flag to draw the paths of the objects (SLOW)",
    )
    parser.add_argument(
        "--path-history",
        type=int,
        default=300,
        help="Length of the paths",
    )
    parser.add_argument(
        "--id-size",
        type=float,
        default=None,
        help="Size multiplier of the ids when drawing. Thikness will addapt to size",
    )
    parser.add_argument(
        "--draw-flow",
        dest="draw_flow",
        action="store_true",
        help="Pass this flag to draw the optical flow of the selected points",
    )

    args = parser.parse_args()
    '''
    class args():
        def __init__(self):
            self.transformation = "homography"
            self.
    '''

    model = torch.hub.load("ultralytics/yolov5", args.model)
    model.conf_threshold = 0
    model.iou_threshold = args.iou_threshold
    model.image_size = args.image_size
    model.classes = args.classes

    use_fixed_camera = args.fixed_camera_scale > 0
    tracked_objects = []
    # Process Videos
    for input_path in args.files:
        if args.transformation == "homography":
            transformations_getter = HomographyTransformationGetter()
        elif args.transformation == "translation":
            transformations_getter = TranslationTransformationGetter()
        elif args.transformation == "none":
            transformations_getter = None
        else:
            raise ValueError(f"invalid transformation {args.transformation}")
        if transformations_getter is not None:
            motion_estimator = MotionEstimator(
                max_points=args.max_points,
                min_distance=args.min_distance,
                transformations_getter=transformations_getter,
                draw_flow=args.draw_flow,
            )
        else:
            motion_estimator = None

        if use_fixed_camera:
            fixed_camera = FixedCamera(scale=args.fixed_camera_scale)

        if args.draw_paths:
            path_drawer = AbsolutePaths(max_history=args.path_history, thickness=10)

        video = Video(input_path=input_path)
        show_or_write = (
            video.write
            if args.save
            else partial(video.show, downsample_ratio=args.downsample_ratio)
        )

        tracker = Tracker(
            distance_function="euclidean",
            detection_threshold=args.confidence_threshold,
            distance_threshold=args.distance_threshold,
            initialization_delay=args.initialization_delay,
            hit_counter_max=args.hit_counter_max,
        )
        first_frame = None
        for frame in video:
            first_frame = frame
            break
        for frame in video:
            detections = model(frame)
            detections, boxes = yolo_detections_to_norfair_detections(
                detections, args.track_boxes
            )

            mask = None
            if args.mask_detections:
                # create a mask of ones
                mask = np.ones(frame.shape[:2], frame.dtype)
                # set to 0 all detections
                for b in boxes:
                    i = b.astype(int)
                    mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
                if args.track_boxes:
                    for obj in tracked_objects:
                        i = obj.estimate.astype(int)
                        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0

            if motion_estimator is None:
                coord_transformations = None
            else:
                coord_transformations = motion_estimator.update(frame, mask)

            tracked_objects = tracker.update(
                detections=detections, coord_transformations=coord_transformations
            )

            if args.draw_objects:
                draw_points(
                    frame,
                    tracked_objects,
                    id_size=5,
                    hide_dead_points=False,
                )

            if args.absolute_grid:
                draw_absolute_grid(frame, coord_transformations)

            if args.draw_paths:
                frame = path_drawer.draw(
                    first_frame, tracked_objects, coord_transform=coord_transformations
                )

            if use_fixed_camera:
                frame = fixed_camera.adjust_frame(frame, coord_transformations)

            show_or_write(frame)
        #print(coord_transformations.)

if __name__ == "__main__":
    run()
