import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from .utils import pad_and_resize, remove_outliers
from .Config import OCRConfig


class TextDetector:

    def __init__(self, config=None):

        if config is None:
            config = OCRConfig()
        self.config = config

        east_model = self.config.east_model
        if east_model is None:
            east_model = os.path.join(os.path.dirname(__file__), 'models', 'frozen_east_text_detection.pb')
        self.east = cv2.dnn.readNet(east_model)

        self.target_width = self.config.detection_target_width
        self.target_height = self.config.detection_target_height
        self.confidence_threshold = self.config.detection_confidence_threshold
        self.nms_threshold = self.config.detection_nms_threshold

    def detect_text(self, image):

        r_w, r_h, scores, geometry = self.run_nn(image)
        boxes, confidences = self.decode(scores, geometry)
        detections = self.non_max_suppression(r_w, r_h, boxes, confidences)
        detections = self.filter_detections(detections)

        return detections

    def run_nn(self, image):

        r_w, r_h, resized_image = pad_and_resize(image, self.target_width, self.target_height)

        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (self.target_width, self.target_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
        self.east.setInput(blob)
        (scores, geometry) = self.east.forward(layer_names)

        return r_w, r_h, scores, geometry

    def decode(self, scores, geometry):
        boxes = []
        confidences = []

        height = scores.shape[2]
        width = scores.shape[3]

        for y in range(height):

            scores_data = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            angles_data = geometry[0][4][y]
            for x in range(0, width):
                score = scores_data[x]

                if score < self.confidence_threshold:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles_data[x]

                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                offset = ([offset_x + cos_a * x1_data[x] + sin_a * x2_data[x],
                           offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

                p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
                p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                boxes.append((center, (w, h), -1 * angle * 180.0 / np.pi))
                confidences.append(float(score))

        return boxes, confidences

    def non_max_suppression(self, r_w, r_h, boxes, probs):

        if len(boxes) == 0:
            return []

        detections = []
        for index in range(len(boxes)):
            box = cv2.boxPoints(boxes[index])
            confidence = probs[index]
            for j in range(4):
                box[j][0] *= r_w
                box[j][1] *= r_h
            detections += [(box, confidence)]

        boxes = np.array([box for box, _ in detections])
        probs = np.array(probs)

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        p1 = boxes[:, 0]
        p2 = boxes[:, 1]
        p3 = boxes[:, 2]
        p4 = boxes[:, 3]

        indices = np.argsort(probs)
        polygons = [Polygon([ip1, ip2, ip3, ip4, ip1]) for ip1, ip2, ip3, ip4 in zip(p1, p2, p3, p4)]
        areas = np.array([p.area for p in polygons])

        sel_boxes = []
        big_boxes = []
        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            pick.append(i)

            polygon = polygons[i]

            merged = polygon
            group = [last]
            for j in range(last, -1, -1):
                other = indices[j]
                overlap = merged.intersection(polygons[other]).area / areas[other]
                if overlap > self.nms_threshold:
                    merged = cascaded_union([merged, polygons[other]])
                    group += [j]

            sel_boxes += [polygon.exterior.coords[:-1]]
            big_boxes += [merged.minimum_rotated_rectangle.exterior.coords[:-1]]

            indices = np.delete(indices, group)

        sel_boxes = np.array(sel_boxes).astype("int")
        big_boxes = np.array(big_boxes).astype("int")
        detections = list(zip(sel_boxes, big_boxes, probs[pick]))

        return detections

    def filter_detections(self, detections):

        if len(detections) == 0:
            return detections

        detections_heights = []
        all_boxes = None
        for box, _, _ in detections:
            if all_boxes is None:
                all_boxes = box.copy()
            else:
                all_boxes = np.vstack((all_boxes, box))

            box_height = np.min(np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1))

            detections_heights += [box_height]

        big_box = cv2.minAreaRect(all_boxes.astype(np.int32))

        detections_heights = np.array(detections_heights)

        height_inliers = remove_outliers(detections_heights, sigmas=10)

        detections = [detections[i] for i in range(len(detections))
                      if (detections_heights[i] < np.min(big_box[1]) * 0.33 or i in height_inliers)]

        return detections

