import cv2
import numpy as np
import pytesseract
import json

from .utils import DebugUtils, sort_box
from .Config import OCRConfig


class TextRecognizer:

    def __init__(self, config=None):

        if config is None:
            config = OCRConfig()
        self.config = config

        self.debugger = DebugUtils()
        self.debugger.set_debug(self.config.debug)

    def recognize(self, image, boxes):

        lines_text = []
        lines_boxes = []
        for box in boxes:

            w = None
            h = None

            for j in range(4):
                p1 = box[j]
                p2 = box[(j + 1) % 4]
                L = np.linalg.norm(p1-p2)
                if w is None or L > w:
                    w = L
                if h is None or L < h:
                    h = L

            dst_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            warp_mat, _ = cv2.estimateAffine2D(box, dst_box)

            inv_warp = np.linalg.inv(np.vstack((warp_mat, np.array([0, 0, 1]))))
            self.debugger.log(inv_warp)

            roi = cv2.warpAffine(image, warp_mat, (int(w), int(h)))

            self.debugger.imshow("tesseract_roi", roi, display_height=int(h))

            results = pytesseract.image_to_data(roi, config="-l eng",  output_type=pytesseract.Output.DICT)
            self.debugger.log(json.dumps(results, indent=2))

            if self.config.debug:
                cv2.waitKey(0)

            groups = {}
            for i, text in enumerate(results["text"]):
                confidence = float(results["conf"][i])
                if confidence < 60:
                    continue

                line_id = (str(results["line_num"][i]) +
                           str(results["par_num"][i]) +
                           str(results["block_num"][i]) +
                           str(results["page_num"][i]))
                x0 = int(results["left"][i])
                x1 = x0 + int(results["width"][i])
                y0 = int(results["top"][i])
                y1 = y0 + int(results["height"][i])

                if line_id in groups:
                    group = groups[line_id]
                    x0 = group["left"] if group["left"] < x0 else x0
                    x1 = group["right"] if group["right"] > x1 else x1
                    y0 = group["top"] if group["top"] < y0 else y0
                    y1 = group["bottom"] if group["bottom"] > y1 else y1
                    text = " ".join((group["text"], text))

                groups[line_id] = {
                    "left": x0,
                    "right": x1,
                    "top": y0,
                    "bottom": y1,
                    "text": text
                }

            for key in groups:
                group = groups[key]
                lines_text += [group["text"]]
                x0 = group["left"]
                y0 = group["top"]
                x1 = group["right"]
                y1 = group["bottom"]
                box_points = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float)
                lines_boxes += [sort_box(cv2.perspectiveTransform(box_points, inv_warp).squeeze())]

        return lines_text, lines_boxes
