import cv2
import numpy as np

from .utils import DebugUtils, as_data_url
from .TextDetector import TextDetector
from .TextBinarizer import TextBinarizer
from .TextRecognizer import TextRecognizer

from .Config import OCRConfig


class OCR:

    def __init__(self, config=None):

        if config is None:
            config = OCRConfig()
        self.config = config

        self.debugger = DebugUtils()
        self.debugger.set_debug(self.config.debug)

        self.detector = TextDetector(config=self.config)
        self.binarizer = TextBinarizer(config=self.config)
        self.recognizer = TextRecognizer(config=self.config)

    def process_image(self, image, images_as_data_urls=False):

        detections = self.detector.detect_text(image)
        binarized, line_boxes = self.binarizer.binarize(image, detections)
        binarized = np.dstack((binarized, binarized, binarized))
        reco_text, reco_boxes = self.recognizer.recognize(binarized, line_boxes)

        ocr_results = []
        for text, box in zip(reco_text, reco_boxes):
            if text.strip() != "":
                ocr_results += [
                    {
                        "geometry": [(int(p[0]), int(p[1])) for p in box],
                        "text": text
                    }
                ]

        scale = 512. / image.shape[0]
        image = cv2.resize(image, None, fx=scale, fy=scale)

        if self.config.return_decorated:
            decorated = image.copy()

            for (box, roi_box, confidence) in detections:
                for j in range(4):
                    p1 = (int(roi_box[j][0] * scale), int(roi_box[j][1] * scale))
                    p2 = (int(roi_box[(j + 1) % 4][0] * scale), int(roi_box[(j + 1) % 4][1] * scale))
                    cv2.line(decorated, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
                for j in range(4):
                    p1 = (int(box[j][0] * scale), int(box[j][1] * scale))
                    p2 = (int(box[(j + 1) % 4][0] * scale), int(box[(j + 1) % 4][1] * scale))
                    cv2.line(decorated, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            decorated = None

        if self.config.return_binarized:
            for box in reco_boxes:
                for j in range(4):
                    p1 = (int(box[j][0]), int(box[j][1]))
                    p2 = (int(box[(j + 1) % 4][0]), int(box[(j + 1) % 4][1]))
                    cv2.line(binarized, p1, p2, (0, 127, 255), 6, cv2.LINE_AA)
        else:
            binarized = None

        return {
            "decorated": as_data_url(decorated) if images_as_data_urls else decorated,
            "binarized": as_data_url(binarized) if images_as_data_urls else binarized,
            "detections": ocr_results
        }
