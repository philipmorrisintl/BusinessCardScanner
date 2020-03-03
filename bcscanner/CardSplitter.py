import os
import cv2
import numpy as np
import pdf2image

from scipy import stats
from shapely.geometry import Polygon

from .utils import DebugUtils
from .Config import SplitterConfig


class CardSplitter:

    def __init__(self, config=None):

        if config is None:
            config = SplitterConfig()

        self.config = config

        self.debugger = DebugUtils()
        self.debugger.set_debug(config.debug)

        edge_model = config.edge_model
        if edge_model is None:
            edge_model = os.path.join(os.path.dirname(__file__), "models", "structured_edges.yml.gz")

        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection(edge_model)

    def split_raw(self, image_bytes, image_type, scan_mode="One Sided"):

        cards = []

        if "pdf" in image_type.lower():
            pages = pdf2image.convert_from_bytes(image_bytes)
            for i_page, page in enumerate(pages):
                if scan_mode == "Double Sided" and i_page % 2 == 1:
                    continue
                if scan_mode == "One Sided" or (i_page+1) >= len(pages):
                    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    cards += self.split_cards(image)
                else:
                    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    image_next = cv2.cvtColor(np.array(pages[i_page+1]), cv2.COLOR_RGB2BGR)
                    cards += self.split_cards_double_sided(image, image_next)
        else:
            image_np = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
            cards += self.split_cards(image)

        return cards

    def split_cards(self, image, return_boxes=False):

        h, w = image.shape[:2]

        scale = 750. / image.shape[0]
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale)

        self.debugger.imshow("image", image)

        edges = (self.edge_detector.detectEdges(scaled_image.astype(np.float32) /
                                                255.) * 255.).astype(np.uint8)
        edges = cv2.resize(edges, None, fx=1./scale, fy=1./scale)

        self.debugger.imshow("edges", edges)

        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        self.debugger.imshow("threshold", edges)

        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5)))
        self.debugger.imshow("morphed", edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        contours_image = self.debugger.imcopy(image)
        self.debugger.draw_contours(contours_image, contours, (0, 255, 255), 3)
        self.debugger.imshow("contours", contours_image)

        rectangles = []
        approx_contours = []
        rect_areas = []
        for i, c in enumerate(contours):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            approx_contours += [approx]

            rect = cv2.minAreaRect(approx)
            box = np.int0(cv2.boxPoints(rect))
            area = cv2.contourArea(approx)
            if area / (w * h) < 0.01 or area / (w * h) > 0.9:
                continue

            box_area = cv2.contourArea(box)
            if 0.8 < box_area / area < 1.2:
                rectangles += [i]
                rect_areas += [area]
        rect_areas = np.array(rect_areas)

        approx_contours_image = self.debugger.imcopy(image)
        self.debugger.draw_contours(approx_contours_image, approx_contours, (0, 255, 255), 3)
        self.debugger.imshow("approx_contours", approx_contours_image)

        inlier_rectangles = []
        if len(rect_areas) > 0:
            median_area = np.median(rect_areas, axis=0)
            mad_areas = stats.median_absolute_deviation(rect_areas)
            z_score = 0.6745 * np.sqrt((rect_areas - median_area) ** 2) / mad_areas
            inlier_rectangles = [rectangles[i] for i in range(len(rectangles)) if z_score[i] < 5]

        outer_rectangles = []
        for i in inlier_rectangles:
            has_rect_parent = False
            for j in inlier_rectangles:
                if i == j:
                    continue
                # [Next, Previous, First_Child, Parent]
                parent = hierarchy[i][3]
                while parent >= 0 and parent != j:
                    parent = hierarchy[parent][3]
                if parent == j:
                    has_rect_parent = True
            if not has_rect_parent:
                outer_rectangles += [i]

        cards = []
        boxes = []
        rects_image = self.debugger.imcopy(image)

        for r in outer_rectangles:
            rect = cv2.minAreaRect(approx_contours[r])
            box = np.int0(cv2.boxPoints(rect))
            center = np.mean(box, axis=0)
            angles = np.arctan2((box - center)[:, 1], (box - center)[:, 0])
            src_pts = box[np.argsort(angles)].astype("float32")
            r_w = int(np.linalg.norm(src_pts[1] - src_pts[0]))
            r_h = int(np.linalg.norm(src_pts[2] - src_pts[1]))
            dst_pts = np.array([[0, 0], [r_w - 1, 0],
                                [r_w - 1, r_h - 1], [0, r_h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (r_w, r_h))
            cards += [warped]
            boxes += [box]

            self.debugger.draw_box(rects_image, box, (0, 0, 255), 3)

        self.debugger.imshow("rects", rects_image)

        if len(cards) <= 1:
            cards = [image]
            boxes = [(0, 0), (0, h), (w, h), (w, 0)]

        if return_boxes:
            return cards, boxes

        return cards

    def split_cards_double_sided(self, image, image_next):

        comb_cards = []

        cards, boxes = self.split_cards(image, return_boxes=True)
        cards_next, boxes_next = self.split_cards(image_next, return_boxes=True)

        for card, box in zip(cards, boxes):
            best_match = None
            best_overlap = 0
            polygon = Polygon(box)
            for card_next, box_next in zip(cards_next, boxes_next):
                polygon_next = Polygon(box_next)
                overlap = polygon.intersection(polygon_next).area
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = card_next
            if best_match is not None:
                comb_cards += [self.get_one_image([card, best_match])]
            else:
                comb_cards += [card]

        return comb_cards

    @staticmethod
    def get_one_image(images):
        max_width = []
        max_height = 0
        for image in images:
            max_width.append(image.shape[1])
            max_height += image.shape[0]
        w = np.max(max_width)
        h = max_height

        final_image = np.zeros((h, w, 3), dtype=np.uint8)

        current_y = 0
        for image in images:
            final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
            current_y += image.shape[0]

        return final_image
