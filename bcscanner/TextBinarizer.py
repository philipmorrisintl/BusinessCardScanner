import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from sklearn.cluster import KMeans, DBSCAN

from .Config import OCRConfig
from .utils import DebugUtils, sort_box


class TextBinarizer:

    def __init__(self, config=None):

        if config is None:
            config = OCRConfig()
        self.config = config

        self.debugger = DebugUtils()
        self.debugger.set_debug(self.config.debug)

        self.do_visualisation = self.config.debug

        if (256 % config.binarization_rgb_bin_size) != 0:
            raise Exception("RGB bin size should divide 256 into an integer number of bins. Given: {}".
                            format(config.binarization_rgb_bin_size))
        self.dbscan_eps = config.binarization_dbscan_eps
        self.rgb_bin_size = config.binarization_rgb_bin_size
        self.dbscan_min_sample_frac = config.binarization_dbscan_min_sample_frac

        self.roi_x_padding = config.binarization_roi_x_padding
        self.roi_y_padding = config.binarization_roi_y_padding

    def binarize(self, image, detections):

        self.debugger.log("binarize: ", len(detections))

        h, w = image.shape[:2]
        scale = np.sqrt(750*750/(h*w))
        self.debugger.log("--> scale: ", w, h, scale)
        if scale > 1:
            scale = 1
        else:
            image = cv2.resize(image, None, fx=scale, fy=scale)
            h, w = image.shape[:2]

        binarized = np.ones((h, w), np.uint8) * 255

        refined_text_boxes = []

        self.do_visualisation = self.config.debug

        for _, box, confidence in detections:
            if scale < 1:
                box = (box.astype("float") * scale).astype("int")
            _x, _y, _w, _h = cv2.boundingRect(box)
            _x -= int(_w*self.roi_x_padding)
            _w = int(_w*(1 + 2*self.roi_x_padding))
            _y -= int(_h*self.roi_y_padding)
            _h = int(_h*(1 + 2*self.roi_y_padding))
            if _x < 0:
                _x = 0
            if _x+_w > w:
                _w = w - _x
            if _y < 0:
                _y = 0
            if _y+_h > h:
                _h = h - _y
            if _w <= 0 or _h <= 0:
                continue
            roi = image[_y:_y+_h, _x:_x+_w]
            binary_roi, text_box = self._binarize_roi(box.astype("int") - [_x, _y], roi)
            if text_box is not None:
                binarized[_y:_y+_h, _x:_x+_w] = np.bitwise_and(binarized[_y:_y+_h, _x:_x+_w], binary_roi)
                refined_text_boxes += [text_box+(_x, _y)]

        refined_text_boxes = self.refine_text_boxes(refined_text_boxes)

        if scale < 1:
            refined_text_boxes = (np.array(refined_text_boxes).astype("float") / scale).astype("int")
            binarized = cv2.resize(binarized, None, fx=1./scale, fy=1./scale)
            _, binarized = cv2.threshold(binarized, 127, 255, cv2.THRESH_BINARY)

        return binarized, refined_text_boxes

    def _binarize_roi(self, box, roi):

        _h, _w = roi.shape[:2]

        binarized_roi = np.ones((_h, _w), np.uint8)*255

        binarized_layers = []
        layer_components = []

        # Preselection
        for layer, cluster in self._get_affinity_layers(box, roi):

            binarized_layer = np.zeros_like(binarized_roi)
            labels, stats, centroids = self.get_connected_components(layer, cluster)
            components = np.zeros((_h, _w, 3))

            selected_labels = []

            for label, stat in enumerate(stats):
                left = stat[cv2.CC_STAT_LEFT]
                top = stat[cv2.CC_STAT_TOP]
                width = stat[cv2.CC_STAT_WIDTH]
                height = stat[cv2.CC_STAT_HEIGHT]
                right = left + width - 1
                bottom = top + height - 1

                is_good = self._select_good_char_box((_w, _h), (left, top, width, height),
                                                     (labels == label), debug_mode=self.config.debug)

                if self.config.debug:
                    is_good, reason = is_good
                    cv2.rectangle(components, (left, top), (right, bottom), (0, 255, 255), 1, cv2.LINE_AA)
                    if reason == 1:
                        continue
                    cv2.rectangle(components, (left, top), (right, bottom), (0, 0, 255), 1, cv2.LINE_AA)
                    if reason == 2:
                        continue
                    cv2.rectangle(components, (left, top), (right, bottom), (0, 255, 0), 1, cv2.LINE_AA)
                    if reason == 3:
                        continue
                    cv2.rectangle(components, (left, top), (right, bottom), (0, 125, 255), 1, cv2.LINE_AA)
                    if reason == 4:
                        continue
                    cv2.rectangle(components, (left, top), (right, bottom), (0, 255, 125), 1, cv2.LINE_AA)
                    if reason == 5:
                        continue
                    cv2.rectangle(components, (left, top), (right, bottom), (255, 0, 0), 1, cv2.LINE_AA)

                if not is_good:
                    continue

                selected_labels += [label]

            layer_box = np.array([(0, 0), (0, _h), (_w, _h), (_w, 0)])
            stubs = []
            isolated_points = []
            stubs_image = np.zeros((_h, _w, 3), np.uint8)
            words = []
            words_image = np.zeros((_h, _w, 3), np.uint8)
            layer_components += [[labels, stats, centroids, selected_labels,
                                  binarized_layer, components, layer, cluster, layer_box,
                                  stubs, isolated_points, stubs_image, words, words_image]]

        # Reduce Noise
        for component in layer_components:
            self.reduce_noise(component)

            if self.config.debug:
                stats = component[1]
                pre_selected_labels = component[3]
                components = component[5]

                for label in pre_selected_labels:
                    stat = stats[label]
                    left = stat[cv2.CC_STAT_LEFT]
                    top = stat[cv2.CC_STAT_TOP]
                    width = stat[cv2.CC_STAT_WIDTH]
                    height = stat[cv2.CC_STAT_HEIGHT]
                    right = left + width - 1
                    bottom = top + height - 1

                    cv2.rectangle(components, (left, top), (right, bottom), (255, 0, 255), 1, cv2.LINE_AA)

        # Find Stubs
        best_layer_index = None
        best_layer_overlap = 0
        best_layer_char_count = 999
        for i_component, component in enumerate(layer_components):

            self.find_stubs(component, _w, _h)
            stubs = component[9]
            best_stub_index, best_overlap = self.get_best_stub(component, box)
            best_char_count = len(stubs[best_stub_index][2]) if best_stub_index >= 0 else 999

            if best_overlap > 1.5 * best_layer_overlap:
                is_better = True
            elif best_overlap < best_layer_overlap / 1.5:
                is_better = False
            else:
                if best_layer_char_count < best_char_count:
                    is_better = False
                else:
                    is_better = True

            if is_better:
                best_layer_index = i_component
                best_layer_overlap = best_overlap
                best_layer_char_count = best_char_count

            stats = component[1]
            isolated_points = component[10]
            stubs_canvas = component[11]
            words = [stubs[best_stub_index]] if best_stub_index >= 0 else []
            component[12] = words
            words_canvas = component[13]
            if self.config.debug:

                colors = [
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 0, 0),
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255),
                    (0, 127, 255),
                    (0, 255, 127),
                    (127, 0, 255),
                    (255, 0, 127),
                    (127, 255, 0),
                    (255, 127, 0),
                    (0, 0, 125),
                    (0, 125, 0),
                    (125, 0, 0),
                    (255, 255, 255)
                ]
                for label in isolated_points:
                    stat = stats[label]
                    left = stat[cv2.CC_STAT_LEFT]
                    top = stat[cv2.CC_STAT_TOP]
                    width = stat[cv2.CC_STAT_WIDTH]
                    height = stat[cv2.CC_STAT_HEIGHT]
                    right = left + width - 1
                    bottom = top + height - 1
                    center = (left + width/2, top + height/2)
                    cv2.circle(stubs_canvas, (int(center[0]), int(center[1])), 3, (127, 127, 127), -1)
                    cv2.rectangle(stubs_canvas, (left, top), (right, bottom), (127, 127, 127), 1)
                    cv2.circle(words_canvas, (int(center[0]), int(center[1])), 3, (127, 127, 127), -1)
                    cv2.rectangle(words_canvas, (left, top), (right, bottom), (127, 127, 127), 1)

                for ic, line in enumerate(stubs):
                    a, b, stub = line
                    color = colors[ic % len(colors)]
                    for label in stub:
                        stat = stats[label]
                        left = stat[cv2.CC_STAT_LEFT]
                        top = stat[cv2.CC_STAT_TOP]
                        width = stat[cv2.CC_STAT_WIDTH]
                        height = stat[cv2.CC_STAT_HEIGHT]
                        right = left + width - 1
                        bottom = top + height - 1
                        center = (left + width/2, top + height/2)

                        cv2.circle(stubs_canvas, (int(center[0]), int(center[1])), 3, color, -1)
                        cv2.rectangle(stubs_canvas, (left, top), (right, bottom), color, 1)
                    x0 = int(stats[stub[0]][cv2.CC_STAT_LEFT])
                    x1 = int(stats[stub[-1]][cv2.CC_STAT_LEFT] + stats[stub[-1]][cv2.CC_STAT_WIDTH])
                    y0 = int(a*x0 + b)
                    y1 = int(a*x1 + b)
                    cv2.line(stubs_canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

                for ic, line in enumerate(words):
                    a, b, stub = line
                    color = colors[ic % len(colors)]
                    for label in stub:
                        stat = stats[label]
                        left = stat[cv2.CC_STAT_LEFT]
                        top = stat[cv2.CC_STAT_TOP]
                        width = stat[cv2.CC_STAT_WIDTH]
                        height = stat[cv2.CC_STAT_HEIGHT]
                        right = left + width - 1
                        bottom = top + height - 1
                        center = (left + width/2, top + height/2)

                        cv2.circle(words_canvas, (int(center[0]), int(center[1])), 3, color, -1)
                        cv2.rectangle(words_canvas, (left, top), (right, bottom), color, 1)
                    x0 = int(stats[stub[0]][cv2.CC_STAT_LEFT])
                    x1 = int(stats[stub[-1]][cv2.CC_STAT_LEFT] + stats[stub[-1]][cv2.CC_STAT_WIDTH])
                    y0 = int(a*x0 + b)
                    y1 = int(a*x1 + b)
                    cv2.line(words_canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        if best_layer_index is not None:
            best_layer_component = layer_components[best_layer_index]
            best_layer_stats = best_layer_component[1]
            for i_component, component in enumerate(layer_components):

                selected_labels = []
                for a, b, word in component[12]:
                    for label in word:
                        selected_labels += [label]

                if i_component == best_layer_index:
                    component[3] = selected_labels
                    continue

                stats = component[1]
                good_labels = []
                for label in selected_labels:
                    stat = stats[label]
                    left = stat[cv2.CC_STAT_LEFT]
                    top = stat[cv2.CC_STAT_TOP]
                    width = stat[cv2.CC_STAT_WIDTH]
                    height = stat[cv2.CC_STAT_HEIGHT]
                    right = left + width - 1
                    bottom = top + height - 1
                    area = width * height
                    should_break = False
                    for good_label in best_layer_component[3]:
                        good_stat = best_layer_stats[good_label]
                        good_left = good_stat[cv2.CC_STAT_LEFT]
                        good_top = good_stat[cv2.CC_STAT_TOP]
                        good_width = good_stat[cv2.CC_STAT_WIDTH]
                        good_height = good_stat[cv2.CC_STAT_HEIGHT]
                        good_right = good_left + good_width - 1
                        good_bottom = good_top + good_height - 1
                        good_area = good_width * good_height
                        overlap_area = ((min(right, good_right) - max(left, good_left)) *
                                        (min(bottom, good_bottom) - max(top, good_top)))
                        overlap = overlap_area / (area + good_area - overlap_area)
                        if 0.5 < overlap < 1.5:
                            good_labels += selected_labels
                            should_break = True
                            break
                    if should_break:
                        break

                component[3] = good_labels

        text_box = []
        for component in layer_components:
            stats = component[1]
            centroids = component[2]
            good_labels = component[3]
            for label in good_labels:
                stat = stats[label]
                centroid = centroids[label]
                width = stat[cv2.CC_STAT_WIDTH]
                height = stat[cv2.CC_STAT_HEIGHT]
                l_box = [[centroid[0], centroid[1]+height/2],
                         [centroid[0], centroid[1]-height/2],
                         [centroid[0]+width/2, centroid[1]],
                         [centroid[0]-width/2, centroid[1]]]
                text_box += l_box
        if len(text_box) > 0:
            text_box = cv2.boxPoints(cv2.minAreaRect(np.array(text_box).astype(np.int32)))
        else:
            text_box = box

        # Restore punctuation
        expanded_text_box = []
        box_center = np.mean(text_box, axis=0)
        for p in text_box:
            cx = box_center[0]
            cy = box_center[1]
            px = 1.2 * (p[0] - cx) + cx
            py = 1.2 * (p[1] - cy) + cy
            expanded_text_box += [(px, py)]
        expanded_text_box = np.array(expanded_text_box)

        text_box = []
        for component in layer_components:
            stats = component[1]
            centroids = component[2]
            isolated_points = component[10]
            selected_labels = component[3]

            for label in isolated_points:
                if Point(centroids[label]).within(Polygon(expanded_text_box)):
                    selected_labels += [label]

            for label in selected_labels:
                stat = stats[label]
                centroid = centroids[label]
                width = stat[cv2.CC_STAT_WIDTH]
                height = stat[cv2.CC_STAT_HEIGHT]
                l_box = [[centroid[0], centroid[1] + height / 2],
                         [centroid[0], centroid[1] - height / 2],
                         [centroid[0] + width / 2, centroid[1]],
                         [centroid[0] - width / 2, centroid[1]]]
                text_box += l_box

            component[3] = selected_labels
        if len(text_box) > 0:
            text_box = cv2.boxPoints(cv2.minAreaRect(np.array(text_box).astype(np.int32)))
        else:
            text_box = box

        expanded_text_box = []
        box_center = np.mean(text_box, axis=0)
        for p in text_box:
            cx = box_center[0]
            cy = box_center[1]
            px = 1.05 * (p[0] - cx) + cx
            py = 1.05 * (p[1] - cy) + cy
            expanded_text_box += [(px, py)]
        text_box = np.array(expanded_text_box)

        for component in layer_components:
            labels = component[0]
            good_labels = component[3]
            binarized_layer = component[4]
            components = component[5]
            layer = component[6]
            cluster = component[7]
            stubs = component[11]
            words = component[13]

            for label in good_labels:
                binarized_layer[labels == label] = 255

            binarized_layers += [binarized_layer]

            if self.do_visualisation:

                cv2.imshow("roi", roi)
                cv2.imshow("layer", layer)
                cv2.imshow("cluster", cluster)
                cv2.imshow("labels", cv2.applyColorMap((labels*255/labels.max()).astype(np.uint8), cv2.COLORMAP_JET))
                cv2.imshow("components", components)
                cv2.imshow("stubs", stubs)
                cv2.imshow("words", words)
                cv2.imshow("binarized_layer", binarized_layer)
                cv2.imshow("binarized_roi", binarized_roi)

                key = cv2.waitKey(0)
                if key & 0xff == ord('q'):
                    break
                if key & 0xff == ord('s'):
                    self.do_visualisation = False
                    break

                cv2.destroyWindow("roi")
                cv2.destroyWindow("layer")
                cv2.destroyWindow("cluster")
                cv2.destroyWindow("labels")
                cv2.destroyWindow("components")
                cv2.destroyWindow("stubs")
                cv2.destroyWindow("words")
                cv2.destroyWindow("binarized_layer")
                cv2.destroyWindow("binarized_roi")

        for binarized_layer in binarized_layers:
            binarized_roi = np.bitwise_and(binarized_roi, np.bitwise_not(binarized_layer))

        return binarized_roi, text_box

    def _select_good_char_box(self, roi_size, box, bin_layer, debug_mode=False):

        _w, _h = roi_size
        left, top, width, height = box
        min_size = 0.05 * (_h if _h < _w else _w)
        if width < min_size and height < min_size:
            return (False, 1) if debug_mode else False
        if width > 0.5 * _w or (height > 0.9 * _h and width > 0.3 * _w):
            return (False, 2) if debug_mode else False
        aspect_ratio = width / height
        if aspect_ratio < 1. / 15 or aspect_ratio > 15:
            return (False, 3) if debug_mode else False
        n_white = np.count_nonzero(bin_layer)
        if n_white / (width*height) < 0.1:
            return (False, 4) if debug_mode else False
        if n_white / (width*height) > 0.95 and width > _h:
            return (False, 5) if debug_mode else False
        return (True, 0) if debug_mode else True

    def _get_affinity_layers(self, box, roi):

        _h, _w = roi.shape[:2]
        box_mask = np.zeros((_h, _w, 3), np.uint8)
        cv2.fillPoly(box_mask, [box], (255, 255, 255))

        bins = int(256 / self.rgb_bin_size)
        hist, _ = np.histogramdd(np.bitwise_and(box_mask, roi).reshape(-1, 3),
                                 (bins, bins, bins), ((0, 256), (0, 256), (0, 256)))
        X = []
        weights = []
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    if hist[i, j, k] != 0:
                        X += [(i, j, k)]
                        weights += [hist[i, j, k]]
        X = np.array(X)
        weights = np.array(weights)
        Xw = X * np.hstack((weights.reshape(-1, 1),
                            weights.reshape(-1, 1),
                            weights.reshape(-1, 1)))
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=np.sum(weights) * self.dbscan_min_sample_frac)
        dbscan.fit(X, sample_weight=weights)
        clusters = []
        for label in np.unique(dbscan.labels_):
            cluster_indices = dbscan.labels_ == label
            clusters += [np.sum(Xw[cluster_indices], axis=0) /
                         np.sum(weights[cluster_indices])]
        clusters = np.array(clusters) * 256 / bins
        kmeans = KMeans(n_clusters=len(clusters))
        kmeans.cluster_centers_ = clusters
        labels = kmeans.predict(roi.reshape(-1, 3)).reshape(_h, _w)
        for label in np.unique(labels):
            layer = np.zeros((_h, _w), np.uint8)
            cluster = np.zeros_like(roi)
            layer[labels == label] = 255
            cluster[labels == label] = roi[labels == label]
            n_white = cv2.countNonZero(np.bitwise_and(box_mask[:, :, 0], layer))
            n_all = cv2.contourArea(box)
            if n_white > n_all / 2:
                layer = np.bitwise_not(layer)

            yield layer, cluster

    def get_connected_components(self, layer, cluster):

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(layer, cv2.DIST_L2, 5)
        dist_transform = (dist_transform * 255 / dist_transform.max()).astype(np.uint8)
        # _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        _h, _w = layer.shape[:2]
        ath_block_size = int(_h/8)*2+1
        sure_fg = cv2.adaptiveThreshold(dist_transform, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                        ath_block_size, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        labels = cv2.watershed(cluster, markers)
        labels[labels == -1] = 0
        labels[layer == 0] = 0

        stats = []
        centroids = []
        for i_label, label in enumerate(np.sort(np.unique(labels))):
            i, j = np.where(labels == label)
            top, left, bottom, right = i.min(), j.min(), i.max(), j.max()
            m = cv2.moments( (labels == label).astype(np.uint8))
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            stats += [{
                cv2.CC_STAT_LEFT: left,
                cv2.CC_STAT_TOP: top,
                cv2.CC_STAT_WIDTH: right - left + 1,
                cv2.CC_STAT_HEIGHT: bottom - top + 1,
            }]
            centroids += [[x, y]]
            labels[labels == label] = i_label

        if self.do_visualisation:

            cv2.imshow("layer", layer)
            cv2.imshow("cluster", cluster)
            cv2.imshow("dist_transform", dist_transform)
            cv2.imshow("sure_fg", sure_fg)
            cv2.imshow("markers", cv2.applyColorMap((labels * 255 / labels.max()).astype(np.uint8), cv2.COLORMAP_JET))

            key = cv2.waitKey(0)
            if key & 0xff == ord('q'):
                self.do_visualisation = False

            cv2.destroyWindow("layer")
            cv2.destroyWindow("cluster")
            cv2.destroyWindow("dist_transform")
            cv2.destroyWindow("sure_fg")
            cv2.destroyWindow("markers")

        return labels, stats, centroids

    def reduce_noise(self, component):

        stats = component[1]
        preselected_labels = component[3]

        selected_labels = []
        for label in preselected_labels:
            stat = stats[label]
            left = stat[cv2.CC_STAT_LEFT]
            top = stat[cv2.CC_STAT_TOP]
            width = stat[cv2.CC_STAT_WIDTH]
            height = stat[cv2.CC_STAT_HEIGHT]
            right = left + width - 1
            bottom = top + height - 1

            n_overlaps = 0
            is_contained = False
            for j_label in preselected_labels:
                if j_label == label:
                    continue
                j_stat = stats[j_label]
                j_left = j_stat[cv2.CC_STAT_LEFT]
                j_top = j_stat[cv2.CC_STAT_TOP]
                j_width = j_stat[cv2.CC_STAT_WIDTH]
                j_height = j_stat[cv2.CC_STAT_HEIGHT]
                j_right = j_left + j_width - 1
                j_bottom = j_top + j_height - 1
                if j_right < left or j_left > right or j_top > bottom or j_bottom < top:
                    continue
                overlap = ((min(right, j_right) - max(left, j_left) + 1) *
                           (min(bottom, j_bottom) - max(top, j_top) + 1))
                if overlap <= 0:
                    continue
                max_overlap = min((width * height), (j_width * j_height))
                if overlap / max_overlap > 0.25:
                    n_overlaps += 1
                if overlap / max_overlap > 0.95 and (j_width * j_height) > (width * height):
                    is_contained = True
            if n_overlaps > 4 or is_contained:
                continue

            selected_labels += [label]

        component[3] = selected_labels

    def find_stubs(self, component, _w, _h):

        roi_center = np.array((_w / 2, _h / 2))

        stats = component[1]
        centers = [(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH]/2,
                    stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]/2)
                   for stat in stats]

        pre_selected_labels = component[3]

        distances = []
        for label in pre_selected_labels:
            center = centers[label]
            distance = np.linalg.norm(center - roi_center)

            distances += [distance]

        sorted_labels = [pre_selected_labels[idx] for idx in np.argsort(np.array(distances))]
        if 0 < len(sorted_labels) < 3:
            stub = sorted_labels
            points_to_fit = np.array([centers[label] for label in stub])
            a, b = self.fit_line_to_points(points_to_fit)
            component[9] = [(a, b, stub)]
            component[10] = []
            return

        tolerance = _h / 5
        stub_candidates = []
        isolated_points = []
        points = [label for label in sorted_labels]
        rejected = []
        while len(points) > 0:
            seed = points.pop(0)
            stub = [seed]
            unused = []
            while len(points) > 0:
                point = points.pop(0)
                if len(stub) < 2:
                    stub += [point]
                else:
                    points_to_fit = np.array([centers[label] for label in stub])
                    line = np.polyfit(points_to_fit[:, 0], points_to_fit[:, 1], 1)
                    a, b = line[0], line[1]
                    residual = self.distance_to_line(centers[point], (a, b))

                    height = stats[point][cv2.CC_STAT_HEIGHT]
                    point_tolerance = max(tolerance, height / 2)
                    if residual < point_tolerance:
                        stub += [point]
                        points += rejected + unused
                        rejected = []
                        unused = []
                    else:
                        unused += [point]

            if len(stub) == 2:
                points = [seed] + unused
                rejected += [stub.pop()]
            elif len(stub) == 1:
                isolated_points += [seed]
                points = unused + rejected
                rejected = []
            else:
                points_to_fit = np.array([centers[label] for label in stub])
                line = np.polyfit(points_to_fit[:, 0], points_to_fit[:, 1], 1)
                a, b = line[0], line[1]
                stub_candidates += [(a, b, stub)]
                points = rejected + unused
                rejected = []

        component[9] = [(a, b, [stub[idx] for idx in np.argsort([centers[point][0] for point in stub])])
                        for a, b, stub in stub_candidates]
        component[10] = isolated_points

    def get_best_stub(self, component, box):

        stats = component[1]
        stubs = component[9]
        box_poly = Polygon(box)
        best_stub_index = -1
        best_overlap = 0
        for index, (a, b, stub) in enumerate(stubs):
            in_box_area = 0
            for label in stub:
                stat = stats[label]
                top = stat[cv2.CC_STAT_TOP]
                left = stat[cv2.CC_STAT_LEFT]
                width = stat[cv2.CC_STAT_WIDTH]
                height = stat[cv2.CC_STAT_HEIGHT]
                bottom = top + height - 1
                right = left + width - 1

                label_poly = Polygon([[left, top],
                                      [left, bottom],
                                      [right, bottom],
                                      [right, top]])

                in_box_area += label_poly.intersection(box_poly).area

            if in_box_area > best_overlap:
                best_stub_index = index
                best_overlap = in_box_area

        return best_stub_index, best_overlap

    def find_words(self, component):

        stats = component[1]
        stubs = component[9]
        isolated_points = component[10]

        centers = [(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH]/2,
                    stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]/2)
                   for stat in stats]

        words = []
        for a, b, stub in stubs:

            heights = []
            widths = []
            gaps = []
            for i, label in enumerate(stub):
                stat = stats[label]
                left = stat[cv2.CC_STAT_LEFT]
                width = stat[cv2.CC_STAT_WIDTH]
                height = stat[cv2.CC_STAT_HEIGHT]

                i_next = i+1 if (i+1) < len(stub) else -1

                widths += [width]
                heights += [height]
                if i_next >= 0:
                    next_left = stats[stub[i_next]][cv2.CC_STAT_LEFT]
                    gaps += [next_left - left - width]

            median_width = np.median(widths)
            median_gap = np.median(gaps)

            if median_gap < 0:
                isolated_points += stub
                continue

            if median_gap > median_width * 2:
                isolated_points += stub
                continue

            max_gap = min(median_width, max(median_width/4, 2*median_gap))
            large_gaps = np.where(gaps > max_gap)[0]
            break_indices = [0] + [index+1 for index in large_gaps] + [len(stub)]
            for start, end in zip(break_indices[:-1], break_indices[1:]):
                word = stub[start:end]
                word_points = np.array([centers[point] for point in word])
                word_line = np.polyfit(word_points[:, 0], word_points[:, 1], 1)
                word_a, word_b = word_line[0], word_line[1]

                words += [(word_a, word_b, word)]

        component[12] = words

    def distance_to_line(self, P, line):
        a, b = line

        v = np.array([1, a])
        v = v / np.linalg.norm(v)
        beta = np.array([0, b])
        H = np.dot(v, (P - beta)) * v + beta
        residual = np.linalg.norm(P - H)

        return residual

    def get_stub_mean_residuals(self, a, b, points):

        if len(points) <= 2:
            return 0

        residuals = np.array([self.distance_to_line(point, (a, b)) for point in points])
        mean = np.mean(residuals)

        return mean

    def fit_line_to_points(self, points_to_fit):
        if len(points_to_fit) == 0:
            return 0, 0
        if len(points_to_fit) == 1:
            return 0, points_to_fit[0][1]
        if len(points_to_fit) == 2:
            x1, y1 = points_to_fit[0][0], points_to_fit[0][1]
            x2, y2 = points_to_fit[1][0], points_to_fit[1][1]
            if x1 == x2:
                x1 = x2 - np.finfo(np.float32).eps
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            return a, b
        line = np.polyfit(points_to_fit[:, 0], points_to_fit[:, 1], 1)
        a, b = line[0], line[1]
        return a, b

    def refine_text_boxes(self, text_boxes, overlap_threshold=0.1):

        if len(text_boxes) == 0:
            return []

        attempt_merge = True

        sel_boxes = [box for box in text_boxes]
        while attempt_merge:

            attempt_merge = False

            boxes = np.array(sel_boxes)

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            p1 = boxes[:, 0]
            p2 = boxes[:, 1]
            p3 = boxes[:, 2]
            p4 = boxes[:, 3]
            polygons = [Polygon([ip1, ip2, ip3, ip4]) for ip1, ip2, ip3, ip4 in zip(p1, p2, p3, p4)]
            areas = np.array([p.area for p in polygons])
            indices = np.argsort(-areas)

            pick = []
            sel_boxes = []
            while len(indices) > 0:
                last = len(indices) - 1
                i = indices[last]
                pick.append(i)

                polygon = polygons[i]
                overlaps = np.array([polygon.intersection(polygons[other]).area for other in indices[:last]])
                overlaps = overlaps / areas[indices[:last]]

                group = np.concatenate(([last], np.where(overlaps > overlap_threshold)[0]))
                group_box = cascaded_union([polygons[indices[ig]] for ig in group]).minimum_rotated_rectangle
                sel_boxes += [sort_box(group_box.exterior.coords[:-1])]

                indices = np.delete(indices, group)

                if len(group) > 1:
                    attempt_merge = True

        sel_boxes = np.array(sel_boxes).astype("int")

        self.debugger.log("refined: ", np.array(text_boxes).shape, sel_boxes.shape)

        return sel_boxes
