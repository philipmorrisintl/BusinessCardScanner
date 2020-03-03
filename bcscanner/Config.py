
class ConfigBase:

    def get_attr(self, key, default, prefix="", **kwargs):
        return kwargs[prefix+key] if prefix+key in kwargs else kwargs[key] if key in kwargs else default


class SplitterConfig(ConfigBase):

    def __init__(self, **kwargs):
        prefix = "splitter_"
        self.debug = self.get_attr("debug", False, prefix=prefix, **kwargs)
        self.edge_model = self.get_attr("edge_model", None, prefix=prefix, **kwargs)


class OCRConfig(ConfigBase):

    def __init__(self, **kwargs):
        prefix = "ocr_"
        self.debug = self.get_attr("debug", False, prefix=prefix, **kwargs)
        self.return_decorated = self.get_attr("return_decorated", True, prefix=prefix, **kwargs)
        self.return_binarized = self.get_attr("return_binarized", False, prefix=prefix, **kwargs)
        self.east_model = self.get_attr("east_model", None, prefix=prefix, **kwargs)

        self.detection_target_width = self.get_attr("detection_target_width", 640, prefix=prefix, **kwargs)
        self.detection_target_height = self.get_attr("detection_target_height", 640, prefix=prefix, **kwargs)
        self.detection_confidence_threshold = self.get_attr("detection_confidence_threshold", 0.4, prefix=prefix, **kwargs)
        self.detection_nms_threshold = self.get_attr("detection_nms_threshold", 0.75, prefix=prefix, **kwargs)

        self.binarization_dbscan_eps = self.get_attr("binarization_dbscan_eps", 3.5, prefix=prefix, **kwargs)
        self.binarization_dbscan_min_sample_frac = self.get_attr("binarization_dbscan_min_sample_frac", 0.05,
                                                                 prefix=prefix, **kwargs)
        self.binarization_rgb_bin_size = self.get_attr("binarization_rgb_bin_size", 4, prefix=prefix, **kwargs)
        self.binarization_roi_x_padding = self.get_attr("binarization_roi_x_padding", 0.4, prefix=prefix, **kwargs)
        self.binarization_roi_y_padding = self.get_attr("binarization_roi_y_padding", 0.0, prefix=prefix, **kwargs)


class Config(ConfigBase):

    def __init__(self, **kwargs):

        self.splitter = SplitterConfig(**kwargs)
        self.ocr = OCRConfig(**kwargs)

        self.debug = self.get_attr("debug", False, **kwargs)
        self.ner_model = self.get_attr("ner_model", None, **kwargs)
