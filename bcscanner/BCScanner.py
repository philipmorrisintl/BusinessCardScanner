
from .utils import DebugUtils
from .CardSplitter import CardSplitter
from .OCR import OCR
from .NER import NER
from .Config import Config


class BCScanner:

    def __init__(self, config=Config()):

        self.config = config
        self.debugger = DebugUtils()
        self.debugger.set_debug(config.debug)
        self.splitter = CardSplitter(config=config.splitter)
        self.ocr = OCR(config=config.ocr)
        self.ner = NER(config=config)

    def scan_image(self, image_bytes, image_type, scan_mode="One Sided", images_as_data_urls=False):

        results = []

        for card in self.splitter.split_raw(image_bytes, image_type, scan_mode):
            results += [self.process_image(card, images_as_data_urls)]

        return results

    def process_image(self, card, images_as_data_urls):

        ocr_results = self.ocr.process_image(card, images_as_data_urls)

        lines = [result["text"] for result in ocr_results["detections"]]
        ner_results = self.ner.parse(lines)

        return {
            "status": "success",
            "NER": ner_results,
            "OCR": ocr_results
        }
