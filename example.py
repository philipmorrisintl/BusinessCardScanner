# Copyright Philip Morris Products S.A. 2019

import os
import cv2
import json

import bcscanner


file_names = [
    "sample_images/bcard0001.jpg",
    "sample_images/bcard0002.jpg",
    "sample_images/bcard0003.jpg",
    "sample_images/bcard0004.jpg",
    "sample_images/bcard0005.jpg",
    "sample_images/bcard0006.jpg",
    "sample_images/bcard0007.jpg",
    "sample_images/bcard0008.jpg",
    "sample_images/multiScan.pdf",
]

config = bcscanner.Config(
    debug=True,
    splitter_debug=False,
    splitter_edge_model=None,
    ocr_debug=False,
    ocr_return_decorated=True,
    ocr_return_binarized=True,
    ocr_east_model=None,
    ocr_detection_target_width=640,
    ocr_detection_target_height=640,
    ocr_detection_confidence_threshold=0.4,
    ocr_detection_nms_threshold=0.75,
    ocr_binarization_dbscan_eps=1.5,
    ocr_binarization_dbscan_min_sample_frac=0.05,
    ocr_binarization_rgb_bin_size=4,
    ocr_binarization_roi_x_padding=0.4,
    ocr_binarization_roi_y_padding=0.0
)
scanner = bcscanner.BCScanner(config=config)

for file_name in file_names:

    should_break = False

    with open(file_name, 'rb') as image_file:

        _, image_type = os.path.splitext(file_name)
        results = scanner.scan_image(image_file.read(), image_type)

        print("Analyzing", file_name)
        print(" ==> Found {} sub-images".format(len(results)))
        for ires, result in enumerate(results):

            print("   -- Result # {}".format(ires))
            print(json.dumps(result["NER"], indent=2))
            print("===================================")
            print(json.dumps(result["OCR"]["detections"], indent=2))
            print("-----------------------------------")

            decorated = result["OCR"]["decorated"]
            binarized = result["OCR"]["binarized"]

            scale = 512. / decorated.shape[0]
            decorated = cv2.resize(decorated, None, fx=scale, fy=scale)
            scale = 512. / binarized.shape[0]
            binarized = cv2.resize(binarized, None, fx=scale, fy=scale)

            cv2.imshow("decorated", decorated)
            cv2.imshow("binarized", binarized)

            key = cv2.waitKey(0)
            if key & 0xff == ord('q'):
                should_break = True
                break

    if should_break:
        break
