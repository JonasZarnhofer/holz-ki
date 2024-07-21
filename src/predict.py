import cv2
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.structures import instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os.path
import numpy

from config import custom_config, setup, TEST, TRAIN

setup()
dataset_dicts = DatasetCatalog.get(TEST)

cfg = custom_config()

RESULT_DIR = "result/images"
os.makedirs(RESULT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = "result/model/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
predictor = DefaultPredictor(cfg)
for d in dataset_dicts:
    basename = os.path.basename(d["file_name"])
    name = os.path.splitext(basename)[0]
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_masks)
    # print(outputs["instances"].scores)
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = [
        "ast",
        "ast_punkt",
        "ast_faul",
        "splint",
        "riss",
        "verfärbung",
        "ungehobelt",
    ]
    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v = v.get_image()[:, :, ::-1]
    cv2.imwrite(RESULT_DIR + name + ".jpg", v)
    cv2.waitKey(0)
