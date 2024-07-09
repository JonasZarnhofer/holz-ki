from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

TEST = "test"
TRAIN = "train"

register_coco_instances(
    TEST, {}, "data/oak_data_2/test/oak_test.json", "data/oak_data_2/test"
)

register_coco_instances(
    TRAIN, {}, "data/oak_data_2/train/train_oak.json", "data/oak_data_2/train"
)


load_coco_json(
    "data/oak_data_2/test/oak_test.json",
    "data/oak_data_2/test",
    TEST,
    {},
)

load_coco_json(
    "data/oak_data_2/train/train_oak.json",
    "data/oak_data_2/train",
    TRAIN,
    {},
)


def custom_config():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"

    cfg.DATASETS.TRAIN = (TRAIN,)
    cfg.DATASETS.TEST = TEST  # no metrics implemented for this dataset

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = (
        100000  # 300 iterations seems good enough, but you can certainly train longer
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 3 classes (data, fig, hazelnut)

    cfg.OUTPUT_DIR = "result"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def train(cfg):
    from detectron2.engine import DefaultTrainer

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


from detectron2.data import DatasetCatalog

dataset_dicts = DatasetCatalog.get(TEST)

from detectron2.utils.logger import setup_logger
import cv2
from detectron2.structures import instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# from google.colab.patches import cv2_imshow
import os.path
import numpy

# path = "/content/result/*jpg"
cfg = custom_config()
train(cfg)
# cfg.MODEL.WEIGHTS = "/content/Deckschichten_Version2_skaliert/model_final.pth"
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
    cv2.imwrite("result/" + name + ".jpg", v)
    cv2.waitKey(0)
