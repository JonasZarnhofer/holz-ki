from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances, load_coco_json
import os


TEST = "test"
TRAIN = "train"


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

    cfg.OUTPUT_DIR = "result/model"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def setup():
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
