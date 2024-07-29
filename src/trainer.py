from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import (
    build_detection_train_loader,
    build_detection_test_loader,
)

from mapper import CustomDatasetMapper


class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(
            cfg, mapper=CustomDatasetMapper(cfg, is_train=True)
        )

    def build_test_loader(self, cfg):
        return build_detection_test_loader(
            cfg, mapper=CustomDatasetMapper(cfg, is_train=False)
        )
