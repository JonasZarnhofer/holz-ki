from detectron2.engine import DefaultTrainer
from config import custom_config, setup, TEST, TRAIN

from trainer import CustomTrainer


def train(cfg):
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


setup()
cfg = custom_config()
train(cfg)
