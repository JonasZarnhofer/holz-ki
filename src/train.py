from detectron2.engine import DefaultTrainer
from config import custom_config, setup, TEST, TRAIN




def train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

setup()
cfg = custom_config()
train(cfg)
