from io import BytesIO
from tempfile import SpooledTemporaryFile
from detectron2.engine import DefaultTrainer
from detectron2.data.dataset_mapper import DatasetMapper

from config import custom_config, setup, TEST, TRAIN

from mapper import CustomDatasetMapper

from detectron2.utils.file_io import PathManager
from PIL import Image
import requests


setup()
cfg = custom_config()

CustomDatasetMapper(cfg, is_train=True)
