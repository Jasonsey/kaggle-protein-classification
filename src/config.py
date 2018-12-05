"""
Config File
"""
from utils.tools import init_path


CUDA_VISIBLE_DEVICES = '2'

CLASSES = 28
BATCH_SIZE = 8   # 单GPU上的batch大小
EPOCHS = 100
EARLEY_STOP = 20
TARGET_SIZE = (512, 512)

COLORS = ['red', 'green', 'blue', 'yellow']

# -------------------logger config ----------------
LOGGING_NAME = 'protein_inception_resnet_pretrain'
LOGGING_HOME = '../data/output/log/' + LOGGING_NAME


def init_config():
    global GPUS, BATCH_SIZE
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))
    BATCH_SIZE *= GPUS
    init_path([LOGGING_HOME])


init_config()
