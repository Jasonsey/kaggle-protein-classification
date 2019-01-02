"""
Config File
"""
from pathlib import Path


CUDA_VISIBLE_DEVICES = '3'
CLASSES = 28
BATCH_SIZE = 16  # 单GPU上的batch大小
FORWARD_BATCH_SIZE = 32
EPOCHS = 1000
EARLEY_STOP = 100
TARGET_SIZE = (512, 512)

COLORS = ['red', 'green', 'blue', 'yellow']

# -------------------logger config ----------------
LOGGING_NAME = 'protein_maxmodel_4_0'
# LOGGING_NAME = 'protein_maxmodel_5_0'
# LOGGING_NAME = 'protein_maxmodel_predict_4_0'
# LOGGING_NAME = 'protein_maxmodel_predict_5_0'


LOGGING_HOME = '../data/output/log/' + LOGGING_NAME


def init_path(paths: list):
    '''创建路径，不存在则新建
    '''
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


def init_config():
    global GPUS, BATCH_SIZE
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))
    BATCH_SIZE *= GPUS
    init_path([LOGGING_HOME])


init_config()
