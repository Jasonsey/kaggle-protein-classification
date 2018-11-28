"""
Config File
"""

CLASSES = 28
BATCH_SIZE = 8   # 单GPU上的batch大小
CUDA_VISIBLE_DEVICES = '1'
TARGET_SIZE = (512, 512)

# -------------------logger config ----------------
LOGGING_HOME = '../data/output/log'
LOGGING_NAME = 'protein'


def init_config():
    global GPUS, BATCH_SIZE
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))
    BATCH_SIZE *= GPUS

init_config()
