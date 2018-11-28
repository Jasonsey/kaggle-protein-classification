import os

from api.restnet50.train import main as res50_train

import config
from utils.logging_tools import set_logger


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    set_logger(config.LOGGING_NAME)
    
    res50_train()


if __name__ == "__main__":
    main()
