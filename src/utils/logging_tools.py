import logging
from pathlib import Path

import config
from utils.tools import init_path


def set_logger(log_name='protein'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # 总开关

    init_path([config.LOGGING_HOME])
    fh = logging.FileHandler(Path(config.LOGGING_HOME) / (log_name + '.run.log'))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info('\n%s %s %s\n' % ('='*20, 'Starting Server...', '='*20))


if __name__ == "__main__":
    set_logger()
    loggerrr = logging.getLogger('protein')
    loggerrr.info((1,2,'111'))
