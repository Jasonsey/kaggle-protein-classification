import os

from api.result.detection import ModelDetection
from api.maxmodel.train import main as maxmodel_train

import setting
from commons.logging_tools import set_logger


def main(flag):
    os.environ["CUDA_VISIBLE_DEVICES"] = setting.CUDA_VISIBLE_DEVICES
    set_logger(setting.LOGGING_NAME)
    
    if flag == 1:
        maxmodel_train()
    elif flag == 2:
        # predict maxmodel_4_0
        detection = ModelDetection(modelfile='ckpt_model.val_f1.best.h5', model_home='../data/output/models/protein_maxmodel_4_0', colors=['red', 'green', 'blue', 'yellow'])
        detection.predict(batch_size=setting.FORWARD_BATCH_SIZE)
    elif flag == 3:
        # predict maxmodel_5_0
        detection = ModelDetection(modelfile='ckpt_model.val_f1.best.h5', model_home='../data/output/models/protein_maxmodel_5_0', colors=['red', 'green', 'blue', 'yellow'])
        detection.predict(batch_size=setting.FORWARD_BATCH_SIZE)


if __name__ == "__main__":
    main(1)
