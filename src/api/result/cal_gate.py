import os
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from keras.models import load_model, Model
from keras.losses import binary_crossentropy
from multiprocessing.pool import Pool

from commons.metrics import binary_accuracy, binary_crossentropy, binary_recall
from database.read_db import load_database_for_result, ProteinSequence
import setting


LOGGER = logging.getLogger(setting.LOGGING_NAME)
custom_objects={
    'loss': binary_crossentropy,
    'binary_accuracy': binary_accuracy,
    'binary_precision': binary_recall
}


class CalculateGate(object):
    '''detect array by saved model file
    '''
    def __init__(self, modelfile, db_path='../data/input', colors=['red', 'green', 'blue'], use_cache=True):
        LOGGER.info('Start calculating gates')
        if isinstance(modelfile, Model):
            self.model = modelfile
        elif isinstance(modelfile, (str, Path)):
            modelfile = str(Path('../data/output/models') / modelfile)
            self.model = load_model(modelfile, custom_objects=custom_objects)

        self.db_path = Path(db_path)
        self.colors = colors
        self.df = pd.read_csv(Path(db_path) / 'train.csv')

        self.gates = None
        self.gates_path = self.db_path / 'submit' / (setting.LOGGING_NAME + '_gates.npy')
        self.use_cache = use_cache
        self.generator = None

    def load_database(self):
        self.generator = ProteinSequence(self.db_path, colors=self.colors, cal_gate=True)

    def save_gate(self):
        if self.gates is not None:
            np.save(self.gates_path, self.gates)
    
    def load_gate(self):
        self.gates = np.load(self.gates_path)

    def calculate(self, batch_size):
        if self.use_cache and self.gates_path.exists():
            LOGGER.info('Using gates cache')
            self.load_gate()
            return self.gates

        self.load_database()
        LOGGER.info('start get result')
        res = self.model.predict_generator(self.generator, workers=os.cpu_count(), verbose=1)
        LOGGER.info('model result got')
        LOGGER.info('res shape: %s' % str(res.shape))

        self.labels = self.generator.labels
        LOGGER.info('labels shape: %s' % str(self.labels.shape))
        pool = Pool()
        gates = pool.map(_calculate, ((res[:, i], self.labels[:, i]) for i in range(setting.CLASSES)))
        self.gates = np.asarray(gates)
        LOGGER.info('gates calculated')

        self.save_gate()
        LOGGER.info('gates saved')
        return self.gates
        

def _calculate(x_y):
    x, y = x_y
    x_max, x_min = x.max(), x.min()
    x = (x - x_min) / (x_max - x_min)
    f1_score = np.zeros((100, ), dtype='float32')
    for ii in range(100):
        gate = ii / 100
        y_ = (x > gate).astype('float32')
        tp = np.sum(y * y_)
        sum_p = np.sum(y)
        sum_p_ = np.sum(y_)
        precision = tp / sum_p
        recall = tp / sum_p_
        if precision + recall == 0:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_score[ii] = f1
        # LOGGER.info('tp: %s, sum_p: %s, sum_p_: %s' % (tp, sum_p, sum_p_))

    best_gate = np.argmax(f1_score) / 100
    LOGGER.info('best gate: %s' % f1_score[np.argmax(f1_score)])
    best_gate = best_gate * (x_max - x_min) + x_min
    return best_gate
