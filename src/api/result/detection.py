import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.losses import binary_crossentropy
from sklearn.preprocessing import MultiLabelBinarizer

from commons.metrics import binary_accuracy, binary_crossentropy, binary_recall, focal_loss
from database.read_db import load_database_for_result, ProteinSequence
import setting
from .cal_gate import CalculateGate


LOGGER = logging.getLogger(setting.LOGGING_NAME)


class ModelDetection(object):
    '''detect array by saved model file
    '''
    def __init__(self, modelfile, model_home='../data/output/models', db_path='../data/input', colors=['red', 'green', 'blue']):
        self.modelfile = str(Path(model_home) / modelfile)
        self.db_path = Path(db_path)
        self.colors = colors
        self.df = pd.read_csv(Path(db_path) / 'sample_submission.csv')
        self.generator = None

    def load_database(self):
        self.generator = ProteinSequence(self.db_path, colors=self.colors, cal_gate=False)

    def save_csv(self):
        self.df.to_csv(self.db_path / 'submit' / (setting.LOGGING_NAME + '_submit.csv'), index=None)
        
    def predict(self, batch_size, gate=0.5, calculate_gate=True):
        model = load_model(
            self.modelfile,
            custom_objects={
                'loss': binary_crossentropy,
                'focal_loss': focal_loss,
                'binary_accuracy': binary_accuracy,
                'binary_precision': binary_recall,
                'binary_recall': binary_recall
            }
        )
        model.summary()
        LOGGER.info('model loaded')

        self.load_database()
        LOGGER.info('start get result')
        res = model.predict_generator(self.generator, workers=os.cpu_count(), verbose=1)
        LOGGER.info('model result got')

        if calculate_gate:
            cal = CalculateGate(model, db_path=self.db_path, colors=self.colors, use_cache=False)
            gate = cal.calculate(batch_size=batch_size)
        LOGGER.info('shape: %s' % str(res.shape))

        # get the best match 4 classes, but if x1<0, it will be ignored
        x1 = (res > gate).astype(np.int)
        # x2 = (res - gate) / (1 - gate)
        # index = np.argsort(x2)[:, -10:]
        # mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))
        # x2 = mlb.fit_transform(index)
        res = x1
        
        LOGGER.info('result got')

        submission = []
        submission_append = submission.append
        for row in res:
            subrow = ' '.join([str(i) for i in np.nonzero(row)[0]])
            submission_append(subrow)

        self.df['Predicted'] = submission
        self.save_csv()
        LOGGER.info('dataframe saved')
        
        
