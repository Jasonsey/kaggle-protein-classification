"""
train.py
"""

from pathlib import Path
from multiprocessing import cpu_count

from easydict import EasyDict

from utils.callbacks import get_callbacks
from utils.tools import init_path
from database import read_db
from .model import MODEL
import config


def train(model, input_dataset: EasyDict, model_direction, pretrain_model):
    callbacks_list = get_callbacks(
        model_direction,
        epochsize=input_dataset.epoch_size,
        batchsize=input_dataset.batch_size
    )
    if pretrain_model:
        model.load_weights(pretrain_model)

    model.fit_generator(
        input_dataset.train,
        steps_per_epoch=input_dataset.train_steps,
        epochs=10000,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        verbose=1,
        workers=cpu_count()-1,
        use_multiprocessing=True,
        max_queue_size=20,
        callbacks=callbacks_list)
    return model


def main():
    database_path = '../data/input'
    model_direction = '../data/output/models/'
    batch_size = config.BATCH_SIZE
    pretrain_model = None
    init_path([model_direction])

    input_dataset = read_db.read_database2(
        path=database_path,
        batch_size=batch_size,
        use_cache=True,
        colors=['red', 'green', 'blue']
    )

    model = MODEL(input_shape=input_dataset.input_shape, class_weight=input_dataset.class_weight)
    model = train(model, input_dataset, model_direction, pretrain_model)


if __name__ == '__main__':
    main()
