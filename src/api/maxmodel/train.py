"""
train.py
"""

from pathlib import Path
from multiprocessing import cpu_count

from easydict import EasyDict

from commons.callbacks import get_callbacks
from commons.tools import init_path
from database import read_db
from .model import MODEL
import setting


def train(model, input_dataset: EasyDict, model_direction, pretrain_model):
    callbacks_list = get_callbacks(
        model_direction,
        epochsize=input_dataset.epoch_size,
        batchsize=setting.BATCH_SIZE
    )
    if pretrain_model:
        model.load_weights(pretrain_model)

    model.fit_generator(
        input_dataset.train,
        epochs=setting.EPOCHS,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        class_weight=input_dataset.class_weight,
        max_queue_size=50,
        verbose=1,
        workers=cpu_count(),
        use_multiprocessing=True,
        callbacks=callbacks_list)

    return model


def main():
    database_path = '../data/input/moredata'
    model_direction = '../data/output/models/'
    batch_size = setting.BATCH_SIZE
    pretrain_model = None
    init_path([model_direction])

    input_dataset = read_db.read_database5(
        path=database_path,
        batch_size=batch_size,
        colors=setting.COLORS,
    )

    model = MODEL(input_shape=input_dataset.input_shape, class_weight=input_dataset.class_weight)
    model = train(model, input_dataset, model_direction, pretrain_model)


if __name__ == '__main__':
    main()
