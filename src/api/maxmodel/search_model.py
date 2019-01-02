"""
train.py
"""

from pathlib import Path
from multiprocessing import cpu_count

from easydict import EasyDict
import pandas as pd

from commons.callbacks import get_callbacks
from commons.tools import init_path
from database import read_db
from .model import MODEL
import setting


def train(model, input_dataset: EasyDict, model_direction, pretrain_model):
    if pretrain_model:
        model.load_weights(pretrain_model)

    history = model.fit_generator(
        input_dataset.train,
        epochs=1,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        class_weight=input_dataset.class_weight,
        max_queue_size=50,
        verbose=1)

    return history


def main():
    database_path = '../data/input'
    model_direction = '../data/output/models/'
    batch_size = setting.BATCH_SIZE
    pretrain_model = None
    init_path([model_direction])

    input_dataset = read_db.read_database5(
        path=database_path,
        batch_size=batch_size,
        colors=setting.COLORS,
        use_datagen=False
    )
    train_loss, val_loss = [], []
    lambd_list = [0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
    for i in lambd_list:
        model = MODEL(input_shape=input_dataset.input_shape, class_weight=input_dataset.class_weight, lambd=i, deep=4)
        history = train(model, input_dataset, model_direction, pretrain_model)
        print(history.history['loss'])
        print(history.history['val_loss'])
        train_loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])
        del model
    
    df = pd.DataFrame({
        'train_loss': train_loss,
        'val_loss': val_loss
    })
    df.to_csv('../data/output/log/model_search_lambd_deep4.csv')
    print('OK')
        

if __name__ == '__main__':
    main()
