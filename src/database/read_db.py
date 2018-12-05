import logging
import asyncio
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from easydict import EasyDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing.pool import Pool

from utils.tools import deprocess_img, init_path, get_array_label, MyImageDataGenerator
import config


LOGGER = logging.getLogger(config.LOGGING_NAME)


def load_database(path: Path, colors=['green']):
    train_csv = path / 'train.csv'
    train_path = path / 'train'

    df = pd.read_csv(train_csv)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2)

    def get_database(data_frame):
        imgs, labels = [], []
        
        LOGGER.info('Creating Futures')
        pool = Pool()
        results = pool.map(get_array_label, ((train_path, row, colors) for _, row in data_frame.iterrows()))
        
        LOGGER.info('Getting Future Results')
        img_append, labels_append = imgs.append, labels.append
        i = 0
        for result in results:
            LOGGER.debug('Getting num.%s' % i)
            img_array, label = result
            img_append(img_array)
            labels_append(label)
            i += 1
        
        LOGGER.info('Formatting Data and Labels')
        imgs = np.asarray(imgs)
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        return imgs, labels

    imgs_train, labels_train = get_database(df_train)
    imgs_test, labels_test = get_database(df_test)

    LOGGER.info('Database Loaded')
    return imgs_train, imgs_test, labels_train, labels_test


def datagen(x_train, y_train, batch_size=128):
    epoch_size = len(y_train)
    if epoch_size % batch_size < batch_size / config.GPUS:    # 使用多GPU时，可能出现其中1个GPU 0 batchsize问题
        x_train = x_train[:-(epoch_size % batch_size)]
        y_train = y_train[:-(epoch_size % batch_size)]
    epoch_size = len(y_train)
    if epoch_size % batch_size:
        train_steps = int(epoch_size / batch_size) + 1
    else:
        train_steps = int(epoch_size / batch_size)
        
    train_datagen = ImageDataGenerator(
        rescale=None,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)

    return train_generator, train_steps, epoch_size


def read_database(path: Union[str, Path], batch_size=128, use_cache=True, cache_home='../data/output/cache', colors=['green']):
    '''
    读取数据集的接口
    '''
    path = Path(path)
    cache_home = Path(cache_home)
    init_path([cache_home])

    db_name = 'db_protein'
    for color in colors:
        db_name += ('_' + color)
    cache_path = Path(cache_home) / (db_name + '.pkl')

    if use_cache and cache_path.exists():
        LOGGER.info('Using database cache')
        with cache_path.open('rb') as f:
            dataset = pickle.load(f)
    else:
        x_train, x_test, y_train, y_test = load_database(path, colors=colors)
        dataset = EasyDict({
            'train': {
                'data': x_train,
                'labels': y_train
            },
            'test': {
                'data': x_test,
                'labels': y_test
            }
        })
        with cache_path.open('wb') as f:
            pickle.dump(dataset, f, protocol=4)

    train_generator, train_steps, epoch_size = datagen(
        x_train=dataset.train.data,
        y_train=dataset.train.labels,
        batch_size=batch_size
    )
    dataset = EasyDict({
        'train': train_generator, 
        'test': {
            'data': dataset.test.data,
            'labels': dataset.test.labels
        },
        'train_steps': train_steps,
        'epoch_size': epoch_size,
        'input_shape': dataset.train.data[0].shape,
        'batch_size': batch_size
    })

    LOGGER.info('All Database Read!')
    return dataset


def read_database2(path: Union[str, Path], batch_size=128, colors=['green'], **kwargs):
    '''使用生成器从文件读取图片
    '''
    path = Path(path)
    train_csv = path / 'train.csv'
    directory = str(path / 'train')
    df = pd.read_csv(train_csv)

    color_dict = dict(green='g', red='r', blue='b', yellow='y')
    color_layer = ''
    for color in colors:
        color_layer += color_dict[color]

    validation_split = 0.2
    train_datagen = MyImageDataGenerator(
        rescale=1/255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split)
    train_generator = train_datagen.flow_flow_mydataframe(
        dataframe=df,
        directory=directory,
        subset='training',
        x_col="Id", y_col="Target",
        target_size=config.TARGET_SIZE, color_layer=color_layer,
        batch_size=batch_size)
    x_test, y_test = train_generator.get_validation_data()

    temp = train_generator.samples * (1 - validation_split)
    epoch_size = int(temp) if temp == int(temp) else int(temp) + 1
    temp = epoch_size / batch_size
    train_steps = int(temp) if temp == int(temp) else int(temp) + 1

    class_num = list(train_generator.label_num_dict.items())
    class_num.sort(key=lambda item: item[0])
    class_weight = np.zeros((len(class_num),), dtype='float32')
    for i in range(len(class_num)):
        num = 1. if class_num[i][1] == 0 else class_num[i][1]
        class_weight[i] = train_generator.samples / (num * 10)

    dataset = EasyDict({
        'train': train_generator, 
        'test': {
            'data': x_test,
            'labels': y_test
        },
        'train_steps': train_steps,
        'epoch_size': epoch_size,
        'input_shape': x_test[0].shape,
        'batch_size': batch_size,
        'class_weight': class_weight
    })

    LOGGER.info('All Database Read!')
    
    return dataset

