import os
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
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from multiprocessing.pool import Pool as Pool

from commons.tools import deprocess_img, init_path, get_array_label, MyImageDataGenerator
import setting
from api.encoder_model.detection import predict


LOGGER = logging.getLogger(setting.LOGGING_NAME)


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


def load_database3(path: Path, encoder: str, colors=['green']):
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

    LOGGER.info('Encoding input data')
    imgs_train = predict(imgs_train, encoder)
    imgs_test = predict(imgs_test, encoder)
    
    LOGGER.info('Database Loaded')
    return imgs_train, imgs_test, labels_train, labels_test


def load_database_for_result(path: Path, colors=['green'], cal_gate=False):
    '''读取待检测图片到内存中
    '''
    if not cal_gate:
        test_csv = path / 'sample_submission.csv'
        test_path = path / 'test'
    else:
        test_csv = path / 'train.csv'
        test_path = path / 'train'

    df = pd.read_csv(test_csv)

    def get_database(data_frame):
        imgs = []
        labels = []
        
        LOGGER.info('Creating Futures')
        mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))
        for _, row in data_frame.iterrows():
            params = (test_path, row, colors)
            img, label = get_array_label(params)
            img = img
            label = mlb.fit_transform(label)
            imgs.append(img)
            labels.append(label)
            if len(labels) == setting.FORWARD_BATCH_SIZE:
                x = np.asarray(imgs)
                y = np.asarray(labels)
                imgs, labels = [], []
                yield x, y
        if len(labels) != 0:
            x = np.asarray(imgs)
            y = np.asarray(labels)
            yield x, y
        
    generater = get_database(df)

    LOGGER.info('Database Loaded')
    return generater


class ProteinSequence(Sequence):
    '''迭代返回batch的图片信息

    用于计算最终结果并提交时使用
    '''
    def __init__(self, path: Path, colors=['green'], cal_gate=False, batch_size=None, predict=True):
        if cal_gate:
            db_file = path / 'train.csv'
            self.db_folder = path / 'train'
        else:
            db_file = path / 'sample_submission.csv'
            self.db_folder = path / 'test'
        self.db = pd.read_csv(db_file)
        self.cal_gate = cal_gate
        self.colors = colors
        self.labels = self.get_labels() if cal_gate else None
        if predict:
            self.batch_size = setting.FORWARD_BATCH_SIZE if batch_size is None else batch_size
        else:
            self.batch_size = setting.BATCH_SIZE if batch_size is None else batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.db) / self.batch_size))
    
    def __getitem__(self, idx):
        imgs = []
        ys = []
        df = self.db.iloc[idx * self.batch_size:idx * self.batch_size + self.batch_size]     # remenber loc is [start,stop)
        mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))
        i = idx * self.batch_size
        for _, row in df.iterrows():
            params = (self.db_folder, row, self.colors)
            img, label = get_array_label(params)
            img = img
            label = mlb.fit_transform((label,))
            imgs.append(img)
            ys.append(label)
        imgs = np.asarray(imgs)
        if len(ys) > 1:
            ys = np.concatenate(ys)
        elif len(ys) == 1:
            ys = ys[0]

        if predict:
            return imgs
        else:
            return imgs, ys
    
    def get_labels(self):
        mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))
        labels = self.db.get('Target')
        if labels is None:
            raise ValueError('Please check your data, the labels should not be None')
        labels = labels.map(lambda x: [int(i) for i in x.split()]).tolist()
        labels = mlb.fit_transform(labels)
        return labels


class ProteinSequenceTrain(Sequence):
    '''迭代返回batch的图片信息

    用于训练网络时使用
    '''
    def __init__(self, path: Path, db: pd.DataFrame, colors=['green'], batch_size=None, is_train=True):
        self.img_folder = path / 'train'

        self.db = db
        self.colors = colors
        self.batch_size = setting.BATCH_SIZE if batch_size is None else batch_size
        self.labels = self.get_labels()

        if is_train:
            self.train_datagen = ImageDataGenerator(
                rotation_range=360,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
        else:
            self.train_datagen = None
    
    def __len__(self):
        return int(np.ceil(len(self.db) / self.batch_size))
    
    def __getitem__(self, idx):
        xs = []
        ys = []
        df = self.db.iloc[idx * self.batch_size:idx * self.batch_size + self.batch_size - 1]     # remenber loc is [start,stop], loc means labels
        mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))

        for _, row in df.iterrows():
            params = (self.img_folder, row, self.colors)
            img, label = get_array_label(params, self.train_datagen)
            label = mlb.fit_transform((label,))
            xs.append(img)
            ys.append(label)
        xs = np.asarray(xs)
        if len(ys) > 1:
            ys = np.concatenate(ys)
        elif len(ys) == 1:
            ys = ys[0]
        else:
            raise ValueError('length of ys should not less than 1')
        return xs, ys
    
    def get_labels(self):
        mlb = MultiLabelBinarizer(classes=list(range(setting.CLASSES)))
        labels = self.db.get('Target')
        if labels is None:
            raise ValueError('Please check your data, the labels should not be None')
        labels = labels.map(lambda x: [int(i) for i in x.split()]).tolist()
        labels = mlb.fit_transform(labels)
        return labels


def datagen(x_train, y_train, batch_size=128):
    epoch_size = len(y_train)
    if epoch_size % batch_size < batch_size / setting.GPUS:    # 使用多GPU时，可能出现其中1个GPU 0 batchsize问题
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
    train_generator = train_datagen.flow_from_mydataframe(
        dataframe=df,
        directory=directory,
        subset='training',
        x_col="Id", y_col="Target",
        target_size=setting.TARGET_SIZE, color_layer=color_layer,
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



def read_database3(path: Union[str, Path], batch_size=128, use_cache=True, cache_home='../data/output/cache', colors=['red', 'green', 'blue'], encoder='InceptionResNet50'):
    '''
    读取数据集的接口
    '''
    path = Path(path)
    cache_home = Path(cache_home)
    init_path([cache_home])

    db_name = encoder + '_cache'
    for color in colors:
        db_name += ('_' + color)
    cache_path = Path(cache_home) / (db_name + '.pkl')

    if use_cache and cache_path.exists():
        LOGGER.info('Using database cache')
        with cache_path.open('rb') as f:
            dataset = pickle.load(f)
    else:
        x_train, x_test, y_train, y_test = load_database3(path, colors=colors, encoder=encoder)
        num_class = np.sum(y_train, axis=0, dtype='float32')
        class_weight = len(y_train) / (num_class * 10)

        dataset = EasyDict({
            'train': {
                'data': x_train,
                'labels': y_train
            },
            'test': {
                'data': x_test,
                'labels': y_test
            },
            'input_shape': x_train[0].shape,
            'batch_size': batch_size,
            'epoch_size': len(y_train),
            'class_weight': class_weight
        })
        with cache_path.open('wb') as f:
            pickle.dump(dataset, f, protocol=4)

    LOGGER.info('All Database Read!')
    return dataset


def read_database4(path: Union[str, Path], batch_size=128, use_cache=True, cache_home='../data/output/cache', colors=['red', 'green', 'blue'], encoder='InceptionResNet50'):
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
            dataset = joblib.load(f)
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
            joblib.dump(dataset, f, protocol=4)

    num_class = np.sum(dataset.train.labels, axis=0)
    class_weight = len(dataset.train.labels) / (num_class * 10)

    train_datagen = MyImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow(
        x=dataset.train.data,
        y=dataset.train.labels,
        batch_size=batch_size
    )

    if not len(dataset.train.labels) % batch_size:
        train_steps = len(dataset.train.labels) // batch_size
    else:
        train_steps = len(dataset.train.labels) // batch_size 

    dataset = EasyDict({
        'train': train_generator, 
        'test': {
            'data': dataset.test.data,
            'labels': dataset.test.labels
        },
        'train_steps': train_steps,
        'epoch_size': len(dataset.train.labels),
        'input_shape': dataset.test.data[0].shape,
        'batch_size': batch_size,
        'class_weight': class_weight
    })

    LOGGER.info('All Database Read!')
    return dataset


def read_database5(path: Union[str, Path], batch_size=128, colors=['red', 'green', 'blue'], use_datagen=True):
    '''利用sequence 从本地读取数据

    作为自定义模型的数据集读取入口
    '''
    path = Path(path)
    db_file = path / 'train.csv'
    df = pd.read_csv(db_file)
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify = df['Target'].map(lambda x: x[:3] if '27' not in x else '0')
    )
    train_generator = ProteinSequenceTrain(path, df_train, colors, batch_size, is_train=use_datagen)
    test_generator = ProteinSequenceTrain(path, df_test, colors, batch_size, is_train=False)

    train_labels = train_generator.labels
    num_class = np.sum(train_labels, axis=0)
    class_weight = len(train_labels) / (num_class * 10)

    LOGGER.info('start getting test data')
    pool = Pool(os.cpu_count())
    results = pool.map(_read_db5_map, ((x, test_generator) for x in range(len(test_generator))))
    LOGGER.info('result got')
    x_test, y_test = [], []
    x_test_append, y_test_append = x_test.append, y_test.append
    for res in results:
        img, label = res
        x_test_append(img)
        y_test_append(label)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    LOGGER.info('test data got')

    dataset = EasyDict({
        'train': train_generator, 
        'test': {
            'data': x_test,
            'labels': y_test
        },
        'epoch_size': len(df_train),
        'input_shape': x_test[0].shape,
        'class_weight': class_weight
    })
    return dataset


def _read_db5_map(index_generator):
    index, generator = index_generator
    return generator[index]
