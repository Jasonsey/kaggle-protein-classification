import logging

from keras.applications import ResNet50
from keras.layers import Dense, Input, Conv2D, ZeroPadding2D, Dropout, UpSampling2D
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model
from keras.metrics import MSE
from keras.optimizers import SGD

import config
from utils.metrics import get_loss, METRICS


LOGGER = logging.getLogger(config.LOGGING_NAME)


def get_model(input_shape=(512, 512, 3), class_weight=None):
    LOGGER.info(input_shape)
    conv_base = ResNet50(
        pooling='avg',
        weights=None,
        include_top=False,
        input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(config.CLASSES, activation='sigmoid', name='fc28'))

    res = []
    model.summary(print_fn=res.append)
    LOGGER.info('\n' + '\n'.join(res) + '\n')
    
    if config.GPUS > 1:
        LOGGER.info('Using multi GPU')
        model = multi_gpu_model(model, gpus=config.GPUS, cpu_relocation=True)
        model.summary()

    model.compile(loss=get_loss(class_weight), optimizer='adam', metrics=METRICS)
    return model
    

def get_model2(input_shape=[224, 224, 3], class_weight=None):
    LOGGER.info('Net Input Shape: %s' % str(input_shape))

    conv_base = ResNet50(
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    for layer in conv_base.layers:
        layer.trainable = False

    img_input = Input(shape=input_shape)
    # x = Conv2D(3, (1,1), padding='same', input_shape=input_shape, name='conv1_change')(img_input)
    x = conv_base(img_input)
    x = Dropout(0.5)(x)
    # start_add = False
    # for layer in conv_base.layers:
    #     LOGGER.info(layer.name)
    #     if layer.name == 'conv1':
    #         start_add = True
    #     if start_add:
    #         layer.trainable = False
    #         x = layer(x)
    x = Dense(config.CLASSES, activation='sigmoid', name='fc28')(x)

    model = Model(inputs=img_input, outputs=x, name='resnet_3_channel')

    res = []
    model.summary(print_fn=res.append)
    LOGGER.info('\n' + '\n'.join(res) + '\n')
    
    if config.GPUS > 1:
        LOGGER.info('Using multi GPU')
        model = multi_gpu_model(model, gpus=config.GPUS, cpu_relocation=True)
        model.summary()

    model.compile(loss=get_loss(class_weight), optimizer='adam', metrics=METRICS)
    return model


MODEL = get_model
