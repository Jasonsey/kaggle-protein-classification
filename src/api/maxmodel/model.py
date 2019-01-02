import logging

from keras.applications import ResNet50, InceptionV3, VGG16
from keras.layers import Dense, Input, Conv2D, ZeroPadding2D, Dropout, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model
from keras import layers, regularizers
from keras.utils import multi_gpu_model
from keras.metrics import MSE
from keras.optimizers import SGD, Adam

import setting
from commons.metrics import METRICS, LOSS
from api.applications.inception_resnet_v2 import InceptionResNetV2

    
LOGGER = logging.getLogger(setting.LOGGING_NAME)


def get_model(input_shape=(1024, ), class_weight=None, deep=4, lambd=0):
    LOGGER.info('Net Input Shape: %s' % str(input_shape))

    net_input = layers.Input(shape=input_shape)
    x = conv_bn(net_input, deep=deep, lambd=lambd)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    net_output = layers.Dense(setting.CLASSES, activation='sigmoid', name='fc28')(x)
    model = Model(inputs=net_input, outputs=net_output, name='maxmodel')

    res = []
    model.summary(print_fn=res.append)
    LOGGER.info('\n' + '\n'.join(res) + '\n')

    # opt = SGD(lr=0.03, momentum=0.9, decay=1e-4)
    opt = Adam(lr=0.0001)
    model.compile(loss=LOSS, optimizer=opt, metrics=METRICS)
    return model


def conv_bn(input_tensor, deep=4, lambd=0.0, ks=3, start_filters=32):
    x = input_tensor
    for i in range(deep):
        for _ in range(2):
            x = layers.Conv2D(start_filters * 2 ** i, ks, padding='same', kernel_regularizer=regularizers.l2(lambd))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        x = layers.MaxPool2D((2, 2), (2, 2))(x)
    return x


MODEL = get_model
