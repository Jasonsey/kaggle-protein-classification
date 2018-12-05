import logging

from keras.applications import ResNet50, InceptionV3
from keras.layers import Dense, Input, Conv2D, ZeroPadding2D, Dropout, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model
from keras.metrics import MSE
from keras.optimizers import SGD

import config
from utils.metrics import get_loss, METRICS
from api.applications.inception_resnet_v2 import InceptionResNetV2

    
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

    conv_base = InceptionResNetV2(
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    for layer in conv_base.layers:
        layer.trainable = False

    img_input = Input(shape=input_shape)
    x = conv_base(img_input)
    x = Dropout(0.2)(x)
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


def get_model_inceptionrestnet(input_shape=[512,512,4], class_weight=None):
    LOGGER.info('Net Input Shape: %s' % str(input_shape))

    conv_base = InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    for layer in conv_base.layers:
        if layer.name != 'conv2d_1':
            layer.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(config.CLASSES, activation='sigmoid', name='fc28'))

    res = []
    model.summary(print_fn=res.append)
    LOGGER.info('\n' + '\n'.join(res) + '\n')
    
    if config.GPUS > 1:
        LOGGER.info('Using multi GPU')
        model = multi_gpu_model(model, gpus=config.GPUS, cpu_relocation=True)
        model.summary()

    sgd = SGD(lr=0.03, momentum=0.9, decay=1e-4)
    model.compile(loss=get_loss(class_weight), optimizer=sgd, metrics=METRICS)
    return model


MODEL = get_model_inceptionrestnet


if __name__ == "__main__":
    get_model_inceptionrestnet()