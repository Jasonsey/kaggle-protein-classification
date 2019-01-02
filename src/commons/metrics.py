from typing import List

from keras import backend as K
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.losses import binary_crossentropy


def binary_recall(y_true, y_pred):
    '''计算有出现的类别中被成功识别出来的概率
    '''
    tp = K.sum(K.round(y_true * y_pred), axis=-1)
    sum_true = K.sum(y_true, axis=-1)
    precision = tp / sum_true
    return precision


def weight_loss(y_true, y_pred, class_weight=None):
    output = K.binary_crossentropy(y_true, y_pred)
    if class_weight is not None:
        output *= class_weight

    y_p = K.sum(y_true, axis=1)
    y_n = K.sum(K.cast(K.equal(y_true, 0.), K.floatx()), axis=1)
    y_n, y_p = K.reshape(y_n, (-1, 1)), K.reshape(y_p, (-1, 1))

    one = K.ones((1, K.shape(y_true)[1]))
    y_p *= one
    y_n *= one

    output *= (y_true * y_n + (1.0 - y_true) * y_p) / (y_n + y_p)

    output = K.mean(output, axis=1)
    return output


def focal_loss(y_true, y_pred, gamma=2):
    y_ = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred) + K.epsilon()
    loss = -((1.0 - y_)**gamma)*K.log(y_)

    weight = y_true * 10. + (1.0 - y_true) * 1.
    loss *= weight

    loss = K.mean(loss, axis=-1)
    return loss


def hinge_loss(y_true, y_pred):
    y_ = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    loss = K.maximum(1. - y_, 0.)

    weight = y_true * 15. + (1.0 - y_true) * 1.
    loss *= weight

    loss = K.mean(loss, axis=-1)
    return loss


def get_loss(class_weight=None):
    if class_weight is None:
        loss = binary_crossentropy
    else:
        def loss(y_true, y_pred):
            return weight_loss(y_true, y_pred, class_weight)
    return loss


METRICS: List = [binary_accuracy, binary_recall]
LOSS = focal_loss

