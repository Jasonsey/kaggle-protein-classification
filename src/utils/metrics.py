from typing import List

from keras import backend as K
from keras.metrics import binary_accuracy, binary_crossentropy


def binary_precision(y_true, y_pred):
    '''计算有出现的类别中被成功识别出来的概率
    '''
    tp = K.sum(K.round(y_true * y_pred), axis=-1)
    sum_true = K.sum(y_true)
    precision = tp / sum_true
    return precision


def weight_loss(y_true, y_pred, class_weight=None):
    output = K.binary_crossentropy(y_true, y_pred)
    if class_weight is not None:
        output *= class_weight

    y_p = K.sum(y_true)
    y_n = K.sum(K.cast(K.equal(y_true, K.constant(0.)), K.floatx()))

    output *= (y_true * y_n + K.cast(K.equal(y_true, K.constant(0.)), K.floatx()) * y_p) / (y_n + y_p)
    output = K.mean(output)
    return output


def get_loss(class_weight=None):
    if class_weight is None:
        loss = binary_crossentropy
    else:
        def loss(y_true, y_pred):
            return weight_loss(y_true, y_pred, class_weight)
    return loss


METRICS: List = [binary_accuracy, binary_precision]

