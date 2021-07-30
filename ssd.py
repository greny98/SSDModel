from feature_extractor import feature_extractor
from libs import *


def create_loc(feature_maps, configs):
    locs = []
    for i in range(len(feature_maps)):
        predict = L.Conv2D(
            filters=4 * len(configs['aspect_ratios']),
            kernel_size=3, activation='relu', padding='same',
            kernel_regularizer=init.HeNormal())(feature_maps[i])
        n_samples, h, w, c = predict.get_shape()
        predict = L.Reshape(target_shape=(h * w * int(c / 4), 4))(predict)
        locs.append(predict)
    locs_tensor = L.Concatenate(axis=1, name='location')(locs)
    locs_tensor = tf.clip_by_value(locs_tensor, clip_value_min=0., clip_value_max=1.)
    return locs_tensor


def create_conf(feature_maps, configs):
    conf = []
    for i in range(len(feature_maps)):
        predict = L.Conv2D(
            filters=(configs['n_classes'] + 1) * configs['default_boxes'],
            kernel_size=3, activation='relu', padding='same',
            kernel_regularizer=init.HeNormal())(feature_maps[i])
        n_samples, h, w, c = predict.get_shape()
        predict = L.Reshape(target_shape=(h * w * int(c / 2), 2))(predict)
        conf.append(predict)
    conf_tensor = L.Concatenate(axis=1)(conf)
    conf_tensor = L.Softmax(name='confident')(conf_tensor)
    return conf_tensor


def detector(feature_maps, configs=None):
    if configs is None:
        configs = configs
    conf = create_conf(feature_maps, configs)
    loc = create_loc(feature_maps, configs)
    return conf, loc


def ssd(configs):
    inputs = L.Input(shape=(320, 320, 3,))
    feature_maps = feature_extractor()(inputs)
    conf, loc = detector(feature_maps, configs)
    return models.Model(inputs, outputs={'conf': conf, 'loc': loc})


@tf.function
def training_step_fn(image, gtruth, dboxes):
    pass
