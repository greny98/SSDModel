from feature_extractor import feature_extractor
from libs import *


def create_loc(feature_maps, configs):
    """
    Predict offset for default boxes from feature maps
    :param feature_maps:
    :param configs:
    :return:
    """
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
    """
    Predict confidents for default boxes from feature maps
    :param feature_maps:
    :param configs:
    :return:
    """
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
    """
    Combination of predicting offsets and confidents
    :param feature_maps:
    :param configs:
    :return:
    """
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
def training_step_fn(model: models.Model, images,
                     gtruths, alpha=1., batch_size=4,
                     optimizer=tf.keras.optimizers.Adam()):
    with tf.GradientTape() as tape:
        # From ground truths
        conf = gtruths['conf']
        loc = gtruths['loc']
        box_labels = gtruths['labels']
        # Predictions
        predicts = model(images)
        pred_conf = predicts['conf']
        pred_loc = predicts['loc']
        # Calc loc loss
        loc_loss = tf.keras.losses.huber(loc, pred_loc)
        loc_loss = loc_loss * box_labels
        mean_loc_loss = tf.reduce_sum(loc_loss, axis=-1, keepdims=True)
        mean_loc_loss = tf.reduce_mean(mean_loc_loss) * alpha
        # Calc confident loss
        conf_loss = tf.keras.losses.categorical_crossentropy(conf, pred_conf)
        pos_idx = tf.where(box_labels > 0)
        neg_idx = tf.where(box_labels <= 0)
        mean_conf_loss = []
        for i in range(batch_size):
            # Calc positive conf
            pos_batch = pos_idx[pos_idx[:, 0] == i][:, 1]
            pos_loss = tf.gather(conf_loss[i, :], pos_batch)
            pos_loss = tf.reduce_sum(pos_loss)
            # Calc negative conf (3 * n_pos)
            neg_batch = neg_idx[neg_idx[:, 0] == i][:, 1]
            neg_loss = tf.gather(conf_loss[i, :], neg_batch)
            neg_loss = tf.sort(neg_loss, direction='DESCENDING')
            neg_loss = tf.reduce_sum(neg_loss[:300])
            total_conf_loss = pos_loss + neg_loss
            mean_conf_loss.append(total_conf_loss)
        mean_conf_loss = tf.reduce_mean(mean_conf_loss)
        total_loss = mean_conf_loss + mean_loc_loss
        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return mean_conf_loss, mean_loc_loss, total_loss
