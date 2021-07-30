from libs import *


def convert_coor_to_center(coors):
    """
    Convert (xmin, ymin, xmax, ymax) -> (cx, cy, w, h)
    """
    center = np.zeros_like(coors, dtype=np.float32)
    center[:, 0] = 0.5 * coors[:, 0] + 0.5 * coors[:, 2]  # cx
    center[:, 2] = 0.5 * coors[:, 1] + 0.5 * coors[:, 3]  # cy
    center[:, 1] = coors[:, 2] - coors[:, 0]  # w
    center[:, 3] = coors[:, 3] - coors[:, 1]  # h
    return center


def convert_center_to_coor(center):
    """
    Convert (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
    """
    coors = np.zeros_like(center, dtype=np.float32)
    coors[:, 0] = center[:, 0] - 0.5 * center[:, 2]  # xmin
    coors[:, 2] = center[:, 0] + 0.5 * center[:, 2]  # xmax
    coors[:, 1] = center[:, 1] - 0.5 * center[:, 3]  # ymin
    coors[:, 3] = center[:, 1] + 0.5 * center[:, 3]  # ymax
    return coors


def create_offset(truth, dbox):
    offset = np.zeros_like(truth, dtype=np.float)
    offset[0] = (truth[0] - dbox[0]) / dbox[2]  # cx
    offset[1] = (truth[1] - dbox[1]) / dbox[3]  # cy
    offset[2] = np.log(truth[2] / dbox[2])  # w
    offset[3] = np.log(truth[3] / dbox[3])  # h
    return offset


def create_default_boxes(configs):
    dboxes = []
    m = len(configs['steps'])
    min_scale = configs['min_scale']
    max_scale = configs['max_scale']
    sk_step = (max_scale - min_scale) / (m - 1)
    sks = [min_scale + sk_step * i for i in range(m)]
    for k, fk in enumerate(configs['steps']):
        for i, j in product(range(int(fk)), repeat=2):
            cx = (j + 0.5) / fk
            cy = (i + 0.5) / fk
            sk = sks[k]
            for ar in configs['aspect_ratios']:
                w = sk * np.sqrt(ar)
                h = sk / np.sqrt(ar)
                dboxes.append([cx, cy, w, h])
    return tf.clip_by_value(tf.convert_to_tensor(dboxes, dtype=tf.float32), clip_value_min=0., clip_value_max=1.)


def decode_bbox(dboxes, predicts):
    bboxes = tf.zeros_like(dboxes)
    bboxes[:, :, 0] = predicts[:, :, 0] * dboxes[:, :, 2] + dboxes[:, :, 0]
    bboxes[:, :, 1] = predicts[:, :, 1] * dboxes[:, :, 3] + dboxes[:, :, 1]
    bboxes[:, :, 2] = tf.exp(predicts[:, :, 2]) * dboxes[:, :, 2]
    bboxes[:, :, 3] = tf.exp(predicts[:, :, 3]) * dboxes[:, :, 3]
    return bboxes


def cal_iou(dbox, truth_boxes):
    dbox = tf.cast(dbox, dtype=tf.float32)
    dbox_coor = convert_center_to_coor(dbox.numpy())
    truth_coor = convert_center_to_coor(truth_boxes.numpy())
    inter = np.zeros_like(truth_boxes)
    inter[:, 0] = np.maximum(dbox_coor[:, 0], truth_coor[:, 0])  # xmin
    inter[:, 1] = np.maximum(dbox_coor[:, 1], truth_coor[:, 1])  # ymin
    inter[:, 2] = np.minimum(dbox_coor[:, 2], truth_coor[:, 2])  # xmax
    inter[:, 3] = np.minimum(dbox_coor[:, 3], truth_coor[:, 3])  # ymax
    inter_area = np.maximum(inter[:, 2] - inter[:, 0], 0) * np.maximum(inter[:, 3] - inter[:, 1], 0)
    area1 = (dbox_coor[:, 2] - dbox_coor[:, 0]) * (dbox_coor[:, 3] - dbox_coor[:, 1])
    area2 = (truth_coor[:, 2] - truth_coor[:, 0]) * (truth_coor[:, 3] - truth_coor[:, 1])
    union = area1 + area2 - inter_area
    return inter_area / union


def matching(truth_bboxes, dboxes):
    info = []
    for dbox in dboxes.numpy():
        ious = cal_iou(np.array([dbox]), truth_bboxes)
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        info.append((max_iou_idx, max_iou))
    return info
