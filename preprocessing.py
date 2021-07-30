from lxml import etree
import os
import numpy as np
import tensorflow as tf

from box_utils import convert_coor_to_center, matching, create_offset


def convert_xml_to_dict(xml_path, image_dir=None, label_to_idx=None):
    """
    Read XML file, get info of image and save into dictionary
        - Read XML file
        - Get following info:
            + folder + filename => image path
            + width, height
            + xmin, ymin, xmax, ymax => cx, cy, w, h => normalize
            + label (categorical), name (text)
    :param xml_path:
    :param image_dir:
    :param label_to_idx:
    :return:
    """
    image_info = {}
    tree = etree.parse(xml_path).getroot()
    # get img_path
    filename = tree.find('filename').text
    if image_dir is None:
        image_dir = tree.find('folder').text
    image_info['path'] = os.path.join(image_dir, filename)
    # size image
    size = tree.find('size')
    image_info['width'] = int(size.find('width').text)
    image_info['height'] = int(size.find('height').text)
    # boxes
    objects = tree.findall('object')
    names = []
    label_idx = []
    bboxes = []
    for o in objects:
        name = o.find('name').text
        names.append(name)
        label_idx.append(label_to_idx[name])
        bbox = o.find('bndbox')
        xmin = float(bbox.find('xmin').text) / image_info['width']
        xmax = float(bbox.find('xmax').text) / image_info['width']
        ymin = float(bbox.find('ymin').text) / image_info['height']
        ymax = float(bbox.find('ymax').text) / image_info['height']
        center_box = convert_coor_to_center(np.array([[xmin, ymin, xmax, ymax]]))
        bboxes.append(center_box[0])
    image_info['bboxes'] = bboxes
    image_info['labels'] = label_idx
    image_info['label_names'] = names
    return image_info


def preprocessing_image(img_path):
    """
    Preprocessing image
        + Read file => decode jpeg => cast to float
        + Resize (320, 320)
        + Normalize: image / 127.5 - 1.
    :param img_path:
    :return:
    """
    raw_img = tf.io.read_file(img_path)
    decoded = tf.image.decode_jpeg(raw_img, channels=3)
    decoded = tf.cast(decoded, dtype=tf.float32)
    preprocessed_img = tf.image.resize(decoded, size=(320, 320))
    preprocessed_img = preprocessed_img / 127.5 - 1
    return preprocessed_img


class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom DataGenerator
        - Initialize with
            + List of info
            + Batch Size
            + Default boxes: Used to calc offset from truth boxes
            and label for each box
            + Training or not
        - Get mini-batch: Each selected index, read image => matching to find best truth box for
        each default box => calc offsets and label for each default box from truth box
        - Shuffle (indexes) on end epoch
    """
    def __init__(self, info_dicts, batch_size, dboxes, n_labels=1, is_training=True):
        self.info_dicts = info_dicts
        self.batch_size = batch_size
        self.is_training = is_training
        self.dboxes = dboxes
        self.samples = np.arange(0, len(info_dicts)).astype(np.int)
        self.n_labels = n_labels + 1

    def __len__(self):
        return np.ceil(len(self.info_dicts) / self.batch_size).astype(np.int)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        return self.__data_generation(start_idx, end_idx)

    def __data_generation(self, start_idx, end_idx):
        selected = self.samples[start_idx: end_idx]
        images = []
        batch_offsets = []
        batch_conf = []
        for i in selected:
            info = self.info_dicts[i]
            image = preprocessing_image(info['path'])
            bboxes = info['bboxes']
            labels = info['labels']
            matching_boxes = matching(tf.convert_to_tensor(bboxes), self.dboxes)
            offsets = []
            conf = []
            for i, dbox in enumerate(self.dboxes):
                idx, iou = matching_boxes[i]
                offsets.append(create_offset(bboxes[idx], dbox))
                if iou >= 0.5:
                    conf.append(tf.one_hot(labels[idx], depth=self.n_labels))
                else:
                    conf.append(tf.one_hot(0, self.n_labels))
            batch_offsets.append(offsets)
            batch_conf.append(conf)
            images.append(image)
        return images, {'loc': tf.convert_to_tensor(batch_offsets),
                        'conf': tf.convert_to_tensor(batch_conf, dtype=tf.int32)}

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.samples)


def create_ds(xml_dir, label_to_idx, n_labels,
              dboxes, image_dir=None, batch_size=None, is_training=True):
    """
    Create Data generator from XML
        - Create empty list to store info of images
        - Read each .xml file in xml_dir
        - Convert info to dictionary (convert_xml_to_dict)
        - Append info to list
        - Create DataGenerator from
            + info list
            + batch_size
            + # of labels (used to one hot label)
    :param xml_dir:
    :param label_to_idx:
    :param image_dir:
    :param batch_size:
    :param is_training:
    :return:
    """
    image_infos = []

    # Read and handle xml file
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        image_info = convert_xml_to_dict(
            xml_path=os.path.join(xml_dir, xml_file),
            image_dir=image_dir,
            label_to_idx=label_to_idx)
        image_infos.append(image_info)

    # Create dataset
    ds = DataGenerator(
        image_infos, n_labels=n_labels,
        batch_size=batch_size, is_training=is_training,
        dboxes=dboxes)
    return ds
