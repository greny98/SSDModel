from preprocessing import create_ds
import ssd
import box_utils

label_to_idx = {
    'licence': 1,
    'plate': 1
}

configs = {
    'default_boxes': 4,
    'aspect_ratios': [1., 2., 3., 4.],
    'steps': [40, 20, 10, 5, 2, 1],
    'n_classes': 1,
    'image_size': 320,
    'min_scale': 0.15,
    'max_scale': 0.8
}

if __name__ == '__main__':
    ds = create_ds('data/alpr/annotations',
                   label_to_idx,
                   n_labels=configs['n_classes'],
                   dboxes=box_utils.create_default_boxes(configs),
                   batch_size=4,
                   image_dir='data/alpr/images')
    model = ssd.ssd(configs)
