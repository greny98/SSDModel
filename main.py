from libs import *
import preprocessing as pre
import ssd
import box_utils

label_to_idx = {
    'licence': 1,
    'plate': 1
}

configs = {
    'default_boxes': 4,
    'aspect_ratios': [1.5, 2.5, 3.5, 4.5],
    'steps': [40, 20, 10, 5, 2, 1],
    'n_classes': 1,
    'image_size': 320,
    'min_scale': 0.15,
    'max_scale': 0.8
}

if __name__ == '__main__':
    dboxes = box_utils.create_default_boxes(configs)
    epochs = 2
    batch_size = 4
    ds = pre.create_ds('data/alpr/annotations', label_to_idx,
                       n_labels=configs['n_classes'],
                       dboxes=dboxes,
                       batch_size=batch_size,
                       image_dir='data/alpr/images')
    model = ssd.ssd(configs)

    for e in range(epochs):
        mean_loc_losses = []
        mean_conf_losses = []
        total_losses = []
        for images, gtruths in ds:
            losses = ssd.training_step_fn(model, images,
                                          gtruths, batch_size=batch_size)
            print('conf_loss:', losses[0])
            print('loc_loss: ', losses[1])
            print('total_loss:', losses[2])

