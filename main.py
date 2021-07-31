from libs import *
import preprocessing as pre
import ssd
import box_utils
import argparse

label_to_idx = {
    'licence': 1,
    'plate': 1
}

configs = {
    'default_boxes': 5,
    'aspect_ratios': [1., 1.5, 2.5, 3.5, 4.5],
    'steps': [20, 10, 5, 2, 1],
    'n_classes': 1,
    'image_size': 320,
    'min_scale': 0.2,
    'max_scale': 0.8
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--xml_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    args = vars(parser.parse_args())
    print(args)
    dboxes = box_utils.create_default_boxes(configs)
    epochs = args['epochs']
    batch_size = args['batch_size']
    ds = pre.create_ds(args['xml_dir'], label_to_idx,
                       n_labels=configs['n_classes'],
                       dboxes=dboxes,
                       batch_size=batch_size,
                       image_dir=args['image_dir'])
    model = ssd.ssd(configs)

    for e in range(epochs):
        mean_loc_losses = []
        mean_conf_losses = []
        total_losses = []
        step = 1
        print(f'Epoch {e}')
        for images, gtruths in ds:
            losses = ssd.training_step_fn(model, images,
                                          gtruths, batch_size=batch_size)
            losses = [loss.numpy() for loss in losses]
            losses_dict = {
                'conf_loss': losses[0],
                'loc_loss': losses[1],
                'total_loss': losses[2]
            }
            print(f'Step {step}: ', losses_dict)
            mean_conf_losses.append(losses[0])
            mean_loc_losses.append(losses[1])
            total_losses.append(losses[2])
            step += 1
        losses_dict = {
            'conf_loss': np.average(mean_conf_losses),
            'loc_loss': np.average(mean_loc_losses),
            'total_loss': np.average(total_losses)
        }
        print('Average: ', losses_dict)
        print('======================')
    model.save('model.h5')
