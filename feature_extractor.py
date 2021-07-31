from libs import *

vgg16 = App.VGG16(include_top=False, weights='imagenet', input_shape=(320, 320, 3,))

layer_dict = {
    'block4': 'block4_conv3',  # (40,40,512)
    'block5': 'block5_conv3',  # (20,20,512)
}


def feature_extractor():
    """
    Create new model for extract features and get output of several layers
    :return:
    """
    feature_map_out = vgg16.output  # (10,10,512)
    feature_map_b5 = vgg16.get_layer(layer_dict['block5']).output
    print(feature_map_b5.shape)
    # Extend model
    extend_b1 = L.Conv2D(
        filters=256, kernel_size=3, activation='relu',
        padding='same', kernel_initializer=init.HeUniform())(feature_map_out)
    extend_b1 = L.MaxPool2D()(extend_b1)  # (5,5,256)
    extend_b2 = L.Conv2D(
        filters=256, kernel_size=3, activation='relu',
        padding='same', kernel_initializer=init.HeUniform())(extend_b1)
    extend_b2 = L.MaxPool2D()(extend_b2)  # (2,2,256)
    extend_b3 = L.Conv2D(
        filters=256, kernel_size=3, activation='relu',
        padding='same', kernel_initializer=init.HeUniform())(extend_b2)
    extend_b3 = L.MaxPool2D()(extend_b3)  # (1,1,256)
    outputs = [feature_map_b5, feature_map_out, extend_b1, extend_b2, extend_b3]
    return models.Model(vgg16.input, outputs=outputs, name='feature_extractor')

