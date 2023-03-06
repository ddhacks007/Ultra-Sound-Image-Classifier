import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import argparse
import properties
import uuid
np.random.seed(123)
tf.random.set_seed(123)



def get_model():
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(properties.img_w, properties.img_h, properties.depth))

    for layer in vgg_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    model = get_model()
    img_width, img_height = properties.img_w, properties.img_h
    train_data_dir = args['data_dir']
    validation_data_dir = args['data_dir']
    epochs = properties.epochs
    batch_size = properties.batch_size
    dsplit=properties.dsplit
    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=properties.rotation_range,  
        width_shift_range=properties.width_shift_range,  
        height_shift_range=properties.height_shift_range,  
        shear_range=properties.shear_range,  
        zoom_range=properties.zoom_range,  
        horizontal_flip=properties.should_hflip,  
        fill_mode=properties.fill_mode,  
        validation_split=properties.dsplit)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',

        )

    validation_datagen = ImageDataGenerator(rescale=1./255.,
        preprocessing_function = None,validation_split=dsplit)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
        )
    monitor = 'val_accuracy'
    model_weight_path = f'vgg_cam_v2-{uuid.uuid4()}.h5'
    checkpoint = ModelCheckpoint(model_weight_path, monitor=monitor, save_best_only=True, mode='max', verbose=1)
    print(model_weight_path, ' is the model weight path')
    model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])


def eval(img, weight_path):
    model = get_model()
    model.load_weights(weight_path)
    return model.predict(img).argmax(axis=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-d', '--data_dir', required=True, help='path to input dataset'
    )
    args = vars(ap.parse_args())
    train()
