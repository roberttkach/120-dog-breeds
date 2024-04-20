import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_dir(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def create_datagen():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )
    return datagen


def process_images(data_dir, datagen):
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=4,
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=4,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )
    return train_generator, validation_generator
