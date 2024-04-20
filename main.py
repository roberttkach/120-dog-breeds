import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from utils import get_dir, create_datagen, process_images

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

data_dir = r'data\images'
breed_folders = get_dir(data_dir)

datagen = create_datagen()
train_generator, validation_generator = process_images(data_dir, datagen)

model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),
    Flatten(),
    Dense(120, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=3
)

model.save('120dogs.h5')
X_test, y_test = validation_generator.next()

# ====================== VIEWING
model = tf.keras.models.load_model('120dogs.h5')
predictions = model.predict(X_test)
for i in range(len(predictions)):
    y_pred = np.argmax(predictions[i])
    print(f'Predicted: {y_pred} Real: {np.argmax(y_test[i])}')
