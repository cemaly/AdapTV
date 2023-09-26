'''
--------------------------------------------------------------------------------
NOTE:

Although we examined and experimented with the models individually by hyper-tuning
the parameters, for the sake of simplicity, we have consolidated it into a single
file for hyperparameter tuning. It's essential to recognize that we treated every
model as a unique entity, endeavoring to find its optimal configuration.

Our intention was to avoid unnecessary complexities and provide a singular file
that can efficiently tune hyperparameters for all classifiers in use. However,
it's crucial to understand that this script will train numerous classifiers
and, consequently, will consume substantial computational resources.

Ensure you have adequate resources available prior to executing this script.
This will furnish us with a preliminary assessment to determine the most
apt classifier for our specific icon classification task at hand.

--------------------------------------------------------------------------------
'''


import os
import tensorflow as tf
from keras import Model
from keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet101V2, MobileNet, MobileNetV2, VGG16, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from kerastuner import RandomSearch
import pandas as pd


# Constants
TRAINING_DIR = 'data/grayscale/train'
NUM_CLASSES = len(os.listdir(TRAINING_DIR))

# Hyperparameter ranges
HP_IMAGE_SIZE = [40, 96, 128]
HP_BATCH_SIZE = [16, 32, 64]
HP_EPOCHS = [5, 10, 20, 35, 50]

# Model builder for Keras Tuner
def build_model(hp):
    image_size = hp.Choice('image_size', values=HP_IMAGE_SIZE)
    model_choice = hp.Choice(
        'model',
        values=['ResNet50', 'ResNet101V2', 'MobileNet', 'MobileNetV2', 'VGG16', 'Xception']
    )

    if model_choice == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_choice == 'ResNet101V2':
        base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_choice == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_choice == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_choice == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_choice == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    base_model.trainable = False

    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Data Augmentation setup
def setup_data(image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

# Create an empty dataframe to store results
results_df = pd.DataFrame(columns=['Model', 'Image Size', 'Batch Size', 'Epochs', 'Dense Layer Units', 'Learning Rate', 'Best Validation Accuracy', 'Training Accuracy'])

# Run tuning
for batch_size in HP_BATCH_SIZE:
    for image_size in HP_IMAGE_SIZE:
        for epochs in HP_EPOCHS:
            print(f"Training with batch_size={batch_size}, image_size={image_size}, epochs={epochs}")

            train_generator, validation_generator = setup_data(image_size, batch_size)

            tuner = RandomSearch(
                build_model,
                objective='val_accuracy',
                max_trials=5,
                directory='output_dir',
                project_name=f'batch_{batch_size}_img_{image_size}_ep_{epochs}'
            )

            tuner.search(train_generator,
                         epochs=epochs,
                         validation_data=validation_generator,
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

            best_hps = tuner.get_best_hyperparameters()[0]
            best_model_name = str(best_hps.get('model'))

            # Get training and validation accuracy
            best_model = tuner.get_best_models(num_models=1)[0]
            val_accuracy = best_model.evaluate(validation_generator, verbose=0)[1]
            train_accuracy = best_model.evaluate(train_generator, verbose=0)[1]

            # Save results to dataframe
            results_df = results_df.append({
                'Model': best_model_name,
                'Image Size': image_size,
                'Batch Size': batch_size,
                'Epochs': epochs,
                'Dense Layer Units': best_hps.get('units'),
                'Learning Rate': best_hps.get('learning_rate'),
                'Best Validation Accuracy': val_accuracy,
                'Training Accuracy': train_accuracy
            }, ignore_index=True)

# Save results dataframe to a CSV file
results_df.to_csv('models_hyperparameters_performance.csv', index=False)


