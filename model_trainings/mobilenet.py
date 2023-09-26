import numpy as np
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import BatchNormalization

# Constants
WIDTH, HEIGHT = 128, 128  # MobileNet default size is 224x224, but for icons, a smaller size might be sufficient
TARGET_SIZE = (HEIGHT, WIDTH)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001

# Data Augmentation and Preprocessing
train_data_gen = ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    validation_split=0.2,  # 20% of data as validation
    preprocessing_function=mobilenet.preprocess_input
)

train_gen = train_data_gen.flow_from_directory(
    'data/grayscale/train',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_data_gen.flow_from_directory(
    'data/grayscale/valid',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

N_CLASSES = train_gen.num_classes

# Model Architecture with increased complexity
base_model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(HEIGHT, WIDTH, 3))

# Extend the model
x = GlobalAveragePooling2D()(base_model.output)

x = Dense(4096, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)

x = Dense(2048, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)

x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

predictions = Dense(N_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the top layers of the base model
for layer in base_model.layers[:15]:  # freezing all but the last 40 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
checkpointer = ModelCheckpoint(filepath='trained_models/resnet_{}_{}_{}.h5'.format(EPOCHS, N_CLASSES, WIDTH), save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Model Training
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop, checkpointer, reduce_lr]
)

print("Training Completed!")

