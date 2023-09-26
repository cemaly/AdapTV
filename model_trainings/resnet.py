import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# Constants
width, height = 80, 80
target_size = (height, width)
num_channels = 3  # Assuming you have 3 channels after preprocessing
input_shape = (height, width, num_channels)
epochs = 20
lr = 0.0001
batch_size = 64
n_classes = 51

# Optimizer
opt = optimizers.Adam(learning_rate=lr)

# Base Model
base_model = resnet_v2.ResNet101V2(include_top=False,
                                   weights='imagenet',
                                   input_shape=input_shape)

# Model Extension
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.25)(x)
y = Dense(n_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=y)

# Data Augmentation and Preprocessing
train_data_generator = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=resnet_v2.preprocess_input
)

train_generator = train_data_generator.flow_from_directory(
    'data/grayscale/train',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Fine-tuning Setup
for layer in base_model.layers[:15]:
    layer.trainable = False

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# Convert class weights to a dictionary
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    class_weight=class_weight_dict,
    epochs=epochs
)

# Save the Model
model_name = 'resnet_{}_{}_{}.h5'.format(epochs, n_classes, width)
model.save('trained_models/' + model_name)
