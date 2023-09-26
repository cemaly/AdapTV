from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet

# Constants
WIDTH, HEIGHT = 128, 128
TARGET_SIZE = (HEIGHT, WIDTH)
BATCH_SIZE = 64  # same as training batch_size, but you can modify if needed

# Load the saved model
model = load_model('trained_models/mobilenet_50_51_128.h5')

# Prepare test data using ImageDataGenerator
test_data_gen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

test_gen = test_data_gen.flow_from_directory(
    'data/grayscale/test',  # adjust to your test data directory
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

print(test_gen.num_classes)

# Evaluate the model
loss, accuracy = model.evaluate(test_gen)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
