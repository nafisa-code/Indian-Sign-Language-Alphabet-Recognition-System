import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
BASE_DIR = r"C:\ISL\dataset"
TEST_DIR = os.path.join(BASE_DIR, "test")
model = tf.keras.models.load_model(r"C:\ISL\isl_model.keras")
test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
loss, accuracy = model.evaluate(test_data)
print(f"\n Test Accuracy: {accuracy*100:.2f}%")
input("Press Enter to exit...")